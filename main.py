import cv2
import numpy as np
import yaml
import json
import gzip
import logging
import time
import threading
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import sys
from datetime import datetime

# Add src/ to sys.path so all core modules are loaded seamlessly
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.camera_service import CameraService, CameraConfig
from src.detector import YOLODetector
from src.classifier import Classifier
from src.conveyor_controller import ConveyorController
import src.db_manager as db
import src.image_logger as image_logger

# Anomaly models
from src.draem_model import DRAEM


class _NoConveyor:
    sock = None
    host = "N/A"
    port = 0
    def send(self, cmd): return False
    def disconnect(self): pass
    def is_connected(self): return False
    def reconnect(self): return False
    def wait_for_inspection(self, timeout=None): return False
    def send_result(self, result): return False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("main")

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cam_cfg = CameraConfig(
    rtsp_url=cfg["camera"]["rtsp_url"],
    width=cfg["camera"].get("width", 1280),
    height=cfg["camera"].get("height", 720),
    fps=cfg["camera"].get("fps", 15),
)

camera = CameraService(cam_cfg)
detector = YOLODetector(
    model_path=cfg["yolo"]["model"],
    conf=cfg["yolo"]["confidence"],
    iou=cfg["yolo"].get("iou", 0.45),
    device=cfg["yolo"].get("device", "cuda:0"),
    half=cfg["yolo"].get("half", True),
    max_det=cfg["yolo"].get("max_det", 1),
    imgsz=cfg["yolo"].get("imgsz", 960),
)

anomaly_algo = "draem"
draem = None

if "draem" in cfg:
    try:
        draem = DRAEM(
            model_path=cfg["draem"]["model"],
            threshold=cfg["draem"].get("threshold", 0.25),
            device=cfg["draem"].get("device", "cuda:0"),
        )
        logger.info("DRAEM model initialized")
    except Exception as e:
        logger.error(f"Failed to load DRAEM model: {e}")

classifier = Classifier()
conveyor = ConveyorController(
    host=cfg["esp"]["ip"],
    port=cfg["esp"].get("conveyor_port", 5000),
)
conveyor.connect()
if not conveyor.sock:
    logger.warning("ESP not connected at startup, will keep reconnecting in background...")
else:
    logger.info(f"ESP connected at {conveyor.host}:{conveyor.port}")

# Initialize database and restore active session
db.init_db()
active_session = db.get_or_create_active_session()
sess_stats = db.get_session_stats(active_session["id"])
classifier.restore_counts(sess_stats["last_can_number"], sess_stats["good"], sess_stats["ng"])
history = deque(db.get_session_records(active_session["id"]), maxlen=500)
logger.info(f"Loaded active test session #{active_session['id']}: {active_session['name']} (Count: {sess_stats['total']}/{active_session['target_count']})")

capture_dir = Path("captures")
capture_dir.mkdir(exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
results_file = results_dir / f"inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
results_fh = open(results_file, "a", encoding="utf-8")
running = True
processing_enabled = True
conveyor_running = False

display_frame = None
display_lock = threading.Lock()
last_verdict = ""
last_status_text = "Starting..."

DETECT_CONF = cfg["yolo"].get("confidence", 0.45)
inspect_delay = cfg["camera"].get("inspect_delay", 1.0)

# MJPEG stream tuning. A full-resolution 30 FPS MJPEG feed easily needs
# 3-5 MB/s, which saturates Tailscale Funnel / home upload links and makes
# remote dashboards lag. Lower these values if the stream is still choppy.
SERVER_CFG = cfg.get("server", {})
STREAM_WIDTH = int(SERVER_CFG.get("stream_width", 640))  # 0 = keep original size
STREAM_FPS = max(1.0, float(SERVER_CFG.get("stream_fps", 8)))
STREAM_QUALITY = int(SERVER_CFG.get("stream_quality", 65))


def save_inspection_result(data):
    results_fh.write(json.dumps(data) + "\n")
    results_fh.flush()
    try:
        db.save_record(
            session_id=active_session["id"],
            can_number=data.get("id", 0),
            timestamp=data.get("timestamp", datetime.now().strftime("%H:%M:%S")),
            verdict=data.get("verdict", "NG"),
            score=data.get("score", 0.0),
            defects=data.get("defects", []),
            details=data.get("details", {}),
            image_file=data.get("image_file")
        )
    except Exception as e:
        logger.error(f"Error saving to sqlite: {e}")

def update_display(frame, text, color=(255, 255, 255)):
    global display_frame
    if frame is None:
        return
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    with display_lock:
        display_frame = frame


def do_inspection(trigger_time: float, retry_detect: bool = True):
    global display_frame, last_verdict, last_status_text, conveyor_running

    logger.info(f"WAITING_INSPECTION received, waiting {inspect_delay:.2f}s for product to reach inspection point")
    last_status_text = "Product arriving..."
    wait_until = trigger_time + inspect_delay
    while running:
        remaining = wait_until - time.time()
        if remaining <= 0:
            break
        conveyor.read_line(timeout=0.05)
        if not conveyor.is_connected():
            break
        frame = camera.get_frame()
        update_display(frame, "Product arriving...", (0, 200, 255))
        time.sleep(0.02)
    if not conveyor.is_connected():
        return

    last_status_text = "Inspecting..."

    frame = camera.get_frame_near(wait_until)
    if frame is None:
        frame = camera.get_frame()
    if frame is None:
        logger.warning("No camera frame available -> sending NG")
        conveyor.send("NG")
        time.sleep(1.0)
        conveyor.clear_buffer()
        return
    update_display(frame.copy(), "Inspecting...", (0, 200, 255))

    dets = detector.detect(frame)

    if not dets and retry_detect:
        logger.info("No detection on first attempt, retrying once...")
        time.sleep(0.3)
        frame = camera.get_frame()
        if frame is not None:
            dets = detector.detect(frame)

    if not dets:
        logger.warning("No product detected in inspection zone (False trigger) -> Sending OK to clear gate safely")
        conveyor.send("OK")
        last_verdict = ""
        last_status_text = "Kich hoat ao (Khong co lon) -> Mo can gat"
        if frame is not None:
            cv2.putText(frame, "NO CAN DETECTED -> AUTO CLEAR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
            with display_lock:
                display_frame = frame
        time.sleep(1.0)
        conveyor.clear_buffer()
        return

    det = dets[0]
    conf = det["confidence"]
    roi = det.get("roi")
    mask = det.get("mask")

    if conf < DETECT_CONF or roi is None or roi.size == 0:
        logger.warning(f"Low confidence or empty ROI (conf={conf:.3f}) -> NG")
        conveyor.send("NG")
        last_verdict = "NG"
        last_status_text = f"Low conf: {conf:.2f} -> NG"
        ins_result = classifier.classify(roi if (roi is not None and roi.size > 0) else np.zeros((100, 100, 3), dtype=np.uint8), {
            "algo": "yolo", "defect": True, "score": 1.0, "reason": f"YOLO confidence {conf:.3f} < threshold {DETECT_CONF}"
        })
        result_data = {
            "id": ins_result.product_id,
            "timestamp": datetime.fromtimestamp(ins_result.timestamp).strftime("%H:%M:%S"),
            "verdict": "NG",
            "yolo_confidence": round(conf, 4),
            "anomaly_score": 1.0,
            "draem_score": None,
            "defects": ["low_confidence"],
            "score": 1.0,
            "details": {"reason": f"YOLO confidence {conf:.3f} < threshold {DETECT_CONF}"},
        }
        history.append(result_data)
        save_inspection_result(result_data)
        if frame is not None:
            cv2.putText(frame, f"LOW CONF (NG) {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with display_lock:
                display_frame = frame
        time.sleep(1.5)
        conveyor.clear_buffer()
        return

    # Anomaly inference (DRAEM)
    if draem is not None:
        model_res = draem.check(roi, mask=mask, need_heatmap=True)
        model_tag = "DRAEM"
    else:
        model_res = {"algo": "none", "defect": False, "score": 0.0, "reason": "No model loaded"}
        model_tag = "NONE"

    ins_result = classifier.classify(roi, model_res)
    verdict = ins_result.details["verdict"]

    last_verdict = verdict
    last_status_text = f"{verdict}"

    conveyor.send("OK" if verdict == "GOOD" else "NG")
    conveyor_running = (verdict == "GOOD")

    x1, y1, x2, y2 = det["bbox"]
    color = (0, 255, 0) if verdict == "GOOD" else (0, 0, 255)
    label = f"{model_tag}:{model_res['score']:.2f} {verdict}"

    result_display = frame
    cv2.rectangle(result_display, (x1, y1), (x2, y2), color, 3)
    cv2.putText(result_display, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    heatmap = model_res.get("heatmap")
    if verdict == "NG" and heatmap is not None:
        hm = cv2.resize(heatmap, (x2 - x1, y2 - y1))
        result_display[y1:y2, x1:x2] = hm

    with display_lock:
        display_frame = result_display

    card_filename = None
    try:
        threshold_val = draem.threshold if draem else 0.25
        card_filename = image_logger.create_scientific_analysis_card(
            can_number=ins_result.product_id,
            verdict=verdict,
            score=float(model_res.get("score", 0.0)),
            threshold=threshold_val,
            yolo_conf=float(conf),
            roi=roi,
            recon_img=model_res.get("recon_img"),
            discr_map=model_res.get("discr_map"),
            pure_heatmap=model_res.get("pure_heatmap"),
            heatmap_overlay=(model_res.get("overlay") if model_res.get("overlay") is not None else heatmap),
            raw_frame=frame,
            timestamp_str=datetime.fromtimestamp(ins_result.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info(f"Saved analysis card for Can #{ins_result.product_id} ({verdict}): {card_filename}")
    except Exception as e:
        logger.error(f"Error creating scientific card: {e}")

    result_data = {
        "id": ins_result.product_id,
        "timestamp": datetime.fromtimestamp(ins_result.timestamp).strftime("%H:%M:%S"),
        "verdict": verdict,
        "yolo_confidence": round(conf, 4),
        "anomaly_score": model_res["score"],
        "draem_score": model_res["score"] if anomaly_algo == "draem" else None,
        "defects": ins_result.details["defects_found"],
        "score": ins_result.score,
        "details": ins_result.details["algorithm_results"],
        "image_file": card_filename,
    }
    history.append(result_data)

    save_inspection_result(result_data)

    # Cooldown: cho lon vua kiem tra di qua khoi camera va vung thanh chan hoan toan
    cooldown_s = 2.0
    cooldown_until = time.time() + cooldown_s
    while running and time.time() < cooldown_until:
        f = camera.get_frame()
        tag = "GOOD -> PASSING" if verdict == "GOOD" else "NG -> EJECTING"
        update_display(f, f"{tag}", (0, 255, 0) if verdict == "GOOD" else (0, 0, 255))
        time.sleep(0.05)

    # Xoa sach toan bo tin nhan ton dong trong socket TCP
    # (triet tieu 100% loi kich hoat trung lap 2 lan cho cung 1 chiec lon)
    conveyor.clear_buffer()


def inspection_worker():
    global display_frame, last_verdict, last_status_text, conveyor_running
    retry_detect = True

    while running:
        if not processing_enabled:
            last_status_text = "Processing disabled (Live preview only)"
            if conveyor.is_connected():
                msg = conveyor.read_line(timeout=0.01)
                if msg and ("TRIGGER_CAMERA" in msg or "WAITING_INSPECTION" in msg):
                    logger.info("Inspection disabled: letting can pass without grading")
                    conveyor.send("OK")
            frame = camera.get_frame()
            update_display(frame, "Live Preview (AI Idle)", (100, 200, 255))
            time.sleep(0.05)
            continue

        if not conveyor.is_connected():
            last_status_text = "ESP disconnected, reconnecting..."
            frame = camera.get_frame()
            update_display(frame, "ESP disconnected - live view", (100, 100, 255))
            if not conveyor.reconnect():
                time.sleep(1)
                continue
            logger.info("ESP reconnected")

        last_status_text = "Waiting for product..."

        trigger_time = None
        while running and processing_enabled:
            msg = conveyor.read_line(timeout=0.02)
            if msg:
                if "TRIGGER_CAMERA" in msg or "WAITING_INSPECTION" in msg:
                    logger.info("ESP Trigger received: %s", msg)
                    trigger_time = time.time()
                    break
                elif "[DIAG]" in msg or "STATE=" in msg or "SLOW_" in msg:
                    logger.info("ESP: %s", msg)
            if not conveyor.is_connected():
                break
            frame = camera.get_frame()
            update_display(frame, "Waiting for product signal from ESP...", (200, 200, 200))

        if not processing_enabled or trigger_time is None:
            continue

        try:
            do_inspection(trigger_time, retry_detect)
        except Exception as e:
            logger.exception("Error in inspection cycle: %s", e)
            last_verdict = "NG"
            last_status_text = f"Error: {e}"
            ins_result = classifier.classify(np.zeros((100, 100, 3), dtype=np.uint8), {
                "algo": "system", "defect": True, "score": 1.0, "reason": f"Inspection error: {e}"
            })
            result_data = {
                "id": ins_result.product_id,
                "timestamp": datetime.fromtimestamp(ins_result.timestamp).strftime("%H:%M:%S"),
                "verdict": "NG",
                "yolo_confidence": 0,
                "anomaly_score": 1.0,
                "draem_score": None,
                "defects": ["system_error"],
                "score": 1.0,
                "details": {"reason": f"System error: {e}"},
            }
            history.append(result_data)
            save_inspection_result(result_data)
            try:
                conveyor.send("NG")
            except Exception:
                pass
            time.sleep(1.0)
            conveyor.clear_buffer()


inspection_thread = threading.Thread(target=inspection_worker, daemon=True)
inspection_thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if conveyor.sock:
        logger.info(f"Conveyor controller connected to {conveyor.host}:{conveyor.port}")
    logger.info("System started")
    yield
    results_fh.close()
    logger.info("System stopped")


app = FastAPI(title="Industrial Defect Detection", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/captures", StaticFiles(directory="captures"), name="captures")


stream_jpeg = None
stream_seq = 0
stream_cond = threading.Condition()


def stream_encoder_worker():
    """Encode the latest dashboard frame once and share it with all viewers.

    Encoding inside every StreamingResponse generator would use one CPU core
    per viewer and upload the full-resolution feed. One lower-bitrate stream
    keeps remote (Internet) dashboards responsive.
    """
    global stream_jpeg, stream_seq
    last_frame = None
    interval = 1.0 / STREAM_FPS
    while running:
        started = time.time()
        with display_lock:
            frame = display_frame
        if frame is None or frame is last_frame:
            time.sleep(0.02)
            continue
        last_frame = frame

        if STREAM_WIDTH > 0 and frame.shape[1] > STREAM_WIDTH:
            new_h = max(1, int(frame.shape[0] * STREAM_WIDTH / frame.shape[1]))
            frame = cv2.resize(frame, (STREAM_WIDTH, new_h), interpolation=cv2.INTER_AREA)

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        if ok:
            with stream_cond:
                stream_jpeg = jpeg.tobytes()
                stream_seq += 1
                stream_cond.notify_all()

        elapsed = time.time() - started
        if elapsed < interval:
            time.sleep(interval - elapsed)


def frame_generator():
    last_seq = -1
    while running:
        with stream_cond:
            if stream_seq == last_seq:
                stream_cond.wait(timeout=1.0)
                continue
            data = stream_jpeg
            last_seq = stream_seq
        if data:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                   + data + b"\r\n")


threading.Thread(target=stream_encoder_worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def index():
    static_dir = Path("static")
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/video")
async def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/status")
async def get_status():
    fps = camera.get_fps()
    return {
        "good": classifier.good_count,
        "ng": classifier.ng_count,
        "total": classifier.product_count,
        "rate": round(classifier.good_count / classifier.product_count * 100, 1) if classifier.product_count > 0 else 100,
        "fps": fps,
        "conveyor": "running" if conveyor_running else "stopped",
        "conveyor_connected": conveyor.is_connected(),
        "processing": processing_enabled,
        "status_text": last_status_text,
        "last_verdict": last_verdict,
        "algo": "draem",
        "draem_threshold": draem.threshold if draem else 0.25,
        "conf_threshold": detector.conf,
        "session_id": active_session["id"],
        "session_name": active_session["name"],
        "target_count": active_session.get("target_count", 100),
    }


@app.get("/history")
async def get_history(request: Request):
    # Compact JSON + gzip keeps the 2-5s polling cheap on slow/mobile links.
    payload = json.dumps(list(history), separators=(",", ":")).encode("utf-8")
    headers = {"Cache-Control": "no-store"}
    if len(payload) > 1024 and "gzip" in request.headers.get("accept-encoding", "").lower():
        payload = gzip.compress(payload, compresslevel=5)
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
    return Response(content=payload, media_type="application/json", headers=headers)


@app.post("/session/new")
async def new_session(data: dict = {}):
    global active_session
    name = data.get("name")
    target = int(data.get("target_count", 100))
    active_session = db.create_new_session(name=name, target_count=target)
    classifier.reset()
    history.clear()
    logger.info(f"Created new session: {active_session['name']} (target: {target})")
    return {"status": "ok", "session": active_session}


@app.get("/session/export")
async def export_csv():
    from fastapi.responses import Response
    csv_data = db.export_session_csv_text(active_session["id"])
    filename = f"report_session_{active_session['id']}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/record/delete_last")
async def delete_last_record():
    deleted = db.delete_last_record(active_session["id"])
    if not deleted:
        return {"status": "error", "message": "Không có lon nào để xóa"}
    sess_stats = db.get_session_stats(active_session["id"])
    classifier.restore_counts(sess_stats["last_can_number"], sess_stats["good"], sess_stats["ng"])
    global history
    history = deque(db.get_session_records(active_session["id"]), maxlen=500)
    logger.info(f"Deleted last record #{deleted['can_number']} ({deleted['verdict']}). Restored count to {sess_stats['total']}")
    return {"status": "ok", "deleted": deleted, "stats": sess_stats}


@app.post("/control/start")
async def control_start():
    global processing_enabled
    processing_enabled = True
    if conveyor.is_connected():
        conveyor.send("AI_ON")
    logger.info("AI inspection enabled")
    return {"status": "started"}


@app.post("/control/stop")
async def control_stop():
    global processing_enabled
    processing_enabled = False
    if conveyor.is_connected():
        conveyor.send("AI_OFF")
    logger.info("AI inspection disabled")
    return {"status": "stopped"}


@app.post("/conveyor/start")
async def conveyor_start():
    global conveyor_running
    if not conveyor.is_connected():
        return {"status": "error", "message": "Conveyor controller not connected"}
    conveyor.send("RESET")
    conveyor.send("START")
    conveyor_running = True
    logger.info("Conveyor started (fault cleared)")
    return {"status": "ok", "message": "Conveyor started"}


@app.post("/conveyor/stop")
async def conveyor_stop():
    global conveyor_running
    if not conveyor.is_connected():
        return {"status": "error", "message": "Conveyor controller not connected"}
    conveyor.send("STOP")
    conveyor_running = False
    logger.info("Conveyor stopped")
    return {"status": "ok", "message": "Conveyor stopped"}


@app.get("/conveyor/status")
async def conveyor_status():
    return {
        "connected": conveyor.is_connected(),
        "running": conveyor_running,
        "host": conveyor.host,
        "port": conveyor.port,
    }


@app.post("/control/threshold")
async def set_threshold(data: dict):
    global anomaly_algo, DETECT_CONF
    algo = data.get("algo", "")
    value = float(data.get("value", 0.4))
    if algo == "draem" and draem is not None:
        draem.threshold = value
        cfg["draem"]["threshold"] = value
        logger.info(f"Updated DRAEM threshold to {value:.4f}")
    elif algo == "conf":
        detector.conf = value
        DETECT_CONF = value
        cfg["yolo"]["confidence"] = value
        logger.info(f"Updated YOLO confidence threshold to {value:.4f}")
    try:
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except Exception as e:
        logger.warning(f"Could not save config.yaml: {e}")
    return {"algo": algo, "value": value}


@app.post("/reset")
async def reset_counter():
    global active_session
    active_session = db.create_new_session(target_count=active_session.get("target_count", 100))
    classifier.reset()
    history.clear()
    if conveyor.is_connected():
        conveyor.send("RESET")
    return {"status": "reset", "session": active_session}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
