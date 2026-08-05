import cv2
import numpy as np
import yaml
import json
import logging
import time
import threading
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime

from camera_service import CameraService, CameraConfig
from detector import YOLODetector
from patch_core import PatchCore
from classifier import Classifier
from conveyor_controller import ConveyorController


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
patch_core = PatchCore(
    model_path=cfg["patchcore"]["model"],
    threshold=cfg["patchcore"]["threshold"],
    device=cfg["patchcore"].get("device", "cuda:0"),
)
classifier = Classifier()
conveyor = ConveyorController(
    host=cfg["esp"]["ip"],
    port=cfg["esp"].get("conveyor_port", 5000),
)
conveyor.connect()
if not conveyor.sock:
    logger.warning("ESP not available, running without conveyor")
    conveyor = _NoConveyor()
else:
    logger.info(f"ESP connected at {conveyor.host}:{conveyor.port}")

history = deque(maxlen=100)
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

DETECT_CONF = 0.97
inspect_delay = cfg["camera"].get("inspect_delay", 1.0)


def save_inspection_result(data):
    results_fh.write(json.dumps(data) + "\n")
    results_fh.flush()

def update_display(frame, text, color=(255, 255, 255)):
    global display_frame
    if frame is None:
        return
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    with display_lock:
        display_frame = frame


def inspection_worker():
    global display_frame, last_verdict, last_status_text, conveyor_running
    retry_detect = True

    while running:
        if not processing_enabled:
            last_status_text = "Processing disabled"
            frame = camera.get_frame()
            update_display(frame, "Processing disabled", (0, 0, 255))
            time.sleep(0.1)
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
        while running:
            msg = conveyor.read_line(timeout=0.02)
            if msg and ("TRIGGER_CAMERA" in msg or "WAITING_INSPECTION" in msg):
                trigger_time = time.time()
                break
            if not conveyor.is_connected():
                break
            frame = camera.get_frame()
            update_display(frame, "Waiting for product signal from ESP...", (200, 200, 200))

        if trigger_time is None:
            continue

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
            continue

        last_status_text = "Inspecting..."

        frame = camera.get_frame_near(wait_until)
        if frame is None:
            frame = camera.get_frame()
        if frame is None:
            continue
        update_display(frame.copy(), "Inspecting...", (0, 200, 255))

        dets = detector.detect(frame)

        if not dets and retry_detect:
            logger.info("No detection on first attempt, retrying once...")
            time.sleep(0.3)
            frame = camera.get_frame()
            if frame is not None:
                dets = detector.detect(frame)

        if not dets:
            logger.warning("No product detected after retry -> NG")
            conveyor.send("NG")
            last_verdict = "NG"
            last_status_text = "No detection -> NG"
            save_inspection_result({
                "id": classifier.product_count + 1,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "verdict": "NG",
                "yolo_confidence": 0,
                "patchcore_score": None,
                "defects": ["no_detection"],
                "score": 1.0,
                "details": {"reason": "No product detected by YOLO after retry"},
            })
            if frame is not None:
                cv2.putText(frame, "NO DETECT (NG)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with display_lock:
                    display_frame = frame
            continue

        det = dets[0]
        conf = det["confidence"]
        roi = det.get("roi")

        if conf < DETECT_CONF or roi is None or roi.size == 0:
            logger.warning(f"Low confidence or empty ROI (conf={conf:.3f}) -> NG")
            conveyor.send("NG")
            last_verdict = "NG"
            last_status_text = f"Low conf: {conf:.2f} -> NG"
            save_inspection_result({
                "id": classifier.product_count + 1,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "verdict": "NG",
                "yolo_confidence": round(conf, 4),
                "patchcore_score": None,
                "defects": ["low_confidence"],
                "score": 1.0,
                "details": {"reason": f"YOLO confidence {conf:.3f} < threshold {DETECT_CONF}"},
            })
            if frame is not None:
                cv2.putText(frame, f"LOW CONF (NG) {conf:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with display_lock:
                    display_frame = frame
            continue

        pc_result = patch_core.check(roi, need_heatmap=True)
        ins_result = classifier.classify(roi, pc_result)
        verdict = ins_result.details["verdict"]

        last_verdict = verdict
        last_status_text = f"{verdict}"

        conveyor.send("OK" if verdict == "GOOD" else "NG")
        conveyor_running = (verdict == "GOOD")

        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if verdict == "GOOD" else (0, 0, 255)
        label = f"PC:{pc_result['score']:.2f} {verdict}"

        result_display = frame
        cv2.rectangle(result_display, (x1, y1), (x2, y2), color, 3)
        cv2.putText(result_display, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        heatmap = pc_result.get("heatmap")
        if verdict == "NG" and heatmap is not None:
            hm = cv2.resize(heatmap, (x2 - x1, y2 - y1))
            result_display[y1:y2, x1:x2] = cv2.addWeighted(
                result_display[y1:y2, x1:x2], 0.6, hm, 0.4, 0)

        with display_lock:
            display_frame = result_display

        if verdict == "NG":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.imwrite(str(capture_dir / f"{ts}_result.jpg"), result_display)
            if heatmap is not None:
                cv2.imwrite(str(capture_dir / f"{ts}_heatmap.jpg"), heatmap)
            cv2.imwrite(str(capture_dir / f"{ts}_roi.jpg"), roi)
            logger.info(f"Saved NG capture: {ts}")

        result_data = {
            "id": ins_result.product_id,
            "timestamp": datetime.fromtimestamp(ins_result.timestamp).strftime("%H:%M:%S"),
            "verdict": verdict,
            "yolo_confidence": round(conf, 4),
            "patchcore_score": pc_result["score"],
            "defects": ins_result.details["defects_found"],
            "score": ins_result.score,
            "details": ins_result.details["algorithm_results"],
        }
        history.append(result_data)

        save_inspection_result(result_data)


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


def frame_generator():
    last = None
    while running:
        with display_lock:
            frame = display_frame
            if frame is None or frame is last:
                time.sleep(0.05)
                continue
            last = frame
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.get("/", response_class=HTMLResponse)
async def index():
    static_dir = Path("static")
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/video")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


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
    }


@app.get("/history")
async def get_history():
    return list(history)[-50:]


@app.post("/control/start")
async def control_start():
    global processing_enabled
    processing_enabled = True
    return {"status": "started"}


@app.post("/control/stop")
async def control_stop():
    global processing_enabled
    processing_enabled = False
    return {"status": "stopped"}


@app.post("/conveyor/start")
async def conveyor_start():
    global conveyor_running
    if not conveyor.is_connected():
        return {"status": "error", "message": "Conveyor controller not connected"}
    conveyor.send("START")
    conveyor_running = True
    logger.info("Conveyor started")
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
    algo = data.get("algo", "")
    value = data.get("value", 0.5)
    if algo == "patchcore":
        patch_core.threshold = value
    elif algo == "conf":
        detector.conf = value
    return {"algo": algo, "value": value}


@app.post("/reset")
async def reset_counter():
    classifier.reset()
    history.clear()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
