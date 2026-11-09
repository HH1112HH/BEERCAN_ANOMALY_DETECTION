import os
import sys
import re

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import yaml
import cv2
import logging
import argparse
from pathlib import Path

# Setup project root and src in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from camera_service import CameraService, CameraConfig
from conveyor_controller import ConveyorController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cap")


def load_config(config_path: str = "config.yaml") -> dict:
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_target_dir(arg: str) -> Path:
    arg_lower = (arg or "").lower().strip()
    if arg_lower in ["", "ok", "train", "train_ok"]:
        return Path("dataset/train_ok")
    elif arg_lower in ["ng", "test_ng", "error", "defect", "loi"]:
        return Path("dataset/test_ng")
    elif arg_lower in ["test_ok", "test"]:
        return Path("dataset/test_ok")
    else:
        return Path(arg)


def get_next_can_index(save_dir: Path) -> int:
    if not save_dir.exists():
        return 1
    indices = []
    # Quét tất cả các định dạng ảnh phổ biến
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]:
        for p in save_dir.glob(ext):
            matches = re.findall(r"\d+", p.stem)
            if matches:
                try:
                    indices.append(int(matches[-1]))
                except ValueError:
                    pass
    return (max(indices) + 1) if indices else 1


def main():
    parser = argparse.ArgumentParser(description="Hệ thống chụp ảnh RAW thu thập Dataset cho DRAEM")
    parser.add_argument(
        "mode",
        nargs="?",
        default="",
        help="Chế độ chụp / thư mục lưu: 'ok'/'train' (dataset/train_ok) hoặc 'ng'/'test_ng' (dataset/test_ng)"
    )
    parser.add_argument("--dir", type=str, default=None, help="Thư mục lưu ảnh tùy chọn (nếu có)")
    parser.add_argument("--no-esp", action="store_true", help="Chạy chế độ chụp thủ công không kết nối ESP8266")
    args = parser.parse_args()

    cfg = load_config("config.yaml")

    # Xác định thư mục lưu ảnh
    if args.dir:
        save_dir = Path(args.dir)
    else:
        save_dir = resolve_target_dir(args.mode)

    save_dir.mkdir(parents=True, exist_ok=True)

    can_idx = get_next_can_index(save_dir)
    mode_desc = "LON TỐT (TRAIN_OK)" if "train_ok" in str(save_dir) else ("LON LỖI (TEST_NG)" if "test_ng" in str(save_dir) else str(save_dir))

    print("\n" + "=" * 64)
    print(" HỆ THỐNG CHỤP ẢNH RAW LON TRÊN BĂNG CHUYỀN (AUTO INDEX)")
    print("=" * 64)
    print(f" [*] Mục tiêu thu thập    : {mode_desc}")
    print(f" [-] Thư mục lưu ảnh RAW : {save_dir.resolve()}")
    print(f" [#] Tên file tiếp theo   : can_{can_idx}.jpg")
    print(" [i] Cách dùng lệnh:")
    print("      - cap           -> Chụp lon tốt (dataset/train_ok) tiếp tục từ file cao nhất")
    print("      - cap ng        -> Chụp lon lỗi (dataset/test_ng) tiếp tục từ file cao nhất")
    print("      - cap test_ok   -> Chụp lon test (dataset/test_ok)")
    print(" [+] Phím tắt trong cửa sổ:")
    print("      - SPACE / Phím C : Chụp ngay 1 ảnh thủ công")
    print("      - Q / ESC        : Dừng chương trình")
    print(" [>] Tự động: Cảm biến IR nhận lon -> Dừng ổn định -> Chụp -> Cho lon đi")
    print("=" * 64 + "\n")

    # 1. Khởi tạo Camera RTSP
    cam_cfg = cfg.get("camera", {})
    rtsp_url = cam_cfg.get("rtsp_url")
    logger.info(f"Đang kết nối camera RTSP: {rtsp_url}...")
    camera = CameraService(CameraConfig(
        rtsp_url=rtsp_url,
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    ))

    # 2. Khởi tạo kết nối ESP8266
    conveyor = None
    if not args.no_esp:
        esp_cfg = cfg.get("esp", {})
        esp_ip = esp_cfg.get("ip", "192.168.1.49")
        esp_port = esp_cfg.get("conveyor_port", 12345)
        logger.info(f"Đang kết nối ESP8266: {esp_ip}:{esp_port}...")
        conveyor = ConveyorController(
            host=esp_ip,
            port=esp_port,
        )
        conveyor.connect()
        if conveyor.is_connected():
            logger.info("✅ ESP8266 đã kết nối thành công! Đang dọn sạch bộ đệm...")
            time.sleep(0.3)
            conveyor.clear_buffer()
            conveyor.send("START")
            time.sleep(0.3)
            conveyor.clear_buffer()
            logger.info("Bang chuyen san sang! Hay tha lon vao...")
        else:
            logger.warning("Chua ket noi duoc ESP8266. Ban van co the bam phim SPACE de chup thu cong.")

    flash_text = ""
    flash_timer = 0
    captured_count = 0

    def capture_can(frame_raw):
        nonlocal can_idx, captured_count, flash_text, flash_timer
        if frame_raw is None:
            return

        # Lưu trực tiếp ảnh RAW độ phân giải gốc của camera
        save_path = save_dir / f"can_{can_idx}.jpg"
        cv2.imwrite(str(save_path), frame_raw)

        captured_count += 1
        h, w = frame_raw.shape[:2]
        flash_text = f"DA CHUP: can_{can_idx}.jpg (Session: {captured_count})"
        flash_timer = time.time() + 1.2
        logger.info(f"✅ ĐÃ LƯU ẢNH RAW [{captured_count}] -> {save_path.name} ({w}x{h})")
        can_idx += 1

    window_name = f"RAW Capture - {save_dir.name} [SPACE: Snap | Q: Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            # Kiểm tra tín hiệu từ cảm biến IR qua ESP8266
            if conveyor and conveyor.is_connected():
                msg = conveyor.read_line(timeout=0.01)
                # Kích hoạt khi ESP báo lon đã vào vị trí dừng (TRIGGER_CAMERA)
                if msg and ("TRIGGER_CAMERA" in msg or "WAITING_INSPECTION" in msg):
                    logger.info("📡 Tín hiệu từ cảm biến: Lon đã tới vị trí! Đang chờ 500ms cho lon ổn định tuyệt đối...")
                    flash_text = "DANG ON DINH LON..."
                    flash_timer = time.time() + 0.6
                    
                    # Chờ 500ms cho lon dừng hẳn hoàn toàn (như đợt 1-190)
                    time.sleep(0.50)
                    snap_frame = camera.get_frame()
                    if snap_frame is not None:
                        capture_can(snap_frame)
                    
                    # Gửi lệnh OK để băng chuyền tiếp tục chạy lon qua cổng
                    conveyor.send("OK")
                    logger.info("➡️ Đã gửi lệnh OK cho lon đi tiếp.")
                    time.sleep(1.0)
                    conveyor.clear_buffer()

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            display = frame.copy()

            # Hiển thị HUD thông tin trên màn hình
            cv2.rectangle(display, (0, 0), (display.shape[1], 45), (20, 20, 20), -1)
            hud = f"Muc: {save_dir.name} | Da chup dot nay: {captured_count} | Tiep theo: can_{can_idx}.jpg"
            cv2.putText(display, hud, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if flash_timer > time.time():
                cv2.rectangle(display, (0, 45), (display.shape[1], 90), (0, 180, 0), -1)
                cv2.putText(display, flash_text, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                break
            elif key in [32, ord("c"), ord("C")]:  # 32 = Spacebar
                logger.info("👉 Người dùng bấm phím chụp thủ công!")
                capture_can(frame)
                if conveyor and conveyor.is_connected():
                    conveyor.send("OK")
                    time.sleep(1.0)
                    conveyor.clear_buffer()

    except KeyboardInterrupt:
        logger.info("Đã dừng chương trình chụp.")
    finally:
        cv2.destroyAllWindows()
        if conveyor:
            conveyor.disconnect()
        print("\n" + "=" * 60)
        print(f" [OK] HOÀN TẤT! Đã chụp thêm: {captured_count} ảnh RAW.")
        print(f" [-] Vị trí ảnh: {save_dir.resolve()}")
        print(f" [#] File cuối cùng hiện tại: can_{can_idx - 1}.jpg")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
