import os
import sys

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

from camera_service import CameraService, CameraConfig
from conveyor_controller import ConveyorController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cap")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_next_can_index(save_dir: Path) -> int:
    existing = list(save_dir.glob("can_*.jpg"))
    if not existing:
        return 1
    indices = []
    for p in existing:
        stem = p.stem  # e.g. "can_12"
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            indices.append(int(parts[-1]))
    return max(indices) + 1 if indices else len(existing) + 1


def main():
    parser = argparse.ArgumentParser(description="Auto RAW Can Collector for DRAEM Training")
    parser.add_argument("--dir", type=str, default="dataset/train_ok", help="Thu muc luu anh raw (default: dataset/train_ok)")
    parser.add_argument("--no-esp", action="store_true", help="Chay che do chup thu cong khong ket noi ESP8266")
    args = parser.parse_args()

    cfg = load_config("config.yaml")

    save_dir = Path(args.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    can_idx = get_next_can_index(save_dir)
    print("\n" + "=" * 62)
    print(" 📸 HỆ THỐNG CHỤP ẢNH RAW LON TRÊN BĂNG CHUYỀN (KHÔNG CẮT)")
    print("=" * 62)
    print(f" [📁] Thư mục lưu ảnh RAW : {save_dir.resolve()}")
    print(f" [🔢] Tên file bắt đầu     : can_{can_idx}.jpg")
    print(" [⌨️] Phím tắt:")
    print("      - SPACE / Phím C : Chụp ngay lập tức 1 ảnh RAW")
    print("      - Q / ESC        : Thoát chương trình")
    print(" [🔄] Tự động: Lon tới cảm biến -> Chụp RAW -> Cho lon đi tiếp")
    print("=" * 62 + "\n")

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
        conveyor = ConveyorController(
            host=esp_cfg.get("ip", "192.168.1.37"),
            port=esp_cfg.get("conveyor_port", 12345),
        )
        conveyor.connect()
        if conveyor.is_connected():
            logger.info("ESP8266 đã kết nối thành công! Đang đồng bộ và dọn sạch bộ đệm...")
            time.sleep(0.5)
            conveyor.clear_buffer()
            # Gửi lệnh START/OK để đảm bảo băng chuyền ở trạng thái RUNNING sẵn sàng đón lon
            conveyor.send("START")
            time.sleep(0.5)
            conveyor.clear_buffer()
            logger.info("ESP8266 sẵn sàng! Đang chờ lon đến cảm biến...")
        else:
            logger.warning("Chưa thấy ESP8266, bạn vẫn có thể bấm phím SPACE để chụp thủ công.")

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
        flash_text = f"DA CHUP: can_{can_idx}.jpg (Tong: {captured_count})"
        flash_timer = time.time() + 1.2
        logger.info(f"✅ ĐÃ LƯU ẢNH RAW [{captured_count}] -> {save_path.name} ({w}x{h})")
        can_idx += 1

    window_name = "RAW Can Capture [SPACE: Snap | Q: Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            # Kiểm tra tín hiệu từ cảm biến IR qua ESP8266
            if conveyor and conveyor.is_connected():
                msg = conveyor.read_line(timeout=0.01)
                # Chỉ kích hoạt khi ESP báo lon đã vào vị trí dừng (TRIGGER_CAMERA)
                if msg and "TRIGGER_CAMERA" in msg:
                    logger.info("📡 Tín hiệu từ ESP8266: Lon đã tới vị trí! Đang chờ 500ms cho lon dừng hẳn...")
                    flash_text = "DANG CHO LON DUNG HAN..."
                    flash_timer = time.time() + 0.6
                    
                    # Chờ 500ms cho lon dừng hẳn hoàn toàn và RTSP cập nhật frame tĩnh
                    time.sleep(0.5)
                    snap_frame = camera.get_frame()
                    if snap_frame is not None:
                        capture_can(snap_frame)
                    
                    # Gửi lệnh OK để băng chuyền tiếp tục chạy lon qua cổng
                    conveyor.send("OK")
                    logger.info("➡️ Đã gửi lệnh OK cho lon đi qua cổng.")
                    # Chờ 2.0s cho lon đi qua hẳn khỏi cảm biến và xóa sạch buffer
                    time.sleep(2.0)
                    conveyor.clear_buffer()

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            display = frame.copy()

            # Hiển thị HUD thông tin
            cv2.rectangle(display, (0, 0), (display.shape[1], 45), (20, 20, 20), -1)
            hud = f"Thu muc: {save_dir.name} | Da chup RAW: {captured_count} lon | Next: can_{can_idx}.jpg"
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
                    time.sleep(1.5)
                    conveyor.clear_buffer()

    except KeyboardInterrupt:
        logger.info("Đã dừng chương trình chụp.")
    finally:
        cv2.destroyAllWindows()
        if conveyor:
            conveyor.disconnect()
        print("\n" + "=" * 60)
        print(f" [✓] HOÀN TẤT! Đã chụp tổng cộng: {captured_count} ảnh RAW.")
        print(f" [📁] Vị trí ảnh: {save_dir.resolve()}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
