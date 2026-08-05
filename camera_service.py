import cv2
import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    rtsp_url: str
    width: int = 1280
    height: int = 720
    fps: int = 15
    reconnect_delay: float = 2.0

class CameraService:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.frames = deque(maxlen=8)
        self.fps_actual = 0
        self._connect()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera capture started: {self.config.rtsp_url}")

    def _connect(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.config.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            logger.info(f"Connected to camera: {self.config.rtsp_url}")
        except Exception as e:
            logger.error(f"Failed to connect camera: {e}")
            self.cap = None

    def _capture_loop(self):
        while self.running:
            t_cycle = time.time()
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._connect()
                    time.sleep(self.config.reconnect_delay)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame, reconnecting...")
                    self._connect()
                    time.sleep(self.config.reconnect_delay)
                    continue

                now = time.time()
                with self.lock:
                    self.frame = frame
                    self.frames.append((now, frame))

                elapsed = now - t_cycle
                sleep_time = (1.0 / self.config.fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                cycle = time.time() - t_cycle
                if cycle > 0:
                    self.fps_actual = 1.0 / cycle

            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(self.config.reconnect_delay)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_frame_near(self, ts: float, max_age: float = 0.3):
        """Lay khung hinh gan thoi diem ts nhat (dong bo voi thoi diem trigger tu ESP)."""
        with self.lock:
            if not self.frames:
                return None
            best = None
            best_dt = float("inf")
            for t, f in self.frames:
                dt = abs(ts - t)
                if dt < best_dt:
                    best_dt = dt
                    best = f
            if best is None or best_dt > max_age:
                return None
            return best.copy()

    def get_fps(self):
        return round(self.fps_actual, 1)
