import logging
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path: str = "models/weights.pt", conf: float = 0.97, iou: float = 0.45, device: str = "cuda:0", half: bool = True, max_det: int = 1, imgsz: int = 960):
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half and device.startswith("cuda")
        self.max_det = max_det
        self.imgsz = imgsz

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.half = False

        logger.info(f"Loading YOLO segment model: {model_path} on {self.device} (half={self.half}, imgsz={imgsz})")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        if self.half:
            self.model.model.half()

        logger.info(f"YOLO segment model loaded on {self.device}")

    def detect(self, frame: np.ndarray):
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            augment=False,
            max_det=self.max_det,
            imgsz=self.imgsz,
        )
        detections = []
        if len(results) == 0:
            return detections

        r = results[0]
        if r.masks is None or r.boxes is None:
            return detections

        boxes = r.boxes
        masks = r.masks.data if r.masks is not None else None
        fh, fw = frame.shape[:2]

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            cls = int(boxes.cls[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            roi = None
            mask = None
            if masks is not None and i < len(masks):
                mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
                mask_np = cv2.resize(mask_np, (fw, fh))
                mask = (mask_np > 127).astype(np.uint8) * 255
                mask = mask[y1:y2, x1:x2]
                if mask.size == 0:
                    continue
                roi = frame[y1:y2, x1:x2].copy()
                roi = cv2.bitwise_and(roi, roi, mask=mask)
                if roi.size == 0:
                    continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "roi": roi,
                "mask": mask,
            })

        return detections