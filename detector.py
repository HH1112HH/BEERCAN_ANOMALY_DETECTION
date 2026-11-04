import logging
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path: str = "models/weights.pt", conf: float = 0.45, iou: float = 0.45, device: str = "cuda:0", half: bool = True, max_det: int = 5, imgsz: int = 960):
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half and device.startswith("cuda")
        self.max_det = max_det
        self.imgsz = imgsz

        # Tọa độ vùng trạm kiểm tra trên khung hình Dahua (1296 x 2304)
        self.station_cx = 1000
        self.station_cy = 520
        self.zone_x1 = 500
        self.zone_y1 = 50
        self.zone_x2 = 1500
        self.zone_y2 = 1000

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
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes
        fh, fw = frame.shape[:2]

        # Lọc các bounding box nằm trong trạm kiểm tra, sắp xếp ưu tiên lon gần tâm trạm nhất
        candidate_boxes = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            cls = int(boxes.cls[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            bcx = (x1 + x2) // 2
            bcy = (y1 + y2) // 2

            # Chỉ nhận lon nằm trong vùng trạm kiểm tra (bỏ qua lon cũ ở mép ra băng chuyền)
            if self.zone_x1 <= bcx <= self.zone_x2 and self.zone_y1 <= bcy <= self.zone_y2:
                dist = np.hypot(bcx - self.station_cx, bcy - self.station_cy)
                candidate_boxes.append((dist, conf, cls, x1, y1, x2, y2))

        # Nếu không có lon nào trong vùng chuẩn, kiểm tra các box khác
        if not candidate_boxes:
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                candidate_boxes.append((9999.0, conf, cls, x1, y1, x2, y2))

        # Sắp xếp ưu tiên lon ở gần tâm trạm nhất
        candidate_boxes.sort(key=lambda item: item[0])

        for _, conf, cls, x1, y1, x2, y2 in candidate_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            R_search = int(max(x2 - x1, y2 - y1) / 2) + 25
            sq_x1, sq_y1 = max(0, cx - R_search), max(0, cy - R_search)
            sq_x2, sq_y2 = min(fw, cx + R_search), min(fh, cy + R_search)

            sub = frame[sq_y1:sq_y2, sq_x1:sq_x2]
            sub_h, sub_w = sub.shape[:2]

            sub_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            sub_blurred = cv2.GaussianBlur(sub_gray, (9, 9), 2)

            # Dò vành nhôm bạc tròn chuẩn xác (bán kính 340..375 px, tránh liếm vào thân hoa văn xanh)
            circles = cv2.HoughCircles(
                sub_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                param1=60, param2=30,
                minRadius=340, maxRadius=375
            )
            if circles is not None:
                hcx, hcy, hr = circles[0, 0]
                true_cx = sq_x1 + int(hcx)
                true_cy = sq_y1 + int(hcy)
                can_r = int(hr) + 2
            else:
                true_cx = cx
                true_cy = cy
                can_r = 358  # Bán kính chuẩn vành nhôm trên camera Dahua 1296x2304

            # Cắt vuông chuẩn tâm khớp 100% tỷ lệ huấn luyện của DRAEM (R_box = can_r + 4)
            R_box = can_r + 4
            bx1, by1 = max(0, true_cx - R_box), max(0, true_cy - R_box)
            bx2, by2 = min(fw, true_cx + R_box), min(fh, true_cy + R_box)

            tight_crop = frame[by1:by2, bx1:bx2].copy()
            tch, tcw = tight_crop.shape[:2]
            if tch != tcw:
                dim = max(tch, tcw)
                padded = np.zeros((dim, dim, 3), dtype=np.uint8)
                padded[(dim - tch) // 2 : (dim - tch) // 2 + tch, (dim - tcw) // 2 : (dim - tcw) // 2 + tcw] = tight_crop
                tight_crop = padded
                tch, tcw = dim, dim

            circ_mask = np.zeros((tch, tcw), dtype=np.uint8)
            cv2.circle(circ_mask, (tcw // 2, tch // 2), can_r, 255, -1)
            roi = cv2.bitwise_and(tight_crop, tight_crop, mask=circ_mask)
            mask = circ_mask

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "roi": roi,
                "mask": mask,
            })

        return detections
