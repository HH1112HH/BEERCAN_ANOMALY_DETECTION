import cv2
import sys
import yaml
import numpy as np
from pathlib import Path
from detector import YOLODetector
from patch_core import PatchCore

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

detector = YOLODetector(
    model_path=cfg["yolo"]["model"],
    conf=cfg["yolo"]["confidence"],
    iou=cfg["yolo"].get("iou", 0.45),
    device=cfg["yolo"].get("device", "cuda:0"),
    half=cfg["yolo"].get("half", True),
    max_det=cfg["yolo"].get("max_det", 5),
)
patch_core = PatchCore(
    model_path=cfg["patchcore"]["model"],
    threshold=cfg["patchcore"]["threshold"],
    device=cfg["patchcore"].get("device", "cuda:0"),
)

img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
img = cv2.imread(img_path)
if img is None:
    print(f"[ERROR] Khong doc duoc anh: {img_path}")
    sys.exit(1)

print(f"[PATCHCORE] Dang xu ly: {img_path}")
dets = detector.detect(img)

if not dets:
    print("[PATCHCORE] YOLO khong phat hien doi tuong → NG")
    sys.exit(0)

for i, det in enumerate(dets):
    conf = det["confidence"]
    roi = det.get("roi")
    x1, y1, x2, y2 = det["bbox"]

    if conf < 0.97 or roi is None or roi.size == 0:
        print(f"[PATCHCORE] Det {i}: conf={conf:.2f} < 0.97 → NG (bo qua)")
        continue

    result = patch_core.check(roi)
    score = result["score"]
    is_defect = result["defect"]
    verdict = "NG" if is_defect else "OK"

    print(f"[PATCHCORE] Det {i}: score={score:.4f} | threshold={patch_core.threshold} | {verdict}")

    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255) if is_defect else (0, 255, 0), 3)
    label = f"PatchCore: {verdict} (score={score:.3f})"
    cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if is_defect else (0, 255, 0), 2)

    if result.get("heatmap") is not None:
        hm = cv2.resize(result["heatmap"], (x2 - x1, y2 - y1))
        overlay = vis.copy()
        overlay[y1:y2, x1:x2] = cv2.addWeighted(vis[y1:y2, x1:x2], 0.5, hm, 0.5, 0)
        cv2.imshow("PatchCore - Heatmap overlay", overlay)
    else:
        cv2.imshow("PatchCore", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
