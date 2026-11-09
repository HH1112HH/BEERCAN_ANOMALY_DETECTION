import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("image_logger")

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)


def create_scientific_analysis_card(
    can_number: int,
    verdict: str,
    score: float,
    threshold: float,
    yolo_conf: float,
    roi: np.ndarray,
    recon_img: np.ndarray = None,
    discr_map: np.ndarray = None,
    pure_heatmap: np.ndarray = None,
    heatmap_overlay: np.ndarray = None,
    raw_frame: np.ndarray = None,
    timestamp_str: str = None
) -> str:
    """
    Tạo ảnh tổng hợp phân tích khoa học 4 khung hình chuẩn đồ án DRAEM:
    [ 1. Ảnh gốc (Input) ] | [ 2. Ảnh tái tạo (Recon Net) ] | [ 3. Bản đồ lỗi (Discr Net - cmap Hot) ] | [ 4. Phủ nhiệt (Overlay) ]
    Kèm thanh tiêu đề hiển thị đầy đủ thông số cho giảng viên đánh giá.
    """
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    size = 256  # kích thước mỗi panel vuông chuẩn 256x256

    # 1. Chuẩn bị Panel 1: Ảnh gốc ROI crop tròn
    if roi is not None and roi.size > 0:
        p1 = cv2.resize(roi, (size, size))
    else:
        p1 = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(p1, "NO ROI", (size//4, size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)

    # Mặt nạ hình tròn nắp lon
    mask_circle = np.zeros((size, size), dtype=bool)
    Y, X = np.ogrid[:size, :size]
    mask_circle[(X - size//2)**2 + (Y - size//2)**2 <= (int(size * 0.485))**2] = True
    can_mask = (p1.sum(axis=2) > 0) & mask_circle
    p1[~can_mask] = 0

    # 2. Chuẩn bị Panel 2: Ảnh tái tạo từ Reconstructive Net
    if recon_img is not None and recon_img.size > 0:
        p2 = cv2.resize(recon_img, (size, size))
    else:
        p2 = p1.copy()
    p2[~can_mask] = 0

    # 3. Chuẩn bị Panel 3: Bản đồ lỗi pixel từ Discriminative Net (COLORMAP_HOT)
    if discr_map is not None:
        if discr_map.shape[:2] != (size, size):
            pm_res = cv2.resize(discr_map.astype(np.float32), (size, size))
        else:
            pm_res = discr_map.astype(np.float32)
        pm_clean = pm_res * can_mask
        pm_norm = np.clip(pm_clean / (float(threshold) * 1.2 + 1e-6), 0.0, 1.0)
    elif pure_heatmap is not None and pure_heatmap.size > 0:
        gray = cv2.cvtColor(cv2.resize(pure_heatmap, (size, size)), cv2.COLOR_BGR2GRAY)
        pm_norm = (gray / 255.0) * can_mask
    else:
        pm_norm = np.zeros((size, size), dtype=np.float32)

    p3 = cv2.applyColorMap((pm_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    p3[~can_mask] = 0

    # 4. Chuẩn bị Panel 4: Phủ nhiệt khuyết tật (Colormap JET đè lên ảnh gốc)
    if heatmap_overlay is not None and heatmap_overlay.size > 0:
        p4 = cv2.resize(heatmap_overlay, (size, size))
    else:
        jet_hm = cv2.applyColorMap((pm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        jet_hm[~can_mask] = 0
        p4 = cv2.addWeighted(p1, 0.55, jet_hm, 0.45, 0)
    p4[~can_mask] = 0

    # Vẽ khung và nhãn cho từng panel
    def add_panel_label(img, label_text):
        labeled = img.copy()
        cv2.rectangle(labeled, (0, size - 30), (size, size), (15, 23, 42), -1)
        cv2.putText(labeled, label_text, (8, size - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(labeled, (0, 0), (size - 1, size - 1), (80, 90, 110), 1)
        return labeled

    p1_lbl = add_panel_label(p1, "1. Anh goc (Input)")
    p2_lbl = add_panel_label(p2, "2. Tai tao (Recon Net)")
    p3_lbl = add_panel_label(p3, "3. Ban do loi (Discr Net)")
    p4_lbl = add_panel_label(p4, "4. Phu nhiet (Overlay)")

    # Ghép ngang 4 panel
    panels_row = np.hstack([p1_lbl, p2_lbl, p3_lbl, p4_lbl])
    total_w = panels_row.shape[1]

    # Tạo Banner tiêu đề khoa học phía trên
    banner_h = 72
    is_good = (verdict.upper() == "GOOD")
    banner_color = (34, 150, 46) if is_good else (34, 40, 215)
    banner = np.full((banner_h, total_w, 3), banner_color, dtype=np.uint8)

    title_text = f"LON #{can_number:03d}  |  KET QUA: {verdict.upper()} ({'DAT TIEU CHUAN' if is_good else 'PHAT HIEN LOI'})"
    sub_text = f"DRAEM Score: {score:.4f}  (Nguong: {threshold:.2f})  |  YOLO Conf: {yolo_conf:.2f}  |  Thoi gian: {timestamp_str}"

    cv2.putText(banner, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, sub_text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    # Ghép banner và 4 panel thành ảnh hoàn chỉnh
    final_card = np.vstack([banner, panels_row])

    # Tên file lưu chuẩn khoa học
    filename = f"Lon_{can_number:03d}_{verdict.upper()}_score_{score:.3f}.jpg"
    filepath = CAPTURES_DIR / filename
    cv2.imwrite(str(filepath), final_card, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info(f"Saved 4-panel analysis card: {filepath}")

    return filename
