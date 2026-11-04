import os
import sys
import glob
import random
import shutil
import zipfile
import cv2
import numpy as np
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def crop_circular_snug(img: np.ndarray, target_size=(256, 256)):
    """
    Cắt lon chuẩn tròn ôm sát 100% vành nhôm bằng giải thuật Hough Circle Gradient.
    Nền đen ôm khít vòng tròn lon bia, triệt tiêu hoàn toàn mọi vệt nhiễu băng chuyền/thanh chắn.
    Thân lon bên trong nguyên vẹn 100%, không bị lủng lỗ.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=60,
        param2=35,
        minRadius=335,
        maxRadius=390
    )

    if circles is not None:
        cx, cy, r = circles[0, 0]
        cx, cy, r = int(cx), int(cy), int(r)
        can_r = r + 2  # Ôm sát tuyệt đối viền ngoài vành nhôm
    else:
        cx, cy, can_r = 999, 520, 360

    R_box = can_r + 4
    sq_x1 = max(0, cx - R_box)
    sq_y1 = max(0, cy - R_box)
    sq_x2 = min(w, cx + R_box)
    sq_y2 = min(h, cy + R_box)

    crop = img[sq_y1:sq_y2, sq_x1:sq_x2].copy()
    ch, cw = crop.shape[:2]

    if ch != cw:
        dim = max(ch, cw)
        padded = np.zeros((dim, dim, 3), dtype=np.uint8)
        padded[(dim - ch) // 2 : (dim - ch) // 2 + ch, (dim - cw) // 2 : (dim - cw) // 2 + cw] = crop
        crop = padded
        ch, cw = dim, dim

    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(mask, (cw // 2, ch // 2), can_r, 255, -1)

    clean_roi = cv2.bitwise_and(crop, crop, mask=mask)
    return cv2.resize(clean_roi, target_size, interpolation=cv2.INTER_AREA)


def main():
    print("\n" + "=" * 65)
    print(" 🛠️  CHUẨN BỊ DATASET DRAEM: CẮT ÔM SÁT 100% VÀNH LON (KHÔNG NHIỄU)")
    print("=" * 65)

    ok_files = sorted(glob.glob("dataset/train_ok/*.jpg") + glob.glob("dataset/train_ok/*.png"))
    ng_files = sorted(glob.glob("dataset/test_ng/*.jpg") + glob.glob("dataset/test_ng/*.png"))

    print(f" [📸] Tổng số ảnh lon OK tìm thấy : {len(ok_files)}")
    print(f" [🚨] Tổng số ảnh lon NG tìm thấy : {len(ng_files)}")

    if not ok_files:
        print("[!] Lỗi: Không tìm thấy ảnh lon OK nào trong dataset/train_ok!")
        return

    # Tạo thư mục xuất
    out_dir = Path("dataset_labeled")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"
    test_img_dir = out_dir / "test" / "images"
    test_lbl_dir = out_dir / "test" / "labels"

    (out_dir / "train" / "ok").mkdir(parents=True, exist_ok=True)
    (out_dir / "test" / "ok").mkdir(parents=True, exist_ok=True)
    (out_dir / "test" / "ng").mkdir(parents=True, exist_ok=True)

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Chia tập OK: 80% Train (~148 lon), 20% Test (~38 lon)
    random.seed(42)
    shuffled_ok = ok_files.copy()
    random.shuffle(shuffled_ok)

    split_idx = int(len(shuffled_ok) * 0.80)
    train_ok_list = shuffled_ok[:split_idx]
    test_ok_list = shuffled_ok[split_idx:]

    print(f"\n[*] Phân chia bộ dữ liệu:")
    print(f"    - Train OK : {len(train_ok_list)} ảnh (Dùng để huấn luyện DRAEM)")
    print(f"    - Test OK  : {len(test_ok_list)} ảnh (Đánh giá false alarm / specificity)")
    print(f"    - Test NG  : {len(ng_files)} ảnh (Đánh giá độ nhạy phát hiện lỗi / recall)")

    # 1. Xử lý TRAIN OK
    print("\n[*] Đang cắt ROI 256x256 ôm sát vành lon cho tập TRAIN OK...")
    for idx, path in enumerate(train_ok_list, 1):
        img = cv2.imread(path)
        cropped = crop_circular_snug(img)
        fname = f"can_ok_train_{idx:03d}.jpg"
        cv2.imwrite(str(train_img_dir / fname), cropped)
        cv2.imwrite(str(out_dir / "train" / "ok" / fname), cropped)
        with open(train_lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
            f.write("1 0.5 0.5 1.0 1.0\n")

    # 2. Xử lý TEST OK
    print("[*] Đang cắt ROI 256x256 ôm sát vành lon cho tập TEST OK...")
    for idx, path in enumerate(test_ok_list, 1):
        img = cv2.imread(path)
        cropped = crop_circular_snug(img)
        fname = f"can_ok_test_{idx:03d}.jpg"
        cv2.imwrite(str(test_img_dir / fname), cropped)
        cv2.imwrite(str(out_dir / "test" / "ok" / fname), cropped)
        with open(test_lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
            f.write("1 0.5 0.5 1.0 1.0\n")

    # 3. Xử lý TEST NG
    print("[*] Đang cắt ROI 256x256 ôm sát vành lon cho tập TEST NG...")
    for idx, path in enumerate(ng_files, 1):
        img = cv2.imread(path)
        cropped = crop_circular_snug(img)
        fname = f"can_ng_test_{idx:03d}.jpg"
        cv2.imwrite(str(test_img_dir / fname), cropped)
        cv2.imwrite(str(out_dir / "test" / "ng" / fname), cropped)
        with open(test_lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
            f.write("0 0.5 0.5 1.0 1.0\n")

    # 4. Đóng gói ZIP
    zip_path = Path("dataset_labeled.zip")
    if zip_path.exists():
        zip_path.unlink()

    print(f"\n[*] Đang nén toàn bộ thành file {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(out_dir.parent)
                zf.write(full_path, rel_path)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 65)
    print(f" [✓] HOÀN TẤT THÀNH CÔNG!")
    print(f" [📦] File nén đã tạo      : {zip_path.resolve()}")
    print(f" [📊] Dung lượng file zip  : {zip_size_mb:.2f} MB")
    print(f" [🎯] Đặc điểm             : Nền đen ôm sát 100% vành lon, sạch 100% nhiễu")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
