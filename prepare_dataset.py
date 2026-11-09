import os
import sys
import glob
import random
import shutil
import zipfile
import argparse
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

# Ensure working directory is project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


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


def generate_can_variants(roi_256: np.ndarray, num_variants: int = 10):
    """
    Tăng cường dữ liệu chuẩn vật lý cho nắp lon tròn đối xứng:
    - Xoay đa góc (Rotational Symmetry 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    - Biến thiên độ sáng/tương phản nhẹ (±5% mô phỏng dao động đèn LED)
    - Lật gương đối xứng (Horizontal Flip)
    Bảo đảm 100% nền đen tuyền xung quanh đĩa tròn.
    """
    center = (128, 128)
    radius = 125
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    variants = []

    # 1. 8 Góc xoay tròn (cách nhau 45 độ)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for ang in angles:
        if ang == 0:
            rot = roi_256.copy()
        else:
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            rot = cv2.warpAffine(roi_256, M, (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            rot = cv2.bitwise_and(rot, rot, mask=mask)

        # Thêm biến thiên ánh sáng tự nhiên cho một số góc
        if ang == 135:
            lit = cv2.convertScaleAbs(rot, alpha=1.05, beta=6)
            lit = cv2.bitwise_and(lit, lit, mask=mask)
            variants.append(("rot135_bright", lit))
        elif ang == 225:
            lit = cv2.convertScaleAbs(rot, alpha=0.95, beta=-6)
            lit = cv2.bitwise_and(lit, lit, mask=mask)
            variants.append(("rot225_dim", lit))
        else:
            variants.append((f"rot{ang:03d}", rot))

    # 2. Lật gương ngang (Horizontal Flip)
    flipped = cv2.flip(roi_256, 1)
    variants.append(("flip_h", flipped))

    # 3. Lật gương kết hợp xoay 90 độ
    M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    flip_rot = cv2.warpAffine(flipped, M90, (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    flip_rot = cv2.bitwise_and(flip_rot, flip_rot, mask=mask)
    variants.append(("flip_h_rot090", flip_rot))

    if num_variants <= len(variants):
        return variants[:num_variants]
    return variants


def main():
    parser = argparse.ArgumentParser(description="Đóng gói và tăng cường dữ liệu huấn luyện DRAEM")
    parser.add_argument("--aug-factor", type=int, default=10, help="Số biến thể sinh ra từ mỗi lon gốc (mặc định: 10)")
    parser.add_argument("--no-aug", action="store_true", help="Tắt tăng cường dữ liệu (chỉ giữ ảnh gốc 1:1)")
    args = parser.parse_args()

    aug_factor = 1 if args.no_aug else max(1, args.aug_factor)

    print("\n" + "=" * 70)
    print(" CHUẨN BỊ DATASET DRAEM: CẮT ÔM SÁT 100% & TĂNG CƯỜNG DỮ LIỆU CẤP 1")
    print(f"     Hệ số tăng cường dữ liệu : x{aug_factor} (Mỗi lon train sinh {aug_factor} biến thể)")
    print("=" * 70)

    ok_files = sorted(glob.glob("dataset/train_ok/*.jpg") + glob.glob("dataset/train_ok/*.png"))
    ng_files = sorted(glob.glob("dataset/test_ng/*.jpg") + glob.glob("dataset/test_ng/*.png"))

    print(f" [*] Tổng số ảnh lon OK tìm thấy : {len(ok_files)}")
    print(f" [*] Tổng số ảnh lon NG tìm thấy : {len(ng_files)}")

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

    # Chia tập OK: 80% Train, 20% Test
    random.seed(42)
    shuffled_ok = ok_files.copy()
    random.shuffle(shuffled_ok)

    split_idx = int(len(shuffled_ok) * 0.80)
    train_ok_list = shuffled_ok[:split_idx]
    test_ok_list = shuffled_ok[split_idx:]

    expected_train_count = len(train_ok_list) * aug_factor

    print(f"\n[*] Phân chia bộ dữ liệu:")
    print(f"    - Mẫu Train gốc : {len(train_ok_list)} lon")
    print(f"    - Sau tăng cường: {expected_train_count} ảnh (x{aug_factor} xoay tròn, ánh sáng, lật)")
    print(f"    - Test OK       : {len(test_ok_list)} ảnh gốc (Đánh giá false alarm / specificity)")
    print(f"    - Test NG       : {len(ng_files)} ảnh gốc (Đánh giá độ nhạy phát hiện lỗi / recall)")

    # 1. Xử lý TRAIN OK kèm Augmentation
    print(f"\n[*] Đang xử lý & tăng cường tập TRAIN OK (Mục tiêu: {expected_train_count} ảnh)...")
    total_saved = 0
    for idx, path in enumerate(train_ok_list, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        cropped = crop_circular_snug(img)
        if cropped is None:
            continue

        if aug_factor == 1:
            variants = [("orig", cropped)]
        else:
            variants = generate_can_variants(cropped, num_variants=aug_factor)

        for var_name, var_img in variants:
            total_saved += 1
            fname = f"can_ok_train_{idx:03d}_{var_name}.jpg"
            cv2.imwrite(str(train_img_dir / fname), var_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_dir / "train" / "ok" / fname), var_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(train_lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
                f.write("1 0.5 0.5 1.0 1.0\n")

        if idx % 30 == 0 or idx == len(train_ok_list):
            print(f"    -> Đã xử lý {idx}/{len(train_ok_list)} lon gốc (Sinh: {total_saved} ảnh)")

    # 2. Xử lý TEST OK (giữ nguyên gốc)
    print("\n[*] Đang cắt ROI 256x256 ôm sát vành lon cho tập TEST OK...")
    for idx, path in enumerate(test_ok_list, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        cropped = crop_circular_snug(img)
        fname = f"can_ok_test_{idx:03d}.jpg"
        cv2.imwrite(str(test_img_dir / fname), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / "test" / "ok" / fname), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(test_lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
            f.write("1 0.5 0.5 1.0 1.0\n")

    # 3. Xử lý TEST NG (giữ nguyên gốc)
    print("[*] Đang cắt ROI 256x256 ôm sát vành lon cho tập TEST NG...")
    for idx, path in enumerate(ng_files, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        cropped = crop_circular_snug(img)
        fname = f"can_ng_test_{idx:03d}.jpg"
        cv2.imwrite(str(test_img_dir / fname), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / "test" / "ng" / fname), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(test_lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
            f.write("0 0.5 0.5 1.0 1.0\n")

    # 4. Đóng gói ZIP (dataset_labeled.zip)
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

    print("\n" + "=" * 70)
    print(f" [OK] HOÀN TẤT THÀNH CÔNG!")
    print(f" [-] File nén đã tạo        : {zip_path.resolve()}")
    print(f" [-] Dung lượng file zip    : {zip_size_mb:.2f} MB")
    print(f" [*] Thống kê tập ảnh       :")
    print(f"      + Train OK (Augmented): {total_saved:,} ảnh (Từ {len(train_ok_list)} lon thực tế)")
    print(f"      + Test OK (Thực tế)   : {len(test_ok_list):,} ảnh")
    print(f"      + Test NG (Thực tế)   : {len(ng_files):,} ảnh")
    print(f"      --------------------------------------------------")
    print(f"      + TỔNG CỘNG           : {total_saved + len(test_ok_list) + len(ng_files):,} ảnh")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
