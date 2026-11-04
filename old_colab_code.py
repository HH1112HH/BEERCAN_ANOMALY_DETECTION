<USER_REQUEST>
lon train là những lon dạng như ảnh 1, băng chuyền mới là dạng như ảnh 2

vậy có phải là do sai lệch data không??

# ==============================================================================
# BƯỚC 1: KIỂM TRA MÔI TRƯỜNG GPU VÀ CÀI ĐẶT THƯ VIỆN
# ==============================================================================
import os
import glob
import time
import random
import math
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[✓] Môi trường thực thi: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ==============================================================================
# BƯỚC 2: DÙNG TRỰC TIẾP TẬP DỮ LIỆU ROBOFLOW ĐÃ PHÂN CHIA (70-20-10)
# ==============================================================================
zip_name = 'dataset_labeled.zip'

if os.path.exists(zip_name):
    print(f'[*] Đang giải nén {zip_name}...')
    !unzip -q -o {zip_name} -d dataset_labeled
    print('[✓] Giải nén xong!')
else:
    print(f'[!] Không thấy file {zip_name}. Hãy upload file {zip_name} lên Colab trước.')

# Hàm đọc nhãn từ file .txt của Roboflow (Class 1 = OK, Class 0 = NG)
def get_label(img_p):
    lbl_p = img_p.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.exists(lbl_p):
        with open(lbl_p, 'r') as f:
            lines = f.readlines()
            if lines:
                cls_id = int(lines[0].split()[0])
                return 'OK' if cls_id == 1 else 'NG'
    return 'OK'

# 1. Tập TRAIN (Roboflow 70%): DRAEM chỉ lấy các lon OK trong tập Train của Roboflow để học
raw_train_imgs = sorted(glob.glob('dataset_labeled/train/images/*.jpg')) + sorted(glob.glob('dataset_labeled/train/images/*.png'))
train_paths = [p for p in raw_train_imgs if get_label(p) == 'OK']

# 2. Tập TEST (Roboflow 10%): Lấy từ thư mục test/ của Roboflow để đánh giá độc lập
raw_test_imgs = sorted(glob.glob('dataset_labeled/test/images/*.jpg')) + sorted(glob.glob('dataset_labeled/test/images/*.png'))
test_paths = [p for p in raw_test_imgs if get_label(p) == 'OK']  # Lon OK trong Test -> Tính Ngưỡng & Specificity
ng_paths   = [p for p in raw_test_imgs if get_label(p) == 'NG']  # Lon NG trong Test -> Tính Recall & Confusion Matrix

print('\n' + '=' * 58)
print(f'  BỘ DỮ LIỆU ROBOFLOW ĐÃ PHÂN CHIA (70% Train - 10% Test)')
print('  ' + '-'*54)
print(f'  Train (OK từ Roboflow Train) : {len(train_paths):,} ảnh  ← Học từ đây')
print(f'  Test-OK (OK từ Roboflow Test): {len(test_paths):,} ảnh  ← Tính ngưỡng & Specificity')
print(f'  Test-NG (NG từ Roboflow Test): {len(ng_paths):,} ảnh  ← Tính Recall & Confusion Matrix')
print('=' * 58)

# ==============================================================================
# BƯỚC 3: HÀM CẮT DÙNG DYNAMIC TRACKING BAO TRÙM TOÀN BỘ MIỆNG LON BIA + VÀNH NGOÀI
# ==============================================================================
def crop_circular_can(image_input, target_size=(256, 256), padding=12, apply_mask=True):
    if image_input is None:
        return None, None
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        return None, None
    if img is None:
        return None, None

    h, w = img.shape[:2]

    # 1. Chuyển Grayscale & Làm mờ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Vùng làm việc trên băng tải trung tâm
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(roi_mask, (int(w * 0.20), int(h * 0.10)), (int(w * 0.80), int(h * 0.90)), 255, -1)

    # 3. Phân ngưỡng lọc vành nhôm sáng miệng lon
    _, thresh = cv2.threshold(blurred, 85, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Tìm Contour TOÀN BỘ VÀNH MIỆNG LON NGOÀI CÙNG
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bán kính mặc định bao trùm toàn bộ miệng lon (R=0.34*h)
    best_cx, best_cy, best_r = int(w * 0.54), int(h * 0.47), int(h * 0.34)

    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            (x, y), rad = cv2.minEnclosingCircle(c)
            # Lấy bán kính vành miệng lon ngoài cùng (22% -> 42% h)
            if 0.22 * h < rad < 0.42 * h and area > 6000:
                valid_contours.append((area, x, y, rad))

        if valid_contours:
            valid_contours.sort(key=lambda item: item[0], reverse=True)
            _, x, y, rad = valid_contours[0]
            # Mở rộng thêm padding để ôm trọn cả vành miệng lon ngoài cùng
            best_cx, best_cy, best_r = int(x), int(y), int(rad) + padding

    # 5. Mặt nạ tròn ôm trọn TOÀN BỘ MIỆNG LON (Vành ngoài + Nắp lon)
    if apply_mask:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (best_cx, best_cy), best_r, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)

    # 6. Cắt vuông 256x256 bám đuổi theo lon
    x1, y1 = max(0, best_cx - best_r), max(0, best_cy - best_r)
    x2, y2 = min(w, best_cx + best_r), min(h, best_cy + best_r)
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        crop = img

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return resized, (best_cx, best_cy, best_r)

# --- HIỂN THỊ KẾT QUẢ ĐỘNG MIỆNG LON ---
if train_paths:
    sample_img = cv2.imread(train_paths[0])
    cropped, _ = crop_circular_can(train_paths[0])
    if sample_img is not None and cropped is not None:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.title("Ảnh gốc trên băng chuyền")
        plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)); plt.axis('off')
        plt.subplot(1, 2, 2); plt.title("Sau Tiền xử lý (Cắt tròn 256×256)")
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)); plt.axis('off')
        plt.tight_layout(); plt.show()

# ==============================================================================
# BƯỚC 4: SINH LỖI GIẢ CHUẨN TỰ NHIÊN (PERLIN NOISE & SCRATCH ANOMALY)
# ==============================================================================
def generate_perlin_noise_2d(shape, res):
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(a, b, t): return a + t * (b - a)
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    tile = lambda x, d: np.repeat(np.repeat(x, d[0], axis=0), d[1], axis=1)
    g00 = tile(gradients[0:-1, 0:-1], d)
    g10 = tile(gradients[1:,   0:-1], d)
    g01 = tile(gradients[0:-1, 1:  ], d)
    g11 = tile(gradients[1:,   1:  ], d)
    n00 = np.sum(np.dstack((grid[:,:,0],   grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0],   grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    t = fade(grid)
    return np.sqrt(2) * lerp(lerp(n00, n10, t[:,:,0]), lerp(n01, n11, t[:,:,0]), t[:,:,1])

# Ngưỡng 0.76 tạo mảng ố Perlin Noise mỏng nhẹ tự nhiên
def generate_perlin_mask(size=256, min_perlin_scale=2, max_perlin_scale=5, threshold=0.76):
    perlin_scalex = 2 ** random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scalex = min(perlin_scalex, size)
    perlin_scaley = min(perlin_scaley, size)
    noise = generate_perlin_noise_2d((size, size), (perlin_scalex, perlin_scaley))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    mask  = (noise > threshold).astype(np.float32)

    if mask.sum() > size * size * 0.15:
        mask = (noise > 0.85).astype(np.float32)
    return mask

def generate_texture(size=256, train_paths=None):
    if train_paths and random.random() > 0.3:
        p = random.choice(train_paths)
        bgr, _ = crop_circular_can(p)
        if bgr is not None:
            tex = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if random.random() > 0.5:
                tex = np.rot90(tex, k=random.randint(1, 3))
            if random.random() > 0.5:
                tex = np.fliplr(tex)
            brightness = random.uniform(0.4, 1.5)
            tex = np.clip(tex * brightness, 0, 1)
            return tex
    tex = np.random.rand(size, size, 3).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (21, 21), 0)
    return tex

def create_synthetic_anomaly(image_rgb_float, train_paths, size=256):
    mask = generate_perlin_mask(size=size)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]

    # Bán kính 0.30 ép Perlin Noise nằm gọn nội trong nắp lon nhôm
    circle_mask = ((X - cx)**2 + (Y - cy)**2 <= (size * 0.30)**2).astype(np.float32)
    mask = mask * circle_mask

    if mask.sum() < size * size * 0.0005:
        return image_rgb_float.copy(), np.zeros((size, size, 1), dtype=np.float32)

    texture = generate_texture(size=size, train_paths=train_paths)
    mask_3ch = mask[:, :, np.newaxis]
    augmented = (1.0 - mask_3ch) * image_rgb_float + mask_3ch * texture
    augmented = np.clip(augmented, 0.0, 1.0)
    return augmented.astype(np.float32), mask[:, :, np.newaxis].astype(np.float32)

def generate_scratch_anomaly(image_rgb_float, size=256):
    augmented = image_rgb_float.copy()
    mask = np.zeros((size, size), dtype=np.float32)
    num_scratches = random.randint(1, 2)
    cy, cx = size // 2, size // 2

    for _ in range(num_scratches):
        points = []
        start_x = random.randint(int(size*0.25), int(size*0.75))
        start_y = random.randint(int(size*0.25), int(size*0.75))
        points.append((start_x, start_y))

        for _ in range(random.randint(2, 4)):
            start_x += random.randint(-20, 20)
            start_y += random.randint(-20, 20)
            points.append((np.clip(start_x, int(size*0.25), int(size*0.75)), np.clip(start_y, int(size*0.25), int(size*0.75))))

        points = np.array(points, np.int32).reshape((-1, 1, 2))
        thickness = random.randint(1, 2)
        is_white_scratch = random.random() > 0.5
        color = (1.0, 1.0, 1.0) if is_white_scratch else (0.1, 0.1, 0.1)

        cv2.polylines(augmented, [points], False, color, thickness, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [points], False, 1.0, thickness + 1)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    Y, X = np.ogrid[:size, :size]
    circle_mask = ((X - cx)**2 + (Y - cy)**2 <= (size * 0.30)**2).astype(np.float32)
    mask = mask * circle_mask

    augmented = (1.0 - mask[:, :, np.newaxis]) * image_rgb_float + mask[:, :, np.newaxis] * augmented
    augmented = np.clip(augmented, 0.0, 1.0)
    return augmented.astype(np.float32), mask[:, :, np.newaxis].astype(np.float32)

def apply_random_anomaly(image_rgb_float, train_paths, size=256, force_type=None):
    types = ['perlin', 'scratch', 'mixed']
    anomaly_type = force_type if force_type else random.choice(types)
    if anomaly_type == 'perlin':
        return create_synthetic_anomaly(image_rgb_float, train_paths, size)
    elif anomaly_type == 'scratch':
        return generate_scratch_anomaly(image_rgb_float, size)
    elif anomaly_type == 'mixed':
        aug_1, mask_1 = create_synthetic_anomaly(image_rgb_float, train_paths, size)
        aug_2, mask_2 = generate_scratch_anomaly(aug_1, size)
        final_mask = np.clip(mask_1 + mask_2, 0.0, 1.0)
        return aug_2, final_mask
    return image_rgb_float, np.zeros((size, size, 1), dtype=np.float32)

# Hiển thị trực quan tạo lỗi giả
if train_paths:
    bgr, _ = crop_circular_can(train_paths[0])
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    display_types = ['perlin', 'perlin', 'scratch', 'mixed']
    titles = ['Perlin Noise 1', 'Perlin Noise 2', 'Vết Xước (Scratch)', 'Kết hợp (Mixed)']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i in range(4):
        aug, mask = apply_random_anomaly(img_rgb, train_paths, size=256, force_type=display_types[i])
        axes[0, i].imshow(aug); axes[0, i].axis('off')
        axes[0, i].set_title(f"Ảnh: {titles[i]}", fontsize=12)
        axes[1, i].imshow(mask[:,:,0], cmap='hot'); axes[1, i].axis('off')
        axes[1, i].set_title(f"Mask: {titles[i]}", fontsize=12)
    plt.suptitle("Kết quả Sinh lỗi giả: Perlin Noise & Scratch (Mỏng nhẹ & Hiển thị chuẩn 100%)\n"
                 "Hàng trên: Ảnh lon sau chồng lỗi  |  Hàng dưới: Mặt nạ nhị phân (Ground Truth)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("synthetic_anomaly_samples.png", dpi=200)
    plt.show()
    print("[✓] Đã lưu minh họa: 'synthetic_anomaly_samples.png'")

# ==============================================================================
# BƯỚC 5: KIẾN TRÚC MẠNG DRAEM (Reconstructive & Discriminative Subnetwork)
# ==============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_ch)
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, 3, padding=1), nn.BatchNorm2d(base_ch * 16), nn.ReLU(True),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1), nn.BatchNorm2d(base_ch * 16), nn.ReLU(True),
        )
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8,  base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8,  base_ch * 4,  base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2,  base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2,  base_ch,      base_ch)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch, 3, 1), nn.Sigmoid())
    def forward(self, x):
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        p4, s4 = self.enc4(p3)
        b  = self.bottleneck(p4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1)

class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=6, base_ch=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_ch)
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, 3, padding=1), nn.BatchNorm2d(base_ch * 16), nn.ReLU(True),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1), nn.BatchNorm2d(base_ch * 16), nn.ReLU(True),
        )
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8,  base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8,  base_ch * 4,  base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2,  base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2,  base_ch,      base_ch)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch, 1, 1), nn.Sigmoid())
    def forward(self, x):
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        p4, s4 = self.enc4(p3)
        b  = self.bottleneck(p4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1)
recon_net = ReconstructiveSubNetwork().to(device)
discr_net = DiscriminativeSubNetwork().to(device)
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"[✓] Reconstructive Network : {count_params(recon_net):,} tham số")
print(f"[✓] Discriminative Network : {count_params(discr_net):,} tham số")

# ==============================================================================
# BƯỚC 6: HÀM MẤT MÁT (LOSS FUNCTIONS)
# ==============================================================================
def ssim_loss(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    mu_x   = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y   = F.avg_pool2d(y, window_size, 1, window_size // 2)
    mu_x2  = mu_x.pow(2);  mu_y2 = mu_y.pow(2);  mu_xy = mu_x * mu_y
    sig_x  = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x2
    sig_y  = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y2
    sig_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2) + 1e-8)
    return 1.0 - ssim_map.mean()

def reconstruction_loss(recon, target):
    l1   = F.l1_loss(recon, target)
    ssim = ssim_loss(recon, target)
    return l1 + ssim, l1.item(), ssim.item()

def focal_loss(pred_mask, gt_mask, gamma=4.0, alpha=0.5):
    eps = 1e-8
    pred_mask = torch.clamp(pred_mask, eps, 1 - eps)
    bce  = -(gt_mask * torch.log(pred_mask) + (1 - gt_mask) * torch.log(1 - pred_mask))
    pt   = torch.where(gt_mask == 1, pred_mask, 1 - pred_mask)
    fl   = alpha * ((1 - pt) ** gamma) * bce
    return fl.mean()

# ==============================================================================
# BƯỚC 7: DATASET VÀ HUẤN LUYỆN MODEL DRAEM
# ==============================================================================
class DRAEMDataset(Dataset):
    def __init__(self, paths, img_size=256, preload=True):
        self.paths    = paths
        self.img_size = img_size
        self.images   = []
        if preload:
            for p in tqdm(paths, desc="Tiền xử lý lon"):
                bgr, _ = crop_circular_can(p, target_size=(img_size, img_size))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                else:
                    rgb = np.array(Image.open(p).convert("RGB").resize((img_size, img_size)), dtype=np.float32) / 255.0
                self.images.append(rgb)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = self.images[idx]
        if random.random() < 0.5:
            aug, mask = apply_random_anomaly(img, self.paths, size=self.img_size)
        else:
            aug = img.copy()
            mask = np.zeros((self.img_size, self.img_size, 1), dtype=np.float32)
        t_aug = torch.from_numpy(aug.transpose(2, 0, 1))
        t_raw = torch.from_numpy(img.transpose(2, 0, 1))
        t_msk = torch.from_numpy(mask.transpose(2, 0, 1))
        return t_aug, t_raw, t_msk, self.paths[idx]
BATCH_SIZE = 8
IMG_SIZE   = 256
NUM_EPOCHS = 100
train_dataset = DRAEMDataset(train_paths, img_size=IMG_SIZE, preload=True)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
optimizer_r = torch.optim.Adam(recon_net.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(discr_net.parameters(), lr=1e-4)
scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=NUM_EPOCHS)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=NUM_EPOCHS)
loss_history = {'total': [], 'recon': [], 'focal': [], 'l1': [], 'ssim': []}
print(f"[*] Bắt đầu huấn luyện DRAEM – {NUM_EPOCHS} Epochs trên {device}...")
start_t = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    recon_net.train(); discr_net.train()
    ep_total, ep_recon, ep_focal = 0.0, 0.0, 0.0
    ep_l1,    ep_ssim            = 0.0, 0.0
    n_samples = 0
    for aug_img, raw_img, gt_mask, _ in train_loader:
        aug_img = aug_img.to(device)
        raw_img = raw_img.to(device)
        gt_mask = gt_mask.to(device)
        recon = recon_net(aug_img)
        disc_input = torch.cat([aug_img, recon.detach()], dim=1)
        pred_mask  = discr_net(disc_input)
        loss_r, l1_val, ssim_val = reconstruction_loss(recon, raw_img)
        loss_d = focal_loss(pred_mask, gt_mask)
        loss   = loss_r + loss_d
        optimizer_r.zero_grad(); optimizer_d.zero_grad()
        loss.backward()
        optimizer_r.step(); optimizer_d.step()
        bs = aug_img.size(0)
        ep_total  += loss.item()   * bs
        ep_recon  += loss_r.item() * bs
        ep_focal  += loss_d.item() * bs
        ep_l1     += l1_val        * bs
        ep_ssim   += ssim_val      * bs
        n_samples += bs
    scheduler_r.step(); scheduler_d.step()
    loss_history['total'].append(ep_total / n_samples)
    loss_history['recon'].append(ep_recon / n_samples)
    loss_history['focal'].append(ep_focal / n_samples)
    loss_history['l1'   ].append(ep_l1    / n_samples)
    loss_history['ssim' ].append(ep_ssim  / n_samples)
    if epoch % 5 == 0 or epoch == NUM_EPOCHS:
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Total: {loss_history['total'][-1]:.4f} | Recon: {loss_history['recon'][-1]:.4f} | Focal: {loss_history['focal'][-1]:.4f}")
elapsed = time.time() - start_t
print(f"\n[✓] Huấn luyện hoàn tất trong {elapsed/60:.2f} phút!")

# Lưu file weights đễ nạp vào máy local
torch.save({
    'recon_state': recon_net.state_dict(),
    'discr_state': discr_net.state_dict(),
}, "draem_fulldata.pth")
print("[✓] Đã lưu trọng số mô hình: 'draem_fulldata.pth'")

# Biểu đồ hàm loss
epochs_arr = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(10, 4.5))
plt.plot(epochs_arr, loss_history['total'], label='Total Loss', color='crimson', lw=2.5)
plt.plot(epochs_arr, loss_history['recon'], label='Reconstruction Loss (L1 + SSIM)', color='royalblue', lw=1.8, linestyle='--')
plt.plot(epochs_arr, loss_history['focal'], label='Focal Loss (Discrimination)', color='forestgreen', lw=1.8, linestyle='--')
plt.title("Đường cong hội tụ hàm mất mát DRAEM (Loss Convergence Curve)", fontsize=13, fontweight='bold')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout(); plt.savefig("draem_loss_curve.png", dpi=300); plt.show()
# ==============================================================================
# BƯỚC 8: TÍNH ĐIỂM ANOMALY SCORE & XÁC ĐỊNH NGƯỠNG PHÂN LOẠI
# ==============================================================================
recon_net.eval(); discr_net.eval()
def compute_score(img_path):
    bgr, _ = crop_circular_can(img_path)
    if bgr is None: return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(device)
    with torch.no_grad():
        recon     = recon_net(t)
        disc_in   = torch.cat([t, recon], dim=1)
        pred_mask = discr_net(disc_in)
    flat  = pred_mask.view(-1)
    k     = max(1, int(len(flat) * 0.01))
    return float(torch.topk(flat, k=k).values.mean().item())
test_normal_scores = []
for p in tqdm(test_paths, desc="Test-OK", leave=False):
    s = compute_score(p)
    if s is not None: test_normal_scores.append(s)
test_normal_scores = np.array(test_normal_scores)
mean_s = float(np.mean(test_normal_scores))
std_s  = float(np.std(test_normal_scores))
threshold = float(np.percentile(test_normal_scores, 98.0)) + 0.1 * std_s
print(f"[✓] Điểm bất thường TB (lon chuẩn) : {mean_s:.4f} ± {std_s:.4f}")
print(f"[★] NGƯỠNG PHÂN LOẠI (THRESHOLD)   : {threshold:.4f}")
ng_scores = []
for p in tqdm(ng_paths, desc="Test-NG", leave=False):
    s = compute_score(p)
    if s is not None: ng_scores.append(s)
ng_scores = np.array(ng_scores)
print(f"[✓] Score TB của NG : {ng_scores.mean():.4f} ± {ng_scores.std():.4f}")
# Cập nhật checkpoint lưu kèm Threshold
ckpt = torch.load("draem_fulldata.pth", map_location=device)
ckpt['threshold'] = threshold
torch.save(ckpt, "draem_fulldata.pth")
# Trực quan hóa phân phối điểm bất thường
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
all_min = min(test_normal_scores.min(), ng_scores.min())
all_max = max(test_normal_scores.max(), ng_scores.max())
bins = np.linspace(all_min, all_max, 50)
ax = axes[0]
ax.hist(test_normal_scores, bins=bins, alpha=0.65, color='#2196F3', label=f'OK (n={len(test_normal_scores):,})')
ax.hist(ng_scores, bins=bins, alpha=0.65, color='#F44336', label=f'NG (n={len(ng_scores):,})')
ax.axvline(threshold, color='orange', lw=2.5, ls='--', label=f'Ngưỡng τ = {threshold:.4f}')
ax.set_title('Histogram Anomaly Score', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
ax2 = axes[1]
x_range = np.linspace(all_min, all_max, 300)
for scores, color, label in [(test_normal_scores, '#2196F3', 'OK'), (ng_scores, '#F44336', 'NG')]:
    kde = gaussian_kde(scores)
    ax2.plot(x_range, kde(x_range), color=color, lw=2.5, label=label)
ax2.axvline(threshold, color='orange', lw=2.5, ls='--'); ax2.set_title('KDE Mật độ xác suất', fontweight='bold'); ax2.legend()
ax3 = axes[2]
bp = ax3.boxplot([test_normal_scores, ng_scores], patch_artist=True, widths=0.5)
ax3.axhline(threshold, color='orange', lw=2.5, ls='--')
ax3.set_xticklabels(['LON OK', 'LON NG']); ax3.set_title('Box Plot OK vs NG', fontweight='bold')
plt.tight_layout()
plt.savefig('draem_score_distribution.png', dpi=300)
plt.show()

# ==============================================================================
# BƯỚC 9: TRỰC QUAN HÓA HEATMAP LỖI PIXEL-LEVEL
# ==============================================================================
def inspect_draem(image_path_or_arr, recon_net, discr_net, threshold, device):
    if isinstance(image_path_or_arr, str):
        bgr, _ = crop_circular_can(image_path_or_arr)
    else:
        bgr = image_path_or_arr
    if bgr is None: return None
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t    = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        recon     = recon_net(t)
        disc_in   = torch.cat([t, recon], dim=1)
        pred_mask = discr_net(disc_in)
    latency = (time.time() - t0) * 1000
    diff_map = torch.abs(t - recon)[0].cpu().numpy().mean(axis=0)
    norm_map = np.clip(diff_map / (threshold * 0.3), 0.0, 1.0)
    heatmap  = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap[bgr.sum(axis=2) == 0] = 0
    overlay  = cv2.addWeighted(bgr, 0.55, heatmap, 0.45, 0)
    flat  = pred_mask.view(-1)
    k     = max(1, int(len(flat) * 0.01))
    score = float(torch.topk(flat, k=k).values.mean().item())
    is_ng = score > threshold
    return {
        'orig': rgb, 'recon': recon[0].cpu().numpy().transpose(1,2,0),
        'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
        'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        'mask': norm_map, 'score': score, 'is_ng': is_ng, 'latency': latency
    }
samples_ok = [inspect_draem(p, recon_net, discr_net, threshold, device) for p in test_paths[:2]]
samples_ng = [inspect_draem(p, recon_net, discr_net, threshold, device) for p in ng_paths[:2]]
fig, axes = plt.subplots(4, 4, figsize=(15, 14))
cols = ["Ảnh cắt tròn gốc", "Ảnh tái tạo", "Bản đồ lỗi pixel", "Phủ màu Overlay"]
for ax, c in zip(axes[0], cols): ax.set_title(c, fontsize=10, fontweight='bold')
for row, (r, lbl) in enumerate(zip(samples_ok, ["OK lon 1", "OK lon 2"])):
    axes[row, 0].imshow(r['orig']); axes[row, 0].set_ylabel(f"{lbl}\nScore: {r['score']:.4f}\n🟢 OK", fontsize=9)
    axes[row, 1].imshow(np.clip(r['recon'], 0, 1))
    axes[row, 2].imshow(r['mask'], cmap='hot', vmin=0, vmax=1)
    axes[row, 3].imshow(r['overlay'])
    for ax in axes[row]: ax.axis('off')
for row, (r, lbl) in enumerate(zip(samples_ng, ["NG lon 1", "NG lon 2"])):
    label_text = '🔴 NG (Phát hiện)' if r['is_ng'] else '⚠️ NG (Bỏ sót)'
    axes[2+row, 0].imshow(r['orig']); axes[2+row, 0].set_ylabel(f"{lbl}\nScore: {r['score']:.4f}\n{label_text}", fontsize=9)
    axes[2+row, 1].imshow(np.clip(r['recon'], 0, 1))
    axes[2+row, 2].imshow(r['mask'], cmap='hot', vmin=0, vmax=1)
    axes[2+row, 3].imshow(r['overlay'])
    for ax in axes[2+row]: ax.axis('off')
plt.suptitle("DRAEM – Trực quan hóa Heatmap pixel-level", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig("draem_heatmap_comparison.png", dpi=200); plt.show()

# ==============================================================================
# BƯỚC 10: ĐÁNH GIÁ ĐỘ CHÍNH XÁC TOÀN DIỆN & VẼ MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ==============================================================================
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
y_true, y_pred, all_scores, all_times = [], [], [], []
def run_draem_inference(img_path, true_label):
    res = inspect_draem(img_path, recon_net, discr_net, threshold, device)
    if res is None: return
    y_true.append(true_label)
    y_pred.append(1 if res['is_ng'] else 0)
    all_scores.append(res['score'])
    all_times.append(res['latency'])
for p in tqdm(test_paths, desc="Lon OK", leave=False): run_draem_inference(p, 0)
for p in tqdm(ng_paths,   desc="Lon NG", leave=False): run_draem_inference(p, 1)
y_true = np.array(y_true); y_pred = np.array(y_pred); all_scores = np.array(all_scores)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, zero_division=0)
rec   = recall_score(y_true, y_pred, zero_division=0)
spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1    = f1_score(y_true, y_pred, zero_division=0)
fpr_r, tpr_r, _ = roc_curve(y_true, all_scores)
auroc = auc(fpr_r, tpr_r)
avg_ms = np.mean(all_times)
print("\n" + "=" * 62)
print("      BẢNG KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH DRAEM")
print("=" * 62)
display(pd.DataFrame([
    {"Chỉ số": "True Positive (TP)", "Giá trị": f"{tp} lon", "Ý nghĩa": "Lon LỖI → đoán đúng là LỖI"},
    {"Chỉ số": "True Negative (TN)", "Giá trị": f"{tn} lon", "Ý nghĩa": "Lon TỐT → đoán đúng là TỐT"},
    {"Chỉ số": "Accuracy – Độ chính xác", "Giá trị": f"{acc*100:.2f}%", "Ý nghĩa": "(TP+TN) / Tổng mẫu"},
    {"Chỉ số": "Precision – Độ chuẩn xác", "Giá trị": f"{prec*100:.2f}%", "Ý nghĩa": "TP / (TP+FP)"},
    {"Chỉ số": "Recall – Độ nhạy", "Giá trị": f"{rec*100:.2f}%", "Ý nghĩa": "TP / (TP+FN)"},
    {"Chỉ số": "F1-Score", "Giá trị": f"{f1:.4f}", "Ý nghĩa": "2×P×R / (P+R)"},
    {"Chỉ số": "AUROC", "Giá trị": f"{auroc*100:.2f}%", "Ý nghĩa": "Năng lực phân tách OK/NG"},
    {"Chỉ số": "Tốc độ suy luận", "Giá trị": f"{avg_ms:.1f} ms ({1000/avg_ms:.0f} FPS)", "Ý nghĩa": "Real-time trên dây chuyền"},
]))
# Biểu đồ Confusion Matrix + ROC
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
annot = np.array([
    [f"TN={tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP={fp}\n({fp/(tn+fp)*100:.1f}%)"],
    [f"FN={fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP={tp}\n({tp/(fn+tp)*100:.1f}%)"]
])
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Dự đoán: OK","Dự đoán: NG"], yticklabels=["Thực tế: OK","Thực tế: NG"],
            linewidths=2, linecolor="white", annot_kws={"size":13,"weight":"bold"}, ax=axes[0])
axes[0].set_title(f"Confusion Matrix\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}", fontweight='bold')
axes[1].plot(fpr_r, tpr_r, color='crimson', lw=2.5, label=f"ROC Curve (AUROC = {auroc:.4f})")
axes[1].plot([0,1],[0,1], color='steelblue', lw=1.5, ls='--')
axes[1].set_title("Đường cong ROC", fontweight='bold'); axes[1].legend(loc='lower right')
plt.tight_layout(); plt.savefig("draem_danh_gia_mo_hinh.png", dpi=300); plt.show()

đây là code làm ở colab data cũ 










</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-03T13:08:02+07:00.

The user's current state is as follows:
Other open documents:
- d:\bc_moi\draem_model.py (LANGUAGE_UNSPECIFIED)
- d:\bc_moi\main.py (LANGUAGE_UNSPECIFIED)
- d:\bc_moi\static\index.html (LANGUAGE_UNSPECIFIED)
- d:\bc_moi\config.yaml (LANGUAGE_UNSPECIFIED)
- d:\bc_moi\test1.ino (LANGUAGE_UNSPECIFIED)

The user has uploaded 2 image(s):
- C:/Users/84382/.gemini/antigravity/brain/852dc058-b23c-4e0e-9ed1-bf0ce02e6d26/.user_uploaded/media_1793685839807_1026c2c6.jpg
- C:/Users/84382/.gemini/antigravity/brain/852dc058-b23c-4e0e-9ed1-bf0ce02e6d26/.user_uploaded/media_1793685923191_324cb05b.png
You can embed these images in an artifact if you need the USER to review them.
</ADDITIONAL_METADATA>