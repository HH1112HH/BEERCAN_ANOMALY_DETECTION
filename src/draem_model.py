import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 3, 1),
            nn.Sigmoid()
        )

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
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Sigmoid()
        )

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


class DRAEM:
    """
    DRAEM Anomaly Detection Model (Chuẩn gốc bài báo & Colab)
    - ReconstructiveSubNetwork (3 -> 3 với Sigmoid)
    - DiscriminativeSubNetwork (6 -> 1 với Sigmoid)
    - Triệt tiêu 100% vòng sáng đèn bằng sai số tái tạo abs(t - recon)
    """
    def __init__(self, model_path: str = "models/draem.pth", threshold: float = 0.25, device: str = "cuda:0"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for DRAEM but not available, falling back to CPU")
            device = "cpu"

        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.img_size = 256

        logger.info(f"Loading DRAEM checkpoint from {model_path} onto {self.device}...")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model_recon = ReconstructiveSubNetwork(in_channels=3, base_ch=32).to(self.device)
        self.model_discr = DiscriminativeSubNetwork(in_channels=6, base_ch=32).to(self.device)

        self.model_recon.load_state_dict(ckpt["recon_state"])
        self.model_discr.load_state_dict(ckpt["discr_state"])

        self.model_recon.eval()
        self.model_discr.eval()

        self.colab_threshold = float(ckpt.get("threshold", 0.25))
        logger.info(f"DRAEM loaded successfully! Model threshold: {self.colab_threshold:.4f}, Active: {self.threshold:.4f}")

    def _preprocess(self, roi: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, roi: np.ndarray, mask: np.ndarray = None, need_heatmap: bool = True) -> dict:
        return self.check(roi, mask=mask, need_heatmap=need_heatmap)

    def check(self, roi: np.ndarray, mask: np.ndarray = None, need_heatmap: bool = True) -> dict:
        if roi is None or roi.size == 0:
            return {
                "algo": "draem",
                "defect": True,
                "score": 1.0,
                "threshold": self.threshold,
                "reason": "Empty ROI",
                "heatmap": None,
            }

        input_tensor = self._preprocess(roi)

        with torch.no_grad():
            rec = self.model_recon(input_tensor)
            inp_discr = torch.cat([input_tensor, rec], dim=1)
            pred_mask = self.model_discr(inp_discr)

        # Tính anomaly score bằng Top 1% pixel lỗi cao nhất (Chuẩn bài báo & code gốc)
        flat = pred_mask.view(-1)
        k = max(1, int(len(flat) * 0.01))
        anomaly_score = float(torch.topk(flat, k=k).values.mean().item())

        is_defect = anomaly_score > self.threshold

        result = {
            "algo": "draem",
            "defect": is_defect,
            "score": round(anomaly_score, 4),
            "threshold": self.threshold,
            "reason": (
                f"DRAEM score={anomaly_score:.3f} > threshold={self.threshold:.2f} (NG)"
                if is_defect
                else f"DRAEM score={anomaly_score:.3f} <= threshold={self.threshold:.2f} (OK)"
            ),
        }

        if need_heatmap:
            # 1. Ảnh tái tạo từ Reconstructive Net (loại bỏ hoàn toàn khuyết tật)
            rec_np = np.clip(rec[0].cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)
            rec_bgr = cv2.cvtColor((rec_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # 2. Bản đồ lỗi Discriminative Net (chuẩn bài báo)
            pm = pred_mask[0, 0].cpu().numpy()

            # BÍ QUYẾT TRIỆT TIÊU VÒNG SÁNG: Hiệu số tái tạo giữa ảnh gốc và ảnh tái tạo
            diff_map = torch.abs(input_tensor - rec)[0].cpu().numpy().mean(axis=0)
            norm_map = np.clip(diff_map / (self.threshold * 0.3 + 1e-6), 0.0, 1.0)

            # Mặt nạ thân lon loại bỏ nền ngoài
            roi_256 = cv2.resize(roi, (self.img_size, self.img_size))
            can_mask = (roi_256.sum(axis=2) > 0)
            norm_map[~can_mask] = 0.0

            # Bản đồ nhiệt thuần túy chuẩn khoa học (JET):
            # Vùng bình thường (norm=0) -> XANH DƯƠNG (Cold)
            # Vùng khuyết tật (norm->1) -> ĐỎ/VÀNG RỰC (Hot)
            # Nền ngoài thân lon -> ĐEN
            heatmap_jet = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_jet[~can_mask] = 0

            # Ảnh phủ nhiệt bán trong suốt (Overlay)
            overlay = cv2.addWeighted(roi_256, 0.55, heatmap_jet, 0.45, 0)
            overlay[~can_mask] = 0

            if mask is not None and mask.size > 0:
                if mask.shape[:2] != (self.img_size, self.img_size):
                    m_resized = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                else:
                    m_resized = mask
                overlay[m_resized == 0] = 0
                heatmap_jet[m_resized == 0] = 0

            h_orig, w_orig = roi.shape[:2]
            result["recon_img"] = rec_bgr
            result["discr_map"] = pm
            result["pure_heatmap"] = cv2.resize(heatmap_jet, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            result["overlay"] = cv2.resize(overlay, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            result["heatmap"] = result["overlay"]

        return result
