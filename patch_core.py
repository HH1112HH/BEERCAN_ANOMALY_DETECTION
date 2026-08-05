import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

logger = logging.getLogger(__name__)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(model.children())[:8])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, mod in self.features._modules.items():
            x = mod(x)
            if name in ['5', '6']:
                outs.append(x)
        return outs


class PatchCore:
    def __init__(self, model_path: str, threshold: float = 0.87, device: str = "cuda:0"):
        self.threshold = threshold
        self.device = device
        self.amp = device.startswith("cuda") and torch.cuda.is_available()

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.coreset = ckpt["coreset"].to(device)
        self.feat_shape = ckpt["feat_shape"]
        self.img_size = ckpt["img_size"]
        self.k_nearest = ckpt["k_nearest"]

        self.backbone = _Backbone().to(device)
        self.backbone.eval()

        if self.amp:
            self.coreset = self.coreset.half()

        self.coreset_norm = F.normalize(self.coreset, dim=1)

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        if self.amp:
            self._mean = self._mean.half()
            self._std = self._std.half()

        logger.info(f"PatchCore loaded: coreset {self.coreset.shape}, img_size={self.img_size}, k={self.k_nearest}")

    def _apply_clahe(self, roi: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _preprocess(self, roi: np.ndarray):
        roi = cv2.resize(roi, (self.img_size, self.img_size))
        roi = self._apply_clahe(roi)
        img_tensor = torch.from_numpy(roi).permute(2, 0, 1).float().div(255.0).to(self.device)
        if self.amp:
            img_tensor = img_tensor.half()
        img_tensor = (img_tensor - self._mean) / self._std
        return img_tensor.unsqueeze(0)

    def _embed(self, img_tensor: torch.Tensor):
        f2, f3 = self.backbone(img_tensor)
        f2_up = F.interpolate(f2, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        fusion = torch.cat([f2_up, f3], dim=1)
        B, C, H, W = fusion.shape
        return fusion.permute(0, 2, 3, 1).reshape(-1, C)

    def check(self, roi: np.ndarray, need_heatmap: bool = True):
        if roi.size == 0:
            return {"defect": True, "score": 1.0, "reason": "Empty ROI"}

        img_tensor = self._preprocess(roi)
        with torch.no_grad():
            if self.amp:
                with torch.autocast("cuda"):
                    patches = self._embed(img_tensor)
                    nn_dist = torch.cdist(F.normalize(patches, dim=1), self.coreset_norm, p=2)
            else:
                patches = self._embed(img_tensor)
                nn_dist = torch.cdist(F.normalize(patches, dim=1), self.coreset_norm, p=2)

            nn_dist = nn_dist.topk(self.k_nearest, largest=False)[0]

        mean_dists = torch.mean(nn_dist, dim=1)
        anomaly_score = float(mean_dists[0].cpu().item())

        is_defect = anomaly_score > self.threshold

        result = {
            "defect": is_defect,
            "score": round(anomaly_score, 4),
            "reason": f"PatchCore score={anomaly_score:.3f} > threshold={self.threshold}" if is_defect else f"PatchCore score={anomaly_score:.3f} OK",
        }

        if need_heatmap:
            pixel_scores = mean_dists.view(-1, *self.feat_shape).cpu().numpy()
            heatmap = pixel_scores[0]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.resize(heatmap, (roi.shape[1], roi.shape[0]))
            result["heatmap"] = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return result