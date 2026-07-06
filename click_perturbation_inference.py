
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_reproducibility(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for deterministic inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_dataset_checkpoint(test_csv, checkpoint):

    csv_name = Path(test_csv).name.lower()
    ckpt_name = Path(checkpoint).name.lower()

    logger.info("=" * 70)
    logger.info("Dataset : %s", csv_name)
    logger.info("Checkpoint : %s", ckpt_name)
    logger.info("=" * 70)

    if "lung_crop" in csv_name and "roi" in ckpt_name:
        logger.warning(
            "Possible preprocessing mismatch "
            "(lung_crop dataset with ROI checkpoint)."
        )

    if "roi" in csv_name and "lung_crop" in ckpt_name:
        logger.warning(
            "Possible preprocessing mismatch "
            "(ROI dataset with lung_crop checkpoint)."
        )
# ---------------------------------------------------------------------------
# CLAHE-RATS-Net architecture copied from the trained implementation.
# Architecture is unchanged. Only inference-time click maps are replaced.
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """Two 3x3 convolution blocks with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConv(nn.Module):
    """Residual convolution block used by the trained Stage-2 refiner."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AttentionGate(nn.Module):
    """Attention gate used to filter encoder skip features."""

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ChannelAttention(nn.Module):
    """Channel attention module used by CBAM."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        att = self.sigmoid(avg_out + max_out)
        return x * att


class SpatialAttention(nn.Module):
    """Spatial attention module used by CBAM."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class CBAM(nn.Module):
    """Convolutional block attention module."""

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class AttentionUNet(nn.Module):
    """Attention U-Net used as the Stage-1 coarse branch."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        ch = base_channels
        self.enc1 = DoubleConv(in_channels, ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(ch, ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(ch * 2, ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(ch * 4, ch * 8, dropout=0.1)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(ch * 8, ch * 16, dropout=0.2)
        self.cbam = CBAM(ch * 16)
        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(ch * 8, ch * 8, ch * 4)
        self.dec4 = DoubleConv(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(ch * 4, ch * 4, ch * 2)
        self.dec3 = DoubleConv(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(ch * 2, ch * 2, ch)
        self.dec2 = DoubleConv(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.att1 = AttentionGate(ch, ch, max(ch // 2, 4))
        self.dec1 = DoubleConv(ch * 2, ch)
        self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        b = self.cbam(b)

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        return self.out(d1)


class ResidualRefinementUNet(nn.Module):
    """Stage-2 refinement branch that predicts residual logits."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        ch = base_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResidualConv(in_channels, ch)
        self.enc2 = ResidualConv(ch, ch * 2)
        self.enc3 = ResidualConv(ch * 2, ch * 4)
        self.enc4 = ResidualConv(ch * 4, ch * 8, dropout=0.1)
        self.bottleneck = ResidualConv(ch * 8, ch * 16, dropout=0.2)
        self.cbam = CBAM(ch * 16)
        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(ch * 8, ch * 8, ch * 4)
        self.dec4 = ResidualConv(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(ch * 4, ch * 4, ch * 2)
        self.dec3 = ResidualConv(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(ch * 2, ch * 2, ch)
        self.dec2 = ResidualConv(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.att1 = AttentionGate(ch, ch, max(ch // 2, 4))
        self.dec1 = ResidualConv(ch * 2, ch)
        self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        b = self.cbam(b)

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        return self.out(d1)


class CLAHE_RATS_Net(nn.Module):
    """Two-stage residual attention model used for inference."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.stage1 = AttentionUNet(in_channels=2, out_channels=1, base_channels=base_channels)
        self.stage2 = ResidualRefinementUNet(in_channels=3, out_channels=1, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        clahe_image = x[:, 0:1, :, :]
        click_map = x[:, 1:2, :, :]
        coarse_logits = self.stage1(x)
        coarse_prob = torch.sigmoid(coarse_logits)
        stage2_input = torch.cat([clahe_image, click_map, coarse_prob], dim=1)
        residual_logits = self.stage2(stage2_input)
        final_logits = coarse_logits + residual_logits
        return {
            "coarse_logits": coarse_logits,
            "residual_logits": residual_logits,
            "final_logits": final_logits,
        }


# ---------------------------------------------------------------------------
# Data loading and click perturbation
# ---------------------------------------------------------------------------


def read_gray_image(path: str | Path, roi_size: int | Tuple[int, int] | None = 256) -> np.ndarray:
    """Read PNG/JPG/TIF/NPY grayscale image and normalize to [0,1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)

    if isinstance(roi_size, tuple):
        target_shape = roi_size
    elif roi_size is None:
        target_shape = None
    else:
        target_shape = (int(roi_size), int(roi_size))

    if target_shape is not None and arr.shape != target_shape:
        resample = Image.BILINEAR
        target_h, target_w = target_shape
        arr = np.array(Image.fromarray(arr).resize((target_w, target_h), resample=resample), dtype=np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def get_original_click(row: pd.Series, click_map: np.ndarray | None, mask: np.ndarray) -> Tuple[float, float]:
    """Return original click location as (cx, cy)."""
    if "centroid_x" in row and "centroid_y" in row and pd.notna(row["centroid_x"]) and pd.notna(row["centroid_y"]):
        return float(row["centroid_x"]), float(row["centroid_y"])

    if click_map is not None and float(click_map.max()) > 0:
        cy, cx = np.unravel_index(int(np.argmax(click_map)), click_map.shape)
        return float(cx), float(cy)

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        return float(w // 2), float(h // 2)
    return float(xs.mean()), float(ys.mean())



def generate_gaussian_click(cx: float, cy: float, shape: Tuple[int, int] = (256, 256), sigma: float = 10.0) -> np.ndarray:
    """Generate the same normalized Gaussian click map used by CLAHE-RATS-Net."""
    h, w = shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))
    click = click.astype(np.float32)
    click = click / (click.max() + 1e-8)
    return click


def generate_binary_click(cx: float, cy: float, shape: Tuple[int, int], radius: int = 1) -> np.ndarray:
    """Generate the 3x3 binary click marker used in the saved CLAHE click maps."""
    h, w = shape
    x = int(round(float(cx)))
    y = int(round(float(cy)))
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    click = np.zeros((h, w), dtype=np.float32)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    click[y1:y2, x1:x2] = 1.0
    return click


def perturb_click(cx: float, cy: float, magnitude: int, shape: Tuple[int, int]) -> Tuple[float, float, float, float, float]:
    """Randomly perturb click location and clip the point inside the ROI."""
    h, w = shape
    if magnitude <= 0:
        return cx, cy, 0.0, 0.0, 0.0

    sampled_dx = random.uniform(-magnitude, magnitude)
    sampled_dy = random.uniform(-magnitude, magnitude)
    px = float(np.clip(cx + sampled_dx, 0, w - 1))
    py = float(np.clip(cy + sampled_dy, 0, h - 1))
    effective_dx = px - cx
    effective_dy = py - cy
    distance = math.sqrt(effective_dx ** 2 + effective_dy ** 2)
    return px, py, effective_dx, effective_dy, distance


def build_image_id(row: pd.Series, fallback_index: int) -> str:
    """Create a stable image identifier for result tables."""
    patient = str(row.get("patient_id", ""))
    slice_idx = str(row.get("slice_index", ""))
    variant = str(row.get("variant_id", ""))
    if patient and slice_idx:
        return f"{patient}_slice{slice_idx}_v{variant}"
    if "image_path" in row:
        return Path(str(row["image_path"])).stem
    return f"sample_{fallback_index:05d}"


class ClickPerturbationDataset(Dataset):
    """Dataset that replaces each sample's click map with a perturbed Gaussian map."""

    def __init__(
        self,
        csv_path: str | Path,
        perturbation: int,
        sigma: float,
        roi_size: int | Tuple[int, int] | None = 256,
        click_mode: str = "gaussian",
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.perturbation = int(perturbation)
        self.sigma = float(sigma)
        self.roi_size = roi_size
        self.click_mode = click_mode

        required = {"image_path", "mask_path"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing required column(s) in {self.csv_path}: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        image = read_gray_image(row["image_path"], roi_size=self.roi_size)
        mask = read_gray_image(row["mask_path"], roi_size=self.roi_size)
        mask = (mask > 0.5).astype(np.float32)

        original_click_map = None
        if "click_path" in row and isinstance(row.get("click_path"), str) and Path(row["click_path"]).exists():
            original_click_map = read_gray_image(row["click_path"], roi_size=self.roi_size)

        cx, cy = get_original_click(row, original_click_map, mask)
        px, py, dx, dy, distance = perturb_click(cx, cy, self.perturbation, mask.shape)
        if self.click_mode == "binary":
            click = generate_binary_click(px, py, shape=mask.shape)
        elif self.click_mode == "original" and self.perturbation == 0 and original_click_map is not None:
            click = original_click_map.astype(np.float32)
        elif self.click_mode == "original":
            click = generate_binary_click(px, py, shape=mask.shape)
        else:
            click = generate_gaussian_click(px, py, shape=mask.shape, sigma=self.sigma)
        model_input = np.stack([image, click], axis=0).astype(np.float32)

        return {
            "image": torch.from_numpy(model_input),
            "mask": torch.from_numpy(mask[None, :, :].astype(np.float32)),
            "image_id": build_image_id(row, idx),
            "perturbation": self.perturbation,
            "dx": float(dx),
            "dy": float(dy),
            "click_x": float(px),
            "click_y": float(py),
            "click_map": torch.from_numpy(click[None, :, :].astype(np.float32)),
            "distance": float(distance),
        }

def print_dataset_summary(csv_path):

    df = pd.read_csv(csv_path)

    print("\n")
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Images          : {len(df)}")

    if "mask_pixel_count_crop" in df.columns:

        print(
            f"Mean Mask Area  : "
            f"{df['mask_pixel_count_crop'].mean():.2f}"
        )

        print(
            f"Min Mask Area   : "
            f"{df['mask_pixel_count_crop'].min():.2f}"
        )

        print(
            f"Max Mask Area   : "
            f"{df['mask_pixel_count_crop'].max():.2f}"
        )

    print("=" * 60)

# ---------------------------------------------------------------------------
# Model loading, inference, post-processing, and metrics
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str | Path, device: torch.device, base_channels: int | None = None) -> CLAHE_RATS_Net:
    """Load a trained CLAHE-RATS-Net checkpoint for inference only."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        inferred_base_channels = int(checkpoint.get("base_channels", base_channels or 32))
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        inferred_base_channels = int(base_channels or 32)
        state_dict = checkpoint

    model = CLAHE_RATS_Net(base_channels=inferred_base_channels)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def retain_click_centered_component(binary_mask: np.ndarray, click_map: np.ndarray, min_object_area: int) -> np.ndarray:
    """Remove small objects and retain the connected component closest to the click."""
    pred = (binary_mask > 0).astype(np.uint8)
    labels, num_labels = ndi.label(pred, structure=np.ones((3, 3), dtype=np.uint8))
    if num_labels == 0:
        return pred.astype(np.float32)

    click_y, click_x = np.unravel_index(int(np.argmax(click_map)), click_map.shape)
    best_label = None
    best_score = float("inf")

    cleaned = np.zeros_like(pred, dtype=np.uint8)
    for label_id in range(1, num_labels + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < min_object_area:
            continue
        ys, xs = np.where(component)
        cx = float(xs.mean())
        cy = float(ys.mean())
        dist = math.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)
        score = dist - 0.005 * area
        if score < best_score:
            best_score = score
            best_label = label_id

    if best_label is not None:
        cleaned[labels == best_label] = 1

    closed = ndi.binary_closing(cleaned, structure=np.ones((3, 3), dtype=bool))
    return closed.astype(np.float32)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    """Compute Dice, IoU, precision, recall, and specificity."""
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    tn = float(np.logical_and(pred == 0, gt == 0).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    return {
        "Dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        "IoU": (tp + eps) / (tp + fp + fn + eps),
        "Precision": (tp + eps) / (tp + fp + eps),
        "Recall": (tp + eps) / (tp + fn + eps),
        "Specificity": (tn + eps) / (tn + fp + eps),
    }


def evaluate_dataset(
    model: nn.Module,
    test_csv: str | Path,
    perturbation_levels: Iterable[int],
    device: torch.device,
    threshold: float,
    sigma: float,
    roi_size: int | Tuple[int, int] | None,
    batch_size: int,
    min_object_area: int,
    click_mode: str,
) -> pd.DataFrame:
    """Run inference for every perturbation level and return sample-wise results."""
    rows: List[Dict[str, object]] = []

    for perturbation in perturbation_levels:
        dataset = ClickPerturbationDataset(
            test_csv,
            perturbation=perturbation,
            sigma=sigma,
            roi_size=roi_size,
            click_mode=click_mode,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for batch in tqdm(loader, desc=f"Perturbation ±{perturbation}px", leave=True):
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].cpu().numpy()
            click_maps = batch["click_map"].cpu().numpy()

            with torch.inference_mode():
                outputs = model(images)
                probs = torch.sigmoid(outputs["final_logits"]).detach().cpu().numpy()

            for i in range(images.shape[0]):
                prob = probs[i, 0]
                raw_pred = (prob >= threshold).astype(np.uint8)
                pred = retain_click_centered_component(raw_pred, click_maps[i, 0], min_object_area=min_object_area)
                metrics = compute_metrics(pred, masks[i, 0])
                rows.append(
                    {
                        "Image_ID": batch["image_id"][i],
                        "Perturbation_Level": int(perturbation),
                        "dx": float(batch["dx"][i]),
                        "dy": float(batch["dy"][i]),
                        "Dice": metrics["Dice"],
                        "IoU": metrics["IoU"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "Specificity": metrics["Specificity"],
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries, statistics, saving, and plotting
# ---------------------------------------------------------------------------


def summarize_results(samplewise: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation across the full test set."""
    grouped = samplewise.groupby("Perturbation_Level", sort=True)
    summary = grouped.agg(
        Mean_Dice=("Dice", "mean"),
        Std_Dice=("Dice", "std"),
        Mean_IoU=("IoU", "mean"),
        Std_IoU=("IoU", "std"),
        Mean_Precision=("Precision", "mean"),
        Std_Precision=("Precision", "std"),
        Mean_Recall=("Recall", "mean"),
        Std_Recall=("Recall", "std"),
        Mean_Specificity=("Specificity", "mean"),
        Std_Specificity=("Specificity", "std"),
        N=("Dice", "count"),
    ).reset_index()
    summary = summary.rename(columns={"Perturbation_Level": "Perturbation"})
    return summary


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's dz for paired observations."""
    diff = b - a
    std = float(np.std(diff, ddof=1))
    if std <= 1e-12:
        return float("nan")
    return float(np.mean(diff) / std)


def perform_statistics(samplewise: pd.DataFrame, baseline_level: int = 0) -> pd.DataFrame:
    """Compare perfect centroid performance with each perturbed condition."""
    metrics = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    baseline = samplewise[samplewise["Perturbation_Level"] == baseline_level].copy()
    stat_rows: List[Dict[str, object]] = []

    for level in sorted(samplewise["Perturbation_Level"].unique()):
        if int(level) == int(baseline_level):
            continue
        current = samplewise[samplewise["Perturbation_Level"] == level].copy()
        merged = baseline.merge(current, on="Image_ID", suffixes=("_baseline", "_perturbed"))

        for metric in metrics:
            a = merged[f"{metric}_baseline"].to_numpy(dtype=float)
            b = merged[f"{metric}_perturbed"].to_numpy(dtype=float)
            diff = b - a

            if len(diff) >= 3:
                normality_p = float(stats.shapiro(diff).pvalue)
            else:
                normality_p = float("nan")

            if np.isfinite(normality_p) and normality_p > 0.05:
                test_name = "paired t-test"
                test_result = stats.ttest_rel(b, a)
                p_value = float(test_result.pvalue)
            else:
                test_name = "Wilcoxon signed-rank"
                try:
                    test_result = stats.wilcoxon(b, a, zero_method="wilcox")
                    p_value = float(test_result.pvalue)
                except ValueError:
                    p_value = 1.0

            stat_rows.append(
                {
                    "Metric": metric,
                    "Comparison": f"{baseline_level}px vs ±{int(level)}px",
                    "Perturbation_Level": int(level),
                    "Baseline_Mean": float(np.mean(a)),
                    "Perturbed_Mean": float(np.mean(b)),
                    "Mean_Difference": float(np.mean(diff)),
                    "Normality_p": normality_p,
                    "Test": test_name,
                    "p_value": p_value,
                    "Cohens_d": cohens_d_paired(a, b),
                    "N": int(len(diff)),
                }
            )

    return pd.DataFrame(stat_rows)


def save_results(samplewise: pd.DataFrame, summary: pd.DataFrame, statistics_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save all CSV result files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samplewise.to_csv(output_dir / "click_perturbation_samplewise.csv", index=False)
    summary.to_csv(output_dir / "click_perturbation_summary.csv", index=False)
    statistics_df.to_csv(output_dir / "click_perturbation_statistics.csv", index=False)


def plot_results(samplewise: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> None:
    """Create 300 dpi publication-quality plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    levels = sorted(samplewise["Perturbation_Level"].unique())
    dice_groups = [samplewise.loc[samplewise["Perturbation_Level"] == lv, "Dice"].to_numpy() for lv in levels]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.boxplot(dice_groups, labels=[f"{lv}px" for lv in levels], patch_artist=True, showfliers=False)
    for patch in ax.artists:
        patch.set_facecolor("#DCEBFA")
    ax.set_xlabel("Click perturbation level")
    ax.set_ylabel("Dice score")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_dice_vs_perturbation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(summary["Perturbation"], summary["Mean_Dice"], yerr=summary["Std_Dice"], marker="o", capsize=4, linewidth=1.8)
    ax.set_xlabel("Click perturbation level (pixels)")
    ax.set_ylabel("Mean Dice")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "line_mean_dice_with_std.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(summary["Perturbation"], summary["Mean_IoU"], yerr=summary["Std_IoU"], marker="s", capsize=4, linewidth=1.8, color="#B45309")
    ax.set_xlabel("Click perturbation level (pixels)")
    ax.set_ylabel("Mean IoU")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "line_mean_iou_with_std.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-only click perturbation robustness experiment for CLAHE-RATS-Net.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to trained CLAHE-RATS-Net checkpoint.")
    parser.add_argument("--test-csv", required=True, type=str, help="Path to test ROI CSV containing image_path, mask_path, and optional click_path.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory where CSV files and figures will be saved.")
    parser.add_argument("--threshold", default=0.55, type=float, help="Segmentation threshold selected from validation set.")
    parser.add_argument("--sigma", default=10.0, type=float, help="Gaussian click-map sigma in pixels.")
    parser.add_argument("--click-mode", default="gaussian", choices=["gaussian", "binary", "original"], help="Click-map type used for inference. Use binary for the saved 3x3 CLAHE click maps.")
    parser.add_argument("--perturbations", default=[0, 5, 10, 15, 20], nargs="+", type=int, help="Perturbation magnitudes in pixels. Include 0 for perfect centroid baseline.")
    parser.add_argument("--roi-size", default=256, type=int, help="ROI size expected by the model.")
    parser.add_argument("--input-height", default=None, type=int, help="Optional rectangular input height used by the trained model.")
    parser.add_argument("--input-width", default=None, type=int, help="Optional rectangular input width used by the trained model.")
    parser.add_argument("--preserve-size", action="store_true", help="Keep original image/mask dimensions instead of resizing to a square ROI.")
    parser.add_argument("--batch-size", default=8, type=int, help="Inference batch size.")
    parser.add_argument("--min-object-area", default=10, type=int, help="Minimum component area retained during post-processing.")
    parser.add_argument("--base-channels", default=None, type=int, help="Override base channel count if not stored in checkpoint.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducible perturbations.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=device, base_channels=args.base_channels)
    if args.preserve_size:
        input_size = None
    elif args.input_height is not None or args.input_width is not None:
        if args.input_height is None or args.input_width is None:
            raise ValueError("Both --input-height and --input-width must be provided together.")
        input_size = (args.input_height, args.input_width)
    else:
        input_size = args.roi_size

    samplewise = evaluate_dataset(
        model=model,
        test_csv=args.test_csv,
        perturbation_levels=args.perturbations,
        device=device,
        threshold=args.threshold,
        sigma=args.sigma,
        roi_size=input_size,
        batch_size=args.batch_size,
        min_object_area=args.min_object_area,
        click_mode=args.click_mode,
    )
    summary = summarize_results(samplewise)
    statistics_df = perform_statistics(samplewise, baseline_level=0)
    save_results(samplewise, summary, statistics_df, output_dir)
    plot_results(samplewise, summary, output_dir)

    print(f"Saved sample-wise results: {output_dir / 'click_perturbation_samplewise.csv'}")
    print(f"Saved summary results:     {output_dir / 'click_perturbation_summary.csv'}")
    print(f"Saved statistics:          {output_dir / 'click_perturbation_statistics.csv'}")
    print(f"Saved figures to:          {output_dir}")


if __name__ == "__main__":
    main()
