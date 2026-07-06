# =============================================================================
# STEP 6: Compare Baseline Models with Proposed CLAHE-RATS-Net
# =============================================================================

import os
import time
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"ACTUAL_PATH")

ROI_DATASET_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_proposed_comparison"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PRED_DIR = OUTPUT_DIR / "predictions"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROI_DATASET_DIR / "train_roi_clahe.csv"
VAL_CSV = ROI_DATASET_DIR / "val_roi_clahe.csv"
TEST_CSV = ROI_DATASET_DIR / "test_roi_clahe.csv"

SEED = 42
ROI_SIZE = 256

NUM_EPOCHS = 250
BATCH_SIZE = 16
NUM_WORKERS = 0

LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 40
SCHEDULER_PATIENCE = 10
GRAD_CLIP = 1.0

USE_AMP = True
USE_TTA = True

REMOVE_SMALL_OBJECTS = True
USE_CLICK_CENTERED_COMPONENT = True
MIN_OBJECT_AREA = 10

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

BASE_CHANNELS = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run all models. To test only one or two models, edit this list.
MODELS_TO_RUN = [
    "UNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "ResUNet",
    "TransUNetLite",
    "CLAHE_RATS_Net"
]


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seed(SEED)


# =============================================================================
# 3. DATASET AND PREPROCESSING
# =============================================================================

def safe_read_image(path):
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if arr is None:
            raise ValueError(f"Could not read image: {path}")

        arr = arr.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def create_click_map_from_mask(mask, sigma=10):
    mask_bin = (mask > 0.5).astype(np.uint8)

    if mask_bin.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    ys, xs = np.where(mask_bin > 0)
    cy = int(round(np.mean(ys)))
    cx = int(round(np.mean(xs)))

    h, w = mask.shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
    click = click.astype(np.float32)
    click = click / (click.max() + 1e-8)

    return click


def resize_image_mask_click(image, mask, click, size=256):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    click = cv2.resize(click, (size, size), interpolation=cv2.INTER_LINEAR)

    mask = (mask > 0.5).astype(np.float32)
    click = np.clip(click, 0.0, 1.0).astype(np.float32)

    return image, mask, click


def apply_train_augmentation(image, mask, click):
    # Horizontal flip
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        click = np.fliplr(click).copy()

    # Mild vertical flip
    if random.random() < 0.2:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        click = np.flipud(click).copy()

    # Mild rotation
    if random.random() < 0.35:
        angle = random.uniform(-8, 8)
        h, w = image.shape
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(
            image, mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        mask = cv2.warpAffine(
            mask, mat, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        click = cv2.warpAffine(
            click, mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # Mild brightness / contrast
    if random.random() < 0.30:
        alpha = random.uniform(0.95, 1.05)
        beta = random.uniform(-0.03, 0.03)
        image = np.clip(image * alpha + beta, 0.0, 1.0)

    # Mild Gaussian noise
    if random.random() < 0.10:
        noise = np.random.normal(0, 0.005, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    mask = (mask > 0.5).astype(np.float32)
    click = np.clip(click, 0.0, 1.0).astype(np.float32)

    return image, mask, click


class LungROIDataset(Dataset):
    def __init__(self, csv_path, train=False, roi_size=256):
        self.csv_path = Path(csv_path)
        self.train = train
        self.roi_size = roi_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        print(f"Loaded {self.csv_path.name}: {len(self.df)} samples")
        print(f"Columns: {list(self.df.columns)}")

        if "image_path" not in self.df.columns:
            raise ValueError("CSV must contain image_path column")

        if "mask_path" not in self.df.columns:
            raise ValueError("CSV must contain mask_path column")

        self.has_click = "click_path" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = safe_read_image(row["image_path"])
        mask = safe_read_image(row["mask_path"])
        mask = (mask > 0.5).astype(np.float32)

        if self.has_click and isinstance(row.get("click_path", ""), str) and os.path.exists(row["click_path"]):
            click = safe_read_image(row["click_path"])
        else:
            click = create_click_map_from_mask(mask)

        image, mask, click = resize_image_mask_click(image, mask, click, self.roi_size)

        if self.train:
            image, mask, click = apply_train_augmentation(image, mask, click)

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        click_t = torch.from_numpy(click).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        x = torch.cat([image_t, click_t], dim=0)

        mask_area = int(mask.sum())

        if mask_area < 50:
            size_group = "small_<50"
        elif mask_area <= 200:
            size_group = "medium_50_200"
        else:
            size_group = "large_>200"

        file_name = (
            f"{row.get('patient_id', 'patient')}"
            f"_slice{row.get('slice_index', idx)}"
            f"_v{row.get('variant_id', 0)}"
        )

        return {
            "image": x,
            "mask": mask_t,
            "file_name": file_name,
            "mask_area": mask_area,
            "size_group": size_group
        }


# =============================================================================
# 4. MODEL BLOCKS
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        att = self.sigmoid(avg_out + max_out)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        att = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(att))

        return x * att


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# =============================================================================
# 5. BASELINE MODELS
# =============================================================================

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8, dropout=0.1)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16, dropout=0.2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8, dropout=0.1)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16, dropout=0.2)
        self.cbam = CBAM(base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.att1 = AttentionGate(base, base, max(base // 2, 4))
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        b = self.cbam(b)

        d4 = self.up4(b)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))

        d3 = self.up3(d4)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        return self.out(d1)


class ResUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base=32):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConv(in_channels, base)
        self.enc2 = ResidualConv(base, base * 2)
        self.enc3 = ResidualConv(base * 2, base * 4)
        self.enc4 = ResidualConv(base * 4, base * 8, dropout=0.1)

        self.bottleneck = ResidualConv(base * 8, base * 16, dropout=0.2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = ResidualConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = ResidualConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ResidualConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ResidualConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


class UNetPlusPlus(nn.Module):
    """
    Compact UNet++ implementation for 256x256 ROI segmentation.
    Uses nested skip connections up to 4 encoder levels.
    """

    def __init__(self, in_channels=2, out_channels=1, base=32):
        super().__init__()

        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = DoubleConv(in_channels, nb[0])
        self.conv1_0 = DoubleConv(nb[0], nb[1])
        self.conv2_0 = DoubleConv(nb[1], nb[2])
        self.conv3_0 = DoubleConv(nb[2], nb[3])
        self.conv4_0 = DoubleConv(nb[3], nb[4], dropout=0.2)

        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = DoubleConv(nb[3] + nb[4], nb[3])

        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = DoubleConv(nb[2] * 2 + nb[3], nb[2])

        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = DoubleConv(nb[1] * 3 + nb[2], nb[1])

        self.conv0_4 = DoubleConv(nb[0] * 4 + nb[1], nb[0])

        self.out = nn.Conv2d(nb[0], out_channels, 1)

    def up(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, x0_0)], dim=1))

        return self.out(x0_4)


class TransUNetLite(nn.Module):
    """
    Lightweight TransUNet-style baseline:
    CNN encoder + Transformer bottleneck + U-Net decoder.
    """

    def __init__(self, in_channels=2, out_channels=1, base=32,
                 num_heads=4, num_layers=2):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8, dropout=0.1)

        dim = base * 8

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.bottleneck = DoubleConv(dim, base * 16, dropout=0.2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)            # 256
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 64
        e4 = self.enc4(self.pool(e3))  # 32

        z = self.pool(e4)              # 16, channels=base*8
        b, c, h, w = z.shape

        tokens = z.flatten(2).transpose(1, 2)  # B, HW, C
        tokens = self.transformer(tokens)
        z = tokens.transpose(1, 2).reshape(b, c, h, w)

        z = self.bottleneck(z)

        d4 = self.up4(z)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# =============================================================================
# 6. PROPOSED MODEL: CLAHE-RATS-NET
# =============================================================================

class ResidualRefinementUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResidualConv(in_channels, base)
        self.enc2 = ResidualConv(base, base * 2)
        self.enc3 = ResidualConv(base * 2, base * 4)
        self.enc4 = ResidualConv(base * 4, base * 8, dropout=0.1)

        self.bottleneck = ResidualConv(base * 8, base * 16, dropout=0.2)
        self.cbam = CBAM(base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = ResidualConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = ResidualConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = ResidualConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.att1 = AttentionGate(base, base, max(base // 2, 4))
        self.dec1 = ResidualConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        b = self.cbam(b)

        d4 = self.up4(b)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))

        d3 = self.up3(d4)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        return self.out(d1)


class CLAHE_RATS_Net(nn.Module):
    def __init__(self, base=32):
        super().__init__()

        self.stage1 = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base=base
        )

        self.stage2 = ResidualRefinementUNet(
            in_channels=3,
            out_channels=1,
            base=base
        )

    def forward(self, x):
        clahe = x[:, 0:1]
        click = x[:, 1:2]

        coarse_logits = self.stage1(x)
        coarse_prob = torch.sigmoid(coarse_logits)

        stage2_input = torch.cat([clahe, click, coarse_prob], dim=1)
        residual_logits = self.stage2(stage2_input)

        final_logits = coarse_logits + residual_logits

        return {
            "coarse_logits": coarse_logits,
            "residual_logits": residual_logits,
            "final_logits": final_logits
        }


# =============================================================================
# 7. LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = ((1 - targets) * probs).sum(dim=1)
        fn = (targets * (1 - probs)).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = torch.pow((1 - tversky), self.gamma)

        return loss.mean()


class CombinedSegLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ft = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        ft_loss = self.ft(logits, targets)

        total = (
            0.30 * bce_loss +
            0.45 * dice_loss +
            0.25 * ft_loss
        )

        return total


class UnifiedLoss(nn.Module):
    """
    For baseline models:
        loss = CombinedSegLoss(logits, target)

    For proposed CLAHE-RATS-Net:
        loss = 0.30 * coarse_loss + 0.70 * final_loss
    """

    def __init__(self):
        super().__init__()
        self.seg_loss = CombinedSegLoss()

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            coarse_loss = self.seg_loss(outputs["coarse_logits"], targets)
            final_loss = self.seg_loss(outputs["final_logits"], targets)
            total_loss = 0.30 * coarse_loss + 0.70 * final_loss

            return total_loss, coarse_loss.detach(), final_loss.detach()

        loss = self.seg_loss(outputs, targets)

        return loss, torch.tensor(0.0, device=targets.device), loss.detach()


# =============================================================================
# 8. METRICS AND POST-PROCESSING
# =============================================================================

def get_final_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["final_logits"]
    return outputs


def dice_score_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice.mean().item()


def compute_numpy_metrics(pred, gt, smooth=1e-6):
    pred = pred.astype(np.uint8).flatten()
    gt = gt.astype(np.uint8).flatten()

    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity
    }


def get_click_center_from_map(click_map):
    if click_map is None or click_map.max() <= 0:
        h, w = click_map.shape
        return w // 2, h // 2

    y, x = np.unravel_index(np.argmax(click_map), click_map.shape)
    return int(x), int(y)


def post_process_mask(prob, click_map=None, threshold=0.5):
    pred = (prob >= threshold).astype(np.uint8)

    if REMOVE_SMALL_OBJECTS:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)

        cleaned = np.zeros_like(pred)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= MIN_OBJECT_AREA:
                cleaned[labels == i] = 1

        pred = cleaned.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)

    if num_labels <= 1:
        return pred.astype(np.float32)

    if USE_CLICK_CENTERED_COMPONENT and click_map is not None:
        click_x, click_y = get_click_center_from_map(click_map)

        best_label = None
        best_score = 1e18

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            dist = np.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)

            score = dist - 0.005 * area

            if score < best_score:
                best_score = score
                best_label = i

        pred = (labels == best_label).astype(np.uint8)

    else:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred = (labels == largest_label).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    return pred.astype(np.float32)


# =============================================================================
# 9. MODEL FACTORY
# =============================================================================

def build_model(model_name):
    if model_name == "UNet":
        return UNet(in_channels=2, out_channels=1, base=BASE_CHANNELS)

    if model_name == "AttentionUNet":
        return AttentionUNet(in_channels=2, out_channels=1, base=BASE_CHANNELS)

    if model_name == "UNetPlusPlus":
        return UNetPlusPlus(in_channels=2, out_channels=1, base=BASE_CHANNELS)

    if model_name == "ResUNet":
        return ResUNet(in_channels=2, out_channels=1, base=BASE_CHANNELS)

    if model_name == "TransUNetLite":
        return TransUNetLite(in_channels=2, out_channels=1, base=BASE_CHANNELS)

    if model_name == "CLAHE_RATS_Net":
        return CLAHE_RATS_Net(base=BASE_CHANNELS)

    raise ValueError(f"Unknown model name: {model_name}")


# =============================================================================
# 10. TRAINING, VALIDATION, EVALUATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_coarse = 0.0
    total_final = 0.0

    progress = tqdm(loader, desc="Training", leave=False)

    for batch in progress:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)
                loss, coarse_loss, final_loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss, coarse_loss, final_loss = criterion(outputs, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf loss detected. Skipping batch.")
            continue

        if USE_AMP and DEVICE.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        final_logits = get_final_logits(outputs)
        dice = dice_score_from_logits(final_logits, masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        total_coarse += float(coarse_loss.item())
        total_final += float(final_loss.item())

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}"
        })

    n = max(len(loader), 1)

    return total_loss / n, total_dice / n, total_coarse / n, total_final / n


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_coarse = 0.0
    total_final = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)
                loss, coarse_loss, final_loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss, coarse_loss, final_loss = criterion(outputs, masks)

        final_logits = get_final_logits(outputs)
        dice = dice_score_from_logits(final_logits, masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        total_coarse += float(coarse_loss.item())
        total_final += float(final_loss.item())

    n = max(len(loader), 1)

    return total_loss / n, total_dice / n, total_coarse / n, total_final / n


@torch.no_grad()
def predict_with_tta(model, image_tensor):
    model.eval()

    probs = []

    outputs = model(image_tensor)
    probs.append(torch.sigmoid(get_final_logits(outputs)))

    if USE_TTA:
        # Horizontal flip
        x_flip = torch.flip(image_tensor, dims=[3])
        outputs_flip = model(x_flip)
        prob_flip = torch.sigmoid(get_final_logits(outputs_flip))
        prob_flip = torch.flip(prob_flip, dims=[3])
        probs.append(prob_flip)

        # Vertical flip
        y_flip = torch.flip(image_tensor, dims=[2])
        outputs_yflip = model(y_flip)
        prob_yflip = torch.sigmoid(get_final_logits(outputs_yflip))
        prob_yflip = torch.flip(prob_yflip, dims=[2])
        probs.append(prob_yflip)

    avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

    return avg_prob


@torch.no_grad()
def evaluate_thresholds(model, loader, model_name, split_name="val", save_predictions=False):
    model.eval()

    all_summary_rows = []
    all_sample_rows = []

    for threshold in THRESHOLDS:
        rows = []

        for batch in tqdm(loader, desc=f"{model_name} {split_name} threshold {threshold}", leave=False):
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks_np = batch["mask"].cpu().numpy()
            file_names = batch["file_name"]
            mask_areas = batch["mask_area"]
            size_groups = batch["size_group"]

            probs = predict_with_tta(model, images)
            probs_np = probs.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()

            for i in range(probs_np.shape[0]):
                prob = probs_np[i, 0]
                gt = masks_np[i, 0]
                click_map = images_np[i, 1]

                pred = post_process_mask(
                    prob=prob,
                    click_map=click_map,
                    threshold=threshold
                )

                metrics = compute_numpy_metrics(pred, gt)

                row = {
                    "model": model_name,
                    "split": split_name,
                    "threshold": threshold,
                    "file_name": file_names[i],
                    "mask_area": int(mask_areas[i]),
                    "size_group": size_groups[i],
                }

                row.update(metrics)
                rows.append(row)
                all_sample_rows.append(row)

                if save_predictions:
                    pred_save_dir = PRED_DIR / model_name / split_name / f"threshold_{threshold:.2f}"
                    pred_save_dir.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_prob.png"), (prob * 255).astype(np.uint8))
                    cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_pred.png"), (pred * 255).astype(np.uint8))
                    cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_gt.png"), (gt * 255).astype(np.uint8))

        df = pd.DataFrame(rows)

        summary = {
            "model": model_name,
            "split": split_name,
            "threshold": threshold,
            "dice_mean": df["dice"].mean(),
            "dice_std": df["dice"].std(),
            "iou_mean": df["iou"].mean(),
            "iou_std": df["iou"].std(),
            "precision_mean": df["precision"].mean(),
            "precision_std": df["precision"].std(),
            "recall_mean": df["recall"].mean(),
            "recall_std": df["recall"].std(),
            "specificity_mean": df["specificity"].mean(),
            "specificity_std": df["specificity"].std(),
        }

        all_summary_rows.append(summary)

    return pd.DataFrame(all_summary_rows), pd.DataFrame(all_sample_rows)


def save_checkpoint(model, optimizer, epoch, val_dice, model_name, path):
    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "val_dice": float(val_dice),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": SEED,
        "roi_size": ROI_SIZE,
        "base_channels": BASE_CHANNELS
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, path):
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


# =============================================================================
# 11. TRAIN SINGLE MODEL
# =============================================================================

def train_and_evaluate_model(model_name, train_loader, val_loader, test_loader):
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 80)

    set_seed(SEED)

    model_output_dir = OUTPUT_DIR / model_name
    model_log_dir = model_output_dir / "logs"
    model_ckpt_dir = model_output_dir / "checkpoints"

    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_log_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = UnifiedLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    best_val_dice = -1.0
    best_epoch = 0
    early_stop_counter = 0

    best_model_path = model_ckpt_dir / f"best_{model_name}.pth"

    log_records = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_dice, train_coarse, train_final = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler
        )

        val_loss, val_dice, val_coarse, val_final = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion
        )

        scheduler.step(val_dice)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print("-" * 80)
        print(f"{model_name} | Epoch [{epoch}/{NUM_EPOCHS}] | LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.6f} | Val Dice:   {val_dice:.4f}")
        print(f"Train coarse/final loss: {train_coarse:.6f} / {train_final:.6f}")
        print(f"Val coarse/final loss:   {val_coarse:.6f} / {val_final:.6f}")
        print(f"Time: {epoch_time:.2f}s")

        log_records.append({
            "model": model_name,
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "train_coarse_loss": train_coarse,
            "train_final_loss": train_final,
            "val_coarse_loss": val_coarse,
            "val_final_loss": val_final,
            "epoch_time_sec": epoch_time
        })

        pd.DataFrame(log_records).to_csv(model_log_dir / "training_log.csv", index=False)
        pd.DataFrame(log_records).to_excel(model_log_dir / "training_log.xlsx", index=False)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            early_stop_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_dice=val_dice,
                model_name=model_name,
                path=best_model_path
            )

            print(f"New best model saved. Best Val Dice: {best_val_dice:.6f}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stopping counter: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Dice during training: {best_val_dice:.6f}")

    print("\nLoading best checkpoint for threshold evaluation...")
    checkpoint = load_checkpoint(model, best_model_path)
    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Loaded val dice: {checkpoint['val_dice']:.6f}")

    print("\nValidation threshold evaluation...")
    val_summary, val_samples = evaluate_thresholds(
        model=model,
        loader=val_loader,
        model_name=model_name,
        split_name="val",
        save_predictions=False
    )

    val_summary.to_csv(model_log_dir / "validation_threshold_summary.csv", index=False)
    val_summary.to_excel(model_log_dir / "validation_threshold_summary.xlsx", index=False)

    val_samples.to_csv(model_log_dir / "validation_samplewise_metrics.csv", index=False)
    val_samples.to_excel(model_log_dir / "validation_samplewise_metrics.xlsx", index=False)

    best_val_row = val_summary.loc[val_summary["dice_mean"].idxmax()]
    best_threshold = float(best_val_row["threshold"])

    print(val_summary)
    print(f"\nBest validation threshold for {model_name}: {best_threshold}")

    print("\nTest threshold evaluation...")
    test_summary, test_samples = evaluate_thresholds(
        model=model,
        loader=test_loader,
        model_name=model_name,
        split_name="test",
        save_predictions=True if model_name == "CLAHE_RATS_Net" else False
    )

    test_summary.to_csv(model_log_dir / "test_threshold_summary.csv", index=False)
    test_summary.to_excel(model_log_dir / "test_threshold_summary.xlsx", index=False)

    test_samples.to_csv(model_log_dir / "test_samplewise_metrics.csv", index=False)
    test_samples.to_excel(model_log_dir / "test_samplewise_metrics.xlsx", index=False)

    print(test_summary)

    best_test_row = test_summary[test_summary["threshold"] == best_threshold].iloc[0]

    test_best_thr = test_samples[test_samples["threshold"] == best_threshold].copy()

    if len(test_best_thr) > 0:
        size_summary = (
            test_best_thr
            .groupby("size_group")
            .agg({
                "dice": ["mean", "std", "count"],
                "iou": ["mean", "std"],
                "precision": ["mean", "std"],
                "recall": ["mean", "std"]
            })
        )

        size_summary.to_excel(model_log_dir / "test_sizewise_summary.xlsx")
        print("\nSize-wise test summary:")
        print(size_summary)

    final_row = {
        "model": model_name,
        "seed": SEED,
        "roi_size": ROI_SIZE,
        "input": "CLAHE ROI + click map",
        "best_epoch": best_epoch,
        "best_val_training_dice": best_val_dice,
        "best_val_threshold": best_threshold,

        "best_val_dice": best_val_row["dice_mean"],
        "best_val_iou": best_val_row["iou_mean"],
        "best_val_precision": best_val_row["precision_mean"],
        "best_val_recall": best_val_row["recall_mean"],
        "best_val_specificity": best_val_row["specificity_mean"],

        "best_test_dice": best_test_row["dice_mean"],
        "best_test_iou": best_test_row["iou_mean"],
        "best_test_precision": best_test_row["precision_mean"],
        "best_test_recall": best_test_row["recall_mean"],
        "best_test_specificity": best_test_row["specificity_mean"],

        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "loss": "BCE + Dice + Focal Tversky",
        "postprocessing": "click-centered connected component + remove small objects",
        "use_tta": USE_TTA
    }

    pd.DataFrame([final_row]).to_csv(model_log_dir / "final_summary.csv", index=False)
    pd.DataFrame([final_row]).to_excel(model_log_dir / "final_summary.xlsx", index=False)

    return final_row


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BASELINE AND PROPOSED MODEL COMPARISON")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"ROI dataset directory: {ROI_DATASET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print("\nConfiguration:")
    print(f"ROI_SIZE: {ROI_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"LR: {LR}")
    print(f"PATIENCE: {PATIENCE}")
    print(f"THRESHOLDS: {THRESHOLDS}")
    print(f"MIN_OBJECT_AREA: {MIN_OBJECT_AREA}")
    print(f"MODELS_TO_RUN: {MODELS_TO_RUN}")

    train_dataset = LungROIDataset(TRAIN_CSV, train=True, roi_size=ROI_SIZE)
    val_dataset = LungROIDataset(VAL_CSV, train=False, roi_size=ROI_SIZE)
    test_dataset = LungROIDataset(TEST_CSV, train=False, roi_size=ROI_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    final_rows = []

    for model_name in MODELS_TO_RUN:
        row = train_and_evaluate_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

        final_rows.append(row)

        comparison_df = pd.DataFrame(final_rows)
        comparison_df.to_csv(LOG_DIR / "final_comparison_summary_progress.csv", index=False)
        comparison_df.to_excel(LOG_DIR / "final_comparison_summary_progress.xlsx", index=False)

    comparison_df = pd.DataFrame(final_rows)

    comparison_df = comparison_df.sort_values(
        by="best_test_dice",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df)

    comparison_df.to_csv(LOG_DIR / "final_comparison_summary.csv", index=False)
    comparison_df.to_excel(LOG_DIR / "final_comparison_summary.xlsx", index=False)

    print("\nAll comparison outputs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
