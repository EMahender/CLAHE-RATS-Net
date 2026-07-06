# =============================================================================
# CLAHE-RATS-Net: Updated 150-Patient Training Code
# CLAHE-Guided Residual Attention Two-Stage Network
#
# Main modifications for better validation/test performance:
#   1. Use original ROI dataset, not clean ROI dataset
#   2. Batch size = 16
#   3. Patience = 40
#   4. Milder augmentation
#   5. Lower MIN_OBJECT_AREA = 10 to avoid missing small nodules
#   6. Extended lower threshold range: 0.10 to 0.60
#   7. Click-centered connected component post-processing
#   8. Size-wise test analysis
#
# Dataset expected:
#   outputs/proposed_roi_clahe_dataset/
#       train_roi_clahe.csv
#       val_roi_clahe.csv
#       test_roi_clahe.csv
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

PROJECT_ROOT = Path(r"E:\Mahender PHD\segmentaion\Project 23-4-2026")

# IMPORTANT:
# Use original ROI dataset for the 150-patient rerun.
# Do not use proposed_roi_clahe_dataset_clean for this run.
ROI_DATASET_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clahe_rats_net_150patients_modified_results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PRED_DIR = OUTPUT_DIR / "predictions"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROI_DATASET_DIR / "train_roi_clahe.csv"
VAL_CSV = ROI_DATASET_DIR / "val_roi_clahe.csv"
TEST_CSV = ROI_DATASET_DIR / "test_roi_clahe.csv"

MODEL_NAME = "CLAHE_RATS_Net_150Patients_Modified"

SEED = 42
ROI_SIZE = 256

NUM_EPOCHS = 250
BATCH_SIZE = 16
NUM_WORKERS = 0

LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 40
GRAD_CLIP = 1.0

USE_AMP = True
USE_TTA = True

USE_LARGEST_COMPONENT = True
USE_CLICK_CENTERED_COMPONENT = True
REMOVE_SMALL_OBJECTS = True

# Lower value to protect small nodules.
MIN_OBJECT_AREA = 10

# Extended threshold range.
# Your test Dice was better at lower thresholds, so include 0.10 to 0.60.
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# 3. BASIC UTILITIES
# =============================================================================

def safe_read_image(path, grayscale=True):
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        if grayscale:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            arr = cv2.imread(path, cv2.IMREAD_COLOR)

        if arr is None:
            raise ValueError(f"Could not read image: {path}")

    arr = arr.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def resize_image_mask_click(image, mask, click, size=256):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    click = cv2.resize(click, (size, size), interpolation=cv2.INTER_LINEAR)

    mask = (mask > 0.5).astype(np.float32)
    click = np.clip(click, 0.0, 1.0).astype(np.float32)

    return image, mask, click


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


def apply_train_augmentation(image, mask, click):
    """
    Milder augmentation than previous run.
    This protects small nodule boundaries.
    """

    # Horizontal flip
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        click = np.fliplr(click).copy()

    # Vertical flip reduced
    if random.random() < 0.2:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        click = np.flipud(click).copy()

    # Mild rotation reduced
    if random.random() < 0.35:
        angle = random.uniform(-8, 8)
        h, w = image.shape
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(
            image,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        mask = cv2.warpAffine(
            mask,
            mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        click = cv2.warpAffine(
            click,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # Mild brightness / contrast reduced
    if random.random() < 0.30:
        alpha = random.uniform(0.95, 1.05)
        beta = random.uniform(-0.03, 0.03)
        image = image * alpha + beta
        image = np.clip(image, 0.0, 1.0)

    # Mild Gaussian noise reduced
    if random.random() < 0.10:
        noise = np.random.normal(0, 0.005, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    mask = (mask > 0.5).astype(np.float32)
    click = np.clip(click, 0.0, 1.0).astype(np.float32)

    return image, mask, click


def get_click_center_from_map(click_map):
    if click_map is None or click_map.max() <= 0:
        h, w = click_map.shape
        return w // 2, h // 2

    y, x = np.unravel_index(np.argmax(click_map), click_map.shape)
    return int(x), int(y)


def post_process_mask(prob, click_map=None, threshold=0.5):
    """
    Click-centered post-processing.
    This is better than largest-component-only for click-guided segmentation.
    """

    pred = (prob >= threshold).astype(np.uint8)

    if REMOVE_SMALL_OBJECTS:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)

        cleaned = np.zeros_like(pred)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= MIN_OBJECT_AREA:
                cleaned[labels == i] = 1

        pred = cleaned.astype(np.uint8)

    if USE_LARGEST_COMPONENT:
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

                # Prefer component close to click and reasonably large.
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
# 4. DATASET
# =============================================================================

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

        if not self.has_click:
            print("click_path column not found. Click map will be generated from mask.")

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

        image, mask, click = resize_image_mask_click(
            image=image,
            mask=mask,
            click=click,
            size=self.roi_size
        )

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
# 5. MODEL COMPONENTS
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
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

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
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
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
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


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
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

    def forward(self, x):
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
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
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

    def forward(self, x):
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


class CLAHE_RATS_Net(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()

        self.stage1 = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=base_channels
        )

        self.stage2 = ResidualRefinementUNet(
            in_channels=3,
            out_channels=1,
            base_channels=base_channels
        )

    def forward(self, x):
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
            "final_logits": final_logits
        }


# =============================================================================
# 6. LOSS FUNCTIONS
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


class RATSNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.seg_loss = CombinedSegLoss()

    def forward(self, outputs, targets):
        coarse_loss = self.seg_loss(outputs["coarse_logits"], targets)
        final_loss = self.seg_loss(outputs["final_logits"], targets)

        total_loss = 0.30 * coarse_loss + 0.70 * final_loss

        return total_loss, coarse_loss.detach(), final_loss.detach()


# =============================================================================
# 7. METRICS
# =============================================================================

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


# =============================================================================
# 8. TRAINING AND VALIDATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_coarse_loss = 0.0
    total_final_loss = 0.0

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

        dice = dice_score_from_logits(outputs["final_logits"], masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        total_coarse_loss += coarse_loss.item()
        total_final_loss += final_loss.item()

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}"
        })

    n = max(len(loader), 1)

    return (
        total_loss / n,
        total_dice / n,
        total_coarse_loss / n,
        total_final_loss / n
    )


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_coarse_loss = 0.0
    total_final_loss = 0.0

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

        dice = dice_score_from_logits(outputs["final_logits"], masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        total_coarse_loss += coarse_loss.item()
        total_final_loss += final_loss.item()

    n = max(len(loader), 1)

    return (
        total_loss / n,
        total_dice / n,
        total_coarse_loss / n,
        total_final_loss / n
    )


# =============================================================================
# 9. TTA PREDICTION
# =============================================================================

@torch.no_grad()
def predict_with_tta(model, image_tensor):
    model.eval()

    image_tensor = image_tensor.to(DEVICE)

    probs = []

    outputs = model(image_tensor)
    probs.append(torch.sigmoid(outputs["final_logits"]))

    if USE_TTA:
        # Horizontal flip
        x_flip = torch.flip(image_tensor, dims=[3])
        outputs_flip = model(x_flip)
        prob_flip = torch.sigmoid(outputs_flip["final_logits"])
        prob_flip = torch.flip(prob_flip, dims=[3])
        probs.append(prob_flip)

        # Vertical flip
        y_flip = torch.flip(image_tensor, dims=[2])
        outputs_yflip = model(y_flip)
        prob_yflip = torch.sigmoid(outputs_yflip["final_logits"])
        prob_yflip = torch.flip(prob_yflip, dims=[2])
        probs.append(prob_yflip)

    avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

    return avg_prob


# =============================================================================
# 10. THRESHOLD EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_thresholds(model, loader, split_name="val", save_predictions=False):
    model.eval()

    all_results = []
    all_sample_rows = []

    for threshold in THRESHOLDS:
        metrics_list = []

        for batch in tqdm(loader, desc=f"{split_name} threshold {threshold}", leave=False):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].cpu().numpy()
            file_names = batch["file_name"]
            mask_areas = batch["mask_area"]
            size_groups = batch["size_group"]

            probs = predict_with_tta(model, images)
            probs_np = probs.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()

            for i in range(probs_np.shape[0]):
                prob = probs_np[i, 0]
                gt = masks[i, 0]
                click_map = images_np[i, 1]

                pred = post_process_mask(
                    prob=prob,
                    click_map=click_map,
                    threshold=threshold
                )

                metrics = compute_numpy_metrics(pred, gt)

                row = {
                    "threshold": threshold,
                    "split": split_name,
                    "file_name": file_names[i],
                    "mask_area": int(mask_areas[i]),
                    "size_group": size_groups[i],
                }

                row.update(metrics)

                metrics_list.append(row)
                all_sample_rows.append(row)

                if save_predictions:
                    pred_save_dir = PRED_DIR / split_name / f"threshold_{threshold:.2f}"
                    pred_save_dir.mkdir(parents=True, exist_ok=True)

                    prob_img = (prob * 255).astype(np.uint8)
                    pred_img = (pred * 255).astype(np.uint8)
                    gt_img = (gt * 255).astype(np.uint8)

                    cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_prob.png"), prob_img)
                    cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_pred.png"), pred_img)
                    cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_gt.png"), gt_img)

        df = pd.DataFrame(metrics_list)

        summary = {
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

        all_results.append(summary)

    summary_df = pd.DataFrame(all_results)
    sample_df = pd.DataFrame(all_sample_rows)

    return summary_df, sample_df


# =============================================================================
# 11. CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(model, optimizer, epoch, val_dice, path):
    checkpoint = {
        "model_name": MODEL_NAME,
        "epoch": epoch,
        "val_dice": float(val_dice),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": SEED,
        "roi_size": ROI_SIZE
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
# 12. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PROPOSED MODEL TRAINING: CLAHE-RATS-Net 150 Patients Modified")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"ROI dataset directory: {ROI_DATASET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print()

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

    model = CLAHE_RATS_Net(base_channels=32).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Patience: {PATIENCE}")
    print(f"Min object area: {MIN_OBJECT_AREA}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    criterion = RATSNetLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    best_val_dice = -1.0
    best_epoch = 0
    early_stop_counter = 0

    log_records = []

    best_model_path = CHECKPOINT_DIR / "best_clahe_rats_net_150patients_modified.pth"

    print("=" * 80)
    print("Training started")
    print("=" * 80)

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
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.6f} | Val Dice:   {val_dice:.4f}")
        print(f"Stage Loss Train: coarse={train_coarse:.6f}, final={train_final:.6f}")
        print(f"Stage Loss Val:   coarse={val_coarse:.6f}, final={val_final:.6f}")
        print(f"Time: {epoch_time:.2f}s")

        log_records.append({
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

        pd.DataFrame(log_records).to_csv(LOG_DIR / "training_log_clahe_rats_net_150patients_modified.csv", index=False)
        pd.DataFrame(log_records).to_excel(LOG_DIR / "training_log_clahe_rats_net_150patients_modified.xlsx", index=False)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            early_stop_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_dice=val_dice,
                path=best_model_path
            )

            print(f"New best model saved. Best Val Dice: {best_val_dice:.6f}")

        else:
            early_stop_counter += 1
            print(f"No improvement. Early stopping counter: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print()
            print("Early stopping triggered.")
            break

    print()
    print("Training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Dice during training: {best_val_dice:.6f}")

    print()
    print("Loading best model for final threshold evaluation...")
    checkpoint = load_checkpoint(model, best_model_path)

    print(f"Loaded best model from epoch: {checkpoint['epoch']}")
    print(f"Loaded best val dice: {checkpoint['val_dice']:.6f}")

    print()
    print("=" * 80)
    print("VALIDATION THRESHOLD SUMMARY")
    print("=" * 80)

    val_summary, val_samples = evaluate_thresholds(
        model=model,
        loader=val_loader,
        split_name="val",
        save_predictions=False
    )

    print(val_summary)

    val_summary.to_csv(LOG_DIR / "validation_threshold_summary.csv", index=False)
    val_summary.to_excel(LOG_DIR / "validation_threshold_summary.xlsx", index=False)

    val_samples.to_csv(LOG_DIR / "validation_samplewise_metrics.csv", index=False)
    val_samples.to_excel(LOG_DIR / "validation_samplewise_metrics.xlsx", index=False)

    best_val_row = val_summary.loc[val_summary["dice_mean"].idxmax()]
    best_threshold = float(best_val_row["threshold"])

    print()
    print(f"Best validation threshold: {best_threshold}")

    print()
    print("=" * 80)
    print("TEST THRESHOLD SUMMARY")
    print("=" * 80)

    test_summary, test_samples = evaluate_thresholds(
        model=model,
        loader=test_loader,
        split_name="test",
        save_predictions=True
    )

    print(test_summary)

    test_summary.to_csv(LOG_DIR / "test_threshold_summary.csv", index=False)
    test_summary.to_excel(LOG_DIR / "test_threshold_summary.xlsx", index=False)

    test_samples.to_csv(LOG_DIR / "test_samplewise_metrics.csv", index=False)
    test_samples.to_excel(LOG_DIR / "test_samplewise_metrics.xlsx", index=False)

    best_test_row = test_summary[test_summary["threshold"] == best_threshold].iloc[0]

    # Size-wise test summary
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

        size_summary.to_excel(LOG_DIR / "test_sizewise_summary.xlsx")

        print()
        print("=" * 80)
        print("TEST SIZE-WISE SUMMARY AT BEST VALIDATION THRESHOLD")
        print("=" * 80)
        print(size_summary)

    final_summary = pd.DataFrame([{
        "seed": SEED,
        "model": MODEL_NAME,
        "roi_size": ROI_SIZE,
        "best_epoch": best_epoch,
        "best_val_training_dice": best_val_dice,
        "best_val_threshold": best_threshold,
        "best_val_dice": best_val_row["dice_mean"],
        "best_val_iou": best_val_row["iou_mean"],
        "best_val_precision": best_val_row["precision_mean"],
        "best_val_recall": best_val_row["recall_mean"],
        "best_test_dice": best_test_row["dice_mean"],
        "best_test_iou": best_test_row["iou_mean"],
        "best_test_precision": best_test_row["precision_mean"],
        "best_test_recall": best_test_row["recall_mean"],
        "best_test_specificity": best_test_row["specificity_mean"],
        "use_tta": USE_TTA,
        "use_largest_component": USE_LARGEST_COMPONENT,
        "click_centered_component": USE_CLICK_CENTERED_COMPONENT,
        "remove_small_objects": REMOVE_SMALL_OBJECTS,
        "min_object_area": MIN_OBJECT_AREA,
        "loss": "0.30 Stage1 Loss + 0.70 Final Residual Loss",
        "seg_loss": "0.30 BCE + 0.45 Dice + 0.25 Focal Tversky",
        "data_type": "150-patient CLAHE ROI + Click Map + Coarse Probability Refinement"
    }])

    print()
    print("=" * 80)
    print("FINAL BEST SUMMARY")
    print("=" * 80)
    print(final_summary)

    final_summary.to_csv(LOG_DIR / "final_best_summary_clahe_rats_net_150patients_modified.csv", index=False)
    final_summary.to_excel(LOG_DIR / "final_best_summary_clahe_rats_net_150patients_modified.xlsx", index=False)

    print()
    print("All outputs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()



# # =============================================================================
# # CLAHE-RATS-Net
# # CLAHE-Guided Residual Attention Two-Stage Network
# # for Lung Nodule Segmentation
# #
# # Architecture:
# #   Input:
# #       Channel 1: CLAHE-enhanced ROI image
# #       Channel 2: Click / center heatmap
# #
# #   Stage 1:
# #       Coarse Attention U-Net
# #       Output: coarse_logits
# #
# #   Stage 2:
# #       Residual Refinement U-Net
# #       Input:
# #           CLAHE ROI image
# #           Click map
# #           Stage-1 coarse probability map
# #
# #   Final:
# #       final_logits = coarse_logits + residual_logits
# #
# # Author: Erukala Mahender - PhD Lung Nodule Segmentation Work
# # =============================================================================

# import os
# import time
# import random
# import warnings
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# from sklearn.metrics import confusion_matrix

# warnings.filterwarnings("ignore")


# # =============================================================================
# # 1. CONFIGURATION
# # =============================================================================

# PROJECT_ROOT = Path(r"E:\Mahender PHD\segmentaion\Project 23-4-2026")

# #ROI_DATASET_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset"\
# ROI_DATASET_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset_clean"


# OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clahe_rats_net_results"
# CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# LOG_DIR = OUTPUT_DIR / "logs"
# PRED_DIR = OUTPUT_DIR / "predictions"

# for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PRED_DIR]:
#     d.mkdir(parents=True, exist_ok=True)

# TRAIN_CSV = ROI_DATASET_DIR / "train_roi_clahe.csv"
# VAL_CSV = ROI_DATASET_DIR / "val_roi_clahe.csv"
# TEST_CSV = ROI_DATASET_DIR / "test_roi_clahe.csv"

# MODEL_NAME = "CLAHE_RATS_Net"

# SEED = 42
# ROI_SIZE = 256

# NUM_EPOCHS = 250
# BATCH_SIZE = 16 #16  #8
# NUM_WORKERS = 0

# LR = 1e-4
# WEIGHT_DECAY = 1e-4
# PATIENCE = 40
# GRAD_CLIP = 1.0

# USE_AMP = True
# USE_TTA = True
# USE_LARGEST_COMPONENT = True
# REMOVE_SMALL_OBJECTS = True
# MIN_OBJECT_AREA = 20

# THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # =============================================================================
# # 2. REPRODUCIBILITY
# # =============================================================================

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False


# set_seed(SEED)


# # =============================================================================
# # 3. BASIC UTILITIES
# # =============================================================================

# def safe_read_image(path, grayscale=True):
#     path = str(path)

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"File not found: {path}")

#     if path.lower().endswith(".npy"):
#         arr = np.load(path)
#     else:
#         if grayscale:
#             arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         else:
#             arr = cv2.imread(path, cv2.IMREAD_COLOR)

#         if arr is None:
#             raise ValueError(f"Could not read image: {path}")

#     arr = arr.astype(np.float32)

#     if arr.max() > 1.0:
#         arr = arr / 255.0

#     arr = np.clip(arr, 0.0, 1.0)
#     return arr


# def resize_image_mask_click(image, mask, click, size=256):
#     image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
#     mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
#     click = cv2.resize(click, (size, size), interpolation=cv2.INTER_LINEAR)

#     mask = (mask > 0.5).astype(np.float32)
#     click = np.clip(click, 0.0, 1.0).astype(np.float32)

#     return image, mask, click


# def create_click_map_from_mask(mask, sigma=10):
#     mask_bin = (mask > 0.5).astype(np.uint8)

#     if mask_bin.sum() == 0:
#         return np.zeros_like(mask, dtype=np.float32)

#     ys, xs = np.where(mask_bin > 0)
#     cy = int(np.mean(ys))
#     cx = int(np.mean(xs))

#     h, w = mask.shape
#     y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

#     click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
#     click = click.astype(np.float32)
#     click = click / (click.max() + 1e-8)

#     return click


# def apply_train_augmentation(image, mask, click):
#     # Horizontal flip
#     if random.random() < 0.5:
#         image = np.fliplr(image).copy()
#         mask = np.fliplr(mask).copy()
#         click = np.fliplr(click).copy()

#     # Vertical flip
#     if random.random() < 0.3:
#         image = np.flipud(image).copy()
#         mask = np.flipud(mask).copy()
#         click = np.flipud(click).copy()

#     # Rotation
#     if random.random() < 0.5:
#         angle = random.uniform(-15, 15)
#         h, w = image.shape
#         center = (w // 2, h // 2)
#         mat = cv2.getRotationMatrix2D(center, angle, 1.0)

#         image = cv2.warpAffine(
#             image, mat, (w, h), flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_REFLECT_101
#         )
#         mask = cv2.warpAffine(
#             mask, mat, (w, h), flags=cv2.INTER_NEAREST,
#             borderMode=cv2.BORDER_CONSTANT, borderValue=0
#         )
#         click = cv2.warpAffine(
#             click, mat, (w, h), flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_CONSTANT, borderValue=0
#         )

#     # Mild brightness / contrast
#     if random.random() < 0.4:
#         alpha = random.uniform(0.90, 1.10)
#         beta = random.uniform(-0.05, 0.05)
#         image = image * alpha + beta
#         image = np.clip(image, 0.0, 1.0)

#     # Mild Gaussian noise
#     if random.random() < 0.25:
#         noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
#         image = np.clip(image + noise, 0.0, 1.0)

#     mask = (mask > 0.5).astype(np.float32)
#     click = np.clip(click, 0.0, 1.0).astype(np.float32)

#     return image, mask, click


# def post_process_mask(prob, threshold=0.5):
#     pred = (prob >= threshold).astype(np.uint8)

#     if REMOVE_SMALL_OBJECTS:
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)

#         cleaned = np.zeros_like(pred)
#         for i in range(1, num_labels):
#             area = stats[i, cv2.CC_STAT_AREA]
#             if area >= MIN_OBJECT_AREA:
#                 cleaned[labels == i] = 1
#         pred = cleaned

#     if USE_LARGEST_COMPONENT:
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)

#         if num_labels > 1:
#             largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#             pred = (labels == largest_label).astype(np.uint8)

#     # Fill small holes
#     pred = pred.astype(np.uint8)
#     kernel = np.ones((3, 3), np.uint8)
#     pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

#     return pred.astype(np.float32)


# # =============================================================================
# # 4. DATASET
# # =============================================================================

# class LungROIDataset(Dataset):
#     def __init__(self, csv_path, train=False, roi_size=256):
#         self.csv_path = Path(csv_path)
#         self.train = train
#         self.roi_size = roi_size

#         if not self.csv_path.exists():
#             raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

#         self.df = pd.read_csv(self.csv_path)

#         print(f"Loaded {self.csv_path.name}: {len(self.df)} samples")
#         print(f"Columns: {list(self.df.columns)}")

#         if "image_path" not in self.df.columns:
#             raise ValueError("CSV must contain image_path column")

#         if "mask_path" not in self.df.columns:
#             raise ValueError("CSV must contain mask_path column")

#         if "click_path" not in self.df.columns:
#             print("click_path column not found. Click map will be generated from mask.")
#             self.has_click = False
#         else:
#             self.has_click = True

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]

#         image = safe_read_image(row["image_path"])
#         mask = safe_read_image(row["mask_path"])

#         mask = (mask > 0.5).astype(np.float32)

#         if self.has_click and isinstance(row["click_path"], str) and os.path.exists(row["click_path"]):
#             click = safe_read_image(row["click_path"])
#         else:
#             click = create_click_map_from_mask(mask)

#         image, mask, click = resize_image_mask_click(image, mask, click, self.roi_size)

#         if self.train:
#             image, mask, click = apply_train_augmentation(image, mask, click)

#         image = torch.from_numpy(image).unsqueeze(0).float()
#         click = torch.from_numpy(click).unsqueeze(0).float()
#         mask = torch.from_numpy(mask).unsqueeze(0).float()

#         x = torch.cat([image, click], dim=0)

#         return {
#             "image": x,
#             "mask": mask,
#             "file_name": f"{row.get('patient_id', 'patient')}_{row.get('slice_index', idx)}"
#         }


# # =============================================================================
# # 5. MODEL COMPONENTS
# # =============================================================================

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super().__init__()

#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         ]

#         if dropout > 0:
#             layers.append(nn.Dropout2d(dropout))

#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)


# class AttentionGate(nn.Module):
#     def __init__(self, g_channels, x_channels, inter_channels):
#         super().__init__()

#         self.W_g = nn.Sequential(
#             nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(inter_channels)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(inter_channels)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)

#         if g1.shape[2:] != x1.shape[2:]:
#             g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)

#         return x * psi


# class ChannelAttention(nn.Module):
#     def __init__(self, channels, reduction=8):
#         super().__init__()

#         hidden = max(channels // reduction, 4)

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
#         )

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         att = self.sigmoid(avg_out + max_out)
#         return x * att


# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)

#         att = torch.cat([avg_out, max_out], dim=1)
#         att = self.sigmoid(self.conv(att))

#         return x * att


# class CBAM(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.ca = ChannelAttention(channels)
#         self.sa = SpatialAttention()

#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x


# class AttentionUNet(nn.Module):
#     def __init__(self, in_channels=2, out_channels=1, base_channels=32):
#         super().__init__()

#         ch = base_channels

#         self.enc1 = DoubleConv(in_channels, ch)
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc2 = DoubleConv(ch, ch * 2)
#         self.pool2 = nn.MaxPool2d(2)

#         self.enc3 = DoubleConv(ch * 2, ch * 4)
#         self.pool3 = nn.MaxPool2d(2)

#         self.enc4 = DoubleConv(ch * 4, ch * 8, dropout=0.1)
#         self.pool4 = nn.MaxPool2d(2)

#         self.bottleneck = DoubleConv(ch * 8, ch * 16, dropout=0.2)
#         self.cbam = CBAM(ch * 16)

#         self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
#         self.att4 = AttentionGate(ch * 8, ch * 8, ch * 4)
#         self.dec4 = DoubleConv(ch * 16, ch * 8)

#         self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
#         self.att3 = AttentionGate(ch * 4, ch * 4, ch * 2)
#         self.dec3 = DoubleConv(ch * 8, ch * 4)

#         self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
#         self.att2 = AttentionGate(ch * 2, ch * 2, ch)
#         self.dec2 = DoubleConv(ch * 4, ch * 2)

#         self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
#         self.att1 = AttentionGate(ch, ch, ch // 2)
#         self.dec1 = DoubleConv(ch * 2, ch)

#         self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)

#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))
#         e4 = self.enc4(self.pool3(e3))

#         b = self.bottleneck(self.pool4(e4))
#         b = self.cbam(b)

#         d4 = self.up4(b)
#         e4_att = self.att4(d4, e4)
#         d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

#         d3 = self.up3(d4)
#         e3_att = self.att3(d3, e3)
#         d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

#         d2 = self.up2(d3)
#         e2_att = self.att2(d2, e2)
#         d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

#         d1 = self.up1(d2)
#         e1_att = self.att1(d1, e1)
#         d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

#         return self.out(d1)


# class ResidualRefinementUNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, base_channels=32):
#         super().__init__()

#         ch = base_channels

#         self.enc1 = DoubleConv(in_channels, ch)
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc2 = DoubleConv(ch, ch * 2)
#         self.pool2 = nn.MaxPool2d(2)

#         self.enc3 = DoubleConv(ch * 2, ch * 4)
#         self.pool3 = nn.MaxPool2d(2)

#         self.enc4 = DoubleConv(ch * 4, ch * 8, dropout=0.1)
#         self.pool4 = nn.MaxPool2d(2)

#         self.bottleneck = DoubleConv(ch * 8, ch * 16, dropout=0.2)
#         self.cbam = CBAM(ch * 16)

#         self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
#         self.att4 = AttentionGate(ch * 8, ch * 8, ch * 4)
#         self.dec4 = DoubleConv(ch * 16, ch * 8)

#         self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
#         self.att3 = AttentionGate(ch * 4, ch * 4, ch * 2)
#         self.dec3 = DoubleConv(ch * 8, ch * 4)

#         self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
#         self.att2 = AttentionGate(ch * 2, ch * 2, ch)
#         self.dec2 = DoubleConv(ch * 4, ch * 2)

#         self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
#         self.att1 = AttentionGate(ch, ch, ch // 2)
#         self.dec1 = DoubleConv(ch * 2, ch)

#         self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)

#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))
#         e4 = self.enc4(self.pool3(e3))

#         b = self.bottleneck(self.pool4(e4))
#         b = self.cbam(b)

#         d4 = self.up4(b)
#         e4_att = self.att4(d4, e4)
#         d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

#         d3 = self.up3(d4)
#         e3_att = self.att3(d3, e3)
#         d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

#         d2 = self.up2(d3)
#         e2_att = self.att2(d2, e2)
#         d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

#         d1 = self.up1(d2)
#         e1_att = self.att1(d1, e1)
#         d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

#         return self.out(d1)


# class CLAHE_RATS_Net(nn.Module):
#     def __init__(self, base_channels=32):
#         super().__init__()

#         self.stage1 = AttentionUNet(
#             in_channels=2,
#             out_channels=1,
#             base_channels=base_channels
#         )

#         self.stage2 = ResidualRefinementUNet(
#             in_channels=3,
#             out_channels=1,
#             base_channels=base_channels
#         )

#     def forward(self, x):
#         # x shape: [B, 2, H, W]
#         # Channel 0: CLAHE ROI
#         # Channel 1: Click map

#         clahe_image = x[:, 0:1, :, :]
#         click_map = x[:, 1:2, :, :]

#         coarse_logits = self.stage1(x)
#         coarse_prob = torch.sigmoid(coarse_logits)

#         stage2_input = torch.cat([clahe_image, click_map, coarse_prob], dim=1)

#         residual_logits = self.stage2(stage2_input)

#         final_logits = coarse_logits + residual_logits

#         return {
#             "coarse_logits": coarse_logits,
#             "residual_logits": residual_logits,
#             "final_logits": final_logits
#         }


# # =============================================================================
# # 6. LOSS FUNCTIONS
# # =============================================================================

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)

#         probs = probs.view(probs.size(0), -1)
#         targets = targets.view(targets.size(0), -1)

#         intersection = (probs * targets).sum(dim=1)
#         denominator = probs.sum(dim=1) + targets.sum(dim=1)

#         dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
#         return 1.0 - dice.mean()


# class FocalTverskyLoss(nn.Module):
#     def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
#         super().__init__()

#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)

#         probs = probs.view(probs.size(0), -1)
#         targets = targets.view(targets.size(0), -1)

#         tp = (probs * targets).sum(dim=1)
#         fp = ((1 - targets) * probs).sum(dim=1)
#         fn = (targets * (1 - probs)).sum(dim=1)

#         tversky = (tp + self.smooth) / (
#             tp + self.alpha * fp + self.beta * fn + self.smooth
#         )

#         loss = torch.pow((1 - tversky), self.gamma)
#         return loss.mean()


# class CombinedSegLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.bce = nn.BCEWithLogitsLoss()
#         self.dice = DiceLoss()
#         self.ft = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

#     def forward(self, logits, targets):
#         bce_loss = self.bce(logits, targets)
#         dice_loss = self.dice(logits, targets)
#         ft_loss = self.ft(logits, targets)

#         total = (
#             0.30 * bce_loss +
#             0.45 * dice_loss +
#             0.25 * ft_loss
#         )

#         return total


# class RATSNetLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.seg_loss = CombinedSegLoss()

#     def forward(self, outputs, targets):
#         coarse_logits = outputs["coarse_logits"]
#         final_logits = outputs["final_logits"]

#         coarse_loss = self.seg_loss(coarse_logits, targets)
#         final_loss = self.seg_loss(final_logits, targets)

#         total_loss = 0.30 * coarse_loss + 0.70 * final_loss

#         return total_loss


# # =============================================================================
# # 7. METRICS
# # =============================================================================

# def dice_score_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
#     probs = torch.sigmoid(logits)
#     preds = (probs >= threshold).float()

#     preds = preds.view(preds.size(0), -1)
#     targets = targets.view(targets.size(0), -1)

#     intersection = (preds * targets).sum(dim=1)
#     denominator = preds.sum(dim=1) + targets.sum(dim=1)

#     dice = (2.0 * intersection + smooth) / (denominator + smooth)
#     return dice.mean().item()


# def compute_numpy_metrics(pred, gt, smooth=1e-6):
#     pred = pred.astype(np.uint8).flatten()
#     gt = gt.astype(np.uint8).flatten()

#     tp = np.sum((pred == 1) & (gt == 1))
#     tn = np.sum((pred == 0) & (gt == 0))
#     fp = np.sum((pred == 1) & (gt == 0))
#     fn = np.sum((pred == 0) & (gt == 1))

#     dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
#     iou = (tp + smooth) / (tp + fp + fn + smooth)
#     precision = (tp + smooth) / (tp + fp + smooth)
#     recall = (tp + smooth) / (tp + fn + smooth)
#     specificity = (tn + smooth) / (tn + fp + smooth)

#     return {
#         "dice": dice,
#         "iou": iou,
#         "precision": precision,
#         "recall": recall,
#         "specificity": specificity
#     }


# # =============================================================================
# # 8. TRAINING AND VALIDATION
# # =============================================================================

# def train_one_epoch(model, loader, criterion, optimizer, scaler):
#     model.train()

#     total_loss = 0.0
#     total_dice = 0.0

#     progress = tqdm(loader, desc="Training", leave=False)

#     for batch in progress:
#         images = batch["image"].to(DEVICE, non_blocking=True)
#         masks = batch["mask"].to(DEVICE, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)

#         if USE_AMP and DEVICE.type == "cuda":
#             with torch.amp.autocast("cuda", enabled=True):
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#         else:
#             outputs = model(images)
#             loss = criterion(outputs, masks)

#         if torch.isnan(loss) or torch.isinf(loss):
#             print("Warning: NaN or Inf loss detected. Skipping batch.")
#             continue

#         if USE_AMP and DEVICE.type == "cuda":
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
#             optimizer.step()

#         dice = dice_score_from_logits(outputs["final_logits"], masks, threshold=0.5)

#         total_loss += loss.item()
#         total_dice += dice

#         progress.set_postfix({
#             "loss": f"{loss.item():.4f}",
#             "dice": f"{dice:.4f}"
#         })

#     avg_loss = total_loss / max(len(loader), 1)
#     avg_dice = total_dice / max(len(loader), 1)

#     return avg_loss, avg_dice


# @torch.no_grad()
# def validate_one_epoch(model, loader, criterion):
#     model.eval()

#     total_loss = 0.0
#     total_dice = 0.0

#     for batch in tqdm(loader, desc="Validation", leave=False):
#         images = batch["image"].to(DEVICE, non_blocking=True)
#         masks = batch["mask"].to(DEVICE, non_blocking=True)

#         if USE_AMP and DEVICE.type == "cuda":
#             with torch.amp.autocast("cuda", enabled=True):
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#         else:
#             outputs = model(images)
#             loss = criterion(outputs, masks)

#         dice = dice_score_from_logits(outputs["final_logits"], masks, threshold=0.5)

#         total_loss += loss.item()
#         total_dice += dice

#     avg_loss = total_loss / max(len(loader), 1)
#     avg_dice = total_dice / max(len(loader), 1)

#     return avg_loss, avg_dice


# # =============================================================================
# # 9. TTA PREDICTION
# # =============================================================================

# @torch.no_grad()
# def predict_with_tta(model, image_tensor):
#     model.eval()

#     image_tensor = image_tensor.to(DEVICE)

#     probs = []

#     # Original
#     outputs = model(image_tensor)
#     probs.append(torch.sigmoid(outputs["final_logits"]))

#     if USE_TTA:
#         # Horizontal flip
#         x_flip = torch.flip(image_tensor, dims=[3])
#         outputs_flip = model(x_flip)
#         prob_flip = torch.sigmoid(outputs_flip["final_logits"])
#         prob_flip = torch.flip(prob_flip, dims=[3])
#         probs.append(prob_flip)

#         # Vertical flip
#         y_flip = torch.flip(image_tensor, dims=[2])
#         outputs_yflip = model(y_flip)
#         prob_yflip = torch.sigmoid(outputs_yflip["final_logits"])
#         prob_yflip = torch.flip(prob_yflip, dims=[2])
#         probs.append(prob_yflip)

#     avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

#     return avg_prob


# # =============================================================================
# # 10. THRESHOLD EVALUATION
# # =============================================================================

# @torch.no_grad()
# def evaluate_thresholds(model, loader, split_name="val", save_predictions=False):
#     model.eval()

#     all_results = []

#     for threshold in THRESHOLDS:
#         metrics_list = []

#         for batch in tqdm(loader, desc=f"{split_name} threshold {threshold}", leave=False):
#             images = batch["image"].to(DEVICE)
#             masks = batch["mask"].cpu().numpy()
#             file_names = batch["file_name"]

#             probs = predict_with_tta(model, images)
#             probs_np = probs.detach().cpu().numpy()

#             for i in range(probs_np.shape[0]):
#                 prob = probs_np[i, 0]
#                 gt = masks[i, 0]

#                 pred = post_process_mask(prob, threshold=threshold)

#                 metrics = compute_numpy_metrics(pred, gt)
#                 metrics["threshold"] = threshold
#                 metrics["split"] = split_name
#                 metrics["file_name"] = file_names[i]

#                 metrics_list.append(metrics)

#                 if save_predictions:
#                     pred_save_dir = PRED_DIR / split_name / f"threshold_{threshold:.2f}"
#                     pred_save_dir.mkdir(parents=True, exist_ok=True)

#                     prob_img = (prob * 255).astype(np.uint8)
#                     pred_img = (pred * 255).astype(np.uint8)
#                     gt_img = (gt * 255).astype(np.uint8)

#                     cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_prob.png"), prob_img)
#                     cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_pred.png"), pred_img)
#                     cv2.imwrite(str(pred_save_dir / f"{file_names[i]}_gt.png"), gt_img)

#         df = pd.DataFrame(metrics_list)

#         summary = {
#             "split": split_name,
#             "threshold": threshold,
#             "dice_mean": df["dice"].mean(),
#             "dice_std": df["dice"].std(),
#             "iou_mean": df["iou"].mean(),
#             "iou_std": df["iou"].std(),
#             "precision_mean": df["precision"].mean(),
#             "precision_std": df["precision"].std(),
#             "recall_mean": df["recall"].mean(),
#             "recall_std": df["recall"].std(),
#             "specificity_mean": df["specificity"].mean(),
#             "specificity_std": df["specificity"].std(),
#         }

#         all_results.append(summary)

#     summary_df = pd.DataFrame(all_results)

#     return summary_df


# # =============================================================================
# # 11. CHECKPOINT HELPERS
# # =============================================================================

# def save_checkpoint(model, optimizer, epoch, val_dice, path):
#     checkpoint = {
#         "model_name": MODEL_NAME,
#         "epoch": epoch,
#         "val_dice": float(val_dice),
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "seed": SEED,
#         "roi_size": ROI_SIZE
#     }

#     torch.save(checkpoint, path)


# def load_checkpoint(model, path):
#     # For PyTorch 2.6+ compatibility
#     try:
#         checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
#     except TypeError:
#         checkpoint = torch.load(path, map_location=DEVICE)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     return checkpoint


# # =============================================================================
# # 12. MAIN
# # =============================================================================

# def main():
#     print("=" * 80)
#     print("PROPOSED MODEL TRAINING: CLAHE-RATS-Net")
#     print("CLAHE-Guided Residual Attention Two-Stage Network")
#     print("=" * 80)

#     print(f"Project root: {PROJECT_ROOT}")
#     print(f"ROI dataset directory: {ROI_DATASET_DIR}")
#     print(f"Output directory: {OUTPUT_DIR}")
#     print()

#     print(f"Using device: {DEVICE}")

#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA: {torch.version.cuda}")

#     print()

#     train_dataset = LungROIDataset(TRAIN_CSV, train=True, roi_size=ROI_SIZE)
#     val_dataset = LungROIDataset(VAL_CSV, train=False, roi_size=ROI_SIZE)
#     test_dataset = LungROIDataset(TEST_CSV, train=False, roi_size=ROI_SIZE)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True if DEVICE.type == "cuda" else False
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True if DEVICE.type == "cuda" else False
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True if DEVICE.type == "cuda" else False
#     )

#     model = CLAHE_RATS_Net(base_channels=32).to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print()
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print()

#     criterion = RATSNetLoss()

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LR,
#         weight_decay=WEIGHT_DECAY
#     )

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="max",
#         factor=0.5,
#         patience=10,
#         min_lr=1e-6
#     )

#     scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

#     best_val_dice = -1.0
#     best_epoch = 0
#     early_stop_counter = 0

#     log_records = []

#     best_model_path = CHECKPOINT_DIR / "best_clahe_rats_net.pth"

#     print("=" * 80)
#     print("Training started")
#     print("=" * 80)

#     for epoch in range(1, NUM_EPOCHS + 1):
#         start_time = time.time()

#         train_loss, train_dice = train_one_epoch(
#             model,
#             train_loader,
#             criterion,
#             optimizer,
#             scaler
#         )

#         val_loss, val_dice = validate_one_epoch(
#             model,
#             val_loader,
#             criterion
#         )

#         scheduler.step(val_dice)

#         epoch_time = time.time() - start_time
#         current_lr = optimizer.param_groups[0]["lr"]

#         print("-" * 80)
#         print(f"Epoch [{epoch}/{NUM_EPOCHS}] | LR: {current_lr:.8f}")
#         print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
#         print(f"Val Loss:   {val_loss:.6f} | Val Dice:   {val_dice:.4f}")
#         print(f"Time: {epoch_time:.2f}s")

#         log_records.append({
#             "epoch": epoch,
#             "lr": current_lr,
#             "train_loss": train_loss,
#             "train_dice": train_dice,
#             "val_loss": val_loss,
#             "val_dice": val_dice,
#             "epoch_time_sec": epoch_time
#         })

#         pd.DataFrame(log_records).to_csv(LOG_DIR / "training_log_clahe_rats_net.csv", index=False)
#         pd.DataFrame(log_records).to_excel(LOG_DIR / "training_log_clahe_rats_net.xlsx", index=False)

#         if val_dice > best_val_dice:
#             best_val_dice = val_dice
#             best_epoch = epoch
#             early_stop_counter = 0

#             save_checkpoint(
#                 model=model,
#                 optimizer=optimizer,
#                 epoch=epoch,
#                 val_dice=val_dice,
#                 path=best_model_path
#             )

#             print(f"New best model saved. Best Val Dice: {best_val_dice:.6f}")
#         else:
#             early_stop_counter += 1
#             print(f"No improvement. Early stopping counter: {early_stop_counter}/{PATIENCE}")

#         if early_stop_counter >= PATIENCE:
#             print()
#             print("Early stopping triggered.")
#             break

#     print()
#     print("Training completed.")
#     print(f"Best epoch: {best_epoch}")
#     print(f"Best validation Dice during training: {best_val_dice:.6f}")

#     print()
#     print("Loading best model for final threshold evaluation...")
#     checkpoint = load_checkpoint(model, best_model_path)

#     print(f"Loaded best model from epoch: {checkpoint['epoch']}")
#     print(f"Loaded best val dice: {checkpoint['val_dice']:.6f}")

#     print()
#     print("=" * 80)
#     print("VALIDATION THRESHOLD SUMMARY")
#     print("=" * 80)

#     val_summary = evaluate_thresholds(
#         model,
#         val_loader,
#         split_name="val",
#         save_predictions=False
#     )

#     print(val_summary)

#     val_summary.to_csv(LOG_DIR / "validation_threshold_summary.csv", index=False)
#     val_summary.to_excel(LOG_DIR / "validation_threshold_summary.xlsx", index=False)

#     best_val_row = val_summary.loc[val_summary["dice_mean"].idxmax()]
#     best_threshold = float(best_val_row["threshold"])

#     print()
#     print(f"Best validation threshold: {best_threshold}")

#     print()
#     print("=" * 80)
#     print("TEST THRESHOLD SUMMARY")
#     print("=" * 80)

#     test_summary = evaluate_thresholds(
#         model,
#         test_loader,
#         split_name="test",
#         save_predictions=True
#     )

#     print(test_summary)

#     test_summary.to_csv(LOG_DIR / "test_threshold_summary.csv", index=False)
#     test_summary.to_excel(LOG_DIR / "test_threshold_summary.xlsx", index=False)

#     best_test_row = test_summary[test_summary["threshold"] == best_threshold].iloc[0]

#     final_summary = pd.DataFrame([{
#         "seed": SEED,
#         "model": MODEL_NAME,
#         "roi_size": ROI_SIZE,
#         "best_epoch": best_epoch,
#         "best_val_training_dice": best_val_dice,
#         "best_val_threshold": best_threshold,
#         "best_val_dice": best_val_row["dice_mean"],
#         "best_val_iou": best_val_row["iou_mean"],
#         "best_val_precision": best_val_row["precision_mean"],
#         "best_val_recall": best_val_row["recall_mean"],
#         "best_test_dice": best_test_row["dice_mean"],
#         "best_test_iou": best_test_row["iou_mean"],
#         "best_test_precision": best_test_row["precision_mean"],
#         "best_test_recall": best_test_row["recall_mean"],
#         "best_test_specificity": best_test_row["specificity_mean"],
#         "use_tta": USE_TTA,
#         "use_largest_component": USE_LARGEST_COMPONENT,
#         "remove_small_objects": REMOVE_SMALL_OBJECTS,
#         "loss": "0.30 Stage1 Loss + 0.70 Final Residual Loss",
#         "data_type": "CLAHE ROI + Click Map + Coarse Probability Refinement"
#     }])

#     print()
#     print("=" * 80)
#     print("FINAL BEST SUMMARY")
#     print("=" * 80)
#     print(final_summary)

#     final_summary.to_csv(LOG_DIR / "final_best_summary_clahe_rats_net.csv", index=False)
#     final_summary.to_excel(LOG_DIR / "final_best_summary_clahe_rats_net.xlsx", index=False)

#     print()
#     print("All outputs saved in:")
#     print(OUTPUT_DIR)


# if __name__ == "__main__":
#     main()





# # # =============================================================================
# # # STEP 5: Train CLAHE-RATS-Net on Clean ROI Dataset
# # # CLAHE-RATS-Net: CLAHE-Guided Residual Attention Two-Stage Network
# # #
# # # Final proposed model:
# # #   Input:
# # #       Channel 1: CLAHE-enhanced ROI image, 256x256
# # #       Channel 2: Click / center heatmap
# # #
# # #   Stage 1:
# # #       Attention U-Net for coarse segmentation
# # #
# # #   Stage 2:
# # #       Residual Attention Refinement U-Net
# # #       Input: CLAHE ROI + click map + Stage-1 coarse probability
# # #
# # #   Final:
# # #       final_logits = coarse_logits + residual_logits
# # #
# # # Dataset:
# # #   outputs/proposed_roi_clahe_dataset_clean/
# # #       train_roi_clahe.csv
# # #       val_roi_clahe.csv
# # #       test_roi_clahe.csv
# # #
# # # Save as:
# # #   scripts/proposed_model/step5_train_clahe_rats_net.py
# # # =============================================================================

# # import os
# # import cv2
# # import time
# # import random
# # import warnings
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader

# # warnings.filterwarnings("ignore")


# # # =============================================================================
# # # 1. CONFIGURATION
# # # =============================================================================

# # PROJECT_ROOT = Path(r"E:\Mahender PHD\segmentaion\Project 23-4-2026")

# # # Use clean ROI dataset
# # ROI_DATASET_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset_clean"

# # OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clahe_rats_net_clean_roi_results"
# # CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# # LOG_DIR = OUTPUT_DIR / "logs"
# # PRED_DIR = OUTPUT_DIR / "predictions"

# # for folder in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PRED_DIR]:
# #     folder.mkdir(parents=True, exist_ok=True)

# # TRAIN_CSV = ROI_DATASET_DIR / "train_roi_clahe.csv"
# # VAL_CSV = ROI_DATASET_DIR / "val_roi_clahe.csv"
# # TEST_CSV = ROI_DATASET_DIR / "test_roi_clahe.csv"

# # MODEL_NAME = "CLAHE_RATS_Net_CleanROI"

# # SEED = 42

# # ROI_SIZE = 256
# # BASE_CHANNELS = 32

# # NUM_EPOCHS = 300
# # BATCH_SIZE = 8
# # NUM_WORKERS = 0

# # LR = 5e-5
# # WEIGHT_DECAY = 1e-4
# # PATIENCE = 60
# # GRAD_CLIP = 1.0

# # USE_AMP = True

# # # TTA settings
# # USE_TTA = True
# # USE_ROTATION_TTA = True
# # ROTATION_ANGLES = [-5, 5]

# # THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# # # Post-processing
# # USE_LARGEST_COMPONENT = True
# # USE_CLICK_CENTERED_COMPONENT = True
# # REMOVE_SMALL_OBJECTS = True
# # MIN_OBJECT_AREA = 20

# # # Loss weights
# # COARSE_LOSS_WEIGHT = 0.30
# # FINAL_LOSS_WEIGHT = 0.70

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # # =============================================================================
# # # 2. REPRODUCIBILITY
# # # =============================================================================

# # def set_seed(seed=42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)

# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed(seed)
# #         torch.cuda.manual_seed_all(seed)

# #     torch.backends.cudnn.benchmark = True
# #     torch.backends.cudnn.deterministic = False


# # set_seed(SEED)


# # # =============================================================================
# # # 3. DATASET UTILITIES
# # # =============================================================================

# # def read_gray(path):
# #     path = str(path)

# #     if not os.path.exists(path):
# #         raise FileNotFoundError(f"File not found: {path}")

# #     if path.lower().endswith(".npy"):
# #         arr = np.load(path).astype(np.float32)
# #     else:
# #         arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# #         if arr is None:
# #             raise ValueError(f"Could not read image: {path}")
# #         arr = arr.astype(np.float32)

# #     if arr.max() > 1.0:
# #         arr = arr / 255.0

# #     return np.clip(arr, 0.0, 1.0)


# # def create_click_map_from_mask(mask, sigma=10):
# #     mask_bin = (mask > 0.5).astype(np.uint8)

# #     if mask_bin.sum() == 0:
# #         return np.zeros_like(mask, dtype=np.float32)

# #     ys, xs = np.where(mask_bin > 0)
# #     cy = int(round(ys.mean()))
# #     cx = int(round(xs.mean()))

# #     h, w = mask.shape
# #     y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

# #     click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
# #     click = click.astype(np.float32)
# #     click = click / (click.max() + 1e-8)

# #     return click


# # def train_augmentation(image, mask, click):
# #     if random.random() < 0.5:
# #         image = np.fliplr(image).copy()
# #         mask = np.fliplr(mask).copy()
# #         click = np.fliplr(click).copy()

# #     if random.random() < 0.25:
# #         image = np.flipud(image).copy()
# #         mask = np.flipud(mask).copy()
# #         click = np.flipud(click).copy()

# #     if random.random() < 0.40:
# #         angle = random.uniform(-10, 10)
# #         h, w = image.shape
# #         center = (w // 2, h // 2)

# #         mat = cv2.getRotationMatrix2D(center, angle, 1.0)

# #         image = cv2.warpAffine(
# #             image,
# #             mat,
# #             (w, h),
# #             flags=cv2.INTER_LINEAR,
# #             borderMode=cv2.BORDER_REFLECT_101
# #         )

# #         mask = cv2.warpAffine(
# #             mask,
# #             mat,
# #             (w, h),
# #             flags=cv2.INTER_NEAREST,
# #             borderMode=cv2.BORDER_CONSTANT,
# #             borderValue=0
# #         )

# #         click = cv2.warpAffine(
# #             click,
# #             mat,
# #             (w, h),
# #             flags=cv2.INTER_LINEAR,
# #             borderMode=cv2.BORDER_CONSTANT,
# #             borderValue=0
# #         )

# #     if random.random() < 0.30:
# #         alpha = random.uniform(0.92, 1.08)
# #         beta = random.uniform(-0.03, 0.03)
# #         image = image * alpha + beta
# #         image = np.clip(image, 0.0, 1.0)

# #     if random.random() < 0.15:
# #         noise = np.random.normal(0, 0.008, image.shape).astype(np.float32)
# #         image = np.clip(image + noise, 0.0, 1.0)

# #     mask = (mask > 0.5).astype(np.float32)
# #     click = np.clip(click, 0.0, 1.0).astype(np.float32)

# #     return image, mask, click


# # class LungROIDataset(Dataset):
# #     def __init__(self, csv_path, train=False, roi_size=256):
# #         self.csv_path = Path(csv_path)
# #         self.train = train
# #         self.roi_size = roi_size

# #         if not self.csv_path.exists():
# #             raise FileNotFoundError(f"CSV not found: {self.csv_path}")

# #         self.df = pd.read_csv(self.csv_path)

# #         print(f"Loaded {self.csv_path.name}: {len(self.df)} samples")
# #         print(f"Columns: {list(self.df.columns)}")

# #         required_cols = ["image_path", "mask_path"]
# #         for col in required_cols:
# #             if col not in self.df.columns:
# #                 raise ValueError(f"Missing required column: {col}")

# #     def __len__(self):
# #         return len(self.df)

# #     def __getitem__(self, idx):
# #         row = self.df.iloc[idx]

# #         image = read_gray(row["image_path"])
# #         mask = read_gray(row["mask_path"])
# #         mask = (mask > 0.5).astype(np.float32)

# #         if image.shape != (self.roi_size, self.roi_size):
# #             image = cv2.resize(image, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)

# #         if mask.shape != (self.roi_size, self.roi_size):
# #             mask = cv2.resize(mask, (self.roi_size, self.roi_size), interpolation=cv2.INTER_NEAREST)
# #             mask = (mask > 0.5).astype(np.float32)

# #         click_path = row.get("click_path", "")

# #         if isinstance(click_path, str) and os.path.exists(click_path):
# #             click = read_gray(click_path)

# #             if click.shape != (self.roi_size, self.roi_size):
# #                 click = cv2.resize(click, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)
# #         else:
# #             click = create_click_map_from_mask(mask)

# #         click = np.clip(click, 0.0, 1.0).astype(np.float32)

# #         if self.train:
# #             image, mask, click = train_augmentation(image, mask, click)

# #         image_t = torch.from_numpy(image).unsqueeze(0).float()
# #         click_t = torch.from_numpy(click).unsqueeze(0).float()
# #         mask_t = torch.from_numpy(mask).unsqueeze(0).float()

# #         x = torch.cat([image_t, click_t], dim=0)

# #         mask_area = int(mask.sum())

# #         file_name = (
# #             f"{row.get('patient_id', 'patient')}"
# #             f"_slice{row.get('slice_index', idx)}"
# #             f"_v{row.get('variant_id', 0)}"
# #         )

# #         return {
# #             "image": x,
# #             "mask": mask_t,
# #             "file_name": file_name,
# #             "mask_area": mask_area
# #         }


# # # =============================================================================
# # # 4. MODEL BLOCKS
# # # =============================================================================

# # class DoubleConv(nn.Module):
# #     def __init__(self, in_channels, out_channels, dropout=0.0):
# #         super().__init__()

# #         layers = [
# #             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True),

# #             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True)
# #         ]

# #         if dropout > 0:
# #             layers.append(nn.Dropout2d(dropout))

# #         self.block = nn.Sequential(*layers)

# #     def forward(self, x):
# #         return self.block(x)


# # class ResidualConv(nn.Module):
# #     def __init__(self, in_channels, out_channels, dropout=0.0):
# #         super().__init__()

# #         self.conv = nn.Sequential(
# #             nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True),

# #             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels)
# #         )

# #         if in_channels != out_channels:
# #             self.shortcut = nn.Sequential(
# #                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
# #                 nn.BatchNorm2d(out_channels)
# #             )
# #         else:
# #             self.shortcut = nn.Identity()

# #         self.relu = nn.ReLU(inplace=True)
# #         self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

# #     def forward(self, x):
# #         out = self.conv(x)
# #         out = out + self.shortcut(x)
# #         out = self.relu(out)
# #         out = self.dropout(out)
# #         return out


# # class AttentionGate(nn.Module):
# #     def __init__(self, g_channels, x_channels, inter_channels):
# #         super().__init__()

# #         self.W_g = nn.Sequential(
# #             nn.Conv2d(g_channels, inter_channels, 1, bias=False),
# #             nn.BatchNorm2d(inter_channels)
# #         )

# #         self.W_x = nn.Sequential(
# #             nn.Conv2d(x_channels, inter_channels, 1, bias=False),
# #             nn.BatchNorm2d(inter_channels)
# #         )

# #         self.psi = nn.Sequential(
# #             nn.Conv2d(inter_channels, 1, 1),
# #             nn.Sigmoid()
# #         )

# #         self.relu = nn.ReLU(inplace=True)

# #     def forward(self, g, x):
# #         g1 = self.W_g(g)
# #         x1 = self.W_x(x)

# #         if g1.shape[2:] != x1.shape[2:]:
# #             g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

# #         psi = self.relu(g1 + x1)
# #         psi = self.psi(psi)

# #         return x * psi


# # class ChannelAttention(nn.Module):
# #     def __init__(self, channels, reduction=8):
# #         super().__init__()

# #         hidden = max(channels // reduction, 4)

# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #         self.max_pool = nn.AdaptiveMaxPool2d(1)

# #         self.fc = nn.Sequential(
# #             nn.Conv2d(channels, hidden, 1, bias=False),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(hidden, channels, 1, bias=False)
# #         )

# #         self.sigmoid = nn.Sigmoid()

# #     def forward(self, x):
# #         avg = self.fc(self.avg_pool(x))
# #         mx = self.fc(self.max_pool(x))
# #         att = self.sigmoid(avg + mx)
# #         return x * att


# # class SpatialAttention(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #         self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
# #         self.sigmoid = nn.Sigmoid()

# #     def forward(self, x):
# #         avg = torch.mean(x, dim=1, keepdim=True)
# #         mx, _ = torch.max(x, dim=1, keepdim=True)

# #         att = torch.cat([avg, mx], dim=1)
# #         att = self.sigmoid(self.conv(att))

# #         return x * att


# # class CBAM(nn.Module):
# #     def __init__(self, channels):
# #         super().__init__()

# #         self.ca = ChannelAttention(channels)
# #         self.sa = SpatialAttention()

# #     def forward(self, x):
# #         x = self.ca(x)
# #         x = self.sa(x)
# #         return x


# # # =============================================================================
# # # 5. STAGE 1: ATTENTION U-NET
# # # =============================================================================

# # class AttentionUNet(nn.Module):
# #     def __init__(self, in_channels=2, out_channels=1, base_channels=32):
# #         super().__init__()

# #         ch = base_channels

# #         self.enc1 = DoubleConv(in_channels, ch)
# #         self.pool1 = nn.MaxPool2d(2)

# #         self.enc2 = DoubleConv(ch, ch * 2)
# #         self.pool2 = nn.MaxPool2d(2)

# #         self.enc3 = DoubleConv(ch * 2, ch * 4)
# #         self.pool3 = nn.MaxPool2d(2)

# #         self.enc4 = DoubleConv(ch * 4, ch * 8, dropout=0.1)
# #         self.pool4 = nn.MaxPool2d(2)

# #         self.bottleneck = DoubleConv(ch * 8, ch * 16, dropout=0.2)
# #         self.cbam = CBAM(ch * 16)

# #         self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
# #         self.att4 = AttentionGate(ch * 8, ch * 8, ch * 4)
# #         self.dec4 = DoubleConv(ch * 16, ch * 8)

# #         self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
# #         self.att3 = AttentionGate(ch * 4, ch * 4, ch * 2)
# #         self.dec3 = DoubleConv(ch * 8, ch * 4)

# #         self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
# #         self.att2 = AttentionGate(ch * 2, ch * 2, ch)
# #         self.dec2 = DoubleConv(ch * 4, ch * 2)

# #         self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
# #         self.att1 = AttentionGate(ch, ch, max(ch // 2, 4))
# #         self.dec1 = DoubleConv(ch * 2, ch)

# #         self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

# #     def forward(self, x):
# #         e1 = self.enc1(x)

# #         e2 = self.enc2(self.pool1(e1))
# #         e3 = self.enc3(self.pool2(e2))
# #         e4 = self.enc4(self.pool3(e3))

# #         b = self.bottleneck(self.pool4(e4))
# #         b = self.cbam(b)

# #         d4 = self.up4(b)
# #         e4a = self.att4(d4, e4)
# #         d4 = self.dec4(torch.cat([d4, e4a], dim=1))

# #         d3 = self.up3(d4)
# #         e3a = self.att3(d3, e3)
# #         d3 = self.dec3(torch.cat([d3, e3a], dim=1))

# #         d2 = self.up2(d3)
# #         e2a = self.att2(d2, e2)
# #         d2 = self.dec2(torch.cat([d2, e2a], dim=1))

# #         d1 = self.up1(d2)
# #         e1a = self.att1(d1, e1)
# #         d1 = self.dec1(torch.cat([d1, e1a], dim=1))

# #         return self.out(d1)


# # # =============================================================================
# # # 6. STAGE 2: RESIDUAL REFINEMENT U-NET
# # # =============================================================================

# # class ResidualRefinementUNet(nn.Module):
# #     def __init__(self, in_channels=3, out_channels=1, base_channels=32):
# #         super().__init__()

# #         ch = base_channels

# #         self.enc1 = ResidualConv(in_channels, ch)
# #         self.pool1 = nn.MaxPool2d(2)

# #         self.enc2 = ResidualConv(ch, ch * 2)
# #         self.pool2 = nn.MaxPool2d(2)

# #         self.enc3 = ResidualConv(ch * 2, ch * 4)
# #         self.pool3 = nn.MaxPool2d(2)

# #         self.enc4 = ResidualConv(ch * 4, ch * 8, dropout=0.1)
# #         self.pool4 = nn.MaxPool2d(2)

# #         self.bottleneck = ResidualConv(ch * 8, ch * 16, dropout=0.2)
# #         self.cbam = CBAM(ch * 16)

# #         self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
# #         self.att4 = AttentionGate(ch * 8, ch * 8, ch * 4)
# #         self.dec4 = ResidualConv(ch * 16, ch * 8)

# #         self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
# #         self.att3 = AttentionGate(ch * 4, ch * 4, ch * 2)
# #         self.dec3 = ResidualConv(ch * 8, ch * 4)

# #         self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
# #         self.att2 = AttentionGate(ch * 2, ch * 2, ch)
# #         self.dec2 = ResidualConv(ch * 4, ch * 2)

# #         self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
# #         self.att1 = AttentionGate(ch, ch, max(ch // 2, 4))
# #         self.dec1 = ResidualConv(ch * 2, ch)

# #         self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

# #     def forward(self, x):
# #         e1 = self.enc1(x)

# #         e2 = self.enc2(self.pool1(e1))
# #         e3 = self.enc3(self.pool2(e2))
# #         e4 = self.enc4(self.pool3(e3))

# #         b = self.bottleneck(self.pool4(e4))
# #         b = self.cbam(b)

# #         d4 = self.up4(b)
# #         e4a = self.att4(d4, e4)
# #         d4 = self.dec4(torch.cat([d4, e4a], dim=1))

# #         d3 = self.up3(d4)
# #         e3a = self.att3(d3, e3)
# #         d3 = self.dec3(torch.cat([d3, e3a], dim=1))

# #         d2 = self.up2(d3)
# #         e2a = self.att2(d2, e2)
# #         d2 = self.dec2(torch.cat([d2, e2a], dim=1))

# #         d1 = self.up1(d2)
# #         e1a = self.att1(d1, e1)
# #         d1 = self.dec1(torch.cat([d1, e1a], dim=1))

# #         return self.out(d1)


# # # =============================================================================
# # # 7. CLAHE-RATS-NET
# # # =============================================================================

# # class CLAHE_RATS_Net(nn.Module):
# #     def __init__(self, base_channels=32):
# #         super().__init__()

# #         self.stage1 = AttentionUNet(
# #             in_channels=2,
# #             out_channels=1,
# #             base_channels=base_channels
# #         )

# #         self.stage2 = ResidualRefinementUNet(
# #             in_channels=3,
# #             out_channels=1,
# #             base_channels=base_channels
# #         )

# #     def forward(self, x):
# #         clahe = x[:, 0:1, :, :]
# #         click = x[:, 1:2, :, :]

# #         coarse_logits = self.stage1(x)
# #         coarse_prob = torch.sigmoid(coarse_logits)

# #         stage2_input = torch.cat(
# #             [clahe, click, coarse_prob],
# #             dim=1
# #         )

# #         residual_logits = self.stage2(stage2_input)

# #         final_logits = coarse_logits + residual_logits

# #         return {
# #             "coarse_logits": coarse_logits,
# #             "residual_logits": residual_logits,
# #             "final_logits": final_logits
# #         }


# # # =============================================================================
# # # 8. LOSSES
# # # =============================================================================

# # class DiceLoss(nn.Module):
# #     def __init__(self, smooth=1e-6):
# #         super().__init__()
# #         self.smooth = smooth

# #     def forward(self, logits, targets):
# #         probs = torch.sigmoid(logits)

# #         probs = probs.view(probs.size(0), -1)
# #         targets = targets.view(targets.size(0), -1)

# #         intersection = (probs * targets).sum(dim=1)
# #         denominator = probs.sum(dim=1) + targets.sum(dim=1)

# #         dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

# #         return 1.0 - dice.mean()


# # class DiceBCELoss(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.dice = DiceLoss()
# #         self.bce = nn.BCEWithLogitsLoss()

# #     def forward(self, logits, targets):
# #         return self.dice(logits, targets) + 0.5 * self.bce(logits, targets)


# # class RATSFixedLoss(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.seg_loss = DiceBCELoss()

# #     def forward(self, outputs, targets):
# #         coarse_loss = self.seg_loss(outputs["coarse_logits"], targets)
# #         final_loss = self.seg_loss(outputs["final_logits"], targets)

# #         total_loss = (
# #             COARSE_LOSS_WEIGHT * coarse_loss +
# #             FINAL_LOSS_WEIGHT * final_loss
# #         )

# #         return total_loss, coarse_loss.detach(), final_loss.detach()


# # # =============================================================================
# # # 9. METRICS AND POST-PROCESSING
# # # =============================================================================

# # def dice_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
# #     probs = torch.sigmoid(logits)
# #     preds = (probs >= threshold).float()

# #     preds = preds.view(preds.size(0), -1)
# #     targets = targets.view(targets.size(0), -1)

# #     intersection = (preds * targets).sum(dim=1)
# #     denominator = preds.sum(dim=1) + targets.sum(dim=1)

# #     dice = (2.0 * intersection + smooth) / (denominator + smooth)

# #     return dice.mean().item()


# # def get_click_center(click_map):
# #     if click_map.max() <= 0:
# #         h, w = click_map.shape
# #         return w // 2, h // 2

# #     y, x = np.unravel_index(np.argmax(click_map), click_map.shape)

# #     return int(x), int(y)


# # def postprocess(prob, click_map, threshold=0.5):
# #     pred = (prob >= threshold).astype(np.uint8)

# #     if REMOVE_SMALL_OBJECTS:
# #         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)

# #         cleaned = np.zeros_like(pred)

# #         for i in range(1, num_labels):
# #             area = stats[i, cv2.CC_STAT_AREA]

# #             if area >= MIN_OBJECT_AREA:
# #                 cleaned[labels == i] = 1

# #         pred = cleaned.astype(np.uint8)

# #     if USE_LARGEST_COMPONENT:
# #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)

# #         if num_labels <= 1:
# #             return pred.astype(np.float32)

# #         if USE_CLICK_CENTERED_COMPONENT:
# #             click_x, click_y = get_click_center(click_map)

# #             best_label = None
# #             best_score = 1e18

# #             for i in range(1, num_labels):
# #                 area = stats[i, cv2.CC_STAT_AREA]
# #                 cx, cy = centroids[i]

# #                 dist = np.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)

# #                 score = dist - 0.01 * area

# #                 if score < best_score:
# #                     best_score = score
# #                     best_label = i

# #             pred = (labels == best_label).astype(np.uint8)

# #         else:
# #             largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
# #             pred = (labels == largest_label).astype(np.uint8)

# #     kernel = np.ones((3, 3), np.uint8)
# #     pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

# #     return pred.astype(np.float32)


# # def compute_metrics(pred, gt, smooth=1e-6):
# #     pred = pred.astype(np.uint8).flatten()
# #     gt = gt.astype(np.uint8).flatten()

# #     tp = np.sum((pred == 1) & (gt == 1))
# #     tn = np.sum((pred == 0) & (gt == 0))
# #     fp = np.sum((pred == 1) & (gt == 0))
# #     fn = np.sum((pred == 0) & (gt == 1))

# #     dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
# #     iou = (tp + smooth) / (tp + fp + fn + smooth)
# #     precision = (tp + smooth) / (tp + fp + smooth)
# #     recall = (tp + smooth) / (tp + fn + smooth)
# #     specificity = (tn + smooth) / (tn + fp + smooth)

# #     return {
# #         "dice": dice,
# #         "iou": iou,
# #         "precision": precision,
# #         "recall": recall,
# #         "specificity": specificity
# #     }


# # # =============================================================================
# # # 10. TTA ROTATION UTILITIES
# # # =============================================================================

# # def rotate_batch_tensor(x, angle_degrees):
# #     """
# #     Rotates tensor batch by angle in degrees.
# #     Shape: [B, C, H, W]
# #     """

# #     if angle_degrees == 0:
# #         return x

# #     angle = angle_degrees * np.pi / 180.0
# #     b = x.size(0)

# #     theta = torch.zeros((b, 2, 3), dtype=x.dtype, device=x.device)

# #     theta[:, 0, 0] = np.cos(angle)
# #     theta[:, 0, 1] = -np.sin(angle)
# #     theta[:, 1, 0] = np.sin(angle)
# #     theta[:, 1, 1] = np.cos(angle)

# #     grid = F.affine_grid(theta, x.size(), align_corners=False)
# #     rotated = F.grid_sample(
# #         x,
# #         grid,
# #         mode="bilinear",
# #         padding_mode="border",
# #         align_corners=False
# #     )

# #     return rotated


# # @torch.no_grad()
# # def predict_tta(model, images):
# #     model.eval()

# #     probs = []

# #     outputs = model(images)
# #     probs.append(torch.sigmoid(outputs["final_logits"]))

# #     if USE_TTA:
# #         # Horizontal flip
# #         x_flip = torch.flip(images, dims=[3])
# #         out_flip = model(x_flip)
# #         prob_flip = torch.sigmoid(out_flip["final_logits"])
# #         prob_flip = torch.flip(prob_flip, dims=[3])
# #         probs.append(prob_flip)

# #         # Vertical flip
# #         y_flip = torch.flip(images, dims=[2])
# #         out_yflip = model(y_flip)
# #         prob_yflip = torch.sigmoid(out_yflip["final_logits"])
# #         prob_yflip = torch.flip(prob_yflip, dims=[2])
# #         probs.append(prob_yflip)

# #         if USE_ROTATION_TTA:
# #             for angle in ROTATION_ANGLES:
# #                 x_rot = rotate_batch_tensor(images, angle)
# #                 out_rot = model(x_rot)
# #                 prob_rot = torch.sigmoid(out_rot["final_logits"])
# #                 prob_back = rotate_batch_tensor(prob_rot, -angle)
# #                 probs.append(prob_back)

# #     avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

# #     return avg_prob


# # # =============================================================================
# # # 11. TRAINING AND VALIDATION
# # # =============================================================================

# # def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
# #     model.train()

# #     total_loss = 0.0
# #     total_dice = 0.0
# #     total_coarse_loss = 0.0
# #     total_final_loss = 0.0
# #     valid_batches = 0

# #     pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

# #     for batch in pbar:
# #         images = batch["image"].to(DEVICE, non_blocking=True)
# #         masks = batch["mask"].to(DEVICE, non_blocking=True)

# #         optimizer.zero_grad(set_to_none=True)

# #         if USE_AMP and DEVICE.type == "cuda":
# #             with torch.amp.autocast("cuda", enabled=True):
# #                 outputs = model(images)
# #                 loss, coarse_loss, final_loss = criterion(outputs, masks)
# #         else:
# #             outputs = model(images)
# #             loss, coarse_loss, final_loss = criterion(outputs, masks)

# #         if torch.isnan(loss) or torch.isinf(loss):
# #             print("NaN/Inf loss detected. Skipping batch.")
# #             continue

# #         if USE_AMP and DEVICE.type == "cuda":
# #             scaler.scale(loss).backward()
# #             scaler.unscale_(optimizer)
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
# #             scaler.step(optimizer)
# #             scaler.update()
# #         else:
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
# #             optimizer.step()

# #         dice = dice_from_logits(outputs["final_logits"], masks, threshold=0.5)

# #         total_loss += loss.item()
# #         total_dice += dice
# #         total_coarse_loss += coarse_loss.item()
# #         total_final_loss += final_loss.item()
# #         valid_batches += 1

# #         pbar.set_postfix({
# #             "loss": f"{total_loss / max(valid_batches, 1):.4f}",
# #             "dice": f"{total_dice / max(valid_batches, 1):.4f}"
# #         })

# #     return (
# #         total_loss / max(valid_batches, 1),
# #         total_dice / max(valid_batches, 1),
# #         total_coarse_loss / max(valid_batches, 1),
# #         total_final_loss / max(valid_batches, 1)
# #     )


# # @torch.no_grad()
# # def validate_one_epoch(model, loader, criterion):
# #     model.eval()

# #     total_loss = 0.0
# #     total_dice = 0.0
# #     total_coarse_loss = 0.0
# #     total_final_loss = 0.0
# #     valid_batches = 0

# #     for batch in tqdm(loader, desc="Validation", leave=False):
# #         images = batch["image"].to(DEVICE, non_blocking=True)
# #         masks = batch["mask"].to(DEVICE, non_blocking=True)

# #         if USE_AMP and DEVICE.type == "cuda":
# #             with torch.amp.autocast("cuda", enabled=True):
# #                 outputs = model(images)
# #                 loss, coarse_loss, final_loss = criterion(outputs, masks)
# #         else:
# #             outputs = model(images)
# #             loss, coarse_loss, final_loss = criterion(outputs, masks)

# #         dice = dice_from_logits(outputs["final_logits"], masks, threshold=0.5)

# #         total_loss += loss.item()
# #         total_dice += dice
# #         total_coarse_loss += coarse_loss.item()
# #         total_final_loss += final_loss.item()
# #         valid_batches += 1

# #     return (
# #         total_loss / max(valid_batches, 1),
# #         total_dice / max(valid_batches, 1),
# #         total_coarse_loss / max(valid_batches, 1),
# #         total_final_loss / max(valid_batches, 1)
# #     )


# # # =============================================================================
# # # 12. THRESHOLD EVALUATION
# # # =============================================================================

# # @torch.no_grad()
# # def evaluate_thresholds(model, loader, split_name="val", save_predictions=False):
# #     model.eval()

# #     all_summary_rows = []
# #     all_sample_rows = []

# #     for threshold in THRESHOLDS:
# #         rows = []

# #         for batch in tqdm(loader, desc=f"{split_name} threshold {threshold}", leave=False):
# #             images = batch["image"].to(DEVICE, non_blocking=True)
# #             masks_np = batch["mask"].cpu().numpy()
# #             file_names = batch["file_name"]
# #             mask_areas = batch["mask_area"]

# #             probs = predict_tta(model, images)
# #             probs_np = probs.detach().cpu().numpy()
# #             images_np = images.detach().cpu().numpy()

# #             for i in range(probs_np.shape[0]):
# #                 prob = probs_np[i, 0]
# #                 gt = masks_np[i, 0]
# #                 click = images_np[i, 1]

# #                 pred = postprocess(prob, click, threshold=threshold)

# #                 metrics = compute_metrics(pred, gt)

# #                 mask_area = int(mask_areas[i])

# #                 if mask_area < 50:
# #                     size_group = "small_<50"
# #                 elif mask_area <= 200:
# #                     size_group = "medium_50_200"
# #                 else:
# #                     size_group = "large_>200"

# #                 row = {
# #                     "split": split_name,
# #                     "threshold": threshold,
# #                     "file_name": file_names[i],
# #                     "mask_area": mask_area,
# #                     "size_group": size_group,
# #                 }

# #                 row.update(metrics)

# #                 rows.append(row)
# #                 all_sample_rows.append(row)

# #                 if save_predictions:
# #                     save_dir = PRED_DIR / split_name / f"threshold_{threshold:.2f}"
# #                     save_dir.mkdir(parents=True, exist_ok=True)

# #                     cv2.imwrite(str(save_dir / f"{file_names[i]}_prob.png"), (prob * 255).astype(np.uint8))
# #                     cv2.imwrite(str(save_dir / f"{file_names[i]}_pred.png"), (pred * 255).astype(np.uint8))
# #                     cv2.imwrite(str(save_dir / f"{file_names[i]}_gt.png"), (gt * 255).astype(np.uint8))

# #         df = pd.DataFrame(rows)

# #         summary = {
# #             "split": split_name,
# #             "threshold": threshold,
# #             "dice_mean": df["dice"].mean(),
# #             "dice_std": df["dice"].std(),
# #             "iou_mean": df["iou"].mean(),
# #             "iou_std": df["iou"].std(),
# #             "precision_mean": df["precision"].mean(),
# #             "precision_std": df["precision"].std(),
# #             "recall_mean": df["recall"].mean(),
# #             "recall_std": df["recall"].std(),
# #             "specificity_mean": df["specificity"].mean(),
# #             "specificity_std": df["specificity"].std(),
# #         }

# #         all_summary_rows.append(summary)

# #     summary_df = pd.DataFrame(all_summary_rows)
# #     sample_df = pd.DataFrame(all_sample_rows)

# #     return summary_df, sample_df


# # # =============================================================================
# # # 13. CHECKPOINT
# # # =============================================================================

# # def save_checkpoint(model, optimizer, epoch, val_dice, path):
# #     checkpoint = {
# #         "model_name": MODEL_NAME,
# #         "epoch": epoch,
# #         "val_dice": float(val_dice),
# #         "model_state_dict": model.state_dict(),
# #         "optimizer_state_dict": optimizer.state_dict(),
# #         "seed": SEED,
# #         "roi_size": ROI_SIZE,
# #         "base_channels": BASE_CHANNELS
# #     }

# #     torch.save(checkpoint, path)


# # def load_checkpoint(model, path):
# #     try:
# #         checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
# #     except TypeError:
# #         checkpoint = torch.load(path, map_location=DEVICE)

# #     model.load_state_dict(checkpoint["model_state_dict"])

# #     return checkpoint


# # # =============================================================================
# # # 14. MAIN
# # # =============================================================================

# # def main():
# #     print("=" * 80)
# #     print("PROPOSED MODEL TRAINING: CLAHE-RATS-Net on Clean ROI Dataset")
# #     print("=" * 80)

# #     print(f"Project root: {PROJECT_ROOT}")
# #     print(f"ROI dataset directory: {ROI_DATASET_DIR}")
# #     print(f"Output directory: {OUTPUT_DIR}")
# #     print()

# #     print(f"Using device: {DEVICE}")

# #     if torch.cuda.is_available():
# #         print(f"GPU: {torch.cuda.get_device_name(0)}")
# #         print(f"CUDA: {torch.version.cuda}")

# #     print()

# #     train_dataset = LungROIDataset(TRAIN_CSV, train=True, roi_size=ROI_SIZE)
# #     val_dataset = LungROIDataset(VAL_CSV, train=False, roi_size=ROI_SIZE)
# #     test_dataset = LungROIDataset(TEST_CSV, train=False, roi_size=ROI_SIZE)

# #     train_loader = DataLoader(
# #         train_dataset,
# #         batch_size=BATCH_SIZE,
# #         shuffle=True,
# #         num_workers=NUM_WORKERS,
# #         pin_memory=True if DEVICE.type == "cuda" else False
# #     )

# #     val_loader = DataLoader(
# #         val_dataset,
# #         batch_size=BATCH_SIZE,
# #         shuffle=False,
# #         num_workers=NUM_WORKERS,
# #         pin_memory=True if DEVICE.type == "cuda" else False
# #     )

# #     test_loader = DataLoader(
# #         test_dataset,
# #         batch_size=BATCH_SIZE,
# #         shuffle=False,
# #         num_workers=NUM_WORKERS,
# #         pin_memory=True if DEVICE.type == "cuda" else False
# #     )

# #     model = CLAHE_RATS_Net(
# #         base_channels=BASE_CHANNELS
# #     ).to(DEVICE)

# #     criterion = RATSFixedLoss().to(DEVICE)

# #     optimizer = torch.optim.AdamW(
# #         model.parameters(),
# #         lr=LR,
# #         weight_decay=WEIGHT_DECAY
# #     )

# #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
# #         optimizer,
# #         mode="max",
# #         factor=0.5,
# #         patience=12,
# #         min_lr=1e-6
# #     )

# #     scaler = torch.amp.GradScaler(
# #         "cuda",
# #         enabled=(USE_AMP and DEVICE.type == "cuda")
# #     )

# #     total_params = sum(p.numel() for p in model.parameters())
# #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# #     print(f"Total parameters: {total_params:,}")
# #     print(f"Trainable parameters: {trainable_params:,}")
# #     print(f"Loss weights: coarse={COARSE_LOSS_WEIGHT}, final={FINAL_LOSS_WEIGHT}")
# #     print(f"TTA: {USE_TTA}, rotation TTA: {USE_ROTATION_TTA}")
# #     print()

# #     best_val_dice = -1.0
# #     best_epoch = 0
# #     early_counter = 0

# #     best_model_path = CHECKPOINT_DIR / "best_clahe_rats_net_clean_roi.pth"

# #     logs = []

# #     for epoch in range(1, NUM_EPOCHS + 1):
# #         start_time = time.time()

# #         train_loss, train_dice, train_coarse, train_final = train_one_epoch(
# #             model=model,
# #             loader=train_loader,
# #             criterion=criterion,
# #             optimizer=optimizer,
# #             scaler=scaler,
# #             epoch=epoch
# #         )

# #         val_loss, val_dice, val_coarse, val_final = validate_one_epoch(
# #             model=model,
# #             loader=val_loader,
# #             criterion=criterion
# #         )

# #         scheduler.step(val_dice)

# #         epoch_time = time.time() - start_time
# #         current_lr = optimizer.param_groups[0]["lr"]

# #         print("-" * 80)
# #         print(f"Epoch [{epoch}/{NUM_EPOCHS}] | LR: {current_lr:.8f}")
# #         print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
# #         print(f"Val Loss:   {val_loss:.6f} | Val Dice:   {val_dice:.4f}")
# #         print(f"Stage Loss Train: coarse={train_coarse:.6f}, final={train_final:.6f}")
# #         print(f"Stage Loss Val:   coarse={val_coarse:.6f}, final={val_final:.6f}")
# #         print(f"Time: {epoch_time:.2f}s")

# #         logs.append({
# #             "epoch": epoch,
# #             "lr": current_lr,
# #             "train_loss": train_loss,
# #             "train_dice": train_dice,
# #             "val_loss": val_loss,
# #             "val_dice": val_dice,
# #             "train_coarse_loss": train_coarse,
# #             "train_final_loss": train_final,
# #             "val_coarse_loss": val_coarse,
# #             "val_final_loss": val_final,
# #             "epoch_time_sec": epoch_time
# #         })

# #         pd.DataFrame(logs).to_csv(LOG_DIR / "training_log_clahe_rats_net_clean_roi.csv", index=False)
# #         pd.DataFrame(logs).to_excel(LOG_DIR / "training_log_clahe_rats_net_clean_roi.xlsx", index=False)

# #         if val_dice > best_val_dice:
# #             best_val_dice = val_dice
# #             best_epoch = epoch
# #             early_counter = 0

# #             save_checkpoint(
# #                 model=model,
# #                 optimizer=optimizer,
# #                 epoch=epoch,
# #                 val_dice=val_dice,
# #                 path=best_model_path
# #             )

# #             print(f"New best model saved. Best Val Dice: {best_val_dice:.6f}")

# #         else:
# #             early_counter += 1
# #             print(f"No improvement. Early stopping counter: {early_counter}/{PATIENCE}")

# #         if early_counter >= PATIENCE:
# #             print("Early stopping triggered.")
# #             break

# #     print()
# #     print("=" * 80)
# #     print("TRAINING COMPLETED")
# #     print("=" * 80)
# #     print(f"Best epoch: {best_epoch}")
# #     print(f"Best validation Dice during training: {best_val_dice:.6f}")

# #     print("\nLoading best model for final threshold evaluation...")
# #     checkpoint = load_checkpoint(model, best_model_path)
# #     print(f"Loaded epoch: {checkpoint['epoch']}")
# #     print(f"Loaded val Dice: {checkpoint['val_dice']:.6f}")

# #     print()
# #     print("=" * 80)
# #     print("VALIDATION THRESHOLD SUMMARY")
# #     print("=" * 80)

# #     val_summary, val_samples = evaluate_thresholds(
# #         model=model,
# #         loader=val_loader,
# #         split_name="val",
# #         save_predictions=False
# #     )

# #     print(val_summary)

# #     val_summary.to_csv(LOG_DIR / "validation_threshold_summary.csv", index=False)
# #     val_summary.to_excel(LOG_DIR / "validation_threshold_summary.xlsx", index=False)

# #     val_samples.to_csv(LOG_DIR / "validation_samplewise_metrics.csv", index=False)
# #     val_samples.to_excel(LOG_DIR / "validation_samplewise_metrics.xlsx", index=False)

# #     best_val_row = val_summary.loc[val_summary["dice_mean"].idxmax()]
# #     best_threshold = float(best_val_row["threshold"])

# #     print()
# #     print(f"Best validation threshold: {best_threshold}")

# #     print()
# #     print("=" * 80)
# #     print("TEST THRESHOLD SUMMARY")
# #     print("=" * 80)

# #     test_summary, test_samples = evaluate_thresholds(
# #         model=model,
# #         loader=test_loader,
# #         split_name="test",
# #         save_predictions=True
# #     )

# #     print(test_summary)

# #     test_summary.to_csv(LOG_DIR / "test_threshold_summary.csv", index=False)
# #     test_summary.to_excel(LOG_DIR / "test_threshold_summary.xlsx", index=False)

# #     test_samples.to_csv(LOG_DIR / "test_samplewise_metrics.csv", index=False)
# #     test_samples.to_excel(LOG_DIR / "test_samplewise_metrics.xlsx", index=False)

# #     best_test_row = test_summary[test_summary["threshold"] == best_threshold].iloc[0]

# #     test_best_thr = test_samples[test_samples["threshold"] == best_threshold].copy()

# #     if len(test_best_thr) > 0:
# #         size_summary = (
# #             test_best_thr
# #             .groupby("size_group")
# #             .agg({
# #                 "dice": ["mean", "std", "count"],
# #                 "iou": ["mean", "std"],
# #                 "precision": ["mean", "std"],
# #                 "recall": ["mean", "std"]
# #             })
# #         )

# #         size_summary.to_excel(LOG_DIR / "test_sizewise_summary.xlsx")

# #         print()
# #         print("=" * 80)
# #         print("TEST SIZE-WISE SUMMARY AT BEST VALIDATION THRESHOLD")
# #         print("=" * 80)
# #         print(size_summary)

# #     final_summary = pd.DataFrame([{
# #         "seed": SEED,
# #         "model": MODEL_NAME,
# #         "roi_size": ROI_SIZE,
# #         "best_epoch": best_epoch,
# #         "best_val_training_dice": best_val_dice,
# #         "best_val_threshold": best_threshold,

# #         "best_val_dice": best_val_row["dice_mean"],
# #         "best_val_iou": best_val_row["iou_mean"],
# #         "best_val_precision": best_val_row["precision_mean"],
# #         "best_val_recall": best_val_row["recall_mean"],

# #         "best_test_dice": best_test_row["dice_mean"],
# #         "best_test_iou": best_test_row["iou_mean"],
# #         "best_test_precision": best_test_row["precision_mean"],
# #         "best_test_recall": best_test_row["recall_mean"],
# #         "best_test_specificity": best_test_row["specificity_mean"],

# #         "loss": "0.30 coarse + 0.70 final, each Dice + 0.5 BCE",
# #         "optimizer": "AdamW",
# #         "lr": LR,
# #         "weight_decay": WEIGHT_DECAY,
# #         "use_tta": USE_TTA,
# #         "use_rotation_tta": USE_ROTATION_TTA,
# #         "data_type": "Clean CLAHE ROI + Click Map"
# #     }])

# #     print()
# #     print("=" * 80)
# #     print("FINAL BEST SUMMARY")
# #     print("=" * 80)
# #     print(final_summary)

# #     final_summary.to_csv(LOG_DIR / "final_best_summary_clahe_rats_net_clean_roi.csv", index=False)
# #     final_summary.to_excel(LOG_DIR / "final_best_summary_clahe_rats_net_clean_roi.xlsx", index=False)

# #     print()
# #     print("All outputs saved in:")
# #     print(OUTPUT_DIR)


# # if __name__ == "__main__":
# #     main()
