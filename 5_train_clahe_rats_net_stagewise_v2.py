# =============================================================================
# STEP 7: CLAHE-RATS-Net Stagewise V2
# CLAHE-Guided Residual Attention Two-Stage Network
#
# Improvement over previous stagewise version:
#   Stage 1       : Positive ROI only
#   Stage 2       : Positive ROI + 25% negative ROI
#   Final finetune: Positive ROI only
#
# Architecture:
#   Input:
#       Channel 1: CLAHE-enhanced ROI
#       Channel 2: Click/center heatmap
#
#   Stage 1:
#       Coarse Attention U-Net
#
#   Stage 2:
#       Residual Refinement U-Net
#       Input: CLAHE ROI + Click map + Stage-1 coarse probability
#
#   Final:
#       final_logits = coarse_logits + residual_logits
#
# Save as:
# scripts/proposed_model/step7_train_clahe_rats_net_stagewise_v2.py
# =============================================================================

import os
import cv2
import time
import random
import warnings
from pathlib import Path

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

SPLIT_DIR = PROJECT_ROOT / "outputs" / "baseline_clahe_preprocessing" / "dataset_splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clahe_rats_net_stagewise_v2_results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PRED_DIR = OUTPUT_DIR / "predictions"

for folder in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PRED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TRAIN_ROI_CSV = ROI_DATASET_DIR / "train_roi_clahe.csv"
VAL_ROI_CSV = ROI_DATASET_DIR / "val_roi_clahe.csv"
TEST_ROI_CSV = ROI_DATASET_DIR / "test_roi_clahe.csv"

TRAIN_CLAHE_SPLIT_CSV = SPLIT_DIR / "clahe_train.csv"
VAL_CLAHE_SPLIT_CSV = SPLIT_DIR / "clahe_val.csv"
TEST_CLAHE_SPLIT_CSV = SPLIT_DIR / "clahe_test.csv"

MODEL_NAME = "CLAHE_RATS_Net_Stagewise_V2"

SEED = 42
ROI_SIZE = 256
BASE_CHANNELS = 32

BATCH_SIZE = 8
NUM_WORKERS = 0

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 60
FINETUNE_EPOCHS = 150

PATIENCE_STAGE1 = 20
PATIENCE_STAGE2 = 25
PATIENCE_FINETUNE = 50

LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-4
LR_FINETUNE = 5e-6

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

USE_AMP = True
USE_TTA = True

# Stage-specific negative ROI strategy
USE_NEGATIVE_ROI_STAGE1 = False
USE_NEGATIVE_ROI_STAGE2 = True
USE_NEGATIVE_ROI_FINETUNE = False

NEGATIVE_RATIO_STAGE1 = 0.0
NEGATIVE_RATIO_STAGE2 = 0.25
NEGATIVE_RATIO_FINETUNE = 0.0

REMOVE_SMALL_OBJECTS = True
MIN_OBJECT_AREA = 20
USE_CLICK_CENTERED_COMPONENT = True

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

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
# 3. IMAGE UTILITIES
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

    arr = np.clip(arr, 0.0, 1.0)

    return arr


def create_click_map_from_mask(mask, sigma=10):
    mask_bin = (mask > 0.5).astype(np.uint8)

    if mask_bin.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    ys, xs = np.where(mask_bin > 0)

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    h, w = mask.shape

    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
    click = click.astype(np.float32)
    click = click / (click.max() + 1e-8)

    return click


def create_center_click_map(size=256, sigma=12):
    h, w = size, size
    cy, cx = h // 2, w // 2

    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
    click = click.astype(np.float32)
    click = click / (click.max() + 1e-8)

    return click


def resize_to_roi(image, mask, click, size=256):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    click = cv2.resize(click, (size, size), interpolation=cv2.INTER_LINEAR)

    mask = (mask > 0.5).astype(np.float32)
    click = np.clip(click, 0.0, 1.0).astype(np.float32)

    return image, mask, click


def random_negative_crop(image, roi_size=256):
    h, w = image.shape

    if h < roi_size or w < roi_size:
        image = cv2.resize(image, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((roi_size, roi_size), dtype=np.float32)
        return image.astype(np.float32), mask

    max_y = h - roi_size
    max_x = w - roi_size

    y1 = random.randint(0, max_y)
    x1 = random.randint(0, max_x)

    crop_img = image[y1:y1 + roi_size, x1:x1 + roi_size]
    crop_mask = np.zeros((roi_size, roi_size), dtype=np.float32)

    return crop_img.astype(np.float32), crop_mask.astype(np.float32)


def train_augmentation(image, mask, click):
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        click = np.fliplr(click).copy()

    if random.random() < 0.25:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        click = np.flipud(click).copy()

    if random.random() < 0.50:
        angle = random.uniform(-15, 15)
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

    if random.random() < 0.35:
        alpha = random.uniform(0.90, 1.10)
        beta = random.uniform(-0.04, 0.04)
        image = image * alpha + beta
        image = np.clip(image, 0.0, 1.0)

    if random.random() < 0.20:
        noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    mask = (mask > 0.5).astype(np.float32)
    click = np.clip(click, 0.0, 1.0).astype(np.float32)

    return image, mask, click


# =============================================================================
# 4. DATAFRAME PREPARATION
# =============================================================================

def prepare_dataframe_with_negatives(
    positive_roi_csv,
    clahe_split_csv,
    negative_ratio=0.0,
    use_negatives=False,
    split_name="train"
):
    positive_roi_csv = Path(positive_roi_csv)
    clahe_split_csv = Path(clahe_split_csv)

    if not positive_roi_csv.exists():
        raise FileNotFoundError(f"Positive ROI CSV not found: {positive_roi_csv}")

    pos_df = pd.read_csv(positive_roi_csv).copy()
    pos_df["sample_type"] = "positive"

    print()
    print("-" * 80)
    print(f"Preparing dataframe: {split_name}")
    print(f"Positive ROI CSV: {positive_roi_csv}")
    print(f"Positive samples: {len(pos_df)}")

    if not use_negatives or negative_ratio <= 0:
        print(f"{split_name}: Using positive ROI samples only.")
        print(f"{split_name}: Total samples: {len(pos_df)}")
        return pos_df.reset_index(drop=True)

    if not clahe_split_csv.exists():
        print(f"Warning: CLAHE split CSV not found: {clahe_split_csv}")
        print(f"{split_name}: Using positive ROI samples only.")
        return pos_df.reset_index(drop=True)

    full_df = pd.read_csv(clahe_split_csv).copy()

    print(f"CLAHE split CSV: {clahe_split_csv}")
    print(f"Full split rows: {len(full_df)}")
    print(f"Full split columns: {list(full_df.columns)}")

    empty_df = pd.DataFrame()

    if "mask_status" in full_df.columns:
        empty_df = full_df[
            full_df["mask_status"].astype(str).str.upper().isin(
                ["EMPTY", "NEGATIVE", "NO_MASK", "BACKGROUND"]
            )
        ].copy()

    if len(empty_df) == 0 and "mask_pixel_count" in full_df.columns:
        empty_df = full_df[full_df["mask_pixel_count"].fillna(0).astype(float) == 0].copy()

    if len(empty_df) == 0:
        print("Warning: No negative/empty samples found.")
        print(f"{split_name}: Using positive ROI samples only.")
        return pos_df.reset_index(drop=True)

    required_negatives = int(len(pos_df) * negative_ratio)
    required_negatives = min(required_negatives, len(empty_df))

    neg_df = empty_df.sample(n=required_negatives, random_state=SEED).copy()
    neg_df["sample_type"] = "negative"

    if "clahe_image_path" in neg_df.columns:
        neg_df["image_path"] = neg_df["clahe_image_path"]
    elif "image_path" not in neg_df.columns:
        raise ValueError("Negative split CSV must contain clahe_image_path or image_path.")

    if "clahe_mask_path" in neg_df.columns:
        neg_df["mask_path"] = neg_df["clahe_mask_path"]
    elif "mask_path" not in neg_df.columns:
        neg_df["mask_path"] = ""

    if "clahe_click_path" in neg_df.columns:
        neg_df["click_path"] = neg_df["clahe_click_path"]
    elif "click_path" not in neg_df.columns:
        neg_df["click_path"] = ""

    keep_cols = list(pos_df.columns)

    for col in keep_cols:
        if col not in neg_df.columns:
            neg_df[col] = ""

    neg_df = neg_df[keep_cols]

    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"{split_name}: Negative samples added: {len(neg_df)}")
    print(f"{split_name}: Total samples: {len(combined_df)}")

    return combined_df


# =============================================================================
# 5. DATASET
# =============================================================================

class LungROIDataset(Dataset):
    def __init__(self, dataframe, train=False, roi_size=256):
        self.df = dataframe.reset_index(drop=True)
        self.train = train
        self.roi_size = roi_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sample_type = str(row.get("sample_type", "positive"))

        image = safe_read_image(row["image_path"])

        if sample_type == "negative":
            image, mask = random_negative_crop(image, roi_size=self.roi_size)
            click = create_center_click_map(size=self.roi_size, sigma=12)

        else:
            mask = safe_read_image(row["mask_path"])
            mask = (mask > 0.5).astype(np.float32)

            click_path = row.get("click_path", "")

            if isinstance(click_path, str) and os.path.exists(click_path):
                click = safe_read_image(click_path)
            else:
                click = create_click_map_from_mask(mask)

            image, mask, click = resize_to_roi(image, mask, click, self.roi_size)

        if self.train:
            image, mask, click = train_augmentation(image, mask, click)

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        click_t = torch.from_numpy(click).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        x = torch.cat([image_t, click_t], dim=0)

        file_name = f"{row.get('patient_id', 'patient')}_{row.get('slice_index', idx)}_{sample_type}"

        return {
            "image": x,
            "mask": mask_t,
            "file_name": file_name
        }


# =============================================================================
# 6. MODEL BLOCKS
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
            nn.Conv2d(inter_channels, 1, kernel_size=1),
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
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        att = self.sigmoid(avg + mx)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        att = torch.cat([avg, mx], dim=1)
        att = self.sigmoid(self.conv(att))

        return x * att


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
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
        self.att1 = AttentionGate(ch, ch, max(ch // 2, 1))
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

        self.stage2 = AttentionUNet(
            in_channels=3,
            out_channels=1,
            base_channels=base_channels
        )

    def forward_stage1(self, x):
        coarse_logits = self.stage1(x)

        return {
            "coarse_logits": coarse_logits,
            "final_logits": coarse_logits
        }

    def forward_stage2(self, x):
        clahe = x[:, 0:1, :, :]
        click = x[:, 1:2, :, :]

        with torch.no_grad():
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

    def forward_full(self, x):
        clahe = x[:, 0:1, :, :]
        click = x[:, 1:2, :, :]

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

    def forward(self, x, mode="full"):
        if mode == "stage1":
            return self.forward_stage1(x)

        if mode == "stage2":
            return self.forward_stage2(x)

        return self.forward_full(x)


# =============================================================================
# 7. LOSSES
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
        fp = ((1.0 - targets) * probs).sum(dim=1)
        fn = (targets * (1.0 - probs)).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = torch.pow((1.0 - tversky), self.gamma)

        return loss.mean()


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        ft_loss = self.focal_tversky(logits, targets)

        total_loss = (
            0.30 * bce_loss +
            0.45 * dice_loss +
            0.25 * ft_loss
        )

        return total_loss


class RATSStagewiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.seg_loss = SegLoss()

    def forward(self, outputs, targets, phase="full"):
        if phase == "stage1":
            return self.seg_loss(outputs["coarse_logits"], targets)

        if phase == "stage2":
            return self.seg_loss(outputs["final_logits"], targets)

        coarse_loss = self.seg_loss(outputs["coarse_logits"], targets)
        final_loss = self.seg_loss(outputs["final_logits"], targets)

        return 0.25 * coarse_loss + 0.75 * final_loss


# =============================================================================
# 8. METRICS AND POST-PROCESSING
# =============================================================================

def dice_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice.mean().item()


def get_click_center(click_map):
    if click_map.max() <= 0:
        h, w = click_map.shape
        return w // 2, h // 2

    y, x = np.unravel_index(np.argmax(click_map), click_map.shape)

    return int(x), int(y)


def click_centered_postprocess(prob, click_map, threshold=0.5):
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

    if USE_CLICK_CENTERED_COMPONENT:
        click_x, click_y = get_click_center(click_map)

        best_label = None
        best_score = 1e18

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            dist = np.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)

            score = dist - 0.01 * area

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


def compute_metrics(pred, gt, smooth=1e-6):
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
# 9. TRAINING FUNCTIONS
# =============================================================================

def set_trainable_stage(model, phase):
    for param in model.parameters():
        param.requires_grad = True

    if phase == "stage1":
        for param in model.stage2.parameters():
            param.requires_grad = False

    elif phase == "stage2":
        for param in model.stage1.parameters():
            param.requires_grad = False

        for param in model.stage2.parameters():
            param.requires_grad = True

    elif phase == "full":
        for param in model.parameters():
            param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, phase):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    valid_batches = 0

    progress = tqdm(loader, desc=f"Training-{phase}", leave=False)

    for batch in progress:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images, mode=phase)
                loss = criterion(outputs, masks, phase=phase)
        else:
            outputs = model(images, mode=phase)
            loss = criterion(outputs, masks, phase=phase)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected. Skipping batch.")
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

        dice = dice_from_logits(outputs["final_logits"], masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        valid_batches += 1

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}"
        })

    avg_loss = total_loss / max(valid_batches, 1)
    avg_dice = total_dice / max(valid_batches, 1)

    return avg_loss, avg_dice


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, phase):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    valid_batches = 0

    eval_mode = "stage1" if phase == "stage1" else "full"
    loss_phase = "stage1" if phase == "stage1" else "full"

    progress = tqdm(loader, desc=f"Validation-{phase}", leave=False)

    for batch in progress:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images, mode=eval_mode)
                loss = criterion(outputs, masks, phase=loss_phase)
        else:
            outputs = model(images, mode=eval_mode)
            loss = criterion(outputs, masks, phase=loss_phase)

        dice = dice_from_logits(outputs["final_logits"], masks, threshold=0.5)

        total_loss += loss.item()
        total_dice += dice
        valid_batches += 1

    avg_loss = total_loss / max(valid_batches, 1)
    avg_dice = total_dice / max(valid_batches, 1)

    return avg_loss, avg_dice


def save_checkpoint(model, optimizer, epoch, val_dice, phase, path):
    checkpoint = {
        "model_name": MODEL_NAME,
        "phase": phase,
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


def run_training_phase(
    model,
    train_loader,
    val_loader,
    phase,
    epochs,
    lr,
    patience,
    checkpoint_path,
    log_name
):
    print()
    print("=" * 80)
    print(f"TRAINING PHASE: {phase.upper()}")
    print("=" * 80)

    set_trainable_stage(model, phase)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f"Trainable parameters in {phase}: {trainable_params:,}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Patience: {patience}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6
    )

    criterion = RATSStagewiseLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    best_val_dice = -1.0
    best_epoch = 0
    early_counter = 0

    logs = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            phase=phase
        )

        val_loss, val_dice = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            phase=phase
        )

        scheduler.step(val_dice)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print("-" * 80)
        print(f"Phase: {phase} | Epoch [{epoch}/{epochs}] | LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.6f} | Val Dice:   {val_dice:.4f}")
        print(f"Time: {epoch_time:.2f}s")

        logs.append({
            "phase": phase,
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "epoch_time_sec": epoch_time
        })

        log_df = pd.DataFrame(logs)
        log_df.to_csv(LOG_DIR / f"{log_name}.csv", index=False)
        log_df.to_excel(LOG_DIR / f"{log_name}.xlsx", index=False)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            early_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_dice=val_dice,
                phase=phase,
                path=checkpoint_path
            )

            print(f"New best {phase} model saved: Val Dice = {best_val_dice:.6f}")

        else:
            early_counter += 1
            print(f"No improvement. Early stopping counter: {early_counter}/{patience}")

        if early_counter >= patience:
            print(f"Early stopping triggered for phase: {phase}")
            break

    print()
    print(f"Best {phase} epoch: {best_epoch}")
    print(f"Best {phase} validation Dice: {best_val_dice:.6f}")

    load_checkpoint(model, checkpoint_path)

    return best_epoch, best_val_dice


# =============================================================================
# 10. EVALUATION
# =============================================================================

@torch.no_grad()
def predict_with_tta(model, images):
    model.eval()

    probs = []

    outputs = model(images, mode="full")
    probs.append(torch.sigmoid(outputs["final_logits"]))

    if USE_TTA:
        x_flip = torch.flip(images, dims=[3])
        outputs_flip = model(x_flip, mode="full")
        prob_flip = torch.sigmoid(outputs_flip["final_logits"])
        prob_flip = torch.flip(prob_flip, dims=[3])
        probs.append(prob_flip)

        y_flip = torch.flip(images, dims=[2])
        outputs_yflip = model(y_flip, mode="full")
        prob_yflip = torch.sigmoid(outputs_yflip["final_logits"])
        prob_yflip = torch.flip(prob_yflip, dims=[2])
        probs.append(prob_yflip)

    avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

    return avg_prob


@torch.no_grad()
def evaluate_thresholds(model, loader, split_name="val", save_predictions=False):
    model.eval()

    threshold_summaries = []

    for threshold in THRESHOLDS:
        metric_rows = []

        progress = tqdm(loader, desc=f"{split_name}-thr-{threshold}", leave=False)

        for batch in progress:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].cpu().numpy()
            file_names = batch["file_name"]

            probs = predict_with_tta(model, images)

            probs_np = probs.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()

            for i in range(probs_np.shape[0]):
                prob = probs_np[i, 0]
                gt = masks[i, 0]
                click = images_np[i, 1]

                pred = click_centered_postprocess(
                    prob=prob,
                    click_map=click,
                    threshold=threshold
                )

                metrics = compute_metrics(pred, gt)
                metrics["threshold"] = threshold
                metrics["split"] = split_name
                metrics["file_name"] = file_names[i]

                metric_rows.append(metrics)

                if save_predictions:
                    save_dir = PRED_DIR / split_name / f"threshold_{threshold:.2f}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(
                        str(save_dir / f"{file_names[i]}_prob.png"),
                        (prob * 255).astype(np.uint8)
                    )

                    cv2.imwrite(
                        str(save_dir / f"{file_names[i]}_pred.png"),
                        (pred * 255).astype(np.uint8)
                    )

                    cv2.imwrite(
                        str(save_dir / f"{file_names[i]}_gt.png"),
                        (gt * 255).astype(np.uint8)
                    )

        df = pd.DataFrame(metric_rows)

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

        threshold_summaries.append(summary)

    return pd.DataFrame(threshold_summaries)


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CLAHE-RATS-Net STAGEWISE V2 TRAINING")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"ROI dataset directory: {ROI_DATASET_DIR}")
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print()

    print("=" * 80)
    print("Preparing stage-specific datasets")
    print("=" * 80)

    train_stage1_df = prepare_dataframe_with_negatives(
        positive_roi_csv=TRAIN_ROI_CSV,
        clahe_split_csv=TRAIN_CLAHE_SPLIT_CSV,
        negative_ratio=NEGATIVE_RATIO_STAGE1,
        use_negatives=USE_NEGATIVE_ROI_STAGE1,
        split_name="train_stage1_positive_only"
    )

    train_stage2_df = prepare_dataframe_with_negatives(
        positive_roi_csv=TRAIN_ROI_CSV,
        clahe_split_csv=TRAIN_CLAHE_SPLIT_CSV,
        negative_ratio=NEGATIVE_RATIO_STAGE2,
        use_negatives=USE_NEGATIVE_ROI_STAGE2,
        split_name="train_stage2_positive_plus_25_percent_negative"
    )

    train_finetune_df = prepare_dataframe_with_negatives(
        positive_roi_csv=TRAIN_ROI_CSV,
        clahe_split_csv=TRAIN_CLAHE_SPLIT_CSV,
        negative_ratio=NEGATIVE_RATIO_FINETUNE,
        use_negatives=USE_NEGATIVE_ROI_FINETUNE,
        split_name="train_finetune_positive_only"
    )

    val_df = prepare_dataframe_with_negatives(
        positive_roi_csv=VAL_ROI_CSV,
        clahe_split_csv=VAL_CLAHE_SPLIT_CSV,
        negative_ratio=0.0,
        use_negatives=False,
        split_name="val_positive_only"
    )

    test_df = prepare_dataframe_with_negatives(
        positive_roi_csv=TEST_ROI_CSV,
        clahe_split_csv=TEST_CLAHE_SPLIT_CSV,
        negative_ratio=0.0,
        use_negatives=False,
        split_name="test_positive_only"
    )

    train_stage1_dataset = LungROIDataset(train_stage1_df, train=True, roi_size=ROI_SIZE)
    train_stage2_dataset = LungROIDataset(train_stage2_df, train=True, roi_size=ROI_SIZE)
    train_finetune_dataset = LungROIDataset(train_finetune_df, train=True, roi_size=ROI_SIZE)

    val_dataset = LungROIDataset(val_df, train=False, roi_size=ROI_SIZE)
    test_dataset = LungROIDataset(test_df, train=False, roi_size=ROI_SIZE)

    train_stage1_loader = DataLoader(
        train_stage1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    train_stage2_loader = DataLoader(
        train_stage2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    train_finetune_loader = DataLoader(
        train_finetune_dataset,
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

    model = CLAHE_RATS_Net(base_channels=BASE_CHANNELS).to(DEVICE)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    stage1_ckpt = CHECKPOINT_DIR / "best_stage1_positive_only.pth"
    stage2_ckpt = CHECKPOINT_DIR / "best_stage2_positive_plus_negative.pth"
    final_ckpt = CHECKPOINT_DIR / "best_clahe_rats_net_stagewise_v2.pth"

    # -------------------------------------------------------------------------
    # Phase 1: Stage 1, positive ROI only
    # -------------------------------------------------------------------------
    stage1_epoch, stage1_dice = run_training_phase(
        model=model,
        train_loader=train_stage1_loader,
        val_loader=val_loader,
        phase="stage1",
        epochs=STAGE1_EPOCHS,
        lr=LR_STAGE1,
        patience=PATIENCE_STAGE1,
        checkpoint_path=stage1_ckpt,
        log_name="phase1_stage1_positive_only_training_log"
    )

    # -------------------------------------------------------------------------
    # Phase 2: Stage 2, positive ROI + 25% negative ROI
    # -------------------------------------------------------------------------
    stage2_epoch, stage2_dice = run_training_phase(
        model=model,
        train_loader=train_stage2_loader,
        val_loader=val_loader,
        phase="stage2",
        epochs=STAGE2_EPOCHS,
        lr=LR_STAGE2,
        patience=PATIENCE_STAGE2,
        checkpoint_path=stage2_ckpt,
        log_name="phase2_stage2_positive_plus_25_negative_training_log"
    )

    # -------------------------------------------------------------------------
    # Phase 3: Full fine-tuning, positive ROI only
    # -------------------------------------------------------------------------
    final_epoch, final_dice = run_training_phase(
        model=model,
        train_loader=train_finetune_loader,
        val_loader=val_loader,
        phase="full",
        epochs=FINETUNE_EPOCHS,
        lr=LR_FINETUNE,
        patience=PATIENCE_FINETUNE,
        checkpoint_path=final_ckpt,
        log_name="phase3_full_finetune_positive_only_training_log"
    )

    print()
    print("=" * 80)
    print("Training completed")
    print("=" * 80)
    print(f"Stage 1 best epoch: {stage1_epoch}")
    print(f"Stage 1 best val Dice: {stage1_dice:.6f}")
    print(f"Stage 2 best epoch: {stage2_epoch}")
    print(f"Stage 2 best val Dice: {stage2_dice:.6f}")
    print(f"Final best epoch: {final_epoch}")
    print(f"Final best val Dice: {final_dice:.6f}")

    print()
    print("Loading final best model for threshold evaluation...")

    checkpoint = load_checkpoint(model, final_ckpt)

    print(f"Loaded final checkpoint phase: {checkpoint['phase']}")
    print(f"Loaded final checkpoint epoch: {checkpoint['epoch']}")
    print(f"Loaded final checkpoint val Dice: {checkpoint['val_dice']:.6f}")

    # -------------------------------------------------------------------------
    # Validation threshold evaluation
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VALIDATION THRESHOLD SUMMARY")
    print("=" * 80)

    val_summary = evaluate_thresholds(
        model=model,
        loader=val_loader,
        split_name="val",
        save_predictions=False
    )

    print(val_summary)

    val_summary.to_csv(LOG_DIR / "validation_threshold_summary_stagewise_v2.csv", index=False)
    val_summary.to_excel(LOG_DIR / "validation_threshold_summary_stagewise_v2.xlsx", index=False)

    best_val_row = val_summary.loc[val_summary["dice_mean"].idxmax()]
    best_threshold = float(best_val_row["threshold"])

    print()
    print(f"Best validation threshold: {best_threshold}")

    # -------------------------------------------------------------------------
    # Test threshold evaluation
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TEST THRESHOLD SUMMARY")
    print("=" * 80)

    test_summary = evaluate_thresholds(
        model=model,
        loader=test_loader,
        split_name="test",
        save_predictions=True
    )

    print(test_summary)

    test_summary.to_csv(LOG_DIR / "test_threshold_summary_stagewise_v2.csv", index=False)
    test_summary.to_excel(LOG_DIR / "test_threshold_summary_stagewise_v2.xlsx", index=False)

    best_test_row = test_summary[test_summary["threshold"] == best_threshold].iloc[0]

    final_summary = pd.DataFrame([{
        "seed": SEED,
        "model": MODEL_NAME,
        "roi_size": ROI_SIZE,

        "stage1_best_epoch": stage1_epoch,
        "stage1_best_val_dice": stage1_dice,

        "stage2_best_epoch": stage2_epoch,
        "stage2_best_val_dice": stage2_dice,

        "final_best_epoch": final_epoch,
        "final_best_val_training_dice": final_dice,

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

        "use_negative_stage1": USE_NEGATIVE_ROI_STAGE1,
        "negative_ratio_stage1": NEGATIVE_RATIO_STAGE1,

        "use_negative_stage2": USE_NEGATIVE_ROI_STAGE2,
        "negative_ratio_stage2": NEGATIVE_RATIO_STAGE2,

        "use_negative_finetune": USE_NEGATIVE_ROI_FINETUNE,
        "negative_ratio_finetune": NEGATIVE_RATIO_FINETUNE,

        "use_tta": USE_TTA,
        "postprocessing": "Click-centered connected component + remove small objects",
        "loss": "Stage-wise BCE + Dice + Focal Tversky",
        "data_type": "Stage1 Positive ROI; Stage2 Positive+25% Negative; Finetune Positive ROI"
    }])

    print()
    print("=" * 80)
    print("FINAL BEST SUMMARY")
    print("=" * 80)
    print(final_summary)

    final_summary.to_csv(LOG_DIR / "final_best_summary_stagewise_v2.csv", index=False)
    final_summary.to_excel(LOG_DIR / "final_best_summary_stagewise_v2.xlsx", index=False)

    print()
    print("All outputs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
