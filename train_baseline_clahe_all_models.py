# ============================================================
# train_baseline_clahe_all_models.py
# ============================================================

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"ACTUAL_PATH")

SPLIT_DIR = PROJECT_ROOT / "outputs" / "baseline_clahe_preprocessing" / "dataset_splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_clahe_training_results"

DATA_TYPES = ["baseline", "clahe"]

MODELS_TO_RUN = [
    "unet",
    "attention_unet",
    "unetpp",
    "proposed_resatt_shallow"
]

IMAGE_SIZE_INFO = "512x256"

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 20

NEGATIVE_PER_POSITIVE = 2
USE_BALANCED_TRAINING = True

EVAL_POSITIVE_ONLY = True

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# CSV DETECTION
# ============================================================

def find_split_csv(data_type, split_name):
    """
    Robustly detects split files.
    Expected possible names:
      baseline_train.csv
      train_baseline.csv
      clahe_train.csv
      train_clahe.csv
    """

    csv_files = list(SPLIT_DIR.glob("*.csv"))

    candidates = []
    for f in csv_files:
        name = f.name.lower()
        if data_type.lower() in name and split_name.lower() in name:
            candidates.append(f)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No CSV found for data_type={data_type}, split={split_name} in:\n{SPLIT_DIR}\n\n"
            f"Available CSV files:\n" + "\n".join([x.name for x in csv_files])
        )

    return candidates[0]


def prepare_dataframe(csv_path, data_type):
    df = pd.read_csv(csv_path)

    # Remove duplicate columns if any already exist
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Identify positive and empty slices
    if "mask_status" in df.columns:
        df["is_positive"] = df["mask_status"].astype(str).str.lower().eq("positive")
    elif "mask_pixel_count" in df.columns:
        df["is_positive"] = df["mask_pixel_count"] > 0
    else:
        raise ValueError("CSV must contain either mask_status or mask_pixel_count column.")

    # Select correct columns based on baseline or CLAHE
    if data_type == "baseline":
        image_col = "baseline_image_path"
        mask_col = "baseline_mask_path"
        click_col = "baseline_click_path"
    elif data_type == "clahe":
        image_col = "clahe_image_path"
        mask_col = "clahe_mask_path"
        click_col = "clahe_click_path"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    required_cols = [
        "patient_id",
        "slice_index",
        image_col,
        mask_col,
        click_col,
        "is_positive"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column in CSV: {col}\n\n"
                f"Available columns are:\n{list(df.columns)}"
            )

    # IMPORTANT:
    # Do not rename directly inside original dataframe.
    # Create a new clean dataframe to avoid duplicate image_path/mask_path/click_path columns.
    clean_df = pd.DataFrame()

    clean_df["patient_id"] = df["patient_id"].astype(str)
    clean_df["slice_index"] = df["slice_index"].astype(int)

    clean_df["image_path"] = df[image_col].astype(str)
    clean_df["mask_path"] = df[mask_col].astype(str)
    clean_df["click_path"] = df[click_col].astype(str)

    clean_df["is_positive"] = df["is_positive"].astype(bool)

    # Remove missing or invalid path rows
    clean_df = clean_df.dropna(subset=["image_path", "mask_path", "click_path"])

    clean_df = clean_df[
        (clean_df["image_path"].str.lower() != "nan") &
        (clean_df["mask_path"].str.lower() != "nan") &
        (clean_df["click_path"].str.lower() != "nan")
    ].copy()

    # Check whether files really exist
    missing_images = clean_df[~clean_df["image_path"].apply(lambda x: Path(x).exists())]
    missing_masks = clean_df[~clean_df["mask_path"].apply(lambda x: Path(x).exists())]
    missing_clicks = clean_df[~clean_df["click_path"].apply(lambda x: Path(x).exists())]

    if len(missing_images) > 0:
        print(f"\nWarning: {len(missing_images)} image files are missing.")
        print("Example missing image:")
        print(missing_images.iloc[0]["image_path"])

    if len(missing_masks) > 0:
        print(f"\nWarning: {len(missing_masks)} mask files are missing.")
        print("Example missing mask:")
        print(missing_masks.iloc[0]["mask_path"])

    if len(missing_clicks) > 0:
        print(f"\nWarning: {len(missing_clicks)} click files are missing.")
        print("Example missing click:")
        print(missing_clicks.iloc[0]["click_path"])

    clean_df = clean_df[
        clean_df["image_path"].apply(lambda x: Path(x).exists()) &
        clean_df["mask_path"].apply(lambda x: Path(x).exists()) &
        clean_df["click_path"].apply(lambda x: Path(x).exists())
    ].reset_index(drop=True)

    print(f"\nPrepared dataframe for {data_type}: {len(clean_df)} valid rows")
    print(f"Positive slices: {clean_df['is_positive'].sum()}")
    print(f"Empty slices: {(~clean_df['is_positive']).sum()}")

    return clean_df


# ============================================================
# BALANCED TRAINING DATA
# ============================================================

def create_balanced_train_df(df, neg_per_pos=2):
    positive_df = df[df["is_positive"] == True].copy()
    negative_df = df[df["is_positive"] == False].copy()

    n_pos = len(positive_df)
    n_neg_required = min(len(negative_df), n_pos * neg_per_pos)

    negative_sampled = negative_df.sample(
        n=n_neg_required,
        random_state=SEED
    )

    balanced_df = pd.concat([positive_df, negative_sampled], axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return balanced_df


# ============================================================
# DATASET
# ============================================================

class LungClickDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_npy(self, path):
        path = str(path)

        arr = np.load(path).astype(np.float32)

        if arr.ndim == 3:
            arr = np.squeeze(arr)

        return arr    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self._load_npy(row["image_path"])
        mask = self._load_npy(row["mask_path"])
        click = self._load_npy(row["click_path"])

        image = np.clip(image, 0.0, 1.0)
        mask = (mask > 0.5).astype(np.float32)
        click = np.clip(click, 0.0, 1.0)

        x = np.stack([image, click], axis=0)
        y = mask[None, :, :]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return {
            "image": x,
            "mask": y,
            "patient_id": row["patient_id"],
            "slice_index": int(row["slice_index"]),
            "is_positive": bool(row["is_positive"])
        }


# ============================================================
# MODEL BLOCKS
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()

        self.e1 = DoubleConv(in_ch, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.b = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.b(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.d4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# ============================================================
# ATTENTION U-NET
# ============================================================

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()

        self.e1 = DoubleConv(in_ch, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.b = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.d4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.d3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.d2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.d1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.b(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.d4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# ============================================================
# SIMPLE U-NET++
# ============================================================

class UNetPP(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv00 = DoubleConv(in_ch, base)
        self.conv10 = DoubleConv(base, base * 2)
        self.conv20 = DoubleConv(base * 2, base * 4)
        self.conv30 = DoubleConv(base * 4, base * 8)
        self.conv40 = DoubleConv(base * 8, base * 16)

        self.up10 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.up20 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.up30 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.up40 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)

        self.conv01 = DoubleConv(base + base, base)
        self.conv11 = DoubleConv(base * 2 + base * 2, base * 2)
        self.conv21 = DoubleConv(base * 4 + base * 4, base * 4)
        self.conv31 = DoubleConv(base * 8 + base * 8, base * 8)

        self.up11 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.up21 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.up31 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)

        self.conv02 = DoubleConv(base * 2 + base, base)
        self.conv12 = DoubleConv(base * 4 + base * 2, base * 2)
        self.conv22 = DoubleConv(base * 8 + base * 4, base * 4)

        self.up12 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.up22 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)

        self.conv03 = DoubleConv(base * 3 + base, base)
        self.conv13 = DoubleConv(base * 6 + base * 2, base * 2)

        self.up13 = nn.ConvTranspose2d(base * 2, base, 2, 2)

        self.conv04 = DoubleConv(base * 4 + base, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        x01 = self.conv01(torch.cat([x00, self.up10(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up20(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up30(x30)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up40(x40)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up11(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up21(x21)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up31(x31)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up12(x12)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up22(x22)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up13(x13)], dim=1))

        return self.out(x04)


# ============================================================
# PROPOSED MODEL:
# RESIDUAL ATTENTION SHALLOW U-NET
# ============================================================

class ProposedResAttShallowUNet(nn.Module):
    """
    Proposed model for click-guided lung nodule segmentation.

    Input:
        Channel 1: CT image
        Channel 2: click map

    Key idea:
        Residual shallow encoder-decoder + attention gates.
        Suitable for small nodules and limited medical data.
    """

    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.e1 = ResidualConv(in_ch, base)
        self.e2 = ResidualConv(base, base * 2)
        self.e3 = ResidualConv(base * 2, base * 4)

        self.bridge = ResidualConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.d3 = ResidualConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.d2 = ResidualConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.d1 = ResidualConv(base * 2, base)

        self.refine = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        b = self.bridge(self.pool(e3))

        d3 = self.up3(b)
        e3_att = self.att3(d3, e3)
        d3 = self.d3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.d2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.d1(torch.cat([d1, e1_att], dim=1))

        d1 = self.refine(d1)

        return self.out(d1)


# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(model_name):
    if model_name == "unet":
        return UNet(in_ch=2, out_ch=1, base=32)

    if model_name == "attention_unet":
        return AttentionUNet(in_ch=2, out_ch=1, base=32)

    if model_name == "unetpp":
        return UNetPP(in_ch=2, out_ch=1, base=32)

    if model_name == "proposed_resatt_shallow":
        return ProposedResAttShallowUNet(in_ch=2, out_ch=1, base=32)

    raise ValueError(f"Unknown model name: {model_name}")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def dice_loss_from_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================
# METRICS
# ============================================================

def compute_metrics_from_probs(probs, masks, threshold=0.5, eps=1e-7):
    preds = (probs >= threshold).astype(np.uint8)
    masks = (masks >= 0.5).astype(np.uint8)

    pred_flat = preds.reshape(-1)
    mask_flat = masks.reshape(-1)

    tp = np.sum((pred_flat == 1) & (mask_flat == 1))
    fp = np.sum((pred_flat == 1) & (mask_flat == 0))
    fn = np.sum((pred_flat == 0) & (mask_flat == 1))
    tn = np.sum((pred_flat == 0) & (mask_flat == 0))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity
    }


@torch.no_grad()
def evaluate_model(model, loader, thresholds, output_dir, split_name):
    model.eval()

    records = []

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].cpu().numpy()

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        batch_size = probs.shape[0]

        for i in range(batch_size):
            prob_i = probs[i, 0]
            mask_i = masks[i, 0]

            for th in thresholds:
                m = compute_metrics_from_probs(prob_i, mask_i, threshold=th)

                records.append({
                    "patient_id": batch["patient_id"][i],
                    "slice_index": int(batch["slice_index"][i]),
                    "threshold": th,
                    "dice": m["dice"],
                    "iou": m["iou"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "specificity": m["specificity"],
                    "is_positive": bool(batch["is_positive"][i])
                })

    per_image_df = pd.DataFrame(records)

    summary_df = (
        per_image_df
        .groupby("threshold")
        .agg(
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
        )
        .reset_index()
    )

    per_image_csv = output_dir / f"{split_name}_per_image_results.csv"
    summary_csv = output_dir / f"{split_name}_threshold_summary.csv"

    per_image_xlsx = output_dir / f"{split_name}_per_image_results.xlsx"
    summary_xlsx = output_dir / f"{split_name}_threshold_summary.xlsx"

    per_image_df.to_csv(per_image_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    per_image_df.to_excel(per_image_xlsx, index=False)
    summary_df.to_excel(summary_xlsx, index=False)

    return per_image_df, summary_df


# ============================================================
# TRAINING
# ============================================================

def train_one_model(data_type, model_name):
    print("\n" + "=" * 100)
    print(f"Training model: {model_name}")
    print(f"Data type: {data_type}")
    print("=" * 100)

    train_csv = find_split_csv(data_type, "train")
    val_csv = find_split_csv(data_type, "val")
    test_csv = find_split_csv(data_type, "test")

    train_df = prepare_dataframe(train_csv, data_type)
    val_df = prepare_dataframe(val_csv, data_type)
    test_df = prepare_dataframe(test_csv, data_type)

    print(f"\nOriginal train samples: {len(train_df)}")
    print(f"Original val samples:   {len(val_df)}")
    print(f"Original test samples:  {len(test_df)}")

    print(f"Train positive: {train_df['is_positive'].sum()} | Empty: {(~train_df['is_positive']).sum()}")
    print(f"Val positive:   {val_df['is_positive'].sum()} | Empty: {(~val_df['is_positive']).sum()}")
    print(f"Test positive:  {test_df['is_positive'].sum()} | Empty: {(~test_df['is_positive']).sum()}")

    if USE_BALANCED_TRAINING:
        train_df_used = create_balanced_train_df(train_df, neg_per_pos=NEGATIVE_PER_POSITIVE)
    else:
        train_df_used = train_df.copy()

    if EVAL_POSITIVE_ONLY:
        val_df_used = val_df[val_df["is_positive"] == True].copy()
        test_df_used = test_df[test_df["is_positive"] == True].copy()
    else:
        val_df_used = val_df.copy()
        test_df_used = test_df.copy()

    print(f"\nUsed train samples: {len(train_df_used)}")
    print(f"Used val samples:   {len(val_df_used)}")
    print(f"Used test samples:  {len(test_df_used)}")

    run_output_dir = OUTPUT_DIR / data_type / model_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_df_used.to_csv(run_output_dir / "used_train_split.csv", index=False)
    val_df_used.to_csv(run_output_dir / "used_val_split.csv", index=False)
    test_df_used.to_csv(run_output_dir / "used_test_split.csv", index=False)

    train_loader = DataLoader(
        LungClickDataset(train_df_used),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        LungClickDataset(val_df_used),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        LungClickDataset(test_df_used),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = get_model(model_name).to(DEVICE)

    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    early_counter = 0

    train_logs = []

    best_model_path = run_output_dir / f"best_{model_name}_{data_type}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        model.train()

        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"{data_type}-{model_name} Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if torch.isnan(loss):
                print("NaN loss detected. Stopping this run.")
                return None

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)

        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss_total += loss.item() * images.size(0)

        val_loss = val_loss_total / len(val_loader.dataset)

        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        train_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "time_sec": elapsed
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")
        else:
            early_counter += 1
            print(f"No improvement. Early stopping counter: {early_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    log_df = pd.DataFrame(train_logs)
    log_df.to_csv(run_output_dir / "training_log.csv", index=False)
    log_df.to_excel(run_output_dir / "training_log.xlsx", index=False)

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    val_per_image, val_summary = evaluate_model(
        model=model,
        loader=val_loader,
        thresholds=THRESHOLDS,
        output_dir=run_output_dir,
        split_name="val"
    )

    test_per_image, test_summary = evaluate_model(
        model=model,
        loader=test_loader,
        thresholds=THRESHOLDS,
        output_dir=run_output_dir,
        split_name="test"
    )

    best_val_row = val_summary.loc[val_summary["dice_mean"].idxmax()]
    best_threshold = float(best_val_row["threshold"])

    test_best_row = test_summary[test_summary["threshold"] == best_threshold].iloc[0]

    best_summary = pd.DataFrame([{
        "data_type": data_type,
        "model": model_name,
        "image_size": IMAGE_SIZE_INFO,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_threshold": best_threshold,
        "best_val_dice": best_val_row["dice_mean"],
        "best_val_iou": best_val_row["iou_mean"],
        "best_val_precision": best_val_row["precision_mean"],
        "best_val_recall": best_val_row["recall_mean"],
        "best_test_threshold": best_threshold,
        "best_test_dice": test_best_row["dice_mean"],
        "best_test_iou": test_best_row["iou_mean"],
        "best_test_precision": test_best_row["precision_mean"],
        "best_test_recall": test_best_row["recall_mean"],
        "best_test_specificity": test_best_row["specificity_mean"],
        "model_path": str(best_model_path),
        "output_dir": str(run_output_dir)
    }])

    best_summary.to_csv(run_output_dir / "best_summary.csv", index=False)
    best_summary.to_excel(run_output_dir / "best_summary.xlsx", index=False)

    print("\nBest Summary:")
    print(best_summary)

    return best_summary


# ============================================================
# COMBINE ALL RESULTS
# ============================================================

def combine_all_results(all_summaries):
    if len(all_summaries) == 0:
        print("No summaries to combine.")
        return

    final_df = pd.concat(all_summaries, axis=0).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_csv = OUTPUT_DIR / "overall_best_model_comparison.csv"
    final_xlsx = OUTPUT_DIR / "overall_best_model_comparison.xlsx"

    final_df.to_csv(final_csv, index=False)
    final_df.to_excel(final_xlsx, index=False)

    print("\n" + "=" * 100)
    print("OVERALL BEST MODEL COMPARISON")
    print("=" * 100)
    print(final_df)

    print(f"\nSaved final comparison CSV:\n{final_csv}")
    print(f"Saved final comparison Excel:\n{final_xlsx}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print(f"\nSplit directory:\n{SPLIT_DIR}")
    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    all_summaries = []

    for data_type in DATA_TYPES:
        for model_name in MODELS_TO_RUN:
            summary = train_one_model(data_type, model_name)

            if summary is not None:
                all_summaries.append(summary)

    combine_all_results(all_summaries)

    print("\nDone. All experiments completed.")


if __name__ == "__main__":
    main()
