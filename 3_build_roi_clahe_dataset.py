# =============================================================================
# STEP 1: Build Clean CLAHE ROI Dataset for Proposed Model
# =============================================================================
# Purpose:
#   This script reads patient-wise split CSV files generated from baseline/CLAHE
#   preprocessing and creates a clean ROI dataset for lung nodule segmentation.
#
# Input:
#   outputs/baseline_clahe_preprocessing/dataset_splits/clahe_train.csv
#   outputs/baseline_clahe_preprocessing/dataset_splits/clahe_val.csv
#   outputs/baseline_clahe_preprocessing/dataset_splits/clahe_test.csv
#
# Output:
#   outputs/proposed_roi_clahe_dataset_clean/
#       train_roi_clahe.csv
#       val_roi_clahe.csv
#       test_roi_clahe.csv
#       roi_quality_summary.xlsx
#       visual_checks/
#
# Improvements:
#   1. Clean centroid-based ROI crop
#   2. Tumor-border touch rejection
#   3. Small-mask filtering
#   4. Click-map validation
#   5. ROI quality reports
#   6. Optional ROI jitter augmentation for training split
# =============================================================================

import os
import cv2
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"E:\Mahender PHD\segmentaion\Project 23-4-2026")

SPLIT_DIR = PROJECT_ROOT / "outputs" / "baseline_clahe_preprocessing" / "dataset_splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset_clean"
ROI_IMAGE_DIR = OUTPUT_DIR / "images"
ROI_MASK_DIR = OUTPUT_DIR / "masks"
ROI_CLICK_DIR = OUTPUT_DIR / "clicks"
VISUAL_CHECK_DIR = OUTPUT_DIR / "visual_checks"

for folder in [OUTPUT_DIR, ROI_IMAGE_DIR, ROI_MASK_DIR, ROI_CLICK_DIR, VISUAL_CHECK_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLIT_DIR / "clahe_train.csv"
VAL_CSV = SPLIT_DIR / "clahe_val.csv"
TEST_CSV = SPLIT_DIR / "clahe_test.csv"

# ROI settings
ROI_SIZE = 256
BASE_MARGIN = 40

# Quality filtering
MIN_MASK_PIXELS_ROI = 30
MIN_BBOX_WIDTH = 3
MIN_BBOX_HEIGHT = 3
MAX_DISTANCE_FROM_CENTER = 45

# Border tolerance in pixels
BORDER_TOLERANCE = 2

# Jitter augmentation for training only
USE_TRAIN_JITTER = True
TRAIN_JITTER_COUNT = 2
JITTER_SHIFT_PIXELS = 10
JITTER_MARGIN_OPTIONS = [32, 40, 48]

# Visual checking
SAVE_VISUAL_CHECKS = True
MAX_VISUAL_CHECKS_PER_SPLIT = 80

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 2. BASIC UTILITIES
# =============================================================================

def ensure_exists(path, description="file"):
    if not Path(path).exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def read_gray_image(path):
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

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


def save_gray_image(path, arr):
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255).astype(np.uint8)
    cv2.imwrite(str(path), arr_uint8)


def get_required_columns(df, split_name):
    required = [
        "patient_id",
        "slice_index",
        "mask_pixel_count",
        "mask_status",
        "clahe_image_path",
        "clahe_mask_path",
        "clahe_click_path",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nMissing required columns in {split_name} CSV:\n"
            f"{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )


def is_positive_row(row):
    mask_status = str(row.get("mask_status", "")).upper()
    mask_pixels = float(row.get("mask_pixel_count", 0))

    if mask_status in ["POSITIVE", "NODULE", "MASK", "TUMOR"]:
        return True

    if mask_pixels > 0:
        return True

    return False


# =============================================================================
# 3. ROI QUALITY FUNCTIONS
# =============================================================================

def get_mask_bbox(mask):
    mask_bin = (mask > 0.5).astype(np.uint8)

    ys, xs = np.where(mask_bin > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return x1, y1, x2, y2


def get_mask_centroid(mask):
    mask_bin = (mask > 0.5).astype(np.uint8)

    ys, xs = np.where(mask_bin > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    cx = float(xs.mean())
    cy = float(ys.mean())

    return cx, cy


def crop_with_padding(image, mask, center_x, center_y, roi_size=256):
    h, w = image.shape

    half = roi_size // 2

    x1 = int(round(center_x)) - half
    y1 = int(round(center_y)) - half
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        image_pad = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )

        mask_pad = cv2.copyMakeBorder(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        x1 = x1 + pad_left
        x2 = x2 + pad_left
        y1 = y1 + pad_top
        y2 = y2 + pad_top
    else:
        image_pad = image
        mask_pad = mask

    image_roi = image_pad[y1:y2, x1:x2]
    mask_roi = mask_pad[y1:y2, x1:x2]

    if image_roi.shape != (roi_size, roi_size):
        image_roi = cv2.resize(image_roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    if mask_roi.shape != (roi_size, roi_size):
        mask_roi = cv2.resize(mask_roi, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)

    mask_roi = (mask_roi > 0.5).astype(np.float32)

    crop_info = {
        "crop_x1": int(x1 - pad_left),
        "crop_y1": int(y1 - pad_top),
        "crop_x2": int(x2 - pad_left),
        "crop_y2": int(y2 - pad_top),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
    }

    return image_roi.astype(np.float32), mask_roi.astype(np.float32), crop_info


def create_click_map_from_mask(mask, sigma=10):
    mask_bin = (mask > 0.5).astype(np.uint8)

    if mask_bin.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    ys, xs = np.where(mask_bin > 0)

    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))

    h, w = mask.shape

    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    click = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
    click = click.astype(np.float32)
    click = click / (click.max() + 1e-8)

    return click


def analyze_roi_quality(mask_roi, click_roi):
    mask_bin = (mask_roi > 0.5).astype(np.uint8)

    mask_pixels = int(mask_bin.sum())

    bbox = get_mask_bbox(mask_bin)

    if bbox is None:
        return {
            "mask_pixel_count_roi": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "distance_from_roi_center": np.nan,
            "tumor_area_ratio": 0.0,
            "touches_border": True,
            "click_inside_mask": False,
            "roi_quality_status": "REJECT_EMPTY_MASK"
        }

    x1, y1, x2, y2 = bbox

    bbox_width = int(x2 - x1 + 1)
    bbox_height = int(y2 - y1 + 1)

    centroid = get_mask_centroid(mask_bin)
    cx, cy = centroid

    roi_center_x = ROI_SIZE / 2.0
    roi_center_y = ROI_SIZE / 2.0

    distance_from_center = math.sqrt((cx - roi_center_x) ** 2 + (cy - roi_center_y) ** 2)

    tumor_area_ratio = mask_pixels / float(ROI_SIZE * ROI_SIZE)

    touches_border = (
        mask_bin[:BORDER_TOLERANCE, :].sum() > 0 or
        mask_bin[-BORDER_TOLERANCE:, :].sum() > 0 or
        mask_bin[:, :BORDER_TOLERANCE].sum() > 0 or
        mask_bin[:, -BORDER_TOLERANCE:].sum() > 0
    )

    click_y, click_x = np.unravel_index(np.argmax(click_roi), click_roi.shape)
    click_inside_mask = bool(mask_bin[click_y, click_x] > 0)

    reject_reasons = []

    if mask_pixels < MIN_MASK_PIXELS_ROI:
        reject_reasons.append("SMALL_MASK")

    if bbox_width < MIN_BBOX_WIDTH:
        reject_reasons.append("SMALL_BBOX_WIDTH")

    if bbox_height < MIN_BBOX_HEIGHT:
        reject_reasons.append("SMALL_BBOX_HEIGHT")

    if touches_border:
        reject_reasons.append("TOUCHES_BORDER")

    if distance_from_center > MAX_DISTANCE_FROM_CENTER:
        reject_reasons.append("OFF_CENTER")

    if not click_inside_mask:
        reject_reasons.append("CLICK_OUTSIDE_MASK")

    if len(reject_reasons) == 0:
        quality_status = "ACCEPT"
    else:
        quality_status = "REJECT_" + "|".join(reject_reasons)

    return {
        "mask_pixel_count_roi": mask_pixels,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "distance_from_roi_center": float(distance_from_center),
        "tumor_area_ratio": float(tumor_area_ratio),
        "touches_border": bool(touches_border),
        "click_inside_mask": bool(click_inside_mask),
        "roi_quality_status": quality_status
    }


# =============================================================================
# 4. VISUAL CHECK FUNCTIONS
# =============================================================================

def make_overlay(image, mask, color=(0, 0, 255), alpha=0.45):
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

    mask_bin = (mask > 0.5).astype(np.uint8)

    color_layer = np.zeros_like(image_rgb)
    color_layer[mask_bin > 0] = color

    overlay = cv2.addWeighted(image_rgb, 1.0, color_layer, alpha, 0)

    return overlay


def write_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return out


def save_visual_check(split_name, sample_name, image_roi, mask_roi, click_roi, quality):
    overlay = make_overlay(image_roi, mask_roi)

    click_vis = (np.clip(click_roi, 0, 1) * 255).astype(np.uint8)
    click_vis = cv2.applyColorMap(click_vis, cv2.COLORMAP_JET)

    image_vis = cv2.cvtColor((image_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.cvtColor((mask_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    image_vis = write_label(image_vis, "CLAHE ROI")
    mask_vis = write_label(mask_vis, "Mask ROI")
    click_vis = write_label(click_vis, "Click Map")
    overlay = write_label(overlay, quality["roi_quality_status"])

    top = np.hstack([image_vis, mask_vis])
    bottom = np.hstack([click_vis, overlay])
    canvas = np.vstack([top, bottom])

    save_dir = VISUAL_CHECK_DIR / split_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_dir / f"{sample_name}.png"), canvas)


# =============================================================================
# 5. MAIN ROI CREATION FUNCTION
# =============================================================================

def build_roi_for_row(row, split_name, variant_id=0, jitter=False):
    patient_id = str(row["patient_id"])
    slice_index = int(row["slice_index"])

    image_path = row["clahe_image_path"]
    mask_path = row["clahe_mask_path"]

    image = read_gray_image(image_path)
    mask = read_gray_image(mask_path)
    mask = (mask > 0.5).astype(np.float32)

    if mask.sum() <= 0:
        return None

    bbox = get_mask_bbox(mask)
    centroid = get_mask_centroid(mask)

    if bbox is None or centroid is None:
        return None

    cx, cy = centroid

    if jitter:
        shift_x = random.randint(-JITTER_SHIFT_PIXELS, JITTER_SHIFT_PIXELS)
        shift_y = random.randint(-JITTER_SHIFT_PIXELS, JITTER_SHIFT_PIXELS)

        cx = cx + shift_x
        cy = cy + shift_y
    else:
        shift_x = 0
        shift_y = 0

    image_roi, mask_roi, crop_info = crop_with_padding(
        image=image,
        mask=mask,
        center_x=cx,
        center_y=cy,
        roi_size=ROI_SIZE
    )

    click_roi = create_click_map_from_mask(mask_roi, sigma=10)

    quality = analyze_roi_quality(mask_roi, click_roi)

    base_name = f"{patient_id}_slice{slice_index:04d}_v{variant_id}"

    image_save_path = ROI_IMAGE_DIR / split_name / f"{base_name}_image.png"
    mask_save_path = ROI_MASK_DIR / split_name / f"{base_name}_mask.png"
    click_save_path = ROI_CLICK_DIR / split_name / f"{base_name}_click.png"

    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    mask_save_path.parent.mkdir(parents=True, exist_ok=True)
    click_save_path.parent.mkdir(parents=True, exist_ok=True)

    save_gray_image(image_save_path, image_roi)
    save_gray_image(mask_save_path, mask_roi)
    save_gray_image(click_save_path, click_roi)

    output_row = {
        "patient_id": patient_id,
        "slice_index": slice_index,
        "variant_id": variant_id,
        "split": split_name,
        "image_path": str(image_save_path),
        "mask_path": str(mask_save_path),
        "click_path": str(click_save_path),
        "original_image_path": str(image_path),
        "original_mask_path": str(mask_path),
        "original_click_path": str(row.get("clahe_click_path", "")),
        "source_mask_pixel_count": int(row.get("mask_pixel_count", 0)),
        "jitter_used": bool(jitter),
        "jitter_shift_x": int(shift_x),
        "jitter_shift_y": int(shift_y),
    }

    output_row.update(crop_info)
    output_row.update(quality)

    return output_row, image_roi, mask_roi, click_roi, quality, base_name


def process_split(split_name, split_csv, use_jitter=False):
    ensure_exists(split_csv, f"{split_name} split CSV")

    print("\n" + "=" * 80)
    print(f"Processing split: {split_name}")
    print("=" * 80)
    print(f"Reading: {split_csv}")

    df = pd.read_csv(split_csv)

    get_required_columns(df, split_name)

    positive_df = df[df.apply(is_positive_row, axis=1)].copy()
    positive_df = positive_df.reset_index(drop=True)

    print(f"Total rows in split: {len(df)}")
    print(f"Positive rows found: {len(positive_df)}")

    all_rows = []
    visual_count = 0

    for idx, row in tqdm(positive_df.iterrows(), total=len(positive_df), desc=f"{split_name} ROI"):
        variants = [(0, False)]

        if use_jitter:
            for j in range(1, TRAIN_JITTER_COUNT + 1):
                variants.append((j, True))

        for variant_id, jitter in variants:
            try:
                result = build_roi_for_row(
                    row=row,
                    split_name=split_name,
                    variant_id=variant_id,
                    jitter=jitter
                )

                if result is None:
                    continue

                output_row, image_roi, mask_roi, click_roi, quality, base_name = result

                all_rows.append(output_row)

                if SAVE_VISUAL_CHECKS and visual_count < MAX_VISUAL_CHECKS_PER_SPLIT:
                    if variant_id == 0:
                        save_visual_check(
                            split_name=split_name,
                            sample_name=base_name,
                            image_roi=image_roi,
                            mask_roi=mask_roi,
                            click_roi=click_roi,
                            quality=quality
                        )
                        visual_count += 1

            except Exception as e:
                print(f"\nError processing {split_name} row {idx}: {e}")

    roi_df = pd.DataFrame(all_rows)

    if len(roi_df) == 0:
        raise RuntimeError(f"No ROI samples created for split: {split_name}")

    # Save all ROI samples before filtering
    all_csv_path = OUTPUT_DIR / f"{split_name}_roi_clahe_all_with_quality.csv"
    roi_df.to_csv(all_csv_path, index=False)

    # Clean accepted samples only
    clean_df = roi_df[roi_df["roi_quality_status"] == "ACCEPT"].copy()
    clean_df = clean_df.reset_index(drop=True)

    clean_csv_path = OUTPUT_DIR / f"{split_name}_roi_clahe.csv"
    clean_df.to_csv(clean_csv_path, index=False)

    print(f"\n{split_name} ROI all samples: {len(roi_df)}")
    print(f"{split_name} ROI clean ACCEPT samples: {len(clean_df)}")
    print(f"Saved all quality CSV: {all_csv_path}")
    print(f"Saved clean CSV: {clean_csv_path}")

    quality_counts = roi_df["roi_quality_status"].value_counts()
    print("\nROI quality status counts:")
    print(quality_counts)

    return roi_df, clean_df


# =============================================================================
# 6. FINAL SUMMARY
# =============================================================================

def summarize_split(split_name, all_df, clean_df):
    summary = {
        "split": split_name,
        "all_samples": len(all_df),
        "clean_samples": len(clean_df),
        "rejected_samples": len(all_df) - len(clean_df),
        "unique_patients_all": all_df["patient_id"].nunique() if len(all_df) else 0,
        "unique_patients_clean": clean_df["patient_id"].nunique() if len(clean_df) else 0,
        "mean_mask_pixels_clean": clean_df["mask_pixel_count_roi"].mean() if len(clean_df) else 0,
        "median_mask_pixels_clean": clean_df["mask_pixel_count_roi"].median() if len(clean_df) else 0,
        "mean_distance_from_center_clean": clean_df["distance_from_roi_center"].mean() if len(clean_df) else 0,
        "max_distance_from_center_clean": clean_df["distance_from_roi_center"].max() if len(clean_df) else 0,
    }

    return summary


def save_quality_reports(split_data):
    summary_rows = []

    writer_path = OUTPUT_DIR / "roi_quality_summary.xlsx"

    with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
        for split_name, all_df, clean_df in split_data:
            summary_rows.append(summarize_split(split_name, all_df, clean_df))

            all_df.to_excel(writer, sheet_name=f"{split_name}_all_quality", index=False)
            clean_df.to_excel(writer, sheet_name=f"{split_name}_clean", index=False)

            counts = all_df["roi_quality_status"].value_counts().reset_index()
            counts.columns = ["roi_quality_status", "count"]
            counts.to_excel(writer, sheet_name=f"{split_name}_quality_counts", index=False)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    summary_csv = OUTPUT_DIR / "roi_quality_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("ROI QUALITY SUMMARY")
    print("=" * 80)
    print(pd.DataFrame(summary_rows))
    print(f"\nSaved summary Excel: {writer_path}")
    print(f"Saved summary CSV:   {summary_csv}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BUILD CLEAN CLAHE ROI DATASET")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nConfiguration:")
    print(f"ROI_SIZE: {ROI_SIZE}")
    print(f"MIN_MASK_PIXELS_ROI: {MIN_MASK_PIXELS_ROI}")
    print(f"MAX_DISTANCE_FROM_CENTER: {MAX_DISTANCE_FROM_CENTER}")
    print(f"BORDER_TOLERANCE: {BORDER_TOLERANCE}")
    print(f"USE_TRAIN_JITTER: {USE_TRAIN_JITTER}")
    print(f"TRAIN_JITTER_COUNT: {TRAIN_JITTER_COUNT}")

    ensure_exists(TRAIN_CSV, "Train split CSV")
    ensure_exists(VAL_CSV, "Validation split CSV")
    ensure_exists(TEST_CSV, "Test split CSV")

    split_results = []

    train_all_df, train_clean_df = process_split(
        split_name="train",
        split_csv=TRAIN_CSV,
        use_jitter=USE_TRAIN_JITTER
    )

    split_results.append(("train", train_all_df, train_clean_df))

    val_all_df, val_clean_df = process_split(
        split_name="val",
        split_csv=VAL_CSV,
        use_jitter=False
    )

    split_results.append(("val", val_all_df, val_clean_df))

    test_all_df, test_clean_df = process_split(
        split_name="test",
        split_csv=TEST_CSV,
        use_jitter=False
    )

    split_results.append(("test", test_all_df, test_clean_df))

    save_quality_reports(split_results)

    print("\n" + "=" * 80)
    print("FINAL CLEAN ROI DATASET CREATED")
    print("=" * 80)

    print(f"Train clean CSV: {OUTPUT_DIR / 'train_roi_clahe.csv'}")
    print(f"Val clean CSV:   {OUTPUT_DIR / 'val_roi_clahe.csv'}")
    print(f"Test clean CSV:  {OUTPUT_DIR / 'test_roi_clahe.csv'}")

    print("\nVisual inspection images saved in:")
    print(VISUAL_CHECK_DIR)

    print("\nNext step:")
    print("Update your training script ROI_DATASET_DIR as:")
    print(r'ROI_DATASET_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset_clean"')


if __name__ == "__main__":
    main()

# # scripts/proposed_model/step1_build_roi_clahe_dataset.py

# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import cv2
# from tqdm import tqdm


# # ============================================================
# # CONFIGURATION
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SPLIT_DIR = PROJECT_ROOT / "outputs" / "baseline_clahe_preprocessing" / "dataset_splits"

# OUTPUT_DIR = PROJECT_ROOT / "outputs" / "proposed_roi_clahe_dataset"

# DATA_TYPE = "clahe"          # proposed model uses CLAHE images
# ROI_SIZE = (256, 256)        # recommended for nodule ROI
# ROI_MARGIN = 48              # margin around nodule mask
# POSITIVE_ONLY = True         # stage-2/refinement should use positive slices only

# RANDOM_SEED = 42


# # ============================================================
# # UTILITY FUNCTIONS
# # ============================================================

# def find_split_csv(split_name: str, data_type: str):
#     """
#     Automatically find train/val/test split CSV files.
#     Works for filenames like:
#     clahe_train.csv, train_clahe.csv, clahe_train_split.csv, etc.
#     """
#     patterns = [
#         f"*{data_type}*{split_name}*.csv",
#         f"*{split_name}*{data_type}*.csv",
#         f"{data_type}_{split_name}.csv",
#         f"{split_name}_{data_type}.csv",
#     ]

#     candidates = []
#     for p in patterns:
#         candidates.extend(list(SPLIT_DIR.glob(p)))

#     candidates = [c for c in candidates if "summary" not in c.name.lower()]

#     if len(candidates) == 0:
#         raise FileNotFoundError(
#             f"No CSV found for split={split_name}, data_type={data_type} in:\n{SPLIT_DIR}"
#         )

#     candidates = sorted(list(set(candidates)))
#     print(f"{split_name.upper()} CSV selected: {candidates[0]}")
#     return candidates[0]


# def safe_path_value(row, possible_cols):
#     """
#     Safely extract path from row even if different column names are used.
#     """
#     for col in possible_cols:
#         if col in row.index:
#             value = row[col]
#             if pd.notna(value):
#                 return str(value)
#     raise KeyError(f"None of these columns found: {possible_cols}")


# def load_npy(path):
#     path = str(path)
#     arr = np.load(path).astype(np.float32)

#     if arr.ndim == 3:
#         arr = arr.squeeze()

#     return arr


# def normalize_01(arr):
#     arr = arr.astype(np.float32)

#     if arr.max() > 1.0:
#         arr = arr / 255.0

#     arr = np.clip(arr, 0.0, 1.0)
#     return arr


# def get_mask_bbox(mask, margin=48):
#     """
#     Find bounding box around positive mask.
#     """
#     ys, xs = np.where(mask > 0.5)

#     if len(xs) == 0 or len(ys) == 0:
#         return None

#     h, w = mask.shape

#     x1 = max(0, xs.min() - margin)
#     x2 = min(w, xs.max() + margin + 1)
#     y1 = max(0, ys.min() - margin)
#     y2 = min(h, ys.max() + margin + 1)

#     return x1, y1, x2, y2


# def make_square_bbox(x1, y1, x2, y2, h, w):
#     """
#     Convert bounding box to square box while staying inside image.
#     """
#     bw = x2 - x1
#     bh = y2 - y1
#     side = max(bw, bh)

#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2

#     nx1 = cx - side // 2
#     ny1 = cy - side // 2
#     nx2 = nx1 + side
#     ny2 = ny1 + side

#     if nx1 < 0:
#         nx2 -= nx1
#         nx1 = 0

#     if ny1 < 0:
#         ny2 -= ny1
#         ny1 = 0

#     if nx2 > w:
#         shift = nx2 - w
#         nx1 -= shift
#         nx2 = w

#     if ny2 > h:
#         shift = ny2 - h
#         ny1 -= shift
#         ny2 = h

#     nx1 = max(0, nx1)
#     ny1 = max(0, ny1)
#     nx2 = min(w, nx2)
#     ny2 = min(h, ny2)

#     return nx1, ny1, nx2, ny2


# def crop_and_resize(image, mask, click, bbox, roi_size=(256, 256)):
#     x1, y1, x2, y2 = bbox

#     img_crop = image[y1:y2, x1:x2]
#     mask_crop = mask[y1:y2, x1:x2]
#     click_crop = click[y1:y2, x1:x2]

#     img_resized = cv2.resize(img_crop, roi_size, interpolation=cv2.INTER_LINEAR)
#     mask_resized = cv2.resize(mask_crop, roi_size, interpolation=cv2.INTER_NEAREST)
#     click_resized = cv2.resize(click_crop, roi_size, interpolation=cv2.INTER_LINEAR)

#     img_resized = normalize_01(img_resized)
#     mask_resized = (mask_resized > 0.5).astype(np.float32)
#     click_resized = normalize_01(click_resized)

#     return img_resized, mask_resized, click_resized


# def generate_click_from_mask(mask, sigma=8):
#     """
#     If click map is empty or invalid, generate center-click Gaussian from mask.
#     """
#     h, w = mask.shape
#     click = np.zeros((h, w), dtype=np.float32)

#     ys, xs = np.where(mask > 0.5)

#     if len(xs) == 0:
#         return click

#     cx = int(np.mean(xs))
#     cy = int(np.mean(ys))

#     yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
#     click = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
#     click = click.astype(np.float32)

#     return click


# def prepare_output_dirs(split_name):
#     split_dir = OUTPUT_DIR / split_name
#     image_dir = split_dir / "images"
#     mask_dir = split_dir / "masks"
#     click_dir = split_dir / "clicks"

#     image_dir.mkdir(parents=True, exist_ok=True)
#     mask_dir.mkdir(parents=True, exist_ok=True)
#     click_dir.mkdir(parents=True, exist_ok=True)

#     return image_dir, mask_dir, click_dir


# # ============================================================
# # MAIN PROCESSING FUNCTION
# # ============================================================

# def process_split(split_name):
#     csv_path = find_split_csv(split_name, DATA_TYPE)
#     df = pd.read_csv(csv_path)

#     print("\n" + "=" * 80)
#     print(f"Processing split: {split_name}")
#     print(f"Rows before filtering: {len(df)}")
#     print(f"Columns: {list(df.columns)}")

#     image_dir, mask_dir, click_dir = prepare_output_dirs(split_name)

#     output_rows = []

#     for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):

#         image_path = safe_path_value(
#             row,
#             ["image_path", "clahe_image_path", "baseline_image_path"]
#         )

#         mask_path = safe_path_value(
#             row,
#             ["mask_path", "clahe_mask_path", "baseline_mask_path"]
#         )

#         click_path = safe_path_value(
#             row,
#             ["click_path", "clahe_click_path", "baseline_click_path"]
#         )

#         image = normalize_01(load_npy(image_path))
#         mask = load_npy(mask_path)
#         mask = (mask > 0.5).astype(np.float32)

#         try:
#             click = normalize_01(load_npy(click_path))
#         except Exception:
#             click = generate_click_from_mask(mask)

#         if click.max() <= 0 and mask.sum() > 0:
#             click = generate_click_from_mask(mask)

#         mask_pixel_count = int(mask.sum())

#         if POSITIVE_ONLY and mask_pixel_count == 0:
#             continue

#         bbox = get_mask_bbox(mask, margin=ROI_MARGIN)

#         if bbox is None:
#             continue

#         h, w = mask.shape
#         bbox = make_square_bbox(*bbox, h=h, w=w)

#         roi_img, roi_mask, roi_click = crop_and_resize(
#             image=image,
#             mask=mask,
#             click=click,
#             bbox=bbox,
#             roi_size=ROI_SIZE
#         )

#         patient_id = row["patient_id"] if "patient_id" in row.index else "unknown"
#         slice_index = row["slice_index"] if "slice_index" in row.index else idx

#         file_stem = f"{patient_id}_slice_{int(slice_index):04d}"

#         out_img_path = image_dir / f"{file_stem}.npy"
#         out_mask_path = mask_dir / f"{file_stem}.npy"
#         out_click_path = click_dir / f"{file_stem}.npy"

#         np.save(out_img_path, roi_img.astype(np.float32))
#         np.save(out_mask_path, roi_mask.astype(np.float32))
#         np.save(out_click_path, roi_click.astype(np.float32))

#         output_rows.append({
#             "patient_id": patient_id,
#             "slice_index": slice_index,
#             "image_path": str(out_img_path),
#             "mask_path": str(out_mask_path),
#             "click_path": str(out_click_path),
#             "original_image_path": image_path,
#             "original_mask_path": mask_path,
#             "original_click_path": click_path,
#             "mask_pixel_count_original": mask_pixel_count,
#             "mask_pixel_count_roi": int(roi_mask.sum()),
#             "bbox_x1": bbox[0],
#             "bbox_y1": bbox[1],
#             "bbox_x2": bbox[2],
#             "bbox_y2": bbox[3],
#         })

#     out_df = pd.DataFrame(output_rows)
#     out_csv = OUTPUT_DIR / f"{split_name}_roi_clahe.csv"
#     out_xlsx = OUTPUT_DIR / f"{split_name}_roi_clahe.xlsx"

#     out_df.to_csv(out_csv, index=False)
#     out_df.to_excel(out_xlsx, index=False)

#     print(f"Rows after ROI creation: {len(out_df)}")
#     print(f"Saved CSV: {out_csv}")
#     print(f"Saved XLSX: {out_xlsx}")

#     return out_df


# def main():
#     print("Project root:")
#     print(PROJECT_ROOT)

#     print("\nInput split directory:")
#     print(SPLIT_DIR)

#     print("\nOutput ROI dataset directory:")
#     print(OUTPUT_DIR)

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     train_df = process_split("train")
#     val_df = process_split("val")
#     test_df = process_split("test")

#     summary = pd.DataFrame([
#         {
#             "split": "train",
#             "samples": len(train_df),
#             "patients": train_df["patient_id"].nunique() if len(train_df) > 0 else 0,
#             "positive_samples": len(train_df),
#         },
#         {
#             "split": "val",
#             "samples": len(val_df),
#             "patients": val_df["patient_id"].nunique() if len(val_df) > 0 else 0,
#             "positive_samples": len(val_df),
#         },
#         {
#             "split": "test",
#             "samples": len(test_df),
#             "patients": test_df["patient_id"].nunique() if len(test_df) > 0 else 0,
#             "positive_samples": len(test_df),
#         },
#     ])

#     summary_csv = OUTPUT_DIR / "roi_dataset_summary.csv"
#     summary_xlsx = OUTPUT_DIR / "roi_dataset_summary.xlsx"

#     summary.to_csv(summary_csv, index=False)
#     summary.to_excel(summary_xlsx, index=False)

#     print("\n" + "=" * 80)
#     print("ROI DATASET SUMMARY")
#     print("=" * 80)
#     print(summary)

#     print("\nDone.")
#     print(f"All ROI dataset files saved in:\n{OUTPUT_DIR}")


# if __name__ == "__main__":
#     main()
