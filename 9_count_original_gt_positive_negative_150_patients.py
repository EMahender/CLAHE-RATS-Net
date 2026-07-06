# =============================================================================
# STEP 16: Count Original CT Images, Ground Truth Masks, Positive and Negative Images
# =============================================================================

from pathlib import Path
from collections import Counter, defaultdict
import os

import numpy as np
import pandas as pd
import pydicom


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"ACTUAL_PATH")

ORIGINAL_DICOM_ROOT = PROJECT_ROOT / "LIDC" / "LIDC-IDRI"

GT_MASK_ROOT = PROJECT_ROOT / "outputs" / "baseline_clahe_preprocessing"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_distribution_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTWISE_CSV = OUTPUT_DIR / "patientwise_original_gt_positive_negative_150.csv"

SUMMARY_CSV = OUTPUT_DIR / "overall_original_gt_positive_negative_150_summary.csv"


# =============================================================================
# 2. SETTINGS
# =============================================================================

NUM_PATIENTS = 150

# Keep True to select first 150 patient folders in sorted order.
# If False, all available patients will be processed.
USE_FIRST_150_PATIENTS = True

# Ground truth mask folder names generated during preprocessing.
GT_MASK_FOLDER_CANDIDATES = [
    "baseline_masks_npy",
    "masks_npy",
    "clahe_masks_npy",
    "baseline_masks",
    "masks",
    "clahe_masks",
]


# =============================================================================
# 3. DICOM FUNCTIONS
# =============================================================================

def is_dicom_file(path: Path) -> bool:
    """
    Check whether file is a valid DICOM file.
    LIDC-IDRI files may have .dcm or no extension.
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return hasattr(ds, "SOPInstanceUID")
    except Exception:
        return False


def get_dicom_sort_value(ds):
    """
    Sorting priority:
      1. ImagePositionPatient[2]
      2. SliceLocation
      3. InstanceNumber
    """
    try:
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
    except Exception:
        pass

    try:
        if hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
    except Exception:
        pass

    try:
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
    except Exception:
        pass

    return 0.0


def collect_patient_dicom_records(patient_id: str):
    """
    Collect all valid CT DICOM records for a patient.
    """
    patient_dir = ORIGINAL_DICOM_ROOT / patient_id

    records = []

    if not patient_dir.exists():
        return records

    for root, _, filenames in os.walk(patient_dir):
        for fname in filenames:
            fpath = Path(root) / fname

            try:
                if not is_dicom_file(fpath):
                    continue

                ds = pydicom.dcmread(
                    str(fpath),
                    stop_before_pixels=True,
                    force=True
                )

                modality = str(getattr(ds, "Modality", "UNKNOWN"))

                if modality != "CT":
                    continue

                rows = int(getattr(ds, "Rows", -1))
                cols = int(getattr(ds, "Columns", -1))

                if rows <= 0 or cols <= 0:
                    continue

                records.append({
                    "dicom_path": str(fpath),
                    "series_uid": str(getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")),
                    "shape": (rows, cols),
                    "rows": rows,
                    "cols": cols,
                    "instance_number": int(getattr(ds, "InstanceNumber", 0)),
                    "sort_value": get_dicom_sort_value(ds),
                })

            except Exception:
                continue

    return records


def select_dominant_ct_series(records):
    """
    Select dominant CT series to avoid scout/localizer or mixed-shape images.
    """
    if len(records) == 0:
        return []

    series_groups = defaultdict(list)

    for rec in records:
        series_groups[rec["series_uid"]].append(rec)

    series_rank = []

    for series_uid, recs in series_groups.items():
        shape_counts = Counter([r["shape"] for r in recs])
        dominant_shape, dominant_shape_count = shape_counts.most_common(1)[0]

        series_rank.append({
            "series_uid": series_uid,
            "num_slices": len(recs),
            "dominant_shape": dominant_shape,
            "dominant_shape_count": dominant_shape_count,
        })

    series_rank = sorted(
        series_rank,
        key=lambda x: (x["dominant_shape_count"], x["num_slices"]),
        reverse=True
    )

    best_series_uid = series_rank[0]["series_uid"]
    best_shape = series_rank[0]["dominant_shape"]

    selected = [
        r for r in series_groups[best_series_uid]
        if r["shape"] == best_shape
    ]

    selected = sorted(selected, key=lambda x: x["sort_value"])

    return selected


# =============================================================================
# 4. GROUND TRUTH MASK FUNCTIONS
# =============================================================================

def find_gt_mask_folder(patient_id: str):
    """
    Find GT mask folder for one patient.
    """
    patient_gt_dir = GT_MASK_ROOT / patient_id

    if not patient_gt_dir.exists():
        return None

    for folder_name in GT_MASK_FOLDER_CANDIDATES:
        candidate = patient_gt_dir / folder_name

        if candidate.exists() and candidate.is_dir():
            files = list(candidate.glob("*.npy")) + list(candidate.glob("*.png")) + list(candidate.glob("*.jpg"))
            if len(files) > 0:
                return candidate

    # Fallback: search any folder containing mask files
    for root, dirs, files in os.walk(patient_gt_dir):
        root_path = Path(root)

        mask_files = [
            f for f in files
            if f.lower().endswith((".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        if len(mask_files) > 0 and "mask" in root_path.name.lower():
            return root_path

    return None


def list_mask_files(mask_dir: Path):
    """
    List all mask files.
    """
    if mask_dir is None or not mask_dir.exists():
        return []

    files = []

    for ext in ["*.npy", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        files.extend(list(mask_dir.glob(ext)))

    files = sorted(files)

    return files


def read_mask(mask_path: Path):
    """
    Read mask from .npy or image file.
    """
    suffix = mask_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(str(mask_path))

        if arr.ndim > 2:
            arr = np.squeeze(arr)

        arr = arr.astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / 255.0

        mask = (arr > 0.5).astype(np.uint8)

        return mask

    # Image mask
    try:
        import cv2

        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        img = img.astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        mask = (img > 0.5).astype(np.uint8)

        return mask

    except Exception:
        return None


def count_gt_positive_negative(mask_files):
    """
    Count positive and negative masks based on pixel values.
    Positive = mask contains at least one foreground pixel.
    Negative / empty = mask has no foreground pixel.
    """
    total_gt = 0
    positive_gt = 0
    negative_gt = 0
    unreadable_gt = 0

    total_positive_pixels = 0

    for mask_path in mask_files:
        mask = read_mask(mask_path)

        if mask is None:
            unreadable_gt += 1
            continue

        total_gt += 1

        pixel_count = int(mask.sum())
        total_positive_pixels += pixel_count

        if pixel_count > 0:
            positive_gt += 1
        else:
            negative_gt += 1

    return {
        "total_gt_masks": total_gt,
        "positive_gt_masks": positive_gt,
        "negative_empty_gt_masks": negative_gt,
        "unreadable_gt_masks": unreadable_gt,
        "total_positive_mask_pixels": total_positive_pixels,
    }


# =============================================================================
# 5. MAIN PROCESSING
# =============================================================================

def get_patient_ids():
    """
    Get selected patient IDs.
    """
    if not ORIGINAL_DICOM_ROOT.exists():
        raise FileNotFoundError(f"Original DICOM root not found: {ORIGINAL_DICOM_ROOT}")

    patient_dirs = sorted([
        p for p in ORIGINAL_DICOM_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("LIDC-IDRI")
    ])

    if USE_FIRST_150_PATIENTS:
        patient_dirs = patient_dirs[:NUM_PATIENTS]

    patient_ids = [p.name for p in patient_dirs]

    return patient_ids


def process_patient(patient_id: str, serial_no: int):
    """
    Process one patient and return patient-wise summary.
    """

    # Original CT DICOM
    dicom_records_all = collect_patient_dicom_records(patient_id)
    dominant_series_records = select_dominant_ct_series(dicom_records_all)

    total_original_dicom_all = len(dicom_records_all)
    total_original_dicom_dominant = len(dominant_series_records)

    has_original_ct = total_original_dicom_dominant > 0

    # GT masks
    gt_mask_folder = find_gt_mask_folder(patient_id)
    mask_files = list_mask_files(gt_mask_folder) if gt_mask_folder else []

    gt_counts = count_gt_positive_negative(mask_files)

    total_gt_masks = gt_counts["total_gt_masks"]
    positive_gt_masks = gt_counts["positive_gt_masks"]
    negative_gt_masks = gt_counts["negative_empty_gt_masks"]
    unreadable_gt_masks = gt_counts["unreadable_gt_masks"]

    has_gt_masks = total_gt_masks > 0
    has_both_original_and_gt = has_original_ct and has_gt_masks

    # Match status
    if has_original_ct and has_gt_masks:
        if total_original_dicom_dominant == total_gt_masks:
            match_status = "MATCHED_ORIGINAL_AND_GT_COUNT"
        else:
            match_status = "COUNT_MISMATCH_BETWEEN_ORIGINAL_AND_GT"
    elif has_original_ct and not has_gt_masks:
        match_status = "ORIGINAL_ONLY_NO_GT"
    elif not has_original_ct and has_gt_masks:
        match_status = "GT_ONLY_NO_ORIGINAL"
    else:
        match_status = "NO_ORIGINAL_NO_GT"

    # Negative images with respect to GT masks
    # If GT masks are generated for all CT slices:
    #   negative images = empty masks.
    # If GT masks are missing for some CT slices:
    #   unpaired original slices are also reported separately.
    unpaired_original_images = max(total_original_dicom_dominant - total_gt_masks, 0)

    row = {
        "S.No": serial_no,
        "Patient ID": patient_id,

        "Has Original CT": has_original_ct,
        "Has GT Mask": has_gt_masks,
        "Has Both Original and GT": has_both_original_and_gt,

        "Total Original CT Images - All CT DICOM": total_original_dicom_all,
        "Total Original CT Images - Dominant Series": total_original_dicom_dominant,

        "Total GT Mask Images": total_gt_masks,
        "Positive Images / Masks": positive_gt_masks,
        "Negative / Empty Images / Masks": negative_gt_masks,
        "Unreadable GT Masks": unreadable_gt_masks,

        "Unpaired Original Images without GT": unpaired_original_images,

        "Total Positive Mask Pixels": gt_counts["total_positive_mask_pixels"],

        "GT Mask Folder": str(gt_mask_folder) if gt_mask_folder else "NOT_FOUND",
        "Match Status": match_status,
    }

    return row


def create_overall_summary(patientwise_df: pd.DataFrame):
    """
    Create final overall summary.
    """

    total_patients_checked = len(patientwise_df)

    patients_with_original = int(patientwise_df["Has Original CT"].sum())
    patients_with_gt = int(patientwise_df["Has GT Mask"].sum())
    patients_with_both = int(patientwise_df["Has Both Original and GT"].sum())

    total_original_all = int(patientwise_df["Total Original CT Images - All CT DICOM"].sum())
    total_original_dominant = int(patientwise_df["Total Original CT Images - Dominant Series"].sum())

    total_gt = int(patientwise_df["Total GT Mask Images"].sum())
    total_positive = int(patientwise_df["Positive Images / Masks"].sum())
    total_negative = int(patientwise_df["Negative / Empty Images / Masks"].sum())
    total_unreadable = int(patientwise_df["Unreadable GT Masks"].sum())
    total_unpaired_original = int(patientwise_df["Unpaired Original Images without GT"].sum())

    summary_rows = [
        {
            "Item": "Total patients checked",
            "Value": total_patients_checked,
        },
        {
            "Item": "Patients with original CT images",
            "Value": patients_with_original,
        },
        {
            "Item": "Patients with GT masks",
            "Value": patients_with_gt,
        },
        {
            "Item": "Patients containing both original CT and GT masks",
            "Value": patients_with_both,
        },
        {
            "Item": "Total original CT images - all CT DICOM files",
            "Value": total_original_all,
        },
        {
            "Item": "Total original CT images - dominant CT series",
            "Value": total_original_dominant,
        },
        {
            "Item": "Total GT mask images",
            "Value": total_gt,
        },
        {
            "Item": "Total positive images / masks",
            "Value": total_positive,
        },
        {
            "Item": "Total negative / empty images / masks",
            "Value": total_negative,
        },
        {
            "Item": "Total unreadable GT masks",
            "Value": total_unreadable,
        },
        {
            "Item": "Total unpaired original images without GT",
            "Value": total_unpaired_original,
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("COUNT ORIGINAL CT, GROUND TRUTH, POSITIVE AND NEGATIVE IMAGES FOR 150 PATIENTS")
    print("=" * 90)

    print(f"Original DICOM root: {ORIGINAL_DICOM_ROOT}")
    print(f"GT mask root       : {GT_MASK_ROOT}")
    print(f"Output directory   : {OUTPUT_DIR}")

    patient_ids = get_patient_ids()

    print(f"\nPatients selected: {len(patient_ids)}")

    all_rows = []

    for idx, patient_id in enumerate(patient_ids, start=1):
        print(f"[{idx:03d}/{len(patient_ids):03d}] Processing {patient_id}")

        row = process_patient(patient_id, idx)
        all_rows.append(row)

    patientwise_df = pd.DataFrame(all_rows)

    summary_df = create_overall_summary(patientwise_df)

    # Save files
    patientwise_df.to_csv(PATIENTWISE_CSV, index=False)
    patientwise_df.to_excel(PATIENTWISE_XLSX, index=False)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    summary_df.to_excel(SUMMARY_XLSX, index=False)

    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    print(summary_df)

    print("\n" + "=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(f"Patient-wise CSV : {PATIENTWISE_CSV}")
    print(f"Patient-wise XLSX: {PATIENTWISE_XLSX}")
    print(f"Summary CSV      : {SUMMARY_CSV}")
    print(f"Summary XLSX     : {SUMMARY_XLSX}")

    print("\nDone.")


if __name__ == "__main__":
    main()
