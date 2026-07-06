import os
import random
from glob import glob

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = r"ACTUAL_PATH"
PREPROCESS_DIR = os.path.join(PROJECT_ROOT, "outputs", "ACTUAL_PATH")
OUTPUT_SPLIT_DIR = os.path.join(PREPROCESS_DIR, "dataset_splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
CHECK_FILE_EXISTENCE = True


# ============================================================
# HELPERS
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_patient_summary_csv(base_dir):
    """
    Find the patient-level summary CSV, usually:
    all_patients_preprocessing_summary.csv
    """
    preferred = os.path.join(base_dir, "all_patients_preprocessing_summary.csv")
    if os.path.exists(preferred):
        return preferred

    csv_files = glob(os.path.join(base_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base_dir}")

    for f in csv_files:
        if "all_patients" in os.path.basename(f).lower() and "summary" in os.path.basename(f).lower():
            return f

    raise FileNotFoundError(
        "Could not find patient-level summary CSV like "
        "'all_patients_preprocessing_summary.csv'"
    )


def load_master_slice_dataframe(preprocess_dir):
    """
    Build master slice-level dataframe by reading the patient summary CSV
    and then loading each patient's summary_csv file.
    """
    patient_summary_csv = find_patient_summary_csv(preprocess_dir)

    print(f"\nReading patient-level summary CSV:\n{patient_summary_csv}")
    patient_df = pd.read_csv(patient_summary_csv)

    print(f"Loaded patient summary rows: {len(patient_df)}")
    print(f"Columns: {list(patient_df.columns)}\n")

    if "summary_csv" not in patient_df.columns:
        raise ValueError(
            "The patient summary CSV does not contain 'summary_csv' column.\n"
            "So the script cannot locate per-patient slice-level metadata files."
        )

    all_parts = []
    missing_files = []
    empty_files = []

    for _, row in patient_df.iterrows():
        csv_path = row["summary_csv"]

        if not isinstance(csv_path, str) or not csv_path.strip():
            missing_files.append(csv_path)
            continue

        if not os.path.exists(csv_path):
            missing_files.append(csv_path)
            continue

        try:
            df_part = pd.read_csv(csv_path)
            if len(df_part) == 0:
                empty_files.append(csv_path)
                continue

            all_parts.append(df_part)

        except Exception as e:
            print(f"Warning: failed to read {csv_path}\nReason: {e}")

    if missing_files:
        print(f"Warning: {len(missing_files)} patient summary CSV files are missing.")

    if empty_files:
        print(f"Warning: {len(empty_files)} patient summary CSV files are empty.")

    if not all_parts:
        raise ValueError(
            "No valid per-patient slice-level CSV files were loaded.\n"
            "Please check whether preprocessing actually generated patient summary CSV files."
        )

    master_df = pd.concat(all_parts, ignore_index=True)
    print(f"Combined slice-level rows: {len(master_df)}")
    print(f"Combined columns: {list(master_df.columns)}\n")
    return master_df


def validate_required_columns(df):
    required_cols = [
        "patient_id",
        "slice_index",
        "mask_pixel_count",
        "mask_status",
        "baseline_image_path",
        "baseline_mask_path",
        "baseline_click_path",
        "clahe_image_path",
        "clahe_mask_path",
        "clahe_click_path",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the slice-level metadata:\n"
            + "\n".join(missing)
        )


def add_label_columns(df):
    df = df.copy()
    df["mask_pixel_count"] = pd.to_numeric(df["mask_pixel_count"], errors="coerce").fillna(0)
    df["is_positive"] = (df["mask_pixel_count"] > 0).astype(int)
    df["class_name"] = df["is_positive"].map({0: "empty", 1: "positive"})
    return df


def filter_existing_files(df):
    df = df.copy()

    def exists_all(row):
        needed = [
            row["baseline_image_path"],
            row["baseline_mask_path"],
            row["baseline_click_path"],
            row["clahe_image_path"],
            row["clahe_mask_path"],
            row["clahe_click_path"],
        ]
        return all(isinstance(p, str) and os.path.exists(p) for p in needed)

    mask = df.apply(exists_all, axis=1)
    removed = (~mask).sum()
    if removed > 0:
        print(f"Rows removed because files were missing: {removed}")
    return df[mask].reset_index(drop=True)


def patient_level_summary(df):
    summary = (
        df.groupby("patient_id")
        .agg(
            total_slices=("slice_index", "count"),
            positive_slices=("is_positive", "sum"),
        )
        .reset_index()
    )
    summary["has_positive"] = (summary["positive_slices"] > 0).astype(int)
    return summary


def split_list(lst, train_ratio, val_ratio):
    n = len(lst)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))

    if n_train > n:
        n_train = n
    if n_train + n_val > n:
        n_val = n - n_train

    train = lst[:n_train]
    val = lst[n_train:n_train + n_val]
    test = lst[n_train + n_val:]
    return train, val, test


def split_patients(summary_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    pos_patients = summary_df[summary_df["has_positive"] == 1]["patient_id"].tolist()
    neg_patients = summary_df[summary_df["has_positive"] == 0]["patient_id"].tolist()

    rng = random.Random(seed)
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    pos_train, pos_val, pos_test = split_list(pos_patients, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_list(neg_patients, train_ratio, val_ratio)

    train_patients = pos_train + neg_train
    val_patients = pos_val + neg_val
    test_patients = pos_test + neg_test

    rng.shuffle(train_patients)
    rng.shuffle(val_patients)
    rng.shuffle(test_patients)

    return train_patients, val_patients, test_patients


def assign_split(df, train_patients, val_patients, test_patients):
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    df = df.copy()

    def get_split(pid):
        if pid in train_set:
            return "train"
        elif pid in val_set:
            return "val"
        elif pid in test_set:
            return "test"
        return "unknown"

    df["split"] = df["patient_id"].apply(get_split)

    unknown = (df["split"] == "unknown").sum()
    if unknown > 0:
        raise ValueError(f"{unknown} rows got unknown split assignment.")
    return df


def build_variant_df(df, variant="baseline"):
    out = df.copy()

    if variant == "baseline":
        out["image_path"] = out["baseline_image_path"]
        out["mask_path"] = out["baseline_mask_path"]
        out["click_path"] = out["baseline_click_path"]
    elif variant == "clahe":
        out["image_path"] = out["clahe_image_path"]
        out["mask_path"] = out["clahe_mask_path"]
        out["click_path"] = out["clahe_click_path"]
    else:
        raise ValueError("variant must be 'baseline' or 'clahe'")

    out["variant"] = variant

    cols = [
        "patient_id",
        "slice_index",
        "split",
        "variant",
        "lung_side",
        "mask_pixel_count",
        "mask_status",
        "is_positive",
        "class_name",
        "image_path",
        "mask_path",
        "click_path",
        "baseline_image_path",
        "baseline_mask_path",
        "baseline_click_path",
        "clahe_image_path",
        "clahe_mask_path",
        "clahe_click_path",
        "source_slice_path",
        "xml_path",
    ]

    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def save_split_files(df, prefix, out_dir):
    ensure_dir(out_dir)

    all_csv = os.path.join(out_dir, f"{prefix}_all.csv")
    all_xlsx = os.path.join(out_dir, f"{prefix}_all.xlsx")
    df.to_csv(all_csv, index=False)
    df.to_excel(all_xlsx, index=False)

    for split_name in ["train", "val", "test"]:
        part = df[df["split"] == split_name].reset_index(drop=True)
        csv_path = os.path.join(out_dir, f"{prefix}_{split_name}.csv")
        xlsx_path = os.path.join(out_dir, f"{prefix}_{split_name}.xlsx")
        part.to_csv(csv_path, index=False)
        part.to_excel(xlsx_path, index=False)

    print(f"Saved {prefix} split files in:\n{out_dir}\n")


def save_patient_split_summary(train_patients, val_patients, test_patients, out_dir):
    rows = []
    for p in train_patients:
        rows.append({"patient_id": p, "split": "train"})
    for p in val_patients:
        rows.append({"patient_id": p, "split": "val"})
    for p in test_patients:
        rows.append({"patient_id": p, "split": "test"})

    df = pd.DataFrame(rows).sort_values(["split", "patient_id"]).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "patient_split_summary.csv")
    xlsx_path = os.path.join(out_dir, "patient_split_summary.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)


def print_summary(df, name):
    print("=" * 70)
    print(name)
    print("=" * 70)

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        n_patients = sub["patient_id"].nunique()
        n_slices = len(sub)
        n_pos = int(sub["is_positive"].sum())
        n_neg = int((sub["is_positive"] == 0).sum())

        print(
            f"{split_name.upper():<5} | "
            f"Patients: {n_patients:<4} | "
            f"Slices: {n_slices:<6} | "
            f"Positive: {n_pos:<6} | "
            f"Empty: {n_neg:<6}"
        )
    print()


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_SPLIT_DIR)

    # Step 1: build slice-level master dataframe
    df = load_master_slice_dataframe(PREPROCESS_DIR)

    # Step 2: validate columns
    validate_required_columns(df)

    # Step 3: labels
    df = add_label_columns(df)

    # Step 4: optional file existence check
    if CHECK_FILE_EXISTENCE:
        df = filter_existing_files(df)

    print(f"Rows after validation: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}\n")

    # Step 5: patient summary
    patient_summary = patient_level_summary(df)

    # Step 6: patient split
    train_patients, val_patients, test_patients = split_patients(
        patient_summary,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    print("Patient split:")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients:   {len(val_patients)}")
    print(f"Test patients:  {len(test_patients)}\n")

    # Step 7: assign split
    df = assign_split(df, train_patients, val_patients, test_patients)

    # Step 8: create variant-specific dataframes
    baseline_df = build_variant_df(df, variant="baseline")
    clahe_df = build_variant_df(df, variant="clahe")

    # Step 9: save files
    save_split_files(baseline_df, "baseline", OUTPUT_SPLIT_DIR)
    save_split_files(clahe_df, "clahe", OUTPUT_SPLIT_DIR)
    save_patient_split_summary(train_patients, val_patients, test_patients, OUTPUT_SPLIT_DIR)

    # Step 10: print summary
    print_summary(baseline_df, "BASELINE SPLIT SUMMARY")
    print_summary(clahe_df, "CLAHE SPLIT SUMMARY")

    print("Done.")
    print(f"All split files saved in:\n{OUTPUT_SPLIT_DIR}")


if __name__ == "__main__":
    main()
