# =============================================================================
# STEP 8: Statistical Significance Testing
# =============================================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"ACTUAL_PATH")

BASE_DIR = PROJECT_ROOT / "outputs" / "baseline_proposed_comparison"

OUTPUT_DIR = BASE_DIR / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = BASE_DIR / "final_model_comparison_master.csv"

PROPOSED_MODEL = "CLAHE_RATS_Net"

BASELINE_MODELS = [
    "UNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "ResUNet",
    "TransUNetLite"
]

ALL_MODELS = BASELINE_MODELS + [PROPOSED_MODEL]

METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity"
]


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def significance_marker(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def read_master_comparison():
    """
    Reads the master comparison file to identify each model's best validation threshold.
    """

    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    elif MASTER_XLSX.exists():
        df = pd.read_excel(MASTER_XLSX, sheet_name="Main_Comparison")
    else:
        raise FileNotFoundError(
            f"Master comparison file not found:\n{MASTER_CSV}\nor\n{MASTER_XLSX}"
        )

    return df


def get_best_thresholds(master_df):
    """
    Extracts best validation threshold for each model.
    """

    threshold_map = {}

    for _, row in master_df.iterrows():
        model = row.get("Model", row.get("model", None))

        if model is None:
            continue

        if "Best Val Threshold" in row:
            threshold = row["Best Val Threshold"]
        elif "best_val_threshold" in row:
            threshold = row["best_val_threshold"]
        else:
            threshold = None

        if model in ALL_MODELS and pd.notna(threshold):
            threshold_map[model] = float(threshold)

    return threshold_map


def read_test_samplewise_metrics(model_name, best_threshold):
    """
    Reads test_samplewise_metrics.csv for a model and filters by its best validation threshold.
    """

    csv_path = BASE_DIR / model_name / "logs" / "test_samplewise_metrics.csv"
    xlsx_path = BASE_DIR / model_name / "logs" / "test_samplewise_metrics.xlsx"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(
            f"Sample-wise test metrics not found for {model_name}:\n{csv_path}\n{xlsx_path}"
        )

    if "threshold" not in df.columns:
        raise ValueError(f"'threshold' column missing in {model_name} sample-wise file.")

    if "file_name" not in df.columns:
        raise ValueError(f"'file_name' column missing in {model_name} sample-wise file.")

    # Floating point tolerance for threshold matching
    df_thr = df[np.isclose(df["threshold"].astype(float), float(best_threshold), atol=1e-6)].copy()

    if len(df_thr) == 0:
        print(f"Warning: No rows found for {model_name} at threshold {best_threshold}.")
        print("Using threshold with maximum mean Dice from sample-wise file.")

        threshold_means = df.groupby("threshold")["dice"].mean().reset_index()
        best_threshold = float(threshold_means.loc[threshold_means["dice"].idxmax(), "threshold"])
        df_thr = df[np.isclose(df["threshold"].astype(float), best_threshold, atol=1e-6)].copy()

    keep_cols = ["file_name", "threshold"] + [m for m in METRICS if m in df_thr.columns]

    if "size_group" in df_thr.columns:
        keep_cols.append("size_group")

    if "mask_area" in df_thr.columns:
        keep_cols.append("mask_area")

    df_thr = df_thr[keep_cols].copy()
    df_thr["model"] = model_name

    return df_thr, best_threshold


def build_aligned_samplewise_table(threshold_map):
    """
    Builds one table where rows are test images/ROIs and columns are metric_model.
    """

    aligned_df = None
    model_thresholds_used = {}

    for model_name in ALL_MODELS:
        if model_name not in threshold_map:
            print(f"Warning: Best validation threshold not found for {model_name}. Using 0.50.")
            best_threshold = 0.50
        else:
            best_threshold = threshold_map[model_name]

        df_model, used_threshold = read_test_samplewise_metrics(model_name, best_threshold)
        model_thresholds_used[model_name] = used_threshold

        rename_dict = {}

        for metric in METRICS:
            if metric in df_model.columns:
                rename_dict[metric] = f"{metric}_{model_name}"

        df_model = df_model.rename(columns=rename_dict)

        # Keep metadata only once
        metric_cols = [f"{m}_{model_name}" for m in METRICS if f"{m}_{model_name}" in df_model.columns]

        if aligned_df is None:
            base_cols = ["file_name"]

            if "size_group" in df_model.columns:
                base_cols.append("size_group")

            if "mask_area" in df_model.columns:
                base_cols.append("mask_area")

            aligned_df = df_model[base_cols + metric_cols].copy()
        else:
            aligned_df = aligned_df.merge(
                df_model[["file_name"] + metric_cols],
                on="file_name",
                how="inner"
            )

    return aligned_df, model_thresholds_used


def paired_statistics(proposed_values, baseline_values):
    """
    Computes paired t-test, Wilcoxon signed-rank test, Cohen's dz, and 95% CI.
    """

    proposed_values = np.asarray(proposed_values, dtype=np.float64)
    baseline_values = np.asarray(baseline_values, dtype=np.float64)

    valid_mask = np.isfinite(proposed_values) & np.isfinite(baseline_values)

    proposed_values = proposed_values[valid_mask]
    baseline_values = baseline_values[valid_mask]

    diff = proposed_values - baseline_values
    n = len(diff)

    if n < 2:
        return {
            "n": n,
            "mean_proposed": np.nan,
            "mean_baseline": np.nan,
            "mean_difference": np.nan,
            "std_difference": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "cohens_dz": np.nan,
            "shapiro_p": np.nan,
            "paired_t_stat": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan
        }

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # 95% CI for paired mean difference
    if std_diff > 0:
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci95_lower = mean_diff - t_crit * se
        ci95_upper = mean_diff + t_crit * se
        cohens_dz = mean_diff / std_diff
    else:
        ci95_lower = mean_diff
        ci95_upper = mean_diff
        cohens_dz = np.nan

    # Normality test for differences
    if 3 <= n <= 5000:
        try:
            shapiro_p = stats.shapiro(diff).pvalue
        except Exception:
            shapiro_p = np.nan
    else:
        shapiro_p = np.nan

    # Paired t-test
    try:
        paired_t = stats.ttest_rel(proposed_values, baseline_values, nan_policy="omit")
        paired_t_stat = paired_t.statistic
        paired_t_p = paired_t.pvalue
    except Exception:
        paired_t_stat = np.nan
        paired_t_p = np.nan

    # Wilcoxon signed-rank test
    try:
        if np.allclose(diff, 0):
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        else:
            wilcoxon = stats.wilcoxon(
                proposed_values,
                baseline_values,
                zero_method="wilcox",
                alternative="two-sided"
            )
            wilcoxon_stat = wilcoxon.statistic
            wilcoxon_p = wilcoxon.pvalue
    except Exception:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

    return {
        "n": n,
        "mean_proposed": np.mean(proposed_values),
        "mean_baseline": np.mean(baseline_values),
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "cohens_dz": cohens_dz,
        "shapiro_p": shapiro_p,
        "paired_t_stat": paired_t_stat,
        "paired_t_p": paired_t_p,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p
    }


def run_statistical_tests(aligned_df):
    """
    Runs paired t-test and Wilcoxon test for Proposed vs each baseline.
    """

    results = []

    for baseline_model in BASELINE_MODELS:
        for metric in METRICS:
            proposed_col = f"{metric}_{PROPOSED_MODEL}"
            baseline_col = f"{metric}_{baseline_model}"

            if proposed_col not in aligned_df.columns:
                print(f"Warning: Missing proposed column: {proposed_col}")
                continue

            if baseline_col not in aligned_df.columns:
                print(f"Warning: Missing baseline column: {baseline_col}")
                continue

            stats_result = paired_statistics(
                proposed_values=aligned_df[proposed_col].values,
                baseline_values=aligned_df[baseline_col].values
            )

            row = {
                "comparison": f"{PROPOSED_MODEL} vs {baseline_model}",
                "proposed_model": PROPOSED_MODEL,
                "baseline_model": baseline_model,
                "metric": metric
            }

            row.update(stats_result)

            row["paired_t_significance"] = significance_marker(row["paired_t_p"])
            row["wilcoxon_significance"] = significance_marker(row["wilcoxon_p"])

            if pd.notna(row["mean_difference"]):
                if row["mean_difference"] > 0:
                    row["direction"] = "Proposed higher"
                elif row["mean_difference"] < 0:
                    row["direction"] = "Baseline higher"
                else:
                    row["direction"] = "No difference"
            else:
                row["direction"] = "NA"

            results.append(row)

    return pd.DataFrame(results)


def descriptive_statistics(aligned_df):
    """
    Creates descriptive mean/std table for all models and metrics.
    """

    rows = []

    for model_name in ALL_MODELS:
        row = {"model": model_name}

        for metric in METRICS:
            col = f"{metric}_{model_name}"

            if col in aligned_df.columns:
                row[f"{metric}_mean"] = aligned_df[col].mean()
                row[f"{metric}_std"] = aligned_df[col].std()
                row[f"{metric}_median"] = aligned_df[col].median()

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)

    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Proposed model: {PROPOSED_MODEL}")
    print(f"Baseline models: {BASELINE_MODELS}")
    print()

    master_df = read_master_comparison()
    threshold_map = get_best_thresholds(master_df)

    print("Best validation thresholds from master comparison:")
    for model_name in ALL_MODELS:
        print(f"  {model_name}: {threshold_map.get(model_name, 'Not found')}")

    aligned_df, thresholds_used = build_aligned_samplewise_table(threshold_map)

    print()
    print("Thresholds used for sample-wise statistical testing:")
    for model_name, threshold in thresholds_used.items():
        print(f"  {model_name}: {threshold}")

    print()
    print(f"Aligned sample-wise rows: {len(aligned_df)}")

    if len(aligned_df) == 0:
        raise RuntimeError("No aligned sample-wise rows found. Check file_name matching across models.")

    stats_df = run_statistical_tests(aligned_df)
    desc_df = descriptive_statistics(aligned_df)

    # Save outputs
    aligned_csv = OUTPUT_DIR / "aligned_samplewise_metrics.csv"
    stats_csv = OUTPUT_DIR / "statistical_test_results.csv"
    desc_csv = OUTPUT_DIR / "descriptive_statistics.csv"
    excel_path = OUTPUT_DIR / "statistical_test_results.xlsx"

    aligned_df.to_csv(aligned_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    desc_df.to_csv(desc_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Paired_T_and_Wilcoxon", index=False)
        desc_df.to_excel(writer, sheet_name="Descriptive_Stats", index=False)
        aligned_df.to_excel(writer, sheet_name="Aligned_Samplewise", index=False)

        # Formatting
        workbook = writer.book

        for sheet_name in workbook.sheetnames:
            ws = workbook[sheet_name]
            ws.freeze_panes = "A2"

            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            for col_cells in ws.columns:
                max_length = 0
                col_letter = col_cells[0].column_letter

                for cell in col_cells:
                    if cell.value is not None:
                        max_length = max(max_length, len(str(cell.value)))

                ws.column_dimensions[col_letter].width = min(max(max_length + 2, 12), 35)

            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
                cell.alignment = cell.alignment.copy(horizontal="center")

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"

    print()
    print("=" * 80)
    print("STATISTICAL TESTING COMPLETED")
    print("=" * 80)

    print(f"Aligned sample-wise metrics saved to:")
    print(aligned_csv)

    print(f"Statistical test CSV saved to:")
    print(stats_csv)

    print(f"Descriptive statistics CSV saved to:")
    print(desc_csv)

    print(f"Excel file saved to:")
    print(excel_path)

    print()
    print("Important interpretation:")
    print("  *   p < 0.05")
    print("  **  p < 0.01")
    print("  *** p < 0.001")
    print("  ns  not significant")


if __name__ == "__main__":
    main()
