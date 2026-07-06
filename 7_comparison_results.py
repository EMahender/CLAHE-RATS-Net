# =============================================================================
# STEP 7: Collect All Baseline and Proposed Model Results into One Excel/CSV
#
# Reads results from:
#   outputs/baseline_proposed_comparison/
#
# Creates:
#   outputs/baseline_proposed_comparison/final_model_comparison_master.xlsx
#   outputs/baseline_proposed_comparison/final_model_comparison_master.csv
# =============================================================================

import os
from pathlib import Path
import pandas as pd


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"ACTUAL_PATH")

BASE_DIR = PROJECT_ROOT / "outputs" / "baseline_proposed_comparison"
OUTPUT_CSV = BASE_DIR / "final_model_comparison_master.csv"

MODEL_NAMES = [
    "UNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "ResUNet",
    "TransUNetLite",
    "CLAHE_RATS_Net"
]


# =============================================================================
# 2. SAFE FILE READERS
# =============================================================================

def read_csv_or_excel(path_without_ext):
    """
    Reads either .xlsx or .csv file if available.
    Example:
        path_without_ext = model_log_dir / "final_summary"
    """

    xlsx_path = Path(str(path_without_ext) + ".xlsx")
    csv_path = Path(str(path_without_ext) + ".csv")

    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return None


def flatten_columns(df):
    """
    Handles multi-index columns from groupby Excel files.
    """

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if str(c) != "nan"]).strip("_")
            for col in df.columns
        ]

    return df


# =============================================================================
# 3. COLLECT MODEL-WISE RESULTS
# =============================================================================

def collect_results():
    final_rows = []
    val_threshold_rows = []
    test_threshold_rows = []
    train_log_rows = []
    sizewise_rows = []

    for model_name in MODEL_NAMES:
        model_dir = BASE_DIR / model_name
        model_log_dir = model_dir / "logs"

        if not model_log_dir.exists():
            print(f"Warning: Log directory not found for {model_name}: {model_log_dir}")
            continue

        print(f"Collecting results for: {model_name}")

        # ---------------------------------------------------------------------
        # Final summary
        # ---------------------------------------------------------------------
        final_df = read_csv_or_excel(model_log_dir / "final_summary")

        if final_df is not None and len(final_df) > 0:
            final_df["model"] = model_name
            final_rows.append(final_df)
        else:
            print(f"  Warning: final_summary not found for {model_name}")

        # ---------------------------------------------------------------------
        # Validation threshold summary
        # ---------------------------------------------------------------------
        val_df = read_csv_or_excel(model_log_dir / "validation_threshold_summary")

        if val_df is not None and len(val_df) > 0:
            val_df["model"] = model_name
            val_threshold_rows.append(val_df)
        else:
            print(f"  Warning: validation_threshold_summary not found for {model_name}")

        # ---------------------------------------------------------------------
        # Test threshold summary
        # ---------------------------------------------------------------------
        test_df = read_csv_or_excel(model_log_dir / "test_threshold_summary")

        if test_df is not None and len(test_df) > 0:
            test_df["model"] = model_name
            test_threshold_rows.append(test_df)
        else:
            print(f"  Warning: test_threshold_summary not found for {model_name}")

        # ---------------------------------------------------------------------
        # Training log
        # ---------------------------------------------------------------------
        train_df = read_csv_or_excel(model_log_dir / "training_log")

        if train_df is not None and len(train_df) > 0:
            train_df["model"] = model_name
            train_log_rows.append(train_df)
        else:
            print(f"  Warning: training_log not found for {model_name}")

        # ---------------------------------------------------------------------
        # Size-wise summary
        # ---------------------------------------------------------------------
        size_xlsx = model_log_dir / "test_sizewise_summary.xlsx"

        if size_xlsx.exists():
            try:
                size_df = pd.read_excel(size_xlsx)
                size_df = flatten_columns(size_df)
                size_df["model"] = model_name
                sizewise_rows.append(size_df)
            except Exception as e:
                print(f"  Warning: could not read sizewise summary for {model_name}: {e}")

    final_summary = pd.concat(final_rows, ignore_index=True) if final_rows else pd.DataFrame()
    val_thresholds = pd.concat(val_threshold_rows, ignore_index=True) if val_threshold_rows else pd.DataFrame()
    test_thresholds = pd.concat(test_threshold_rows, ignore_index=True) if test_threshold_rows else pd.DataFrame()
    training_logs = pd.concat(train_log_rows, ignore_index=True) if train_log_rows else pd.DataFrame()
    sizewise_summary = pd.concat(sizewise_rows, ignore_index=True) if sizewise_rows else pd.DataFrame()

    return final_summary, val_thresholds, test_thresholds, training_logs, sizewise_summary


# =============================================================================
# 4. CREATE MAIN COMPARISON TABLE
# =============================================================================

def create_main_comparison(final_summary, val_thresholds, test_thresholds, training_logs):
    """
    Creates one clean table for manuscript/report comparison.
    """

    rows = []

    for model_name in MODEL_NAMES:
        row = {"Model": model_name}

        # ---------------------------------------------------------------------
        # Best training epoch and best validation training Dice
        # ---------------------------------------------------------------------
        if not training_logs.empty and "model" in training_logs.columns:
            mlog = training_logs[training_logs["model"] == model_name].copy()

            if len(mlog) > 0 and "val_dice" in mlog.columns:
                best_idx = mlog["val_dice"].idxmax()
                best_train_row = mlog.loc[best_idx]

                row["Best Epoch"] = int(best_train_row.get("epoch", -1))
                row["Best Val Dice During Training"] = best_train_row.get("val_dice", None)
                row["Train Dice at Best Epoch"] = best_train_row.get("train_dice", None)
                row["Train Loss at Best Epoch"] = best_train_row.get("train_loss", None)
                row["Val Loss at Best Epoch"] = best_train_row.get("val_loss", None)
                row["Epoch Time Sec"] = best_train_row.get("epoch_time_sec", None)

        # ---------------------------------------------------------------------
        # Best validation threshold
        # ---------------------------------------------------------------------
        best_threshold = None

        if not val_thresholds.empty and "model" in val_thresholds.columns:
            vdf = val_thresholds[val_thresholds["model"] == model_name].copy()

            if len(vdf) > 0 and "dice_mean" in vdf.columns:
                best_val_idx = vdf["dice_mean"].idxmax()
                best_val_row = vdf.loc[best_val_idx]

                best_threshold = best_val_row.get("threshold", None)

                row["Best Val Threshold"] = best_threshold
                row["Val Dice"] = best_val_row.get("dice_mean", None)
                row["Val IoU"] = best_val_row.get("iou_mean", None)
                row["Val Precision"] = best_val_row.get("precision_mean", None)
                row["Val Recall"] = best_val_row.get("recall_mean", None)
                row["Val Specificity"] = best_val_row.get("specificity_mean", None)

        # ---------------------------------------------------------------------
        # Test result at validation-selected threshold
        # ---------------------------------------------------------------------
        if not test_thresholds.empty and "model" in test_thresholds.columns:
            tdf = test_thresholds[test_thresholds["model"] == model_name].copy()

            if len(tdf) > 0:
                if best_threshold is not None and "threshold" in tdf.columns:
                    matched = tdf[tdf["threshold"] == best_threshold]

                    if len(matched) > 0:
                        best_test_row = matched.iloc[0]
                    else:
                        best_test_row = tdf.loc[tdf["dice_mean"].idxmax()]
                else:
                    best_test_row = tdf.loc[tdf["dice_mean"].idxmax()]

                row["Test Dice"] = best_test_row.get("dice_mean", None)
                row["Test IoU"] = best_test_row.get("iou_mean", None)
                row["Test Precision"] = best_test_row.get("precision_mean", None)
                row["Test Recall"] = best_test_row.get("recall_mean", None)
                row["Test Specificity"] = best_test_row.get("specificity_mean", None)

                # Also keep true best test threshold only for analysis
                best_test_idx = tdf["dice_mean"].idxmax()
                best_test_any = tdf.loc[best_test_idx]

                row["Best Test Threshold Any"] = best_test_any.get("threshold", None)
                row["Best Test Dice Any"] = best_test_any.get("dice_mean", None)

        # ---------------------------------------------------------------------
        # Parameters from final summary
        # ---------------------------------------------------------------------
        if not final_summary.empty and "model" in final_summary.columns:
            fdf = final_summary[final_summary["model"] == model_name].copy()

            if len(fdf) > 0:
                frow = fdf.iloc[0]

                row["Parameters"] = frow.get("parameters", frow.get("trainable_parameters", None))
                row["Trainable Parameters"] = frow.get("trainable_parameters", None)
                row["Input"] = frow.get("input", None)
                row["Loss"] = frow.get("loss", None)
                row["Postprocessing"] = frow.get("postprocessing", None)
                row["Use TTA"] = frow.get("use_tta", None)

        rows.append(row)

    main_df = pd.DataFrame(rows)

    # Sort by test Dice descending
    if "Test Dice" in main_df.columns:
        main_df = main_df.sort_values(by="Test Dice", ascending=False).reset_index(drop=True)

    # Add rank
    main_df.insert(0, "Rank", range(1, len(main_df) + 1))

    return main_df


# =============================================================================
# 5. FORMAT AND EXPORT
# =============================================================================

def export_results(main_df, final_summary, val_thresholds, test_thresholds, training_logs, sizewise_summary):
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Save main CSV
    main_df.to_csv(OUTPUT_CSV, index=False)

    # Save Excel with multiple sheets
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="Main_Comparison", index=False)

        if not final_summary.empty:
            final_summary.to_excel(writer, sheet_name="Raw_Final_Summary", index=False)

        if not val_thresholds.empty:
            val_thresholds.to_excel(writer, sheet_name="All_Val_Thresholds", index=False)

        if not test_thresholds.empty:
            test_thresholds.to_excel(writer, sheet_name="All_Test_Thresholds", index=False)

        if not training_logs.empty:
            training_logs.to_excel(writer, sheet_name="All_Training_Logs", index=False)

        if not sizewise_summary.empty:
            sizewise_summary.to_excel(writer, sheet_name="Sizewise_Test", index=False)

        # ---------------------------------------------------------------------
        # Basic formatting
        # ---------------------------------------------------------------------
        workbook = writer.book

        for sheet_name in workbook.sheetnames:
            ws = workbook[sheet_name]

            # Freeze header
            ws.freeze_panes = "A2"

            # Auto-filter
            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            # Column widths
            for col_cells in ws.columns:
                max_length = 0
                col_letter = col_cells[0].column_letter

                for cell in col_cells:
                    value = cell.value

                    if value is not None:
                        max_length = max(max_length, len(str(value)))

                adjusted_width = min(max(max_length + 2, 12), 35)
                ws.column_dimensions[col_letter].width = adjusted_width

            # Header style
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
                cell.alignment = cell.alignment.copy(horizontal="center")

            # Number formatting
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"

    print("\nDone.")
    print(f"Master Excel saved to: {OUTPUT_EXCEL}")
    print(f"Master CSV saved to:   {OUTPUT_CSV}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("COLLECT BASELINE AND PROPOSED MODEL COMPARISON RESULTS")
    print("=" * 80)

    print(f"Base comparison directory: {BASE_DIR}")

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    final_summary, val_thresholds, test_thresholds, training_logs, sizewise_summary = collect_results()

    print("\nCollected data:")
    print(f"Final summary rows:      {len(final_summary)}")
    print(f"Val threshold rows:      {len(val_thresholds)}")
    print(f"Test threshold rows:     {len(test_thresholds)}")
    print(f"Training log rows:       {len(training_logs)}")
    print(f"Sizewise summary rows:   {len(sizewise_summary)}")

    main_df = create_main_comparison(
        final_summary=final_summary,
        val_thresholds=val_thresholds,
        test_thresholds=test_thresholds,
        training_logs=training_logs
    )

    print("\nMain comparison:")
    print(main_df)

    export_results(
        main_df=main_df,
        final_summary=final_summary,
        val_thresholds=val_thresholds,
        test_thresholds=test_thresholds,
        training_logs=training_logs,
        sizewise_summary=sizewise_summary
    )


if __name__ == "__main__":
    main()
