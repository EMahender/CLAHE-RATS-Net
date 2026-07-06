import os
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import cv2
import xml.etree.ElementTree as ET

from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, disk


# =========================================================
# USER SETTINGS
# =========================================================
RAW_ROOT = r"ACTUAL_PATH"
OUT_ROOT = r"ACTUAL_PATH"

MAX_PATIENTS = 150          # None for all
TARGET_SIZE = (512, 256)   # (width, height)
HU_MIN = -1000
HU_MAX = 400
Z_TOLERANCE = 0.2

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Click generation
CLICK_MODE = "center"      # center | edge | mixed
CLICK_SIZE = 3
MIXED_OFFSET = 20

# Save sample figures
SAVE_SAMPLE_COUNT = 5
FIG_DPI = 300


# =========================================================
# XML HELPERS
# =========================================================
def local_name(tag):
    return tag.split("}")[-1]


def get_text_of_child(elem, child_name, default=None):
    for child in list(elem):
        if local_name(child.tag) == child_name:
            return child.text
    return default


def find_xml_files(patient_dir):
    xml_files = []
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, f))
    return xml_files


def select_main_xml(patient_dir):
    xml_files = find_xml_files(patient_dir)
    if len(xml_files) == 0:
        return None
    xml_files = sorted(xml_files, key=lambda x: os.path.getsize(x), reverse=True)
    return xml_files[0]


def parse_lidc_xml_contours(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    contours = []

    for session in root.iter():
        if local_name(session.tag) != "readingSession":
            continue

        for nodule in list(session):
            if local_name(nodule.tag) != "unblindedReadNodule":
                continue

            nodule_id = get_text_of_child(nodule, "noduleID", default="unknown_nodule")

            for roi in list(nodule):
                if local_name(roi.tag) != "roi":
                    continue

                z_text = get_text_of_child(roi, "imageZposition", default=None)
                incl_text = get_text_of_child(roi, "inclusion", default="TRUE")

                if z_text is None:
                    continue

                try:
                    z_position = float(z_text)
                except Exception:
                    continue

                inclusion = str(incl_text).strip().lower() in ["true", "1", "yes"]

                points = []
                for edge_map in list(roi):
                    if local_name(edge_map.tag) != "edgeMap":
                        continue

                    x_text = get_text_of_child(edge_map, "xCoord", default=None)
                    y_text = get_text_of_child(edge_map, "yCoord", default=None)

                    if x_text is None or y_text is None:
                        continue

                    try:
                        x = int(round(float(x_text)))
                        y = int(round(float(y_text)))
                        points.append((x, y))
                    except Exception:
                        pass

                if inclusion and len(points) >= 3:
                    contours.append({
                        "nodule_id": nodule_id,
                        "z_position": z_position,
                        "points": points
                    })

    return contours


def contours_to_mask(contours_for_slice, image_shape):
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for contour in contours_for_slice:
        pts = np.array(contour["points"], dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

    return mask


# =========================================================
# DICOM HELPERS
# =========================================================
def get_dicom_files(series_dir):
    series_dir = Path(series_dir)
    files = []

    for f in series_dir.iterdir():
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if hasattr(ds, "SOPInstanceUID"):
                    files.append(str(f))
            except Exception:
                pass

    return files


def find_all_series_folders(patient_dir):
    patient_dir = Path(patient_dir)
    series_folders = []

    for study_dir in patient_dir.iterdir():
        if study_dir.is_dir():
            for series_dir in study_dir.iterdir():
                if series_dir.is_dir():
                    n_dcm = len(get_dicom_files(series_dir))
                    if n_dcm > 0:
                        series_folders.append((str(series_dir), n_dcm))

    return series_folders


def select_main_series(patient_dir):
    series_folders = find_all_series_folders(patient_dir)
    if len(series_folders) == 0:
        return None, 0

    series_folders = sorted(series_folders, key=lambda x: x[1], reverse=True)
    return series_folders[0][0], series_folders[0][1]


def load_dicom_slices(series_dir):
    dicom_files = get_dicom_files(series_dir)
    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files found in: {series_dir}")

    slices = [pydicom.dcmread(fp) for fp in dicom_files]

    if hasattr(slices[0], "ImagePositionPatient"):
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    elif hasattr(slices[0], "InstanceNumber"):
        slices.sort(key=lambda s: int(s.InstanceNumber))
    else:
        raise ValueError("Cannot sort DICOM slices.")

    return slices


def dicom_to_hu(ds):
    img = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    if slope != 1:
        img = (img.astype(np.float32) * slope).astype(np.int16)

    img = img + np.int16(intercept)
    return img


# =========================================================
# PREPROCESSING HELPERS
# =========================================================
def clip_and_normalize_hu(img_hu, hu_min=-1000, hu_max=400):
    img = np.clip(img_hu, hu_min, hu_max).astype(np.float32)
    img = (img - hu_min) / float(hu_max - hu_min)
    return img


def make_binary_image(img_hu, threshold=-320):
    binary = img_hu < threshold
    return binary.astype(np.uint8)


def keep_two_largest_regions(binary):
    labeled = label(binary)
    props = regionprops(labeled)

    if len(props) == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    props = sorted(props, key=lambda x: x.area, reverse=True)
    out = np.zeros_like(binary, dtype=np.uint8)

    for p in props[:2]:
        out[labeled == p.label] = 1

    return out


def segment_lung_mask(img_hu):
    binary = make_binary_image(img_hu)
    binary = clear_border(binary)

    lung_mask = keep_two_largest_regions(binary)
    lung_mask = binary_fill_holes(lung_mask).astype(np.uint8)
    lung_mask = binary_closing(lung_mask, disk(3)).astype(np.uint8)
    lung_mask = remove_small_objects(lung_mask.astype(bool), min_size=500).astype(np.uint8)

    return binary.astype(np.uint8), lung_mask.astype(np.uint8)


def split_left_right_lungs(lung_mask):
    h, w = lung_mask.shape
    mid = w // 2
    left = np.zeros_like(lung_mask, dtype=np.uint8)
    right = np.zeros_like(lung_mask, dtype=np.uint8)
    left[:, :mid] = lung_mask[:, :mid]
    right[:, mid:] = lung_mask[:, mid:]
    return left, right


def select_lung_side(image_norm, lung_mask, gt_mask):
    """
    Select lung side containing the nodule.
    Returns cropped image, cropped mask, side name.
    """
    h, w = image_norm.shape
    mid = w // 2

    left_mask, right_mask = split_left_right_lungs(lung_mask)

    left_gt = gt_mask[:, :mid].sum()
    right_gt = gt_mask[:, mid:].sum()

    if left_gt > right_gt:
        side = "left"
        img_side = image_norm[:, :mid]
        mask_side = gt_mask[:, :mid]
        lung_side = left_mask[:, :mid]
    elif right_gt > left_gt:
        side = "right"
        img_side = image_norm[:, mid:]
        mask_side = gt_mask[:, mid:]
        lung_side = right_mask[:, mid:]
    else:
        # fallback: side with more lung area
        if left_mask.sum() >= right_mask.sum():
            side = "left"
            img_side = image_norm[:, :mid]
            mask_side = gt_mask[:, :mid]
            lung_side = left_mask[:, :mid]
        else:
            side = "right"
            img_side = image_norm[:, mid:]
            mask_side = gt_mask[:, mid:]
            lung_side = right_mask[:, mid:]

    img_side = img_side * lung_side
    return img_side.astype(np.float32), mask_side.astype(np.uint8), side


def resize_image_and_mask(image, mask, target_size):
    w, h = target_size
    image_r = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return image_r.astype(np.float32), mask_r.astype(np.uint8)


def apply_clahe_to_norm_image(image_norm, clip_limit=2.0, tile_grid=(8, 8)):
    img_8 = np.clip(image_norm * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out_8 = clahe.apply(img_8)
    out = out_8.astype(np.float32) / 255.0
    return out


# =========================================================
# CLICK MAP
# =========================================================
def get_mask_center(mask):
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0 or len(ys) == 0:
        return h // 2, w // 2
    return int(np.mean(ys)), int(np.mean(xs))


def get_mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def generate_click_map(mask, mode="center", click_size=3, mixed_offset=20):
    h, w = mask.shape
    click = np.zeros((h, w), dtype=np.uint8)

    bbox = get_mask_bbox(mask)
    cy, cx = get_mask_center(mask)

    if bbox is None:
        y, x = cy, cx
    else:
        y_min, y_max, x_min, x_max = bbox

        if mode == "center":
            y, x = cy, cx
        elif mode == "edge":
            choice = np.random.choice(["top", "bottom", "left", "right"])
            if choice == "top":
                y, x = y_min, cx
            elif choice == "bottom":
                y, x = y_max, cx
            elif choice == "left":
                y, x = cy, x_min
            else:
                y, x = cy, x_max
        elif mode == "mixed":
            dy = np.random.randint(-mixed_offset, mixed_offset + 1)
            dx = np.random.randint(-mixed_offset, mixed_offset + 1)
            y = int(np.clip(cy + dy, 0, h - 1))
            x = int(np.clip(cx + dx, 0, w - 1))
        else:
            y, x = cy, cx

    half = click_size // 2
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    click[y1:y2, x1:x2] = 1
    return click.astype(np.uint8)


# =========================================================
# VISUALIZATION
# =========================================================
def save_gray_png(img, save_path):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_path), img)


def save_binary_png(mask, save_path):
    cv2.imwrite(str(save_path), (mask.astype(np.uint8) * 255))


def save_rgb_png(image_rgb, save_path):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), image_bgr)


def create_overlay(image_norm, mask, click_map=None, alpha=0.35):
    image_8 = np.clip(image_norm * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([image_8] * 3, axis=-1).astype(np.uint8)
    overlay = rgb.copy()

    # mask fill: magenta
    fill = np.zeros_like(rgb, dtype=np.uint8)
    fill[..., 0] = 255
    fill[..., 1] = 0
    fill[..., 2] = 180

    overlay[mask > 0] = ((1 - alpha) * rgb[mask > 0] + alpha * fill[mask > 0]).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 215, 0), 2)

    if click_map is not None:
        overlay[click_map > 0] = (0, 255, 255)

    return overlay


# =========================================================
# MAPPING CONTOURS TO SLICES
# =========================================================
def build_slice_metadata(slices, patient_id, xml_path):
    rows = []
    for idx, ds in enumerate(slices):
        z_pos = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else None
        rows.append({
            "patient_id": patient_id,
            "slice_index": idx,
            "z_position": z_pos,
            "slice_path": ds.filename,
            "xml_path": xml_path,
        })
    return pd.DataFrame(rows)


def build_slice_mask_map(contours, metadata_df, z_tolerance=0.2):
    slice_map = {}

    slice_z = metadata_df["z_position"].tolist()
    slice_indices = metadata_df["slice_index"].tolist()

    for contour in contours:
        z = contour["z_position"]
        diffs = [abs(sz - z) if pd.notna(sz) else 1e9 for sz in slice_z]
        min_idx = int(np.argmin(diffs))
        min_diff = diffs[min_idx]

        if min_diff <= z_tolerance:
            slice_index = int(slice_indices[min_idx])
            if slice_index not in slice_map:
                slice_map[slice_index] = []
            slice_map[slice_index].append(contour)

    return slice_map


# =========================================================
# MAIN PATIENT PIPELINE
# =========================================================
def process_one_patient(patient_dir):
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    print(f"\nProcessing patient: {patient_id}")

    series_path, _ = select_main_series(patient_dir)
    xml_path = select_main_xml(patient_dir)

    if series_path is None:
        raise ValueError("No DICOM series found.")
    if xml_path is None:
        raise ValueError("No XML found.")

    slices = load_dicom_slices(series_path)
    metadata_df = build_slice_metadata(slices, patient_id, xml_path)

    contours = parse_lidc_xml_contours(xml_path)
    slice_mask_map = build_slice_mask_map(contours, metadata_df, z_tolerance=Z_TOLERANCE)

    patient_out = Path(OUT_ROOT) / patient_id
    baseline_img_dir = patient_out / "baseline_images_npy"
    baseline_mask_dir = patient_out / "baseline_masks_npy"
    baseline_click_dir = patient_out / "baseline_clicks_npy"

    clahe_img_dir = patient_out / "clahe_images_npy"
    clahe_mask_dir = patient_out / "clahe_masks_npy"
    clahe_click_dir = patient_out / "clahe_clicks_npy"

    preview_dir = patient_out / "sample_previews"

    for d in [
        baseline_img_dir, baseline_mask_dir, baseline_click_dir,
        clahe_img_dir, clahe_mask_dir, clahe_click_dir, preview_dir
    ]:
        d.mkdir(parents=True, exist_ok=True)

    out_rows = []
    sample_saved = 0

    for i, row in metadata_df.iterrows():
        slice_index = int(row["slice_index"])
        ds = pydicom.dcmread(row["slice_path"])
        img_hu = dicom_to_hu(ds)

        if slice_index in slice_mask_map:
            gt_mask = contours_to_mask(slice_mask_map[slice_index], image_shape=img_hu.shape)
        else:
            gt_mask = np.zeros(img_hu.shape, dtype=np.uint8)

        # Baseline preprocessing
        img_norm = clip_and_normalize_hu(img_hu, HU_MIN, HU_MAX)
        _, lung_mask = segment_lung_mask(img_hu)

        img_side, mask_side, side_name = select_lung_side(img_norm, lung_mask, gt_mask)
        img_resized, mask_resized = resize_image_and_mask(img_side, mask_side, TARGET_SIZE)

        click_map = generate_click_map(
            mask_resized,
            mode=CLICK_MODE,
            click_size=CLICK_SIZE,
            mixed_offset=MIXED_OFFSET
        )

        # CLAHE branch
        img_clahe = apply_clahe_to_norm_image(
            img_resized,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid=CLAHE_TILE_GRID
        )

        # Save baseline
        baseline_img_path = baseline_img_dir / f"slice_{slice_index:04d}.npy"
        baseline_mask_path = baseline_mask_dir / f"slice_{slice_index:04d}.npy"
        baseline_click_path = baseline_click_dir / f"slice_{slice_index:04d}.npy"

        np.save(baseline_img_path, img_resized.astype(np.float32))
        np.save(baseline_mask_path, mask_resized.astype(np.uint8))
        np.save(baseline_click_path, click_map.astype(np.uint8))

        # Save clahe
        clahe_img_path = clahe_img_dir / f"slice_{slice_index:04d}.npy"
        clahe_mask_path = clahe_mask_dir / f"slice_{slice_index:04d}.npy"
        clahe_click_path = clahe_click_dir / f"slice_{slice_index:04d}.npy"

        np.save(clahe_img_path, img_clahe.astype(np.float32))
        np.save(clahe_mask_path, mask_resized.astype(np.uint8))
        np.save(clahe_click_path, click_map.astype(np.uint8))

        mask_pixels = int(mask_resized.sum())
        mask_status = "positive" if mask_pixels > 0 else "empty"

        out_rows.append({
            "patient_id": patient_id,
            "slice_index": slice_index,
            "xml_path": xml_path,
            "source_slice_path": row["slice_path"],
            "lung_side": side_name,
            "mask_pixel_count": mask_pixels,
            "mask_status": mask_status,

            "baseline_image_path": str(baseline_img_path),
            "baseline_mask_path": str(baseline_mask_path),
            "baseline_click_path": str(baseline_click_path),

            "clahe_image_path": str(clahe_img_path),
            "clahe_mask_path": str(clahe_mask_path),
            "clahe_click_path": str(clahe_click_path),
        })

        # Save samples
        if mask_status == "positive" and sample_saved < SAVE_SAMPLE_COUNT:
            overlay_base = create_overlay(img_resized, mask_resized, click_map)
            overlay_clahe = create_overlay(img_clahe, mask_resized, click_map)

            save_gray_png(img_resized, preview_dir / f"sample_{sample_saved+1:02d}_baseline.png")
            save_gray_png(img_clahe, preview_dir / f"sample_{sample_saved+1:02d}_clahe.png")
            save_binary_png(mask_resized, preview_dir / f"sample_{sample_saved+1:02d}_mask.png")
            save_binary_png(click_map, preview_dir / f"sample_{sample_saved+1:02d}_click.png")
            save_rgb_png(overlay_base, preview_dir / f"sample_{sample_saved+1:02d}_overlay_baseline.png")
            save_rgb_png(overlay_clahe, preview_dir / f"sample_{sample_saved+1:02d}_overlay_clahe.png")
            sample_saved += 1

    out_df = pd.DataFrame(out_rows)

    csv_path = patient_out / "patient_preprocessing_summary.csv"
    xlsx_path = patient_out / "patient_preprocessing_summary.xlsx"
    out_df.to_csv(csv_path, index=False)
    out_df.to_excel(xlsx_path, index=False)

    return {
        "patient_id": patient_id,
        "num_slices": len(out_df),
        "positive_slices": int((out_df["mask_status"] == "positive").sum()),
        "empty_slices": int((out_df["mask_status"] == "empty").sum()),
        "summary_csv": str(csv_path),
        "summary_xlsx": str(xlsx_path),
        "status": "SUCCESS"
    }


# =========================================================
# MAIN
# =========================================================
def main():
    raw_root = Path(RAW_ROOT)
    patient_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])

    if MAX_PATIENTS is not None:
        patient_dirs = patient_dirs[:MAX_PATIENTS]

    print(f"Processing {len(patient_dirs)} patient(s)...")

    summary_rows = []
    for patient_dir in patient_dirs:
        try:
            result = process_one_patient(patient_dir)
            summary_rows.append(result)
        except Exception as e:
            summary_rows.append({
                "patient_id": patient_dir.name,
                "status": f"FAILED: {str(e)}"
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = Path(OUT_ROOT) / "all_patients_preprocessing_summary.csv"
    summary_xlsx = Path(OUT_ROOT) / "all_patients_preprocessing_summary.xlsx"

    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_excel(summary_xlsx, index=False)

    print("\n=== PREPROCESSING SUMMARY ===")
    print(summary_df["status"].value_counts(dropna=False))
    print(f"\nSaved summary CSV:  {summary_csv}")
    print(f"Saved summary XLSX: {summary_xlsx}")


if __name__ == "__main__":
    main()
