# CLAHE-RATS-Net
CLAHE-RATS-Net: A Click-Guided Two-Stage Residual Attention Network for Lung Nodule Segmentation in CT Images
This repository contains the implementation code used for lung nodule segmentation experiments on CT images. The pipeline builds CLAHE-enhanced nodule ROIs, generates click-guidance maps, trains the proposed two-stage residual attention network, compares it with baseline segmentation models, and exports reviewer-facing analysis tables and figures.
Overview
CLAHE-RATS-Net is a click-guided two-stage segmentation framework:
Preprocessing: CT slices are converted to HU-normalized images, nodule annotations are parsed from LIDC-style XML files, and binary masks are created using pixel-wise annotation fusion.
CLAHE enhancement: Contrast-limited adaptive histogram equalization is applied to improve local nodule visibility.
Patient-level split: Patients are split into training, validation, and testing sets to prevent leakage.
ROI extraction: Positive nodule slices are converted into 256 x 256 ROI crops with click/center heatmaps.
Stage 1: A click-guided attention U-Net produces a coarse nodule mask.
Stage 2: A residual refinement network uses the CLAHE ROI, click map, and Stage-1 probability map to refine the segmentation.
Evaluation: Dice, IoU, precision, recall, specificity, threshold analysis, size-wise analysis, and statistical tests are computed.
