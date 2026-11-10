# Brain Tumor Detection in MRI with Texture Features (GLCM)

A **classical radiomics** pipeline: instead of a neural network, each MRI slice is described by 46 interpretable texture features (Gray-Level Co-occurrence Matrix, GLCM) and intensity statistics, then classified with traditional machine learning models.

The idea: healthy tissue tends to have homogeneous texture (GLCM values concentrated on the diagonal), while tumor tissue is more heterogeneous (values spread away from it). This repo tests that idea on a real dataset of 7,200 MRI slices.

![GLCM healthy vs tumor](assets/glcm_comparison.png)

## Results (held-out test set, 1,600 images)

| Task | Best model | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|---|
| Tumor vs no tumor | Gradient Boosting | **97.3%** | 0.965 | 0.998 |
| 4 classes (glioma, meningioma, pituitary, no tumor) | Gradient Boosting | **90.1%** | 0.897 | 0.973 |

Model comparison (binary task, test set):

| Model | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|
| Logistic Regression | 92.9% | 0.909 | 0.975 |
| SVM (RBF) | 96.6% | 0.956 | 0.994 |
| Random Forest | 96.6% | 0.957 | 0.997 |
| Gradient Boosting | 97.3% | 0.965 | 0.998 |

The model is selected by macro F1 in stratified 5-fold cross-validation on the training set; the test set is only used for the final evaluation.

<p float="left">
  <img src="assets/confusion_multiclass.png" width="48%"/>
  <img src="assets/feature_importance.png" width="48%"/>
</p>

**Takeaways:** no-tumor slices are recognized almost perfectly, meningioma and pituitary tumors exceed 94% recall, and glioma is the hardest class (71%), often confused with meningioma and healthy tissue. The most useful features are **GLCM contrast** at several distances and intensity percentiles.

## Features

- **GLCM** (32 gray levels; distances 1, 2 and 4 px; angles 0°, 45°, 90°, 135°): contrast, dissimilarity, homogeneity, energy, correlation, ASM and entropy. For each distance, the mean and range across angles (the range captures anisotropy).
- **First-order** statistics over the brain mask: mean, standard deviation, skewness, kurtosis, 10th and 90th percentiles, histogram entropy.
- Preprocessing: grayscale, resize to 256×256, crop to the brain with Otsu thresholding.

## Usage

```bash
pip install -r requirements.txt
python src/download_data.py                 # ~7,200 images into ./data
python src/features.py --jobs 4             # -> results/features.csv
python src/train.py                         # -> results/metrics.json + figures
python src/visualize_glcm.py data/Testing/notumor/Te-no_100.jpg data/Testing/glioma/Te-gl_100.jpg
```

Feature extraction for all 7,200 images takes ~50 s on 2 CPU cores. Training and evaluation take a few minutes on CPU.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar), mirrored on [Hugging Face](https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset): 5,600 training and 1,600 test images, balanced across 4 classes. No identical images (MD5 hash) are shared between train and test.

## Limitations

- Features are computed on the whole slice, not on a segmented lesion: the model detects "slices containing a tumor", and part of the signal may come from sequence or acquisition differences between the dataset's sources.
- 2D JPG slices, with no DICOM metadata and no patient-level split.
- Academic project. Not a diagnostic tool.

## Structure

```
src/
  download_data.py   dataset download
  features.py        GLCM + first-order feature extraction
  train.py           cross-validation, evaluation and figures
  visualize_glcm.py  visual GLCM comparison
results/             features.csv, metrics.json and figures
assets/              README images
```
