# Refractive Eye Classifier

A deep-learning project that classifies retinal fundus images into **Normal Vision (0)** or **Myopia (1)** using a fine-tuned **ResNet18** model. The workflow includes dataset cleaning, image preprocessing/augmentation, handling class imbalance, model training/evaluation, and a small demo interface.

## What this project does

-  **Input:** Fundus images (left/right eye) referenced by a cleaned CSV dataset.
-  **Output:** Binary prediction:
   -  `0` = Normal Vision
   -  `1` = Myopia
-  **Grad-Cam** Explain in and showcase the areas of interest.
-  **Model:** Transfer learning with `torchvision.models.resnet18`, with the final fully-connected layer replaced for binary classification.
-  **Imbalance handling:** Uses class weighting via `BCEWithLogitsLoss(pos_weight=...)`.

## Results (from experiments)

From [`main.ipynb`](main.ipynb):

-  **Test Loss:** ~`0.35`
-  **Test Accuracy:** ~`0.91`
-  **Test AUC-ROC:** ~`0.92`

Validation analysis also includes threshold tuning (example shown at threshold `0.8`) and plots:

-  Confusion matrix
-  ROC curve
-  Precision–Recall curve
-  Predicted probability distributions

## Repository structure

-  [`main.ipynb`](main.ipynb) — End-to-end notebook: loading cleaned data, dataset class, augmentation, training setup, evaluation, plots, and conclusions.
-  [`data_cleaning_myopia.ipynb`](data_cleaning_myopia.ipynb) — Data cleaning workflow (produces the cleaned CSV used in training).
-  [`interface.py`](interface.py) — Demo interface code (loads the trained model and runs inference).
-  [`myopia_model.pth`](myopia_model.pth) — Saved PyTorch model weights.
-  [`data/`](data/)
   -  `raw/` — Original dataset artifacts (including ODIR-5K archive and images).
   -  `cleaned/cleaned_odir_N_M.csv` — Cleaned labels/paths used by training.
   -  `cleaned/preprocessed_images/` — Preprocessed images used by the notebook.
-  `demo_images/` — Example images for demo/testing.

## Data pipeline overview

1. **Raw data** lives in [`data/raw/`](data/raw) (ODIR-5K + metadata).
2. **Cleaning** (see [`data_cleaning_myopia.ipynb`](data_cleaning_myopia.ipynb)):
   -  Removes unused columns
   -  Converts multi-label fields into a **binary target** (`M` present → myopia)
3. **Training data** is read from [`data/cleaned/cleaned_odir_N_M.csv`](data/cleaned/cleaned_odir_N_M.csv) and images are loaded from:
   -  [`data/cleaned/preprocessed_images/`](data/cleaned/preprocessed_images/)

## Model + training details

Implemented in [`main.ipynb`](main.ipynb):

-  **Dataset class:** `MyopiaDataset` (loads left/right image filenames and returns image + label)
-  **Transforms:**
   -  Train: rotation, horizontal flip, color jitter, normalization
   -  Validation/Test: tensor conversion + normalization
-  **Split strategy:** stratified train/val/test split
-  **Loss:** `torch.nn.BCEWithLogitsLoss` with `pos_weight` computed from class imbalance
-  **Optimizer:** Adam

## How to run

### 1) Create an environment

Install common dependencies used in [`main.ipynb`](main.ipynb):

```sh
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib seaborn kagglehub
```
