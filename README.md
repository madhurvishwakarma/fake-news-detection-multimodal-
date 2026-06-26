# Multimodal Fake News Detection 

A machine learning pipeline for detecting fake news using both **text** and **image** features from social media datasets (Weibo and Twitter). The project demonstrates that multimodal fusion consistently outperforms single-modality approaches.

---

## Overview

This notebook implements an end-to-end pipeline that:

1. Loads and validates multimodal social media datasets (CSV + images)
2. Extracts image features using a pretrained **ResNet18** CNN
3. Extracts text features using **BERT-family sentence transformers** (with TF-IDF as a fallback)
4. Reduces dimensionality using **PCA** (images) and **TruncatedSVD** (text)
5. Fuses modalities via feature concatenation
6. Trains and evaluates two classifiers: **MLP** and **ExtraTrees (Decision Tree ensemble)**
7. Produces ROC-AUC / Precision-Recall curves and a full **ablation study**

---

## Project Structure

```
PRML_PROJECT.ipynb          # Main notebook (14 sections)
MyDrive/GodiMedia/
  weibo/                    # Weibo dataset (CSV + images)
  twitter/                  # Twitter dataset (CSV + images)
```

---

## Sections

| #   | Section                     | Description                                      |
| --- | --------------------------- | ------------------------------------------------ |
| 1   | Environment Setup           | Imports, Colab/Drive mounting                    |
| 2   | Configuration               | Seeds, dataset selection, file paths             |
| 3   | Load CSV & Validate Schema  | Load train/test CSVs, schema checks              |
| 4   | Data Cleaning & Split       | Missing value handling, stratified splitting     |
| 5   | Image Loading               | Robust image loading from disk or URL            |
| 6   | ResNet18 Feature Extraction | CNN-based image embeddings (512-dim)             |
| 7   | Transformer Text Embeddings | Sentence-BERT embeddings (with TF-IDF fallback)  |
| 8   | Dimensionality Reduction    | PCA for images, TruncatedSVD for text            |
| 9   | Feature Fusion              | Concatenation + covariance visualization         |
| 10  | MLP Classifier              | Training and evaluation                          |
| 11  | Decision Tree Classifier    | ExtraTrees ensemble training and evaluation      |
| 12  | Model Comparison            | Final results summary and logging                |
| 13  | ROC-AUC & PR Curves         | Curve plots for model comparison                 |
| 14  | Ablation Study              | Proves benefit of multimodal fusion vs. unimodal |

---

## Requirements

```bash
pip install torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn requests sentence-transformers
```

**Python 3.8+** is required. GPU is optional but recommended for ResNet18 feature extraction.

---

## Supported Datasets

| Dataset | Language | Label Column | Text Column | Image Column |
| ------- | -------- | ------------ | ----------- | ------------ |
| Weibo   | Chinese  | `label`      | `post_text` | `image_id`   |
| Twitter | English  | `label`      | `post_text` | `image_path` |

Set `DATASET_NAME = 'weibo'` or `'twitter'` in Section 2.

---

## Running in Google Colab

1. Upload the notebook to Colab.
2. Mount Google Drive in Section 1 — the notebook detects Colab automatically.
3. Place your dataset folder at `/content/drive/MyDrive/Godimedia/weibo` (or configure paths in Section 2).
4. Run cells sequentially from top to bottom.

---

## Pipeline Summary

```
CSV Data
   ├── Text column  →  Sentence-BERT  →  TruncatedSVD  ──┐
   └── Image paths  →  ResNet18       →  PCA            ──┴──► Concatenate ──► MLP / ExtraTrees ──► Evaluation
```

---

## Outputs

- Classification report (precision, recall, F1) for both models
- ROC-AUC and Precision-Recall curve plots
- Ablation study table comparing: image-only, text-only, and fused features
- Covariance matrix heatmap of fused feature space

---

## Key Design Choices

**Feature Fusion:** Simple concatenation is used intentionally — it contributes equal weight from each modality, captures complementary information, and keeps the pipeline interpretable.

**ExtraTrees over vanilla Decision Tree:** The ensemble of randomized trees provides better generalization and variance reduction while maintaining the tree-based, interpretable baseline role.

**Ablation Study:** Demonstrates that multimodal fusion is not just additive — it provides measurable accuracy gains over either modality used alone.
