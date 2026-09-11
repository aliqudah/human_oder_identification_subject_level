# Human Odor Identifier Using Machine Learning with Subject-Level Cross-Validation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PLOS ONE Compliant](https://img.shields.io/badge/PLOS_ONE-Code_Sharing_Compliant-success)](https://journals.plos.org/plosone/s/materials-and-software-sharing)

This repository contains the complete, reproducible Python code and analysis pipeline for the research paper: 
**"Towards Building a Human Odor Identifier Using Machine Learning with Subject-Level Cross-Validation"** (Published in *PLOS ONE*).

The code processes raw Gas Chromatography-Mass Spectrometry (GC/MS) text files, applies domain-specific physics-informed signal-level data augmentation, extracts peak-based chromatographic features, and evaluates 12 Machine Learning and Deep Learning classifiers for human subject identification based on body odor (Volatile Organic Compounds - VOCs).

---

## 📑 Table of Contents
1. [Scientific Overview](#-scientific-overview)
2. [Key Features of the Codebase](#-key-features-of-the-codebase)
3. [PLOS ONE Reviewer Revisions](#-plos-one-reviewer-revisions)
4. [Repository Structure](#-repository-structure)
5. [Prerequisites & Installation](#-prerequisites--installation)
6. [Data Preparation](#-data-preparation)
7. [Usage & Execution](#-usage--execution)
8. [Outputs & Visualizations](#-outputs--visualizations)
9. [Methodology Summary](#-methodology-summary)
10. [Citation](#-citation)
11. [License](#-license)

---

## 🔬 Scientific Overview
Human body odor, composed of Volatile Organic Compounds (VOCs), is unique to each individual, resembling a fingerprint. This repository implements an automated e-nose (electronic nose) pipeline to identify individuals from a cohort of **22 human subjects** using **132 skin emanation samples**. 

Samples were collected using three distinct methods:
1. **Arm/Funnel (A)**
2. **Gauze Swab (G)**
3. **Suit (S)**

The pipeline evaluates the efficacy of various pattern recognition techniques in discriminating among the 22 subjects, ultimately demonstrating that ensemble methods (specifically **Random Forest**) can achieve an accuracy of **80.9%** and a Macro-AUC of **98.9%** using a minimal, interpretable set of chromatographic features.

---

## ✨ Key Features of the Codebase
* **Robust GC/MS Parsing:** Automatically parses raw `.TXT` chromatogram outputs (retention times and intensities) from HP ChemStation.
* **Physics-Informed Signal Augmentation:** Implements a custom `VOCSignalAugmenter` class that simulates realistic chromatographic artifacts to expand the training set and prevent overfitting:
  * Baseline noise and low-frequency drift injection
  * Retention time shifting (simulating flow rate/pressure variations)
  * Intensity scaling (simulating injection volume differences)
  * Peak broadening (simulating column diffusion)
  * Column aging simulation (efficiency loss and RT shifts)
* **Advanced Feature Engineering:** Extracts up to 80 features per sample, including top-5 peak characteristics, statistical aggregations (area, height, skewness, kurtosis), and temporal elution patterns.
* **Subject-Level Cross-Validation:** Ensures zero data leakage by strictly separating subjects between training and testing folds (using `StratifiedKFold` mapped to subject IDs).
* **Comprehensive Model Evaluation:** Compares 12 classifiers (Random Forest, CatBoost, Gradient Boosting, SVM, Deep Learning, etc.) using Accuracy, Macro-F1, and Macro-AUC (One-vs-Rest).
* **Explainable AI (XAI):** Generates feature importance plots to identify the most discriminative VOC peaks for subject identification.

---

## 🔍 PLOS ONE Reviewer Revisions
This specific codebase (`subject_code4.py`) has been explicitly updated to address peer-review comments for PLOS ONE (Manuscript ID: PONE-D-25-67677):

1. **Baseline vs. Augmented Comparison (Reviewer #1, Comment 8):** 
   * *Addition:* Added `create_raw_dataset()` and `print_augmentation_comparison()` functions.
   * *Result:* The script now evaluates models on raw, un-augmented data first, then prints a dedicated console table comparing Raw Accuracy/AUC against Augmented Accuracy/AUC to empirically prove the efficacy of the augmentation strategy.
2. **Top-K Feature Importance (Reviewer #1, Comment 7):** 
   * *Addition:* Modified `plot_feature_importance()` to accept a `top_k` parameter.
   * *Result:* The feature importance bar chart now strictly displays only the **Top 10** most important features, preventing visual clutter and improving interpretability.
3. **Robust AUC Calculation Verification (Reviewer #1, Comment 10):** 
   * *Addition:* Enhanced `calculate_metrics_signal_augmented()` with safe fallback handling.
   * *Result:* The macro-average AUC is calculated using a verified One-vs-Rest (OvR) approach. The code explicitly handles edge cases in cross-validation folds where a class might be temporarily missing, ensuring the high Macro-AUC (0.989) is mathematically sound and reproducible.

---

## 📂 Repository Structure
```text
├── code.py          # Main execution script (End-to-End pipeline)
├── VOC Raw Data2/            # Directory containing raw GC/MS .TXT files
│   ├── Subject01/
│   │   ├── A1.TXT            # Arm/Funnel sample
│   │   ├── G1.TXT            # Gauze sample
│   │   └── S1.TXT            # Suit sample
│   └── ... (22 Subject folders)
├── subject_results/          # Auto-generated output directory
│   ├── subject_augmentation_results.xlsx
│   ├── Figure1_Feature_Engineering_Summary.png
│   ├── Figure2_CatBoost_Feature_Importance_Top10.png
│   ├── Figure3_Classifier_Comparison.png
│   ├── Figure4_[Best_Model]_Detailed_Metrics.png
│   ├── Figure5_[Best_Model]_Confusion_Matrix.png
│   ├── Figure6_[Best_Model]_ROC.png
│   └── Figure7_Radar_Chart.png
├── requirements.txt          # Python dependencies
└── README.md
