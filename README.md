# Human Odor Identifier Using Machine Learning with Subject-Level Cross-Validation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PLOS ONE Compliant](https://img.shields.io/badge/PLOS_ONE-Code_Sharing_Compliant-success)](https://journals.plos.org/plosone/s/materials-and-software-sharing)

This repository contains the complete, reproducible Python code and analysis pipeline for the research paper: 
**"Towards Building a Human Odor Identifier Using Machine Learning with Subject-Level Cross-Validation"** (Submitted to *PLOS ONE*).

The code processes raw Gas Chromatography-Mass Spectrometry (GC/MS) text files, applies domain-specific signal-level data augmentation, extracts peak-based chromatographic features, and evaluates multiple Machine Learning and Deep Learning classifiers for human subject identification based on body odor (Volatile Organic Compounds - VOCs).

## 📋 Table of Contents
1. [Features](#-features)
2. [Repository Structure](#-repository-structure)
3. [Prerequisites & Installation](#-prerequisites--installation)
4. [Data Preparation](#-data-preparation)
5. [Usage](#-usage)
6. [Outputs & Visualizations](#-outputs--visualizations)
7. [PLOS ONE Reviewer Revisions](#-plos-one-reviewer-revisions)
8. [Citation](#-citation)
9. [License](#-license)

---

## ✨ Features
* **Robust GC/MS Parsing:** Automatically parses raw `.TXT` chromatogram outputs (retention times and intensities) from HP ChemStation.
* **Domain-Specific Signal Augmentation:** Implements a custom `VOCSignalAugmenter` class that applies realistic chromatographic variations, including:
  * Baseline noise and drift injection
  * Retention time shifting (simulating flow rate variations)
  * Intensity scaling (simulating concentration differences)
  * Peak broadening (simulating column degradation)
  * Column aging simulation
* **Advanced Feature Engineering:** Extracts 80+ features per sample, including top-5 peak characteristics, statistical aggregations, and temporal elution patterns.
* **Subject-Level Cross-Validation:** Ensures no data leakage by strictly separating subjects between training and testing folds.
* **Comprehensive Model Evaluation:** Compares 12 classifiers (Random Forest, CatBoost, Gradient Boosting, SVM, Deep Learning, etc.) using Accuracy, Macro-F1, and Macro-AUC (One-vs-Rest).
* **Baseline vs. Augmented Comparison:** Explicitly evaluates model performance on raw data versus augmented data to quantify the impact of the augmentation strategy.

---

## 📂 Repository Structure
```text
├── subject_code4.py          # Main execution script (End-to-End pipeline)
├── VOC Raw Data2/            # Directory containing raw GC/MS .TXT files (Not included in repo, see Data Preparation)
│   ├── Subject01/
│   │   ├── A1.TXT            # Arm/Funnel sample
│   │   ├── G1.TXT            # Gauze sample
│   │   └── S1.TXT            # Suit sample
│   └── ...
├── subject_results/          # Auto-generated output directory
│   ├── subject_augmentation_results.xlsx
│   ├── Figure1_...png        # Feature Engineering Summary
│   ├── Figure2_...png        # Top 10 Feature Importance
│   ├── Figure3_...png        # Classifier Comparison
│   ├── Figure4_...png        # Best Classifier Detailed Metrics
│   ├── Figure5_...png        # Aggregated Confusion Matrix
│   ├── Figure6_...png        # Aggregated ROC Curves
│   └── Figure7_...png        # Radar Chart
└── README.md
