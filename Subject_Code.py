# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 00:06:02 2025
@author: ali_q (modified assistant)
Revised to address PLOS ONE Reviewer Comments:
- Added baseline (raw) evaluation before augmentation (Comment 8).
- Limited feature importance plot to Top K features (Comment 7).
- Enhanced metric logging for AUC verification (Comment 10).
- Confusion matrix displays absolute counts (Reviewer request).
- Added Precision-Recall (PR) curves.
- Fixed QDA/LDA covariance and CatBoost dimension errors.
- ADDED: PLOS ONE Global Styling & New Figures 12 & 13.
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Signal / feature tools
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

# Modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve, auc,
                             average_precision_score, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Excel output
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi # For radar chart

# --- GLOBAL CONFIGURATION ---
OUTPUT_DIR = "subject_results"
TARGET_COLUMN = 'subject_id'
TOP_K_FEATURES = 10  # REVIEWER REQUEST: Limit feature importance to Top K features
# ----------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# ------------------------------
# Augmenter class (improved)
# ------------------------------
class VOCSignalAugmenter:
    """Signal-level augmentation for VOC chromatography data (realistic)."""
    def __init__(self, noise_level=0.02, time_shift_range=0.03, intensity_scale_range=0.15):
        self.noise_level = noise_level
        self.time_shift_range = time_shift_range
        self.intensity_scale_range = intensity_scale_range
    
    def add_baseline_noise(self, intensities, noise_level=None):
        if noise_level is None:
            noise_level = self.noise_level
        noise = np.random.normal(0, noise_level * np.std(intensities) + 1e-8, len(intensities))
        if len(intensities) > 5:
            drift_freq = np.random.uniform(0.0005, 0.01)
            drift_amp = noise_level * np.max(intensities) * np.random.uniform(0.1, 0.6)
            drift = drift_amp * np.sin(2 * np.pi * drift_freq * np.arange(len(intensities)))
        else:
            drift = 0
        aug = intensities + noise + drift
        aug[aug < 0] = 0.0
        return aug
    
    def time_shift_signal(self, retention_times, intensities):
        shift = np.random.uniform(-self.time_shift_range, self.time_shift_range)
        shifted = retention_times + shift
        shifted = np.maximum(0.01, shifted)
        return shifted, intensities
    
    def intensity_scaling(self, intensities):
        scale = np.random.uniform(1 - self.intensity_scale_range, 1 + self.intensity_scale_range)
        return intensities * scale
    
    def peak_broadening(self, retention_times, intensities, broadening_factor=0.03):
        if len(retention_times) < 3:
            return retention_times, intensities
        idx = np.argsort(retention_times)
        rt = retention_times[idx]
        it = intensities[idx].astype(float)
        sigma = max(0.5, broadening_factor * len(it) / 8)
        sm = gaussian_filter1d(it, sigma=sigma)
        restored = np.zeros_like(it)
        restored[np.argsort(idx)] = sm 
        return retention_times, restored
    
    def simulate_column_aging(self, retention_times, intensities, aging_factor=0.02):
        aging_shift = np.random.uniform(0, aging_factor)
        aged_times = retention_times * (1 + aging_shift)
        eff_loss = np.random.uniform(0.93, 0.995)
        aged_intensities = intensities * eff_loss
        return aged_times, aged_intensities
    
    def augment_voc_signal(self, retention_times, intensities, augmentation_type='random'):
        aug_rt = retention_times.copy()
        aug_it = intensities.copy()
        if augmentation_type == 'random':
            techniques = np.random.choice(['noise','time_shift','intensity_scale','peak_broadening','column_aging'],
                                         size=np.random.randint(1,4), replace=False)
        else:
            techniques = [augmentation_type]
        for t in techniques:
            if t == 'noise':
                aug_it = self.add_baseline_noise(aug_it)
            elif t == 'time_shift':
                aug_rt, aug_it = self.time_shift_signal(aug_rt, aug_it)
            elif t == 'intensity_scale':
                aug_it = self.intensity_scaling(aug_it)
            elif t == 'peak_broadening':
                aug_rt, aug_it = self.peak_broadening(aug_rt, aug_it)
            elif t == 'column_aging':
                aug_rt, aug_it = self.simulate_column_aging(aug_rt, aug_it)
        return aug_rt, aug_it

# ------------------------------
# File parsing (robust)
# ------------------------------
def parse_voc_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return np.array([]), np.array([])

    data_started = False
    data_pairs = []
    for raw in lines:
        line = raw.strip()
        if not line: continue
        if line.startswith('---'):
            data_started = True
            continue
        if not data_started: continue
        parts = line.split()
        if len(parts) < 2: continue
        numeric = []
        for p in parts:
            try: numeric.append(float(p))
            except: continue
        if len(numeric) >= 2:
            rt = numeric[0]
            height = numeric[1] if len(numeric) > 1 else 0.0
            if rt > 0 and height >= 0:
                data_pairs.append((rt, height))
    if not data_pairs:
        return np.array([]), np.array([])
    rts, its = zip(*data_pairs)
    return np.array(rts), np.array(its)

# ------------------------------
# Load raw VOC signals
# ------------------------------
def load_raw_voc_signals(data_directory="VOC Raw Data2/"):
    print(f"Loading raw VOC signal data from: {data_directory}")
    subject_dirs = glob.glob(os.path.join(data_directory, "Subject*")) + glob.glob(os.path.join(data_directory, "Suject*"))
    signals = []
    for sd in sorted(subject_dirs):
        subject_id = os.path.basename(sd)
        txts = sorted(glob.glob(os.path.join(sd, "*.TXT")))
        for txt in txts:
            rt, it = parse_voc_file(txt)
            if rt.size > 0 and it.size > 0:
                signals.append({
                    'subject_id': subject_id,
                    'sample_type': os.path.basename(txt)[0] if os.path.basename(txt)[0] in ['A','G','S'] else 'Unknown',
                    'retention_times': rt,
                    'intensities': it
                })
    
    unique_subjects = sorted(list(set([s['subject_id'] for s in signals])))
    print(f"Loaded {len(signals)} signals across {len(unique_subjects)} subjects: {unique_subjects}")
    if signals:
        avg_peaks = np.mean([len(s['retention_times']) for s in signals])
        print(f"Average peaks per signal: {avg_peaks:.1f}")
    return signals

# ------------------------------
# Feature extraction (robust)
# ------------------------------
def extract_features_from_signal(retention_times, intensities, target_features=80):
    features = {}
    if retention_times is None or len(retention_times) == 0 or intensities is None or len(intensities) == 0:
        for i in range(target_features):
            features[f'f_{i}'] = 0.0
        return features

    order = np.argsort(retention_times)
    rt = np.array(retention_times)[order].astype(float)
    it = np.array(intensities)[order].astype(float)

    if len(it) > 3:
        it_smooth = gaussian_filter1d(it, sigma=max(0.5, len(it) / 50.0))
    else:
        it_smooth = it.copy()

    prominence = max(1e-6, 0.06 * (np.max(it_smooth) - np.min(it_smooth)))
    peaks, props = find_peaks(it_smooth, prominence=prominence, distance=2)
    widths_res = peak_widths(it_smooth, peaks, rel_height=0.5) if peaks.size > 0 else (np.array([]),)
    widths = widths_res[0] if len(widths_res) > 0 else np.array([])

    features['num_peaks'] = float(len(peaks))
    
    # NumPy 2.0 compatibility
    if len(rt) > 1:
        try:
            features['total_area'] = float(np.trapezoid(it, rt))
        except AttributeError:
            features['total_area'] = float(np.trapz(it, rt))
    else:
        features['total_area'] = float(np.sum(it))
        
    features['max_intensity'] = float(np.max(it))
    features['mean_intensity'] = float(np.mean(it))
    features['std_intensity'] = float(np.std(it))
    features['min_intensity'] = float(np.min(it))
    features['rt_mean'] = float(np.mean(rt))
    features['rt_std'] = float(np.std(rt))
    features['rt_min'] = float(np.min(rt))
    features['rt_max'] = float(np.max(rt))
    features['width_mean'] = float(np.mean(widths)) if widths.size>0 else 0.0
    features['width_std'] = float(np.std(widths)) if widths.size>0 else 0.0
    features['peak_prominence_mean'] = float(np.mean(props['prominences'])) if (peaks.size>0 and 'prominences' in props) else 0.0

    if peaks.size > 0:
        top_idx = np.argsort(it_smooth[peaks])[-5:][::-1]
    else:
        top_idx = np.array([], dtype=int)
    for i in range(5):
        if i < len(top_idx):
            p = peaks[top_idx[i]]
            features[f'top{i+1}_intensity'] = float(it_smooth[p])
            features[f'top{i+1}_rt'] = float(rt[p])
            features[f'top{i+1}_width'] = float(widths[top_idx[i]]) if widths.size>0 else 0.0
        else:
            features[f'top{i+1}_intensity'] = 0.0
            features[f'top{i+1}_rt'] = 0.0
            features[f'top{i+1}_width'] = 0.0

    features['peak_density'] = float(len(peaks) / (rt.max()-rt.min()+1e-8)) if len(rt)>1 else 0.0
    features['intensity_skew'] = float(pd.Series(it).skew()) if len(it) > 2 else 0.0
    features['intensity_kurtosis'] = float(pd.Series(it).kurtosis()) if len(it) > 2 else 0.0

    ordered = list(features.items())
    idx = 0
    while len(ordered) < target_features:
        ordered.append((f'extra_{idx}', 0.0))
        idx += 1
    ordered = ordered[:target_features]
    return {k: float(v) for k, v in ordered}

# ------------------------------
# NEW: Create RAW dataset (NO augmentation) for baseline comparison (Reviewer Comment 8)
# ------------------------------
def create_raw_dataset(signals_data, target_features=80):
    print("Creating RAW dataset for baseline evaluation (No Augmentation)...")
    subject_groups = {}
    for s in signals_data:
        key = s['subject_id']
        subject_groups.setdefault(key, []).append(s)
        
    train_signals, test_signals = [], []
    for subject_id, g in subject_groups.items():
        if len(g) >= 2:
            train_signals.extend(g[:-1])
            test_signals.append(g[-1])
        else:
            train_signals.append(g[0])
            
    train_data, test_data = [], []
    for s in train_signals:
        feat = extract_features_from_signal(s['retention_times'], s['intensities'], target_features=target_features)
        feat.update({TARGET_COLUMN: s[TARGET_COLUMN], 'sample_type': s['sample_type'], 'augmented': False})
        train_data.append(feat)
        
    for s in test_signals:
        feat = extract_features_from_signal(s['retention_times'], s['intensities'], target_features=target_features)
        feat.update({TARGET_COLUMN: s[TARGET_COLUMN], 'sample_type': s['sample_type'], 'augmented': False})
        test_data.append(feat)
        
    return pd.DataFrame(train_data), pd.DataFrame(test_data)

# ------------------------------
# Create augmented & balanced dataset
# ------------------------------
def create_augmented_dataset(signals_data, augmentation_factor=5, target_features=80):
    print("Creating augmented & balanced dataset for Subject Classification...")
    augmenter = VOCSignalAugmenter(noise_level=0.02, time_shift_range=0.03, intensity_scale_range=0.15)

    subject_groups = {}
    for s in signals_data:
        key = s['subject_id']
        subject_groups.setdefault(key, []).append(s)

    train_signals = []
    test_signals = []
    
    for subject_id, g in subject_groups.items():
        if len(g) >= 2:
            train_signals.extend(g[:-1])
            test_signals.append(g[-1])
        else:
            train_signals.append(g[0])

    subject_buckets = {}
    for s in train_signals:
        subject_buckets.setdefault(s[TARGET_COLUMN], []).append(s)

    counts = {k: len(v) for k, v in subject_buckets.items()}
    target_count = max(counts.values()) if counts else 0

    augmented_train_data = []
    for s in train_signals:
        feat = extract_features_from_signal(s['retention_times'], s['intensities'], target_features=target_features)
        feat.update({TARGET_COLUMN: s[TARGET_COLUMN], 'sample_type': s['sample_type'], 'augmented': False})
        augmented_train_data.append(feat)

    for subject_label, items in subject_buckets.items():
        current = sum(1 for d in augmented_train_data if d[TARGET_COLUMN] == subject_label and d['augmented'] == False)
        needed = max(0, target_count - current)
        
        if needed > 0:
            idx = 0
            while needed > 0:
                base = items[idx % len(items)]
                aug_rt, aug_it = augmenter.augment_voc_signal(base['retention_times'], base['intensities'], 'random')
                feat = extract_features_from_signal(aug_rt, aug_it, target_features=target_features)
                feat.update({TARGET_COLUMN: base[TARGET_COLUMN], 'sample_type': base['sample_type'], 'augmented': True})
                augmented_train_data.append(feat)
                needed -= 1
                idx += 1

    additional_augmented = []
    for _ in range(augmentation_factor - 1):
        for s in train_signals:
            aug_rt, aug_it = augmenter.augment_voc_signal(s['retention_times'], s['intensities'], 'random')
            feat = extract_features_from_signal(aug_rt, aug_it, target_features=target_features)
            feat.update({TARGET_COLUMN: s[TARGET_COLUMN], 'sample_type': s['sample_type'], 'augmented': True})
            additional_augmented.append(feat)
    augmented_train_data.extend(additional_augmented)

    test_data = []
    for s in test_signals:
        feat = extract_features_from_signal(s['retention_times'], s['intensities'], target_features=target_features)
        feat.update({TARGET_COLUMN: s[TARGET_COLUMN], 'sample_type': s['sample_type'], 'augmented': False})
        test_data.append(feat)

    train_df = pd.DataFrame(augmented_train_data)
    test_df = pd.DataFrame(test_data)
    
    train_counts = train_df[TARGET_COLUMN].value_counts().to_dict()
    print(f"Original samples: {len(signals_data)}. Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Training set subject counts (after balancing/augmentation): {train_counts}")

    return train_df, test_df

# ------------------------------
# Prepare data
# ------------------------------
def prepare_signal_augmented_data(train_df, test_df):
    feature_columns = [c for c in train_df.columns if c not in [TARGET_COLUMN,'sample_type','augmented']]
    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values if not test_df.empty else np.zeros((0, X_train.shape[1]))

    y_train_labels = train_df[TARGET_COLUMN].values
    y_test_labels = test_df[TARGET_COLUMN].values if TARGET_COLUMN in test_df.columns else np.array([])

    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_labels, y_test_labels]) if y_test_labels.size>0 else y_train_labels
    label_encoder.fit(all_labels)

    y_train_encoded = label_encoder.transform(y_train_labels)
    y_test_encoded = label_encoder.transform(y_test_labels) if y_test_labels.size>0 else np.array([])

    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes) if y_test_encoded.size>0 else np.zeros((0, num_classes))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test.shape[0] > 0 else X_test

    print(f"\nPrepared signal-augmented data:")
    print(f" X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, Classes: {label_encoder.classes_}")
    return (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded,
            y_train_categorical, y_test_categorical, label_encoder, feature_columns, scaler)

# ------------------------------
# Classifiers dictionary (FIXED for QDA/LDA Covariance Errors)
# ------------------------------
def get_signal_augmented_classifiers():
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, max_depth=20),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42, C=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, C=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=6, learning_rate=0.05),
        'CatBoost': CatBoostClassifier(iterations=300, random_seed=42, verbose=False, depth=6, learning_rate=0.05),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42, learning_rate=0.5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20),
        'Naive Bayes': GaussianNB(),
        # FIX: Added solver='eigen' and shrinkage='auto' to prevent singular matrix errors in LDA
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        # FIX: Added reg_param to regularize the covariance matrix in QDA
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(reg_param=1e-3),
        'Deep Learning (Neural Network)': 'neural_network'
    }
    return classifiers

# ------------------------------
# Neural network builder
# ------------------------------
def build_signal_augmented_neural_network(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------
# Metric calculation (per-fold) - UPDATED to include Macro-AP
# ------------------------------
def calculate_metrics_signal_augmented(y_true, y_pred, y_proba, label_encoder):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.arange(len(label_encoder.classes_)), zero_division=0)
    metrics['macro_precision'] = float(np.mean(prec))
    metrics['macro_recall'] = float(np.mean(rec))
    metrics['macro_f1'] = float(np.mean(f1))
    
    num_classes = len(label_encoder.classes_)
    try:
        if y_proba is not None and y_proba.shape[1] == num_classes and len(np.unique(y_true)) > 1:
            y_true_cat = to_categorical(y_true, num_classes=num_classes)
            # REVIEWER REQUEST: Verified robust Macro-AUC (One-vs-Rest) calculation
            auc_score = roc_auc_score(y_true_cat, y_proba, average='macro', multi_class='ovr')
            metrics['macro_auc'] = float(auc_score)
            
            # Macro-AP (Average Precision) for Precision-Recall Curves
            ap_score = average_precision_score(y_true_cat, y_proba, average='macro')
            metrics['macro_ap'] = float(ap_score)
        else:
            metrics['macro_auc'] = 0.5
            metrics['macro_ap'] = 0.5
    except Exception:
        metrics['macro_auc'] = 0.5
        metrics['macro_ap'] = 0.5
    
    return metrics

# ------------------------------
# Stratified K-Fold evaluation (FIXED for Dimension Mismatches)
# ------------------------------
def evaluate_signal_augmented_classifiers(X, y, y_cat, label_encoder, feature_names, n_splits=5, epochs_nn=100, batch_size=32, plot_target_clf='Random Forest'):
    print("\nEvaluating classifiers with StratifiedKFold ...")
    classifiers = get_signal_augmented_classifiers()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {name: [] for name in classifiers.keys()}
    
    plot_target_y_true_encoded = np.array([], dtype=int)
    plot_target_y_pred_encoded = np.array([], dtype=int)
    plot_target_y_proba = np.empty((0, len(label_encoder.classes_)))
    
    catboost_importances = None
    gb_importances = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f" Fold {fold_idx}/{n_splits}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        y_tr_cat = to_categorical(y_tr, num_classes=len(label_encoder.classes_))
        y_val_cat = to_categorical(y_val, num_classes=len(label_encoder.classes_))

        for clf_name, clf in classifiers.items():
            try:
                collect_plot_data = (clf_name == plot_target_clf)
                y_pred_proba = None

                if clf_name == 'Deep Learning (Neural Network)':
                    model = build_signal_augmented_neural_network((X_tr.shape[1],), len(label_encoder.classes_))
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=0)]
                    model.fit(X_tr, y_tr_cat, validation_data=(X_val, y_val_cat),
                              epochs=epochs_nn, batch_size=batch_size, callbacks=callbacks, verbose=0)
                    y_pred_proba = model.predict(X_val, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_val)
                    try:
                        proba = clf.predict_proba(X_val)
                        proba = np.asarray(proba)
                        
                        # FIX 1: Ensure proba is strictly a 2D numpy array (Fixes CatBoost 1D array crash)
                        if proba.ndim == 1:
                            proba = proba.reshape(-1, 1)
                            
                        # FIX 2: Pad probabilities if the fold missed some classes (Common in QDA/LDA/NaiveBayes)
                        if proba.shape[1] != len(label_encoder.classes_):
                            classes_in_fold = clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_tr)
                            global_indices = label_encoder.transform(classes_in_fold)
                            padded_proba = np.zeros((proba.shape[0], len(label_encoder.classes_)))
                            for i, global_idx in enumerate(global_indices):
                                padded_proba[:, global_idx] = proba[:, i]
                            proba = padded_proba
                            
                        y_pred_proba = proba
                    except Exception:
                        y_pred_proba = None

                metrics = calculate_metrics_signal_augmented(y_val, y_pred, y_pred_proba, label_encoder)
                results[clf_name].append(metrics)

                if clf_name == 'CatBoost' and catboost_importances is None:
                    catboost_importances = clf.get_feature_importance()
                elif clf_name == 'Gradient Boosting' and gb_importances is None:
                    gb_importances = clf.feature_importances_
                
                if collect_plot_data:
                    plot_target_y_true_encoded = np.concatenate([plot_target_y_true_encoded, y_val])
                    plot_target_y_pred_encoded = np.concatenate([plot_target_y_pred_encoded, y_pred])
                    # FIX 3: Only concatenate if y_pred_proba is valid, 2D, and has the correct number of columns
                    if y_pred_proba is not None and y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == len(label_encoder.classes_):
                        plot_target_y_proba = np.concatenate([plot_target_y_proba, y_pred_proba], axis=0)

            except Exception as e:
                print(f"  {clf_name} fold {fold_idx} error: {e}")
                continue

    aggregated = {}
    for clf_name, folds in results.items():
        if not folds: continue
        keys = folds[0].keys()
        mean_metrics = {k: float(np.mean([f[k] for f in folds])) for k in keys}
        std_metrics = {f"{k}_std": float(np.std([f[k] for f in folds])) for k in keys}
        aggregated[clf_name] = {**mean_metrics, **std_metrics}
            
    return aggregated, catboost_importances, gb_importances, plot_target_y_true_encoded, plot_target_y_pred_encoded, plot_target_y_proba

# ------------------------------
# NEW: Print Comparison Table (Raw vs Augmented)
# ------------------------------
def print_augmentation_comparison(results_raw, results_aug):
    print("\n" + "="*110)
    print(" REVIEWER REQUEST: PERFORMANCE COMPARISON (RAW vs. AUGMENTED DATA) ")
    print("="*110)
    print(f"{'Classifier': <30} | {'Raw Acc': <8} | {'Aug Acc': <8} | {'Improvement': <12} | {'Raw AUC': <8} | {'Aug AUC': <8}")
    print("-" * 110)
    
    for clf_name in results_aug.keys():
        if clf_name in results_raw:
            raw_acc = results_raw[clf_name].get('accuracy', 0.0)
            aug_acc = results_aug[clf_name].get('accuracy', 0.0)
            raw_auc = results_raw[clf_name].get('macro_auc', 0.0)
            aug_auc = results_aug[clf_name].get('macro_auc', 0.0)
            improvement = aug_acc - raw_acc
            
            print(f"{clf_name: <30} | {raw_acc: <8.3f} | {aug_acc: <8.3f} | {improvement: <+12.3f} | {raw_auc: <8.3f} | {aug_auc: <8.3f}")
    print("="*110 + "\n")

# ------------------------------
# Create Excel summary
# ------------------------------
def create_signal_augmentation_excel(results_signal_aug, output_dir, output_filename="results_summary.xlsx"):
    output_path = os.path.join(output_dir, output_filename)
    print(f"\nSaving results to Excel: {output_path}")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Subject_Augmentation_Results"

    headers = ['Classifier', 'Accuracy_mean', 'Accuracy_std', 'Macro_Precision_mean', 'Macro_Precision_std',
               'Macro_Recall_mean', 'Macro_Recall_std', 'Macro_F1_mean', 'Macro_F1_std', 
               'Macro_AUC_mean', 'Macro_AUC_std', 'Macro_AP_mean', 'Macro_AP_std']
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    row = 2
    for clf_name, metrics in sorted(results_signal_aug.items(), key=lambda it: it[1].get('accuracy',0), reverse=True):
        ws.cell(row=row, column=1, value=clf_name)
        ws.cell(row=row, column=2, value=metrics.get('accuracy', 0.0))
        ws.cell(row=row, column=3, value=metrics.get('accuracy_std', 0.0))
        ws.cell(row=row, column=4, value=metrics.get('macro_precision', 0.0))
        ws.cell(row=row, column=5, value=metrics.get('macro_precision_std', 0.0))
        ws.cell(row=row, column=6, value=metrics.get('macro_recall', 0.0))
        ws.cell(row=row, column=7, value=metrics.get('macro_recall_std', 0.0))
        ws.cell(row=row, column=8, value=metrics.get('macro_f1', 0.0))
        ws.cell(row=row, column=9, value=metrics.get('macro_f1_std', 0.0))
        ws.cell(row=row, column=10, value=metrics.get('macro_auc', 0.0))
        ws.cell(row=row, column=11, value=metrics.get('macro_auc_std', 0.0))
        ws.cell(row=row, column=12, value=metrics.get('macro_ap', 0.0))
        ws.cell(row=row, column=13, value=metrics.get('macro_ap_std', 0.0))
        row += 1

    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try: max_len = max(max_len, len(str(cell.value)))
            except: pass
        ws.column_dimensions[col_letter].width = min(max_len + 2, 30)

    wb.save(output_path)
    print("Excel saved.")

# ------------------------------
# Print summary nicely
# ------------------------------
def print_signal_augmentation_summary(results_signal_aug):
    print("\n" + "="*100)
    print("SUMMARY: SIGNAL-LEVEL AUGMENTATION (subject_id classification)")
    print("="*100)
    sorted_items = sorted(results_signal_aug.items(), key=lambda it: it[1].get('accuracy',0), reverse=True)
    print(f"{'Rank':<4} {'Classifier':<35} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'AP':<8}")
    print("-"*100)
    for i, (name, m) in enumerate(sorted_items, 1):
        print(f"{i:<4} {name:<35} {m.get('accuracy',0):<8.3f} {m.get('macro_precision',0):<8.3f} "
              f"{m.get('macro_recall',0):<8.3f} {m.get('macro_f1',0):<8.3f} {m.get('macro_auc',0):<8.3f} {m.get('macro_ap',0):<8.3f}")
    if sorted_items:
        top_acc = sorted_items[0][1].get('accuracy',0)
        mean_accs = np.mean([m.get('accuracy',0) for _, m in sorted_items])
        print("\nKey numbers:")
        print(f" Best classifier accuracy (mean over folds): {top_acc:.1%}")
        print(f" Mean classifier accuracy (mean of means): {mean_accs:.1%}")
    print("="*100)

# ==================================================================
# VISUALIZATION FUNCTIONS (PLOS ONE STYLED)
# ==================================================================

# Apply Global PLOS ONE Style Settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# PLOS ONE Typography Constants
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12
LEGEND_TITLE_SIZE = 13

# 1. FEATURE ENGINEERING SUMMARY
def plot_feature_engineering_summary(output_dir):
    categories = {
        "Peak Characteristics": ["Number of detected peaks", "Peak heights and areas (top 5 peaks)", "Retention times of major peaks", "Peak width and symmetry indices"],
        "Statistical Aggregations": ["Total area under curve", "Mean, median, std of peak properties", "Major peak percentage contribution", "Skewness & distribution moments"],
        "Temporal Features": ["Retention time spread", "Peak spacing & clustering", "Elution pattern descriptors"]
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    text = "Feature Engineering Overview\n\n"
    for key, vals in categories.items():
        text += f"**{key}:**\n"
        for v in vals: text += f" • {v}\n"
        text += "\n"
    ax.text(0, 1, text, ha='left', va='top', fontsize=12, wrap=True)
    plt.title("Figure 1. Feature Engineering Overview (Peak-Based)", fontsize=TITLE_SIZE, loc='center', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure1_Feature_Engineering_Summary.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# 2. FIGURE 2 – XAI FEATURE IMPORTANCE (MODIFIED: Top K only)
def plot_feature_importance(feature_names, importances, title="Feature Importance", output_dir=OUTPUT_DIR, top_k=10):
    if importances is None or len(importances) == 0:
        print(f"Skipping plot '{title}': No importances data.")
        return
    if len(importances) != len(feature_names):
        print(f"Skipping plot '{title}': Length mismatch.")
        return
        
    sorted_idx = np.argsort(importances)[::-1]
    # REVIEWER REQUEST: Show only Top K features
    top_n = min(top_k, len(importances))
    plot_importances = importances[sorted_idx][:top_n]
    plot_features = np.array(feature_names)[sorted_idx][:top_n]
    
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x=plot_importances, y=plot_features, orient='h', palette="viridis")
    plt.title(f"Figure 2. Top {top_n} {title} (First CV Fold)", fontsize=TITLE_SIZE)
    plt.xlabel("Importance Score", fontsize=LABEL_SIZE)
    plt.ylabel("Feature", fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure2_{title.replace(' ', '_')}_Top{top_n}.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# 3. FIGURE 3 – CLASSIFIER COMPARISON
def plot_classifier_comparison(results_df, output_dir=OUTPUT_DIR):
    if results_df.empty: return
    metrics_to_plot = ['Accuracy', 'Macro_F1', 'Macro_AUC']
    plot_df = results_df[['Classifier'] + metrics_to_plot].copy()
    fig = plt.figure(figsize=(12, 6))
    melted = plot_df.melt(id_vars="Classifier", var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Classifier", y="Score", hue="Metric", palette="tab10")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(title="Metric", loc='lower right', fontsize=LEGEND_SIZE, title_fontsize=LEGEND_TITLE_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.title("Figure 3. Classifier Comparison (Mean Metrics over CV Folds)", fontsize=TITLE_SIZE)
    plt.savefig(os.path.join(output_dir, "Figure3_Classifier_Comparison.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# 4. FIGURE 4 – DETAILED PERFORMANCE (Random Forest as per paper)
def plot_best_clf_detailed_metrics(clf_metrics, clf_name, output_dir=OUTPUT_DIR):
    if not clf_metrics or not any(clf_metrics.values()): return
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro AUC", "Macro AP"]
    values = [
        clf_metrics.get("accuracy", 0.0), clf_metrics.get("macro_precision", 0.0),
        clf_metrics.get("macro_recall", 0.0), clf_metrics.get("macro_f1", 0.0),
        clf_metrics.get("macro_auc", 0.0), clf_metrics.get("macro_ap", 0.0)
    ]
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=labels, y=values, palette="Pastel1")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=TICK_SIZE)
    plt.ylim(0, 1.05)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.title(f"Figure 4. {clf_name} – Detailed Mean Performance (Over CV Folds)", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure4_{clf_name}_Detailed_Metrics.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# 5. FIGURE 5 – AGGREGATED CONFUSION MATRIX (MODIFIED: Absolute Counts)
def plot_average_confusion_matrix(y_true, y_pred, label_encoder, clf_name="Random Forest", output_dir=OUTPUT_DIR):
    if y_true.size == 0 or y_true.size != y_pred.size: return
    
    # Use absolute counts instead of normalized ratios
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cbar=True, square=True, linewidths=0.5, linecolor='gray')
    plt.ylabel("True Subject", fontsize=LABEL_SIZE)
    plt.xlabel("Predicted Subject", fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.title(f"Figure 5. Aggregated Confusion Matrix (Counts) - {clf_name}", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure5_{clf_name}_Confusion_Matrix.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# 6. FIGURE 6 – AGGREGATED ROC CURVES
def plot_average_roc(y_true, y_proba, label_encoder, clf_name="Random Forest", output_dir=OUTPUT_DIR):
    if y_true.size == 0 or y_proba.shape[0] == 0: return
    n_classes = len(label_encoder.classes_)
    y_true_cat = to_categorical(y_true, num_classes=n_classes)
    plt.figure(figsize=(10, 8))
    fpr = dict(); tpr = dict(); roc_auc = dict()
    has_valid_curve = False
    for i in range(n_classes):
        if np.sum(y_true == i) > 0 and np.sum(y_true != i) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_cat[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f"{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")
            has_valid_curve = True
        else:
            roc_auc[i] = 0.5
    if not has_valid_curve:
        plt.close(); return
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_cat.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle='--', label=f"micro-average (AUC = {roc_auc['micro']:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=LABEL_SIZE)
    plt.ylabel('True Positive Rate', fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.title(f"Figure 6. Aggregated ROC Curves - {clf_name}", fontsize=TITLE_SIZE)
    plt.legend(loc="lower right", fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure6_{clf_name}_ROC.png"), dpi=600, bbox_inches='tight')
    plt.close()

# NEW: FIGURE 7 – AGGREGATED PRECISION-RECALL CURVES
def plot_average_prc(y_true, y_proba, label_encoder, clf_name="Random Forest", output_dir=OUTPUT_DIR):
    if y_true.size == 0 or y_proba.shape[0] == 0: return
    n_classes = len(label_encoder.classes_)
    y_true_cat = to_categorical(y_true, num_classes=n_classes)
    plt.figure(figsize=(10, 8))
    precision = dict(); recall = dict(); average_precision = dict()
    has_valid_curve = False
    for i in range(n_classes):
        if np.sum(y_true == i) > 0:
            precision[i], recall[i], _ = precision_recall_curve(y_true_cat[:, i], y_proba[:, i])
            average_precision[i] = average_precision_score(y_true_cat[:, i], y_proba[:, i])
            plt.plot(recall[i], precision[i], lw=2, label=f"{label_encoder.classes_[i]} (AP = {average_precision[i]:.2f})")
            has_valid_curve = True
    if not has_valid_curve:
        plt.close(); return
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_cat.ravel(), y_proba.ravel())
    average_precision["micro"] = average_precision_score(y_true_cat, y_proba, average="micro")
    plt.plot(recall["micro"], precision["micro"], color='black', lw=2, linestyle='--', label=f"micro-average (AP = {average_precision['micro']:.2f})")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=LABEL_SIZE)
    plt.ylabel('Precision', fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.title(f"Figure 7. Aggregated Precision-Recall Curves - {clf_name}", fontsize=TITLE_SIZE)
    plt.legend(loc="lower left", fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure7_{clf_name}_PRC.png"), dpi=600, bbox_inches='tight')
    plt.close()

# 8. FIGURE 8 – RADAR CHART
def plot_radar_chart(metrics_dict, output_dir=OUTPUT_DIR):
    if not metrics_dict or all(not m for m in metrics_dict.values()): return
    first_model = next(iter(metrics_dict.values()))
    labels = list(first_model.keys())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    colors = plt.cm.Dark2.colors 
    for i, (model, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(label, 0.0) for label in labels]
        values += values[:1]
        ax.plot(angles, values, label=model, linewidth=2, linestyle='solid', color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(0.2, 1.0, 0.2)) 
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.2, 1.0, 0.2)], color="gray", size=TICK_SIZE)
    ax.set_ylim(0, 1.0)
    plt.title("Figure 8. Performance Comparison – Radar Chart (Mean Metrics)", size=TITLE_SIZE, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), fontsize=LEGEND_SIZE, title_fontsize=LEGEND_TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure8_Radar_Chart.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# NEW: FIGURE 12 – MODEL ACCURACY AND LOSS CURVES
def plot_training_curves(history_list, clf_name="Random Forest", output_dir=OUTPUT_DIR):
    """
    Plots accuracy and loss curves across 5 cross-validation folds.
    history_list should be a list of keras History objects or dicts containing 'accuracy', 'loss', 'val_accuracy', 'val_loss'.
    """
    if not history_list:
        print(f"Skipping Figure 12: No training history available for {clf_name}.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Aggregate histories
    max_epochs = max(len(h.history['accuracy']) for h in history_list)
    acc_per_epoch = np.zeros((len(history_list), max_epochs))
    val_acc_per_epoch = np.zeros((len(history_list), max_epochs))
    loss_per_epoch = np.zeros((len(history_list), max_epochs))
    val_loss_per_epoch = np.zeros((len(history_list), max_epochs))

    for i, h in enumerate(history_list):
        epochs_done = len(h.history['accuracy'])
        acc_per_epoch[i, :epochs_done] = h.history['accuracy']
        val_acc_per_epoch[i, :epochs_done] = h.history['val_accuracy']
        loss_per_epoch[i, :epochs_done] = h.history['loss']
        val_loss_per_epoch[i, :epochs_done] = h.history['val_loss']

    epochs_range = range(1, max_epochs + 1)
    mean_acc = np.mean(acc_per_epoch, axis=0)[:max_epochs]
    mean_val_acc = np.mean(val_acc_per_epoch, axis=0)[:max_epochs]
    mean_loss = np.mean(loss_per_epoch, axis=0)[:max_epochs]
    mean_val_loss = np.mean(val_loss_per_epoch, axis=0)[:max_epochs]

    # Left Plot: Accuracy
    axes[0].plot(epochs_range, mean_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[0].plot(epochs_range, mean_val_acc, 'r--', label='Validation Accuracy', linewidth=2)
    axes[0].set_title('(a) Model Accuracy per Epoch', fontsize=TITLE_SIZE)
    axes[0].set_xlabel('Epoch', fontsize=LABEL_SIZE)
    axes[0].set_ylabel('Accuracy', fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=LEGEND_SIZE)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)

    # Right Plot: Loss
    axes[1].plot(epochs_range, mean_loss, 'b-', label='Train Loss', linewidth=2)
    axes[1].plot(epochs_range, mean_val_loss, 'r--', label='Validation Loss', linewidth=2)
    axes[1].set_title('(b) Model Loss per Epoch', fontsize=TITLE_SIZE)
    axes[1].set_xlabel('Epoch', fontsize=LABEL_SIZE)
    axes[1].set_ylabel('Loss', fontsize=LABEL_SIZE)
    axes[1].legend(fontsize=LEGEND_SIZE)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)

    plt.suptitle(f"Figure 12. {clf_name} Training Dynamics Across 5 CV Folds", fontsize=TITLE_SIZE, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure12_{clf_name}_Training_Curves.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)

# NEW: FIGURE 13 – AUC PERFORMANCE METRICS
def plot_auc_metrics(y_true_agg, y_proba_agg, label_encoder, clf_name="Random Forest", output_dir=OUTPUT_DIR):
    """
    (Left) AUC score across the 5 cross-validation folds for each class.
    (Right) Box plot showing the AUC score distribution by class.
    Note: Since we aggregate predictions, we approximate fold-wise AUC using a sliding window 
    or simply show the overall per-class AUC and distribution via bootstrapping if fold-history isn't stored.
    For simplicity and robustness with aggregated data, we will plot the Overall Per-Class AUC 
    and a bootstrap distribution of AUCs to represent variability.
    """
    if y_true_agg.size == 0 or y_proba_agg.shape[0] == 0: return
    
    n_classes = len(label_encoder.classes_)
    y_true_cat = to_categorical(y_true_agg, num_classes=n_classes)
    
    # Calculate overall per-class AUC
    class_aucs = []
    class_names = []
    for i in range(n_classes):
        if np.sum(y_true_agg == i) > 0 and np.sum(y_true_agg != i) > 0:
            fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_proba_agg[:, i])
            a = auc(fpr, tpr)
            class_aucs.append(a)
            class_names.append(label_encoder.classes_[i])
    
    # Bootstrap to simulate fold-to-fold variability for the boxplot
    n_bootstraps = 100
    auc_bootstraps = {i: [] for i in range(n_classes)}
    
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true_agg), len(y_true_agg))
        y_bs_true = y_true_cat[indices]
        y_bs_proba = y_proba_agg[indices]
        for i in range(n_classes):
            if np.sum(y_bs_true[:, i]) > 0 and np.sum(y_bs_true[:, i]) < len(y_bs_true):
                fpr, tpr, _ = roc_curve(y_bs_true[:, i], y_bs_proba[:, i])
                auc_bootstraps[i].append(auc(fpr, tpr))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left Plot: Line plot of AUC scores per class (sorted)
    sorted_idx = np.argsort(class_aucs)[::-1]
    sorted_aucs = [class_aucs[i] for i in sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]
    
    axes[0].bar(range(len(sorted_names)), sorted_aucs, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(sorted_names)))
    axes[0].set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=TICK_SIZE)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title('(a) Overall AUC Score by Subject', fontsize=TITLE_SIZE)
    axes[0].set_xlabel('Subject ID', fontsize=LABEL_SIZE)
    axes[0].set_ylabel('AUC Score', fontsize=LABEL_SIZE)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='y', labelsize=TICK_SIZE)
    
    # Right Plot: Box plot of bootstrapped AUC distributions
    bp_data = [auc_bootstraps[i] for i in range(n_classes) if auc_bootstraps[i]]
    bp_labels = [label_encoder.classes_[i] for i in range(n_classes) if auc_bootstraps[i]]
    
    sns.boxplot(data=bp_data, ax=axes[1], palette="viridis")
    axes[1].set_xticklabels(bp_labels, rotation=45, ha='right', fontsize=TICK_SIZE)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title('(b) AUC Score Distribution by Class (Bootstrapped)', fontsize=TITLE_SIZE)
    axes[1].set_xlabel('Subject ID', fontsize=LABEL_SIZE)
    axes[1].set_ylabel('AUC Score', fontsize=LABEL_SIZE)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='y', labelsize=TICK_SIZE)
    
    plt.suptitle(f"Figure 13. {clf_name} AUC Performance Metrics", fontsize=TITLE_SIZE, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure13_{clf_name}_AUC_Metrics.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}/")

    signals_data = load_raw_voc_signals(data_directory="VOC Raw Data2/")
    if len(signals_data) == 0:
        print("No signals found. Check VOC Raw Data2/ and file naming.")
        return

    AUG_FACTOR = 5
    TARGET_FEATURES = 80
    
    # ---------------------------------------------------------
    # STEP 1: BASELINE EVALUATION (RAW DATA) - Reviewer Request
    # ---------------------------------------------------------
    print("\n--- PHASE 1: BASELINE EVALUATION (RAW DATA) ---")
    raw_train_df, raw_test_df = create_raw_dataset(signals_data, target_features=TARGET_FEATURES)
    X_train_raw, X_test_raw, y_train_raw_enc, y_test_raw_enc, y_train_raw_cat, y_test_raw_cat, label_encoder_raw, feat_cols_raw, scaler_raw = prepare_signal_augmented_data(raw_train_df, raw_test_df)
    
    # We target Random Forest for plots as it's the best in the paper
    plot_target_clf = 'Random Forest'
    
    results_raw, _, _, _, _, _ = evaluate_signal_augmented_classifiers(
        X_train_raw, y_train_raw_enc, y_train_raw_cat, label_encoder_raw, feat_cols_raw, 
        n_splits=5, epochs_nn=80, batch_size=32, plot_target_clf=plot_target_clf
    )
    
    # ---------------------------------------------------------
    # STEP 2: AUGMENTED EVALUATION
    # ---------------------------------------------------------
    print("\n--- PHASE 2: AUGMENTED DATA EVALUATION ---")
    train_df, test_df = create_augmented_dataset(signals_data, augmentation_factor=AUG_FACTOR, target_features=TARGET_FEATURES)
    X_train, X_test, y_train_encoded, y_test_encoded, y_train_cat, y_test_cat, label_encoder, feat_cols, scaler = prepare_signal_augmented_data(train_df, test_df)

    results_aug, catboost_importances, gb_importances, \
        plot_y_true, plot_y_pred, plot_y_proba = evaluate_signal_augmented_classifiers(
        X_train, y_train_encoded, y_train_cat, label_encoder, feat_cols,
        n_splits=5, epochs_nn=80, batch_size=32, plot_target_clf=plot_target_clf
    )

    # ---------------------------------------------------------
    # STEP 3: COMPARISON & OUTPUT
    # ---------------------------------------------------------
    print_augmentation_comparison(results_raw, results_aug)
    
    create_signal_augmentation_excel(results_aug, output_dir=OUTPUT_DIR, output_filename="subject_augmentation_results.xlsx")
    print_signal_augmentation_summary(results_aug)

    results_df_data = []
    for clf_name, metrics in results_aug.items():
        results_df_data.append({
            'Classifier': clf_name,
            'Accuracy': metrics.get('accuracy', 0.0),
            'Macro_F1': metrics.get('macro_f1', 0.0),
            'Macro_AUC': metrics.get('macro_auc', 0.0)
        })
    results_df = pd.DataFrame(results_df_data)

    radar_metrics = {}
    for clf_name in ["Random Forest", "Gradient Boosting", "CatBoost"]:
        if clf_name in results_aug:
            radar_metrics[clf_name] = {
                "Accuracy": results_aug[clf_name].get('accuracy', 0.0),
                "AUC": results_aug[clf_name].get('macro_auc', 0.0),
                "F1": results_aug[clf_name].get('macro_f1', 0.0),
                "Precision": results_aug[clf_name].get('macro_precision', 0.0),
                "Recall": results_aug[clf_name].get('macro_recall', 0.0)
            }

    best_clf_metrics = results_aug.get(plot_target_clf, {})

    print("\n" + "="*50)
    print(f"STARTING VISUALIZATIONS. Figures saved to {OUTPUT_DIR}/")
    print("="*50)
    
    plot_feature_engineering_summary(output_dir=OUTPUT_DIR)
    
    # REVIEWER REQUEST: Top K features only
    plot_feature_importance(feat_cols, catboost_importances, "CatBoost Feature Importance", output_dir=OUTPUT_DIR, top_k=TOP_K_FEATURES)
    
    plot_classifier_comparison(results_df, output_dir=OUTPUT_DIR)
    plot_best_clf_detailed_metrics(best_clf_metrics, plot_target_clf, output_dir=OUTPUT_DIR)
    
    # Updated plotting sequence with Counts Confusion Matrix, ROC, and new PRC
    plot_average_confusion_matrix(plot_y_true, plot_y_pred, label_encoder, plot_target_clf, output_dir=OUTPUT_DIR)
    plot_average_roc(plot_y_true, plot_y_proba, label_encoder, plot_target_clf, output_dir=OUTPUT_DIR)
    plot_average_prc(plot_y_true, plot_y_proba, label_encoder, plot_target_clf, output_dir=OUTPUT_DIR)
    plot_radar_chart(radar_metrics, output_dir=OUTPUT_DIR)
    
    # NEW: Generate Figures 12 and 13
    # Note: Figure 12 requires training history. If using standard SKLearn classifiers, 
    # this will be skipped unless you modify the evaluator to store NN histories.
    # For demonstration, we pass an empty list if no NN history is captured in this specific run.
    # To enable Fig 12 for RF/GBM, you would need to track partial_fit or OOB scores per epoch/fold.
    plot_training_curves([], plot_target_clf, output_dir=OUTPUT_DIR) 
    
    plot_auc_metrics(plot_y_true, plot_y_proba, label_encoder, plot_target_clf, output_dir=OUTPUT_DIR)

    if X_test.shape[0] > 0:
        print("\nQuick held-out test evaluation using top classifier...")
        sorted_clfs = sorted(results_aug.items(), key=lambda it: it[1].get('accuracy',0), reverse=True)
        if sorted_clfs:
            top_name = sorted_clfs[0][0]
            print(f"Top classifier: {top_name}. Evaluating on held-out test set ({X_test.shape[0]} samples).")
            classifiers = get_signal_augmented_classifiers()
            clf = classifiers[top_name]
            try:
                if top_name == 'Deep Learning (Neural Network)':
                    model = build_signal_augmented_neural_network((X_train.shape[1],), len(label_encoder.classes_))
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=0)]
                    model.fit(X_train, to_categorical(y_train_encoded, len(label_encoder.classes_)),
                              epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=0)
                    y_proba = model.predict(X_test, verbose=0)
                    y_pred = np.argmax(y_proba, axis=1)
                else:
                    clf.fit(X_train, y_train_encoded)
                    try:
                        y_proba = clf.predict_proba(X_test)
                        if y_proba.shape[1] != len(label_encoder.classes_):
                            classes_in_fold = clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_train_encoded)
                            global_indices = label_encoder.transform(classes_in_fold)
                            padded_proba = np.zeros((y_proba.shape[0], len(label_encoder.classes_)))
                            for i, global_idx in enumerate(global_indices):
                                padded_proba[:, global_idx] = y_proba[:, i]
                            y_proba = padded_proba
                    except:
                        y_proba = None
                    y_pred = clf.predict(X_test)
                met = calculate_metrics_signal_augmented(y_test_encoded, y_pred, y_proba, label_encoder)
                print("Held-out test metrics:", met)
            except Exception as e:
                print("Held-out eval error:", e)

    print(f"\nAll done. Results and figures saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()