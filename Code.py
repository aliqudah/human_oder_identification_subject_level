import os
import glob
import numpy as np
import pandas as pd
import warnings
from math import pi

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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- GLOBAL CONFIGURATION ---
OUTPUT_DIR = "subject_results"
TARGET_COLUMN = 'subject_id'
TOP_K_FEATURES = 10  # REVIEWER REQUEST: Limit feature importance to Top K features
# ----------------------------

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
    """Parse a VOC text file. Returns (retention_times, intensities) arrays (may be empty)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return np.array([]), np.array([])
    
    data_started = False
    data_pairs = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('---'):
            data_started = True
            continue
        if not data_started:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        numeric = []
        for p in parts:
            try:
                numeric.append(float(p))
            except:
                continue
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

    if len(rt) > 1:
        try:
            features['total_area'] = float(np.trapezoid(it, rt)) # NumPy 2.0+
        except AttributeError:
            features['total_area'] = float(np.trapz(it, rt))     # NumPy < 2.0
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
    features['width_mean'] = float(np.mean(widths)) if widths.size > 0 else 0.0
    features['width_std'] = float(np.std(widths)) if widths.size > 0 else 0.0
    features['peak_prominence_mean'] = float(np.mean(props['prominences'])) if (peaks.size > 0 and 'prominences' in props) else 0.0

    if peaks.size > 0:
        top_idx = np.argsort(it_smooth[peaks])[-5:][::-1]
    else:
        top_idx = np.array([], dtype=int)
        
    for i in range(5):
        if i < len(top_idx):
            p = peaks[top_idx[i]]
            features[f'top{i+1}_intensity'] = float(it_smooth[p])
            features[f'top{i+1}_rt'] = float(rt[p])
            features[f'top{i+1}_width'] = float(widths[top_idx[i]]) if widths.size > top_idx[i] else 0.0
        else:
            features[f'top{i+1}_intensity'] = 0.0
            features[f'top{i+1}_rt'] = 0.0
            features[f'top{i+1}_width'] = 0.0

    features['peak_density'] = float(len(peaks) / (rt.max() - rt.min() + 1e-8)) if len(rt) > 1 else 0.0
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
# NEW: Create RAW dataset (NO augmentation) for baseline comparison
# ------------------------------
def create_raw_dataset(signals_data, target_features=80):
    """Build raw dataset with subject-level train/test split, NO augmentation."""
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
        current = len(items) 
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
# Prepare data (X, y, scaling, encoding)
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
    
    print("\nPrepared signal-augmented data:")
    print(f" X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, Classes: {label_encoder.classes_}")
    return (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded,
            y_train_categorical, y_test_categorical, label_encoder, feature_columns, scaler)

# ------------------------------
# Classifiers dictionary
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
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
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
# Metric calculation (per-fold)
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
            auc_score = roc_auc_score(y_true_cat, y_proba, average='macro', multi_class='ovr')
            metrics['macro_auc'] = float(auc_score)
        else:
            metrics['macro_auc'] = 0.5
    except Exception:
        metrics['macro_auc'] = 0.5
    return metrics

# ------------------------------
# Stratified K-Fold evaluation
# ------------------------------
def evaluate_signal_augmented_classifiers(X, y, y_cat, label_encoder, feature_names, n_splits=5, epochs_nn=100, batch_size=32):
    print("\nEvaluating classifiers with StratifiedKFold ...")
    classifiers = get_signal_augmented_classifiers()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {name: [] for name in classifiers.keys()}
    catboost_importances = None
    gb_importances = None
    
    clf_aggregation_data = {name: {'y_true': np.array([], dtype=int), 
                                   'y_pred': np.array([], dtype=int), 
                                   'y_proba': np.empty((0, len(label_encoder.classes_)))} 
                            for name in classifiers.keys()}
                            
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f" Fold {fold_idx}/{n_splits}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        y_tr_cat = to_categorical(y_tr, num_classes=len(label_encoder.classes_))
        y_val_cat = to_categorical(y_val, num_classes=len(label_encoder.classes_))
        
        for clf_name, clf in classifiers.items():
            try:
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
                        y_pred_proba = clf.predict_proba(X_val)
                        y_pred_proba = np.asarray(y_pred_proba)
                        if y_pred_proba.ndim == 1:
                            y_pred_proba = y_pred_proba.reshape(-1, 1)
                        if y_pred_proba.shape[1] != len(label_encoder.classes_):
                            classes_in_fold = clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_tr)
                            global_indices = label_encoder.transform(classes_in_fold)
                            padded_proba = np.zeros((y_pred_proba.shape[0], len(label_encoder.classes_)))
                            for i, global_idx in enumerate(global_indices):
                                padded_proba[:, global_idx] = y_pred_proba[:, i]
                            y_pred_proba = padded_proba
                    except Exception:
                        y_pred_proba = None
                        
                metrics = calculate_metrics_signal_augmented(y_val, y_pred, y_pred_proba, label_encoder)
                results[clf_name].append(metrics)
                
                if fold_idx == 1:
                    if clf_name == 'CatBoost' and hasattr(clf, 'get_feature_importance'):
                        catboost_importances = clf.get_feature_importance()
                    elif clf_name == 'Gradient Boosting' and hasattr(clf, 'feature_importances_'):
                        gb_importances = clf.feature_importances_
                        
                clf_aggregation_data[clf_name]['y_true'] = np.concatenate([clf_aggregation_data[clf_name]['y_true'], y_val])
                clf_aggregation_data[clf_name]['y_pred'] = np.concatenate([clf_aggregation_data[clf_name]['y_pred'], y_pred])
                if y_pred_proba is not None and y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == len(label_encoder.classes_):
                    clf_aggregation_data[clf_name]['y_proba'] = np.concatenate([clf_aggregation_data[clf_name]['y_proba'], y_pred_proba], axis=0)
            except Exception as e:
                print(f"  {clf_name} fold {fold_idx} error: {e}")
                continue
                
    aggregated = {}
    for clf_name, folds in results.items():
        if not folds:
            continue
        keys = folds[0].keys()
        mean_metrics = {k: float(np.mean([f[k] for f in folds])) for k in keys}
        std_metrics = {f"{k}_std": float(np.std([f[k] for f in folds])) for k in keys}
        aggregated[clf_name] = {**mean_metrics, **std_metrics}
        
    best_clf_name = max(aggregated.items(), key=lambda x: x[1].get('accuracy', 0.0))[0] if aggregated else None
    print(f"\nBest classifier determined for plotting: {best_clf_name} (Accuracy: {aggregated.get(best_clf_name, {}).get('accuracy', 0.0):.3f})")
    
    best_clf_data = clf_aggregation_data.get(best_clf_name, {'y_true': np.array([]), 'y_pred': np.array([]), 'y_proba': np.empty((0, len(label_encoder.classes_)))})
    return (aggregated, catboost_importances, gb_importances, 
            best_clf_name, best_clf_data['y_true'], best_clf_data['y_pred'], best_clf_data['y_proba'])

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
               'Macro_Recall_mean', 'Macro_Recall_std', 'Macro_F1_mean', 'Macro_F1_std', 'Macro_AUC_mean', 'Macro_AUC_std']
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
        row += 1
        
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value)))
            except:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 2, 30)
    wb.save(output_path)
    print("Excel saved.")

# ------------------------------
# Print summary nicely
# ------------------------------
def print_signal_augmentation_summary(results_signal_aug):
    print("\n" + "= "*50)
    print("SUMMARY: SIGNAL-LEVEL AUGMENTATION (subject_id classification)")
    print("= "*50)
    sorted_items = sorted(results_signal_aug.items(), key=lambda it: it[1].get('accuracy',0), reverse=True)
    print(f"{'Rank': <4} {'Classifier': <35} {'Acc_mean': <10} {'Prec': <10} {'Rec': <10} {'F1': <10} {'AUC': <10}")
    print("-"*100)
    for i, (name, m) in enumerate(sorted_items, 1):
        print(f"{i: <4} {name: <35} {m.get('accuracy',0): <10.3f} {m.get('macro_precision',0): <10.3f}  "
              f"{m.get('macro_recall',0): <10.3f} {m.get('macro_f1',0): <10.3f} {m.get('macro_auc',0): <10.3f}")
    if sorted_items:
        top_acc = sorted_items[0][1].get('accuracy',0)
        mean_accs = np.mean([m.get('accuracy',0) for _, m in sorted_items])
        print("\nKey numbers:")
        print(f" Best classifier accuracy (mean over folds): {top_acc:.1%}")
        print(f" Mean classifier accuracy (mean of means): {mean_accs:.1%}")
    print("= "*50)

# ------------------------------
# VISUALIZATION FUNCTIONS
# ------------------------------
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
        for v in vals:
            text += f" • {v}\n"
        text += "\n"
    ax.text(0, 1, text, ha='left', va='top', fontsize=12, wrap=True)
    plt.title("Figure 1. Feature Engineering Overview (Peak-Based)", fontsize=14, loc='center', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure1_Feature_Engineering_Summary.png"), dpi=300)
    plt.close(fig)

# MODIFIED: Top K Feature Importance
def plot_feature_importance(feature_names, importances, title="Feature Importance", output_dir=OUTPUT_DIR, top_k=10):
    if importances is None or len(importances) == 0:
        print(f"Skipping plot '{title}': No importances data.")
        return
    if len(importances) != len(feature_names):
        print(f"Skipping plot '{title}': Length mismatch.")
        return
        
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(top_k, len(importances))
    plot_importances = importances[sorted_idx][:top_n]
    plot_features = np.array(feature_names)[sorted_idx][:top_n]
    
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x=plot_importances, y=plot_features, orient='h', palette="viridis")
    plt.title(f"Top {top_n} {title} (First CV Fold)", fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure2_{title.replace(' ', '_')}_Top{top_n}.png"), dpi=300)
    plt.close(fig)

def plot_classifier_comparison(results_df, output_dir=OUTPUT_DIR):
    if results_df.empty:
        print("Skipping Classifier Comparison plot: Results DataFrame is empty.")
        return
    metrics_to_plot = ['Accuracy', 'Macro_F1', 'Macro_AUC']
    plot_df = results_df[['Classifier'] + metrics_to_plot].copy()
    fig = plt.figure(figsize=(12, 6))
    melted = plot_df.melt(id_vars="Classifier", var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Classifier", y="Score", hue="Metric", palette="tab10")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(title="Metric", loc='lower right')
    plt.tight_layout()
    plt.title("Figure 3. Classifier Comparison (Mean Metrics over CV Folds)", fontsize=16)
    plt.savefig(os.path.join(output_dir, "Figure3_Classifier_Comparison.png"), dpi=300)
    plt.close(fig)

def plot_best_clf_detailed_metrics(clf_metrics, clf_name, output_dir=OUTPUT_DIR):
    if not clf_metrics or not any(clf_metrics.values()):
        print(f"Skipping {clf_name} Detailed Metrics plot: No metrics data.")
        return
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro AUC"]
    values = [
        clf_metrics.get("accuracy", 0.0),
        clf_metrics.get("macro_precision", 0.0),
        clf_metrics.get("macro_recall", 0.0),
        clf_metrics.get("macro_f1", 0.0),
        clf_metrics.get("macro_auc", 0.0)
    ]
    fig = plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=labels, y=values, palette="Pastel1")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.ylim(0, 1.05)
    plt.title(f"Figure 4. {clf_name} – Detailed Mean Performance (Over CV Folds)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure4_{clf_name}_Detailed_Metrics.png"), dpi=300)
    plt.close(fig)

# MODIFIED: Absolute Counts Confusion Matrix
def plot_average_confusion_matrix(y_true, y_pred, label_encoder, clf_name, output_dir=OUTPUT_DIR):
    if y_true.size == 0 or y_true.size != y_pred.size:
        print(f"Skipping Confusion Matrix for {clf_name}: Inconsistent or empty aggregated data.")
        return
    # Use absolute counts instead of normalized ratios
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cbar=True, square=True, linewidths=0.5, linecolor='gray')
    plt.ylabel("True Subject", fontsize=12)
    plt.xlabel("Predicted Subject", fontsize=12)
    plt.title(f"Figure 5. Aggregated Confusion Matrix (Counts) - {clf_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure5_{clf_name}_Confusion_Matrix.png"), dpi=300)
    plt.close(fig)

# ROC Curves
def plot_average_roc(y_true, y_proba, label_encoder, clf_name, output_dir=OUTPUT_DIR):
    if y_true.size == 0 or y_proba.shape[0] == 0:
        print(f"Skipping ROC plot for {clf_name}: empty y_true or y_proba.")
        return
    n_classes = len(label_encoder.classes_)
    y_true_cat = to_categorical(y_true, num_classes=n_classes)
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
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
        print(f"Skipping ROC plot for {clf_name}: Not enough variability in aggregated data to compute curves.")
        plt.close()
        return
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_cat.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle='--', label=f"micro-average (AUC = {roc_auc['micro']:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"Figure 6. Aggregated ROC Curves - {clf_name}", fontsize=16)
    plt.legend(loc="lower right")  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure6_{clf_name}_ROC.png"), dpi=300)
    plt.close()

# NEW: Precision-Recall Curves
def plot_average_prc(y_true, y_proba, label_encoder, clf_name, output_dir=OUTPUT_DIR):
    if y_true.size == 0 or y_proba.shape[0] == 0:
        print(f"Skipping PRC plot for {clf_name}: empty y_true or y_proba.")
        return
    n_classes = len(label_encoder.classes_)
    y_true_cat = to_categorical(y_true, num_classes=n_classes)
    plt.figure(figsize=(10, 8))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    has_valid_curve = False
    
    for i in range(n_classes):
        if np.sum(y_true == i) > 0:
            precision[i], recall[i], _ = precision_recall_curve(y_true_cat[:, i], y_proba[:, i])
            average_precision[i] = average_precision_score(y_true_cat[:, i], y_proba[:, i])
            plt.plot(recall[i], precision[i], lw=2, label=f"{label_encoder.classes_[i]} (AP = {average_precision[i]:.2f})")
            has_valid_curve = True
            
    if not has_valid_curve:
        print(f"Skipping PRC plot for {clf_name}: Not enough variability in aggregated data to compute curves.")
        plt.close()
        return
        
    # Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_cat.ravel(), y_proba.ravel())
    average_precision["micro"] = average_precision_score(y_true_cat, y_proba, average="micro")
    plt.plot(recall["micro"], precision["micro"], color='black', lw=2, linestyle='--', label=f"micro-average (AP = {average_precision['micro']:.2f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f"Figure 7. Aggregated Precision-Recall Curves - {clf_name}", fontsize=16)
    plt.legend(loc="lower left")  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure7_{clf_name}_PRC.png"), dpi=300)
    plt.close()

def plot_radar_chart(metrics_dict, output_dir=OUTPUT_DIR):
    if not metrics_dict or all(not m for m in metrics_dict.values()):
        print("Skipping Radar Chart: Metrics dictionary is empty or contains no data.")
        return
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
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0.2, 1.0, 0.2)) 
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.2, 1.0, 0.2)], color="gray", size=10)
    ax.set_ylim(0, 1.0)
    plt.title("Figure 8. Performance Comparison – Radar Chart (Mean Metrics)", size=16, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure8_Radar_Chart.png"), dpi=300)
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
        
    TARGET_FEATURES = 80
    AUG_FACTOR = 5
   

    # ---------------------------------------------------------
    # STEP 1: BASELINE EVALUATION (RAW DATA) - Reviewer Request
    # ---------------------------------------------------------
    print("\n--- PHASE 1: BASELINE EVALUATION (RAW DATA) ---")
    raw_train_df, raw_test_df = create_raw_dataset(signals_data, target_features=TARGET_FEATURES)
    X_train_raw, X_test_raw, y_train_raw_enc, y_test_raw_enc, y_train_raw_cat, y_test_raw_cat, label_encoder_raw, feat_cols_raw, scaler_raw = prepare_signal_augmented_data(raw_train_df, raw_test_df)
    
    results_raw, _, _, best_clf_raw, _, _, _ = evaluate_signal_augmented_classifiers(
        X_train_raw, y_train_raw_enc, y_train_raw_cat, label_encoder_raw, feat_cols_raw, n_splits=5, epochs_nn=80, batch_size=32
    )
    
    # ---------------------------------------------------------
    # STEP 2: AUGMENTED EVALUATION
    # ---------------------------------------------------------
    print("\n--- PHASE 2: AUGMENTED DATA EVALUATION ---")
    train_df, test_df = create_augmented_dataset(signals_data, augmentation_factor=AUG_FACTOR, target_features=TARGET_FEATURES)
    X_train, X_test, y_train_encoded, y_test_encoded, y_train_cat, y_test_cat, label_encoder, feat_cols, scaler = prepare_signal_augmented_data(train_df, test_df)
    
    results_aug, catboost_importances, gb_importances, best_clf_name, best_y_true, best_y_pred, best_y_proba = evaluate_signal_augmented_classifiers(
        X_train, y_train_encoded, y_train_cat, label_encoder, feat_cols, n_splits=5, epochs_nn=80, batch_size=32
    )
    
    # ---------------------------------------------------------
    # STEP 3: COMPARISON & OUTPUT
    # ---------------------------------------------------------
    print_augmentation_comparison(results_raw, results_aug)
    
    create_signal_augmentation_excel(results_aug, output_dir=OUTPUT_DIR, output_filename="subject_augmentation_results.xlsx")
    print_signal_augmentation_summary(results_aug)
    
    results_df_data = [{'Classifier': clf, 'Accuracy': m.get('accuracy', 0.0), 'Macro_F1': m.get('macro_f1', 0.0), 'Macro_AUC': m.get('macro_auc', 0.0)} for clf, m in results_aug.items()]
    results_df = pd.DataFrame(results_df_data)
    
    radar_metrics = {}
    for clf_name in ["Gradient Boosting", "CatBoost", "Deep Learning (Neural Network)"]:
        if clf_name in results_aug:
            radar_metrics[clf_name] = {
                "Accuracy": results_aug[clf_name].get('accuracy', 0.0),
                "AUC": results_aug[clf_name].get('macro_auc', 0.0),
                "F1": results_aug[clf_name].get('macro_f1', 0.0),
                "Precision": results_aug[clf_name].get('macro_precision', 0.0),
                "Recall": results_aug[clf_name].get('macro_recall', 0.0)
            }
            
    print("\n" + "="*50)
    print(f"STARTING VISUALIZATIONS. Figures saved to {OUTPUT_DIR}/")
    print("="*50)
    
    plot_feature_engineering_summary(output_dir=OUTPUT_DIR)
    # REVIEWER REQUEST: Top K features only
    plot_feature_importance(feat_cols, catboost_importances, "CatBoost Feature Importance", output_dir=OUTPUT_DIR, top_k=TOP_K_FEATURES)
    plot_classifier_comparison(results_df, output_dir=OUTPUT_DIR)
    plot_best_clf_detailed_metrics(results_aug[best_clf_name], best_clf_name, output_dir=OUTPUT_DIR)
    
    # Updated plotting sequence with Counts Confusion Matrix, ROC, and new PRC
    plot_average_confusion_matrix(best_y_true, best_y_pred, label_encoder, best_clf_name, output_dir=OUTPUT_DIR)
    plot_average_roc(best_y_true, best_y_proba, label_encoder, best_clf_name, output_dir=OUTPUT_DIR)
    plot_average_prc(best_y_true, best_y_proba, label_encoder, best_clf_name, output_dir=OUTPUT_DIR)
    plot_radar_chart(radar_metrics, output_dir=OUTPUT_DIR)
    
    print(f"\nAll done. Results, comparison tables, and figures saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
