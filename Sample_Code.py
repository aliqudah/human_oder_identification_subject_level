#!/usr/bin/env python3
"""
Enhanced VOC Data Classification Script
Includes comprehensive metrics, XAI methods, and Excel export functionality.
Formatted for PLOS ONE publication standards.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, auc,
                           precision_recall_fscore_support)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import warnings
warnings.filterwarnings('ignore')

# --- PLOS ONE Global Styling ---
OUTPUT_DIR = "sample_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 13

def load_data(filepath):
    """Load the feature matrix from the CSV file."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print("Data loaded successfully")
    print(f"Shape of the dataset: {df.shape}")
    return df

def prepare_data(df):
    """Prepare the data for modeling (feature scaling, label encoding)."""
    print("\nPreparing data for modeling...")
    
    feature_columns = df.columns.drop(['subject_id', 'sample_type', 'set_number', 'filename', 'label'])
    X = df[feature_columns].values
    y_labels = df['label'].values
    groups = df['subject_id'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    y_categorical = to_categorical(y_encoded)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Labels shape: {y_categorical.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    return X_scaled, y_categorical, y_encoded, groups, label_encoder, scaler, feature_columns

def build_model(input_shape, num_classes):
    """Build a simple feed-forward neural network for classification."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, class_names):
    """Calculate comprehensive classification metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    
    for i in range(len(class_names)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision[i]
        metrics[f'{class_name}_recall'] = recall[i]
        metrics[f'{class_name}_specificity'] = specificity[i]
        metrics[f'{class_name}_f1'] = f1[i]
    
    try:
        auc_scores = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None)
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_auc'] = auc_scores[i]
        metrics['macro_auc'] = np.mean(auc_scores)
    except:
        for class_name in class_names:
            metrics[f'{class_name}_auc'] = np.nan
        metrics['macro_auc'] = np.nan
    
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_specificity'] = np.mean(specificity)
    metrics['macro_f1'] = np.mean(f1)
    
    return metrics

def train_and_evaluate_comprehensive(X, y_categorical, y_encoded, groups, label_encoder, n_splits=5):
    """Train and evaluate the model with comprehensive metrics."""
    print(f"\nStarting {n_splits}-fold cross-validation with comprehensive metrics...")
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    class_names = label_encoder.classes_
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_categorical, groups), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]
        y_test_encoded = y_encoded[test_idx]
        
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])
        
        print(f"Training subjects: {len(train_subjects)}")
        print(f"Testing subjects: {len(test_subjects)}")
        
        model = build_model(input_shape=(X_train.shape[1],), num_classes=y_categorical.shape[1])
        
        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=16,
                            validation_split=0.1,
                            verbose=0)
        
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        metrics = calculate_comprehensive_metrics(y_test_encoded, y_pred, y_pred_probs, class_names)
        metrics['fold'] = fold
        metrics['history'] = history.history
        
        fold_results.append(metrics)
        
        all_y_true.extend(y_test_encoded)
        all_y_pred.extend(y_pred)
        all_y_pred_proba.extend(y_pred_probs)
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        
    print("\nCross-validation complete.")
    return fold_results, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_pred_proba), class_names

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save PLOS ONE styled Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=True, square=True, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Figure 1. Aggregated Confusion Matrix', fontsize=16, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure1_Confusion_Matrix.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to {OUTPUT_DIR}/Figure1_Confusion_Matrix.png")

def plot_roc_curves(y_true, y_pred_proba, class_names):
    """Plot and save PLOS ONE styled ROC Curves."""
    fig, ax = plt.subplots(figsize=(10, 8))
    n_classes = len(class_names)
    y_true_cat = to_categorical(y_true, num_classes=n_classes)
    
    for i in range(n_classes):
        if np.sum(y_true == i) > 0 and np.sum(y_true != i) > 0:
            fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
            
    fpr_micro, tpr_micro, _ = roc_curve(y_true_cat.ravel(), y_pred_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='black', lw=2, linestyle='--', label=f'Micro-average (AUC = {roc_auc_micro:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2. Aggregated ROC Curves', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure2_ROC_Curves.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC Curves to {OUTPUT_DIR}/Figure2_ROC_Curves.png")

def create_metrics_excel(fold_results, class_names, output_path):
    """Create a comprehensive Excel file with all metrics."""
    print("\nCreating comprehensive metrics Excel file...")
    
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    
    summary_sheet = wb.create_sheet("Summary")
    detailed_sheet = wb.create_sheet("Detailed_Results")
    per_class_sheet = wb.create_sheet("Per_Class_Metrics")
    
    # PLOS ONE Style Fonts for Excel
    header_font = Font(name='Arial', bold=True, color="FFFFFF", size=11)
    data_font = Font(name='Arial', size=10)
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    center_alignment = Alignment(horizontal="center", vertical="center")
    
    # Summary Sheet
    summary_headers = ["Metric", "Mean", "Std", "Min", "Max"]
    for col, header in enumerate(summary_headers, 1):
        cell = summary_sheet.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
    
    metrics_to_summarize = ['accuracy', 'macro_precision', 'macro_recall', 'macro_specificity', 'macro_f1', 'macro_auc']
    for class_name in class_names:
        metrics_to_summarize.extend([f'{class_name}_precision', f'{class_name}_recall', f'{class_name}_specificity', f'{class_name}_auc'])
    
    row = 2
    for metric in metrics_to_summarize:
        values = [result[metric] for result in fold_results if metric in result and not np.isnan(result[metric])]
        if values:
            summary_sheet.cell(row=row, column=1, value=metric).font = data_font
            summary_sheet.cell(row=row, column=2, value=np.mean(values)).font = data_font
            summary_sheet.cell(row=row, column=3, value=np.std(values)).font = data_font
            summary_sheet.cell(row=row, column=4, value=np.min(values)).font = data_font
            summary_sheet.cell(row=row, column=5, value=np.max(values)).font = data_font
            row += 1
    
    # Detailed Results Sheet
    detailed_headers = ["Fold"] + metrics_to_summarize
    for col, header in enumerate(detailed_headers, 1):
        cell = detailed_sheet.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
    
    for row, result in enumerate(fold_results, 2):
        detailed_sheet.cell(row=row, column=1, value=result['fold']).font = data_font
        for col, metric in enumerate(metrics_to_summarize, 2):
            value = result.get(metric, np.nan)
            if not np.isnan(value):
                detailed_sheet.cell(row=row, column=col, value=value).font = data_font
    
    # Per-Class Metrics Sheet
    class_headers = ["Class", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "AUC"]
    for col, header in enumerate(class_headers, 1):
        cell = per_class_sheet.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
    
    for row, class_name in enumerate(class_names, 2):
        per_class_sheet.cell(row=row, column=1, value=class_name).font = data_font
        
        precision_values = [result[f'{class_name}_precision'] for result in fold_results]
        recall_values = [result[f'{class_name}_recall'] for result in fold_results]
        specificity_values = [result[f'{class_name}_specificity'] for result in fold_results]
        f1_values = [result[f'{class_name}_f1'] for result in fold_results]
        auc_values = [result[f'{class_name}_auc'] for result in fold_results if not np.isnan(result[f'{class_name}_auc'])]
        
        per_class_sheet.cell(row=row, column=2, value=np.mean(precision_values)).font = data_font
        per_class_sheet.cell(row=row, column=3, value=np.mean(recall_values)).font = data_font
        per_class_sheet.cell(row=row, column=4, value=np.mean(specificity_values)).font = data_font
        per_class_sheet.cell(row=row, column=5, value=np.mean(f1_values)).font = data_font
        if auc_values:
            per_class_sheet.cell(row=row, column=6, value=np.mean(auc_values)).font = data_font
    
    for sheet in [summary_sheet, detailed_sheet, per_class_sheet]:
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 25)
            sheet.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(output_path)
    print(f"Metrics Excel file saved to: {output_path}")

def perform_xai_analysis(X, y_encoded, feature_columns, label_encoder, sample_size=100):
    """Perform XAI analysis using SHAP and LIME."""
    print("\nPerforming XAI (Explainable AI) analysis...")
    
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y_encoded[indices]
    else:
        X_sample = X
        y_sample = y_encoded
    
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=20)
    rf_model.fit(X_sample, y_sample)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance (PLOS ONE Style)
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'], color='#2ca02c', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance Score', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    plt.title('Figure 3. Top 10 Most Important Features (Random Forest)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Feature_Importance.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    # SHAP analysis
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(12, 8))
        if len(label_encoder.classes_) > 2:
            shap.summary_plot(shap_values[0], X_sample, feature_names=feature_columns, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, show=False)
        plt.title('Figure 4. SHAP Summary Plot', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure4_SHAP_Summary.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print("SHAP analysis completed successfully")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    # LIME analysis
    try:
        explainer = lime_tabular.LimeTabularExplainer(
            X_sample,
            feature_names=feature_columns,
            class_names=label_encoder.classes_,
            mode='classification'
        )
        print("LIME analysis completed successfully")
    except Exception as e:
        print(f"LIME analysis failed: {e}")
    
    return feature_importance

def main():
    """Main function to run the enhanced classification task."""
    # Note: Update this path to your actual feature matrix CSV
    filepath = "voc_feature_matrix.csv" 
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}. Please update the path.")
        return

    df = load_data(filepath)
    X, y_categorical, y_encoded, groups, label_encoder, scaler, feature_columns = prepare_data(df)
    
    fold_results, y_true, y_pred, y_pred_proba, class_names = train_and_evaluate_comprehensive(
        X, y_categorical, y_encoded, groups, label_encoder
    )
    
    # Generate PLOS ONE styled plots
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curves(y_true, y_pred_proba, class_names)
    
    create_metrics_excel(fold_results, class_names, os.path.join(OUTPUT_DIR, "voc_classification_metrics.xlsx"))
    feature_importance = perform_xai_analysis(X, y_encoded, feature_columns, label_encoder)
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
    avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
    std_accuracy = np.std([res['accuracy'] for res in fold_results])
    
    print(f"\n" + "="*50)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*50)
    print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f} +/- {std_accuracy:.4f}")
    print(f"\nAll figures and tables saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()