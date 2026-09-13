import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename to extract sample type and set number.
    Expected format: <Sample Type><Set Number>.TXT
    Where Sample Type: A (FA - Arm sample using Funnel apparatus), G (Gauze swab), S (Suit)
    """
    name = filename.replace('.TXT', '')
    if len(name) >= 2:
        sample_type = name[0]
        set_number = name[1:]
        return sample_type, set_number
    return None, None

def load_voc_file_flexible(filepath):
    """
    Load a VOC data file with flexible parsing to handle variable column formats.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data_lines = lines[2:]
        parsed_data = []
        
        for line_num, line in enumerate(data_lines, start=3):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 10:
                continue
                
            try:
                peak_num = int(parts[0])
                retention_time = float(parts[1])
                
                scan_start_idx = 2
                height_idx = None
                
                for i in range(scan_start_idx, len(parts)):
                    if parts[i] in ['BV', 'VV', 'VB', 'BB'] or (len(parts[i]) >= 2 and parts[i][:2] in ['BV', 'VV', 'VB', 'BB']):
                        height_idx = i + 1
                        break
                    else:
                        try:
                            int(parts[i])
                        except ValueError:
                            height_idx = i + 1
                            break
                
                if height_idx is None or height_idx >= len(parts):
                    continue
                    
                height = float(parts[height_idx])
                area1 = float(parts[height_idx + 1]) if height_idx + 1 < len(parts) else 0
                area2 = float(parts[height_idx + 2]) if height_idx + 2 < len(parts) else 0
                
                percent_max_str = parts[height_idx + 3] if height_idx + 3 < len(parts) else "0%"
                percent_total_str = parts[height_idx + 4] if height_idx + 4 < len(parts) else "0%"
                
                percent_max = float(percent_max_str.replace('%', ''))
                percent_total = float(percent_total_str.replace('%', ''))
                
                parsed_data.append({
                    'peak_num': peak_num,
                    'retention_time': retention_time,
                    'height': height,
                    'area1': area1,
                    'area2': area2,
                    'percent_max': percent_max,
                    'percent_total': percent_total
                })
                
            except (ValueError, IndexError):
                continue
        
        if parsed_data:
            return pd.DataFrame(parsed_data)
        else:
            return None
            
    except Exception:
        return None

def analyze_dataset_structure():
    """Analyze the complete dataset structure and prepare for classification."""
    base_dir = Path('VOC Raw Data2/')
    
    subject_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item.name.startswith('Subject') or item.name.startswith('Suject')):
            subject_dirs.append(item)
    
    subject_dirs.sort()
    print(f"Found {len(subject_dirs)} subject directories")
    
    dataset_info = {
        'subjects': [],
        'sample_types': set(),
        'files_per_subject': defaultdict(list),
        'total_files': 0,
        'successful_files': 0,
        'failed_files': 0
    }
    
    all_data = []
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        dataset_info['subjects'].append(subject_id)
        
        txt_files = list(subject_dir.glob('*.TXT'))
        dataset_info['files_per_subject'][subject_id] = [f.name for f in txt_files]
        dataset_info['total_files'] += len(txt_files)
        
        for txt_file in txt_files:
            sample_type, set_number = parse_filename(txt_file.name)
            if sample_type:
                dataset_info['sample_types'].add(sample_type)
                
                voc_data = load_voc_file_flexible(txt_file)
                if voc_data is not None and len(voc_data) > 0:
                    voc_data['subject_id'] = subject_id
                    voc_data['sample_type'] = sample_type
                    voc_data['set_number'] = set_number
                    voc_data['filename'] = txt_file.name
                    
                    all_data.append(voc_data)
                    dataset_info['successful_files'] += 1
                else:
                    dataset_info['failed_files'] += 1
    
    print(f"Dataset Summary: {dataset_info['successful_files']} successful, {dataset_info['failed_files']} failed.")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        os.makedirs('Data_Figures', exist_ok=True)
        combined_df.to_csv('Data_Figures/voc_combined_dataset.csv', index=False)
        return combined_df, dataset_info
    else:
        return None, dataset_info

def analyze_classification_task(df):
    """Analyze the classification task and data distribution."""
    print("\n" + "="*50)
    print("CLASSIFICATION TASK ANALYSIS")
    print("="*50)
    
    sample_type_counts = df['sample_type'].value_counts()
    print("\nSample Type Distribution:\n", sample_type_counts)
    
    subject_counts = df['subject_id'].value_counts()
    print(f"\nSubjects: {len(subject_counts)}")
    print(f"Samples per subject range: {subject_counts.min()} - {subject_counts.max()}")
    
    return sample_type_counts, subject_counts

def create_feature_matrix(df):
    """Create feature matrices for classification."""
    print("\n" + "="*50)
    print("FEATURE MATRIX CREATION")
    print("="*50)
    
    sample_groups = df.groupby(['subject_id', 'sample_type', 'set_number', 'filename'])
    sample_features, sample_labels, sample_metadata = [], [], []
    
    for (subject_id, sample_type, set_number, filename), group in sample_groups:
        features = {
            'num_peaks': len(group),
            'total_area': group['area1'].sum(),
            'max_height': group['height'].max(),
            'mean_height': group['height'].mean(),
            'std_height': group['height'].std() if len(group) > 1 else 0,
            'max_retention_time': group['retention_time'].max(),
            'mean_retention_time': group['retention_time'].mean(),
            'std_retention_time': group['retention_time'].std() if len(group) > 1 else 0,
            'total_percent_max': group['percent_max'].sum(),
            'max_percent_max': group['percent_max'].max(),
            'mean_percent_max': group['percent_max'].mean(),
            'total_percent_total': group['percent_total'].sum(),
        }
        
        top_peaks = group.nlargest(10, 'height')
        for i, (_, peak) in enumerate(top_peaks.iterrows()):
            features[f'top{i+1}_height'] = peak['height']
            features[f'top{i+1}_retention_time'] = peak['retention_time']
            features[f'top{i+1}_area'] = peak['area1']
            features[f'top{i+1}_percent_max'] = peak['percent_max']
        
        for i in range(len(top_peaks), 10):
            features[f'top{i+1}_height'] = 0
            features[f'top{i+1}_retention_time'] = 0
            features[f'top{i+1}_area'] = 0
            features[f'top{i+1}_percent_max'] = 0
        
        rt_values = group['retention_time'].values
        if len(rt_values) > 0:
            features['rt_range'] = rt_values.max() - rt_values.min()
            features['rt_median'] = np.median(rt_values)
            features['rt_q25'] = np.percentile(rt_values, 25)
            features['rt_q75'] = np.percentile(rt_values, 75)
        else:
            features['rt_range'] = features['rt_median'] = features['rt_q25'] = features['rt_q75'] = 0
        
        sample_features.append(features)
        sample_labels.append(sample_type)
        sample_metadata.append({'subject_id': subject_id, 'sample_type': sample_type, 'set_number': set_number, 'filename': filename})
    
    features_df = pd.DataFrame(sample_features)
    labels_df = pd.DataFrame(sample_labels, columns=['label'])
    metadata_df = pd.DataFrame(sample_metadata)
    final_df = pd.concat([metadata_df, features_df, labels_df], axis=1)
    
    final_df.to_csv('Data_Figures/voc_feature_matrix.csv', index=False)
    print(f"Created feature matrix with {len(final_df)} samples and {len(features_df.columns)} features.")
    return final_df
def visualize_data_distribution(df, feature_df):
    """
    Create publication-quality visualizations of the data distribution, 
    formatted strictly to PLOS ONE style guidelines.
    """
    print("\n" + "="*50)
    print("DATA VISUALIZATION (PLOS ONE STYLE)")
    print("="*50)

    # 1. PLOS ONE Typography & Style Settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    
    # 2. Colorblind-friendly, high-contrast palette
    colors = sns.color_palette("colorblind")
    
    # Explicit PLOS ONE Font Sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12
    LEGEND_TITLE_SIZE = 13

    # 3. Create figure (2 rows, 3 columns matching Figure 5 description)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('VOC Dataset Analysis', fontsize=TITLE_SIZE, fontweight='bold', y=0.98)

    # --- Top-Left: Sample Type Distribution (FIXED LEGENDS & LABELS) ---
    ax1 = axes[0, 0]
    sample_counts = feature_df['sample_type'].value_counts()
    
    # Ensure consistent ordering and handle missing types safely
    ordered_types = ['A', 'G', 'S']
    counts = [sample_counts.get(t, 0) for t in ordered_types]
    
    # Create separate bar artists for each type to enable individual legends
    bars_a = ax1.bar(ordered_types[0], counts[0], color=colors[0], edgecolor='black', linewidth=0.8, label='A = Arm/Funnel')
    bars_g = ax1.bar(ordered_types[1], counts[1], color=colors[1], edgecolor='black', linewidth=0.8, label='G = Gauze')
    bars_s = ax1.bar(ordered_types[2], counts[2], color=colors[2], edgecolor='black', linewidth=0.8, label='S = Suit')
    
    ax1.set_title('Sample Type Distribution', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.set_xlabel('Sample Type', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', rotation=0)
    ax1.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 10)
    
    # Add value labels on bars (FIXED: iterate directly over the containers)
    for container in [bars_a, bars_g, bars_s]:
        for bar in container:
            yval = bar.get_height()
            if yval > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval),
                         ha='center', va='bottom', fontsize=TICK_SIZE, fontweight='bold')
    
    # Now the legend will correctly show all three entries
    ax1.legend(title='Sample Type', fontsize=LEGEND_SIZE, title_fontsize=LEGEND_TITLE_SIZE, 
               loc='lower right', frameon=True, fancybox=True, framealpha=0.95)
    
    sns.despine(ax=ax1)

    # --- Top-Middle: Samples per Subject by Type ---
    ax2 = axes[0, 1]
    subject_sample_counts = feature_df.groupby(['subject_id', 'sample_type']).size().unstack(fill_value=0)
    
    # Ensure consistent column order and presence for stacking
    for col in ['A', 'G', 'S']:
        if col not in subject_sample_counts.columns:
            subject_sample_counts[col] = 0
    subject_sample_counts = subject_sample_counts[['A', 'G', 'S']]
    
    subject_sample_counts.plot(kind='bar', ax=ax2, stacked=True, 
                               color=[colors[0], colors[1], colors[2]], edgecolor='black', linewidth=0.5)
    
    ax2.set_title('Samples per Subject by Type', fontsize=TITLE_SIZE, fontweight='bold')
    ax2.set_xlabel('Subject ID', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=TICK_SIZE)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Sample Type', fontsize=LEGEND_SIZE, title_fontsize=LEGEND_TITLE_SIZE, loc='lower right')
    sns.despine(ax=ax2)

    # --- Top-Right: Distribution of Number of Peaks per Sample ---
    ax3 = axes[0, 2]
    ax3.hist(feature_df['num_peaks'], bins=15, color=colors[3], edgecolor='black', alpha=0.85, linewidth=0.8)
    
    ax3.set_title('Distribution of Number of Peaks per Sample', fontsize=TITLE_SIZE, fontweight='bold')
    ax3.set_xlabel('Number of Peaks', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_ylabel('Frequency (Number of Samples)', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=TICK_SIZE)
    sns.despine(ax=ax3)

    # --- Bottom-Left: Total Area Distribution by Sample Type ---
    ax4 = axes[1, 0]
    sns.boxplot(data=feature_df, x='sample_type', y='total_area', ax=ax4, 
                palette=[colors[0], colors[1], colors[2]], order=['A', 'G', 'S'], linewidth=1.2)
    
    ax4.set_title('Total Area Distribution by Sample Type', fontsize=TITLE_SIZE, fontweight='bold')
    ax4.set_xlabel('Sample Type', fontsize=LABEL_SIZE, fontweight='bold')
    ax4.set_ylabel('Total Area', fontsize=LABEL_SIZE, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=TICK_SIZE)
    ax4.set_xticklabels(['A (Arm)', 'G (Gauze)', 'S (Suit)'])
    sns.despine(ax=ax4)

    # --- Bottom-Middle: Max Height Distribution by Sample Type ---
    ax5 = axes[1, 1]
    sns.boxplot(data=feature_df, x='sample_type', y='max_height', ax=ax5, 
                palette=[colors[0], colors[1], colors[2]], order=['A', 'G', 'S'], linewidth=1.2)
    
    ax5.set_title('Max Height Distribution by Sample Type', fontsize=TITLE_SIZE, fontweight='bold')
    ax5.set_xlabel('Sample Type', fontsize=LABEL_SIZE, fontweight='bold')
    ax5.set_ylabel('Max Height', fontsize=LABEL_SIZE, fontweight='bold')
    ax5.tick_params(axis='both', labelsize=TICK_SIZE)
    ax5.set_xticklabels(['A (Arm)', 'G (Gauze)', 'S (Suit)'])
    sns.despine(ax=ax5)

    # --- Bottom-Right: Mean Retention Time by Sample Type ---
    ax6 = axes[1, 2]
    sns.boxplot(data=feature_df, x='sample_type', y='mean_retention_time', ax=ax6, 
                palette=[colors[0], colors[1], colors[2]], order=['A', 'G', 'S'], linewidth=1.2)
    
    ax6.set_title('Mean Retention Time by Sample Type', fontsize=TITLE_SIZE, fontweight='bold')
    ax6.set_xlabel('Sample Type', fontsize=LABEL_SIZE, fontweight='bold')
    ax6.set_ylabel('Mean Retention Time (min)', fontsize=LABEL_SIZE, fontweight='bold')
    ax6.tick_params(axis='both', labelsize=TICK_SIZE)
    ax6.set_xticklabels(['A (Arm)', 'G (Gauze)', 'S (Suit)'])
    sns.despine(ax=ax6)

    # 4. Final layout adjustments and high-res save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = 'Data_Figures/voc_data_analysis.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight') # 600 DPI exceeds PLOS ONE 300 DPI minimum
    plt.close()

    print(f"Publication-quality visualization saved to: {output_path}")
def prepare_subject_level_splits(feature_df):
    """Prepare subject-level splits for k-fold cross-validation."""
    print("\n" + "="*50)
    print("SUBJECT-LEVEL SPLIT PREPARATION")
    print("="*50)
    
    unique_subjects = feature_df['subject_id'].unique()
    print(f"Total unique subjects: {len(unique_subjects)}")
    
    samples_per_subject = feature_df['subject_id'].value_counts()
    print(f"Samples per subject:\n{samples_per_subject.describe()}")
    
    return {
        'unique_subjects': list(unique_subjects),
        'samples_per_subject': samples_per_subject.to_dict()
    }

def main():
    """Main function to run the complete analysis."""
    print("VOC Dataset Analysis - PLOS ONE Ready Version")
    print("="*50)
    
    os.makedirs('Data_Figures', exist_ok=True)
    combined_df, dataset_info = analyze_dataset_structure()
    
    if combined_df is not None:
        analyze_classification_task(combined_df)
        feature_df = create_feature_matrix(combined_df)
        prepare_subject_level_splits(feature_df)
        visualize_data_distribution(combined_df, feature_df)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print("Files created in 'Data_Figures/' directory:")
        print("- voc_combined_dataset.csv")
        print("- voc_feature_matrix.csv")
        print("- voc_data_analysis.png (PLOS ONE formatted)")
        
        return feature_df, dataset_info
    else:
        print("Analysis failed - no data could be loaded")
        return None, None

if __name__ == "__main__":
    main()