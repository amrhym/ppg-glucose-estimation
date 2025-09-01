#!/usr/bin/env python3
"""
Comprehensive histogram analysis of the PPG glucose estimation dataset.
Visualizes distributions of glucose values, patient demographics, and signal characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from collections import Counter

warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_dataset_info(data_dir="data/PPG_Dataset"):
    """Load all dataset information including glucose values and metadata."""
    
    data_dir = Path(data_dir)
    label_dir = data_dir / "Labels"
    raw_dir = data_dir / "RawData"
    
    # Collect all data
    glucose_values = []
    ages = []
    genders = []
    heights = []
    weights = []
    subject_ids = []
    signal_lengths = []
    
    print("Loading dataset information...")
    
    # Load all label files
    label_files = sorted(label_dir.glob("label_*.csv"))
    
    for label_file in label_files:
        try:
            # Load label data
            df = pd.read_csv(label_file)
            
            if 'Glucose' in df.columns:
                glucose_values.append(df['Glucose'].values[0])
                
                # Extract subject ID
                subject_id = label_file.stem.split('_')[1]
                subject_ids.append(subject_id)
                
                # Load other metadata if available
                if 'Age' in df.columns:
                    ages.append(df['Age'].values[0])
                if 'Gender' in df.columns:
                    genders.append(df['Gender'].values[0])
                if 'Height' in df.columns:
                    heights.append(df['Height'].values[0])
                if 'Weight' in df.columns:
                    weights.append(df['Weight'].values[0])
                
                # Check corresponding signal file
                signal_file = raw_dir / f"signal_{subject_id}.csv"
                if signal_file.exists():
                    signal_data = pd.read_csv(signal_file, header=None)
                    signal_lengths.append(len(signal_data))
                    
        except Exception as e:
            print(f"Error loading {label_file}: {e}")
    
    print(f"Loaded {len(glucose_values)} samples")
    
    return {
        'glucose': glucose_values,
        'ages': ages,
        'genders': genders,
        'heights': heights,
        'weights': weights,
        'subject_ids': subject_ids,
        'signal_lengths': signal_lengths
    }

def create_comprehensive_histograms(data):
    """Create comprehensive histogram visualizations."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle('PPG Glucose Dataset - Comprehensive Distribution Analysis', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # 1. Glucose Distribution (Main plot - larger)
    ax1 = plt.subplot(3, 4, (1, 6))
    glucose_values = data['glucose']
    
    # Create histogram with KDE
    n, bins, patches = ax1.hist(glucose_values, bins=20, color='skyblue', 
                                edgecolor='navy', alpha=0.7, density=True)
    
    # Add KDE curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(glucose_values)
    x_range = np.linspace(min(glucose_values), max(glucose_values), 100)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Add statistics
    mean_glucose = np.mean(glucose_values)
    median_glucose = np.median(glucose_values)
    std_glucose = np.std(glucose_values)
    
    ax1.axvline(mean_glucose, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_glucose:.1f} mg/dL')
    ax1.axvline(median_glucose, color='orange', linestyle='--', linewidth=2,
                label=f'Median: {median_glucose:.1f} mg/dL')
    
    # Add normal range shading
    ax1.axvspan(70, 100, alpha=0.2, color='green', label='Normal Range (70-100)')
    ax1.axvspan(100, 125, alpha=0.2, color='yellow', label='Prediabetic (100-125)')
    ax1.axvspan(125, 200, alpha=0.2, color='red', label='Diabetic (>125)')
    
    ax1.set_xlabel('Glucose Level (mg/dL)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Glucose Distribution (n={len(glucose_values)})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Statistics:\nMean: {mean_glucose:.1f} mg/dL\nMedian: {median_glucose:.1f} mg/dL\n'
    stats_text += f'Std Dev: {std_glucose:.1f} mg/dL\nMin: {min(glucose_values):.0f} mg/dL\n'
    stats_text += f'Max: {max(glucose_values):.0f} mg/dL\nRange: {max(glucose_values)-min(glucose_values):.0f} mg/dL'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # 2. Glucose Range Distribution (Pie Chart)
    ax2 = plt.subplot(3, 4, 3)
    
    # Categorize glucose levels
    normal = sum(1 for g in glucose_values if g < 100)
    prediabetic = sum(1 for g in glucose_values if 100 <= g < 126)
    diabetic = sum(1 for g in glucose_values if g >= 126)
    
    sizes = [normal, prediabetic, diabetic]
    labels = [f'Normal\n(<100)\n{normal}', 
              f'Prediabetic\n(100-125)\n{prediabetic}', 
              f'Diabetic\n(≥126)\n{diabetic}']
    colors = ['green', 'yellow', 'red']
    explode = (0.05, 0.05, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('Glucose Categories', fontsize=12, fontweight='bold')
    
    # 3. Box Plot of Glucose Values
    ax3 = plt.subplot(3, 4, 4)
    box = ax3.boxplot(glucose_values, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_edgecolor('navy')
    
    # Add mean marker
    ax3.scatter([1], [mean_glucose], color='red', s=100, zorder=5, label='Mean')
    
    # Add quartile labels
    q1, q2, q3 = np.percentile(glucose_values, [25, 50, 75])
    ax3.text(1.1, q1, f'Q1: {q1:.1f}', fontsize=9)
    ax3.text(1.1, q2, f'Median: {q2:.1f}', fontsize=9)
    ax3.text(1.1, q3, f'Q3: {q3:.1f}', fontsize=9)
    
    ax3.set_ylabel('Glucose (mg/dL)', fontsize=11)
    ax3.set_title('Glucose Box Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticklabels(['All Samples'])
    
    # 4. Age Distribution (if available)
    if data['ages']:
        ax4 = plt.subplot(3, 4, 7)
        ax4.hist(data['ages'], bins=15, color='coral', edgecolor='darkred', alpha=0.7)
        ax4.set_xlabel('Age (years)', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title(f'Age Distribution (n={len(data["ages"])})', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add mean line
        if data['ages']:
            mean_age = np.mean(data['ages'])
            ax4.axvline(mean_age, color='red', linestyle='--', 
                       label=f'Mean: {mean_age:.1f} years')
            ax4.legend()
    
    # 5. Gender Distribution (if available)
    if data['genders']:
        ax5 = plt.subplot(3, 4, 8)
        gender_counts = Counter(data['genders'])
        
        if gender_counts:
            bars = ax5.bar(gender_counts.keys(), gender_counts.values(), 
                          color=['blue', 'pink'], edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Gender', fontsize=11)
            ax5.set_ylabel('Count', fontsize=11)
            ax5.set_title('Gender Distribution', fontsize=12, fontweight='bold')
            
            # Add count labels on bars
            for bar, count in zip(bars, gender_counts.values()):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
    
    # 6. Height Distribution (if available)
    if data['heights'] and any(h > 0 for h in data['heights']):
        ax6 = plt.subplot(3, 4, 9)
        valid_heights = [h for h in data['heights'] if h > 0]
        if valid_heights:
            ax6.hist(valid_heights, bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            ax6.set_xlabel('Height (cm)', fontsize=11)
            ax6.set_ylabel('Count', fontsize=11)
            ax6.set_title(f'Height Distribution (n={len(valid_heights)})', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
    
    # 7. Weight Distribution (if available)
    if data['weights'] and any(w > 0 for w in data['weights']):
        ax7 = plt.subplot(3, 4, 10)
        valid_weights = [w for w in data['weights'] if w > 0]
        if valid_weights:
            ax7.hist(valid_weights, bins=15, color='lightyellow', edgecolor='orange', alpha=0.7)
            ax7.set_xlabel('Weight (kg)', fontsize=11)
            ax7.set_ylabel('Count', fontsize=11)
            ax7.set_title(f'Weight Distribution (n={len(valid_weights)})', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
    
    # 8. Signal Length Distribution
    if data['signal_lengths']:
        ax8 = plt.subplot(3, 4, 11)
        ax8.hist(data['signal_lengths'], bins=20, color='purple', edgecolor='darkviolet', alpha=0.7)
        ax8.set_xlabel('Signal Length (samples)', fontsize=11)
        ax8.set_ylabel('Count', fontsize=11)
        ax8.set_title('Signal Length Distribution', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # Add expected length line
        expected_length = 21900  # 10 seconds at 2175 Hz
        ax8.axvline(expected_length, color='red', linestyle='--', 
                   label=f'Expected: {expected_length}')
        ax8.legend()
    
    # 9. Glucose vs Subject ID (to show individual variation)
    ax9 = plt.subplot(3, 4, 12)
    
    # Group glucose values by subject
    subject_glucose = {}
    for sid, glucose in zip(data['subject_ids'], data['glucose']):
        subject = sid.split('_')[0] if '_' in sid else sid
        if subject not in subject_glucose:
            subject_glucose[subject] = []
        subject_glucose[subject].append(glucose)
    
    # Calculate mean glucose per subject
    subjects = sorted(subject_glucose.keys())[:20]  # Show first 20 subjects
    mean_glucose_per_subject = [np.mean(subject_glucose[s]) for s in subjects]
    
    bars = ax9.bar(range(len(subjects)), mean_glucose_per_subject, 
                   color='teal', edgecolor='darkslategray', alpha=0.7)
    
    # Color bars based on glucose level
    for i, (bar, glucose) in enumerate(zip(bars, mean_glucose_per_subject)):
        if glucose < 100:
            bar.set_facecolor('green')
        elif glucose < 126:
            bar.set_facecolor('yellow')
        else:
            bar.set_facecolor('red')
    
    ax9.set_xlabel('Subject ID', fontsize=11)
    ax9.set_ylabel('Mean Glucose (mg/dL)', fontsize=11)
    ax9.set_title('Mean Glucose by Subject (First 20)', fontsize=12, fontweight='bold')
    ax9.set_xticks(range(len(subjects)))
    ax9.set_xticklabels(subjects, rotation=45, ha='right')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add normal range line
    ax9.axhline(100, color='blue', linestyle='--', alpha=0.5, label='Normal threshold')
    ax9.axhline(126, color='red', linestyle='--', alpha=0.5, label='Diabetic threshold')
    ax9.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('dataset_histogram_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved comprehensive histogram analysis: dataset_histogram_analysis.png")
    
    plt.show()
    
    return fig

def create_detailed_glucose_histogram(glucose_values):
    """Create a detailed glucose-only histogram with advanced statistics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Glucose Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Standard Histogram with Normal Distribution Overlay
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(glucose_values, bins=25, density=True, 
                                alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(glucose_values)
    x = np.linspace(min(glucose_values), max(glucose_values), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
            label=f'Normal fit\nμ={mu:.1f}, σ={sigma:.1f}')
    
    ax1.set_xlabel('Glucose (mg/dL)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Histogram with Normal Distribution Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Distribution
    ax2 = axes[0, 1]
    sorted_glucose = np.sort(glucose_values)
    cumulative = np.arange(1, len(sorted_glucose) + 1) / len(sorted_glucose)
    
    ax2.plot(sorted_glucose, cumulative, 'b-', linewidth=2)
    ax2.fill_between(sorted_glucose, 0, cumulative, alpha=0.3)
    
    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(glucose_values, p)
        ax2.axvline(val, color='red', linestyle='--', alpha=0.5)
        ax2.text(val, p/100, f'{p}%: {val:.0f}', rotation=45, fontsize=9)
    
    ax2.set_xlabel('Glucose (mg/dL)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(glucose_values, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Test)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Violin Plot with Box Plot Overlay
    ax4 = axes[1, 1]
    parts = ax4.violinplot([glucose_values], positions=[1], widths=0.7, 
                           showmeans=True, showmedians=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Add box plot overlay
    bp = ax4.boxplot([glucose_values], positions=[1], widths=0.3, 
                     patch_artist=True, boxprops=dict(facecolor='white', alpha=0.5))
    
    # Add swarm plot for individual points
    y_jittered = glucose_values + np.random.normal(0, 0.02, len(glucose_values))
    x_jittered = np.ones(len(glucose_values)) + np.random.normal(0, 0.05, len(glucose_values))
    ax4.scatter(x_jittered, glucose_values, alpha=0.3, s=20, color='navy')
    
    ax4.set_ylabel('Glucose (mg/dL)')
    ax4.set_title('Violin Plot with Data Points')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['All Samples'])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('detailed_glucose_histogram.png', dpi=150, bbox_inches='tight')
    print("✅ Saved detailed glucose histogram: detailed_glucose_histogram.png")
    plt.show()
    
    return fig

def print_dataset_statistics(data):
    """Print comprehensive dataset statistics."""
    
    print("\n" + "="*80)
    print("📊 DATASET STATISTICS SUMMARY")
    print("="*80)
    
    glucose_values = data['glucose']
    
    print(f"\n📈 Glucose Statistics:")
    print(f"  • Total samples: {len(glucose_values)}")
    print(f"  • Mean: {np.mean(glucose_values):.2f} mg/dL")
    print(f"  • Median: {np.median(glucose_values):.2f} mg/dL")
    print(f"  • Std Dev: {np.std(glucose_values):.2f} mg/dL")
    print(f"  • Min: {min(glucose_values):.0f} mg/dL")
    print(f"  • Max: {max(glucose_values):.0f} mg/dL")
    print(f"  • Range: {max(glucose_values) - min(glucose_values):.0f} mg/dL")
    
    # Quartiles
    q1, q2, q3 = np.percentile(glucose_values, [25, 50, 75])
    iqr = q3 - q1
    print(f"\n📊 Quartiles:")
    print(f"  • Q1 (25%): {q1:.2f} mg/dL")
    print(f"  • Q2 (50%): {q2:.2f} mg/dL")
    print(f"  • Q3 (75%): {q3:.2f} mg/dL")
    print(f"  • IQR: {iqr:.2f} mg/dL")
    
    # Categories
    normal = sum(1 for g in glucose_values if g < 100)
    prediabetic = sum(1 for g in glucose_values if 100 <= g < 126)
    diabetic = sum(1 for g in glucose_values if g >= 126)
    
    print(f"\n🏥 Clinical Categories:")
    print(f"  • Normal (<100): {normal} ({normal/len(glucose_values)*100:.1f}%)")
    print(f"  • Prediabetic (100-125): {prediabetic} ({prediabetic/len(glucose_values)*100:.1f}%)")
    print(f"  • Diabetic (≥126): {diabetic} ({diabetic/len(glucose_values)*100:.1f}%)")
    
    # Skewness and Kurtosis
    from scipy.stats import skew, kurtosis
    print(f"\n📐 Distribution Shape:")
    print(f"  • Skewness: {skew(glucose_values):.3f}")
    print(f"  • Kurtosis: {kurtosis(glucose_values):.3f}")
    
    # Normality test
    _, p_value = stats.normaltest(glucose_values)
    print(f"  • Normality test p-value: {p_value:.4f}")
    print(f"  • Distribution is {'approximately normal' if p_value > 0.05 else 'not normal'}")
    
    # Subject information
    unique_subjects = len(set(s.split('_')[0] for s in data['subject_ids']))
    print(f"\n👥 Subject Information:")
    print(f"  • Unique subjects: {unique_subjects}")
    print(f"  • Samples per subject: {len(glucose_values) / unique_subjects:.1f}")
    
    if data['ages']:
        valid_ages = [a for a in data['ages'] if a > 0]
        if valid_ages:
            print(f"\n🎂 Age Statistics:")
            print(f"  • Mean age: {np.mean(valid_ages):.1f} years")
            print(f"  • Age range: {min(valid_ages):.0f} - {max(valid_ages):.0f} years")
    
    if data['signal_lengths']:
        print(f"\n📡 Signal Information:")
        print(f"  • Mean signal length: {np.mean(data['signal_lengths']):.0f} samples")
        print(f"  • Expected length: 21,900 samples (10s at 2175 Hz)")
        consistent = sum(1 for l in data['signal_lengths'] if l == 21900)
        print(f"  • Consistent length: {consistent}/{len(data['signal_lengths'])} samples")
    
    print("\n" + "="*80)

def main():
    """Main analysis function."""
    
    print("\n🎯 PPG GLUCOSE DATASET - HISTOGRAM ANALYSIS")
    print("="*80)
    
    # Load dataset
    data = load_dataset_info()
    
    if not data['glucose']:
        print("❌ No glucose data found!")
        return
    
    # Print statistics
    print_dataset_statistics(data)
    
    # Create comprehensive histograms
    print("\n📊 Creating comprehensive histogram visualizations...")
    create_comprehensive_histograms(data)
    
    # Create detailed glucose histogram
    print("\n📈 Creating detailed glucose distribution analysis...")
    create_detailed_glucose_histogram(data['glucose'])
    
    print("\n✅ Analysis complete!")
    print("Generated files:")
    print("  • dataset_histogram_analysis.png - Comprehensive overview")
    print("  • detailed_glucose_histogram.png - Detailed glucose analysis")
    
    return data

if __name__ == "__main__":
    data = main()