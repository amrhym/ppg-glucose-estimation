#!/usr/bin/env python3
"""
Create a frequency histogram of blood glucose levels from the PPG dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_glucose_frequency_histogram():
    """Create a frequency histogram of glucose levels."""
    
    # Load glucose values from dataset
    data_dir = Path("data/PPG_Dataset")
    label_dir = data_dir / "Labels"
    
    glucose_values = []
    
    print("Loading glucose values from dataset...")
    
    # Load all label files
    label_files = sorted(label_dir.glob("label_*.csv"))
    
    for label_file in label_files:
        try:
            df = pd.read_csv(label_file)
            if 'Glucose' in df.columns:
                glucose_values.append(df['Glucose'].values[0])
        except Exception as e:
            continue
    
    print(f"Loaded {len(glucose_values)} glucose measurements")
    
    if not glucose_values:
        print("No glucose data found!")
        return
    
    # Convert to numpy array
    glucose_values = np.array(glucose_values)
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Blood Glucose Level Frequency Distribution', fontsize=16, fontweight='bold')
    
    # ============= Plot 1: Main Frequency Histogram =============
    ax1 = axes[0, 0]
    
    # Create histogram
    n, bins, patches = ax1.hist(glucose_values, bins=20, color='steelblue', 
                                edgecolor='black', alpha=0.7)
    
    # Color bars based on clinical ranges
    for i, patch in enumerate(patches):
        if bins[i] < 100:
            patch.set_facecolor('green')  # Normal
        elif bins[i] < 126:
            patch.set_facecolor('yellow')  # Prediabetic
        else:
            patch.set_facecolor('red')  # Diabetic
    
    # Add frequency labels on top of bars
    for i, rect in enumerate(patches):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                f'{int(n[i])}', ha='center', va='bottom', fontweight='bold')
    
    # Add mean and median lines
    mean_glucose = np.mean(glucose_values)
    median_glucose = np.median(glucose_values)
    
    ax1.axvline(mean_glucose, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_glucose:.1f} mg/dL')
    ax1.axvline(median_glucose, color='darkblue', linestyle='--', linewidth=2,
                label=f'Median: {median_glucose:.1f} mg/dL')
    
    ax1.set_xlabel('Blood Glucose Level (mg/dL)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (Number of Samples)', fontsize=12, fontweight='bold')
    ax1.set_title('Frequency Distribution of Blood Glucose Levels', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Total Samples: {len(glucose_values)}\n'
    stats_text += f'Mean: {mean_glucose:.1f} mg/dL\n'
    stats_text += f'Std Dev: {np.std(glucose_values):.1f} mg/dL\n'
    stats_text += f'Min: {min(glucose_values):.0f} mg/dL\n'
    stats_text += f'Max: {max(glucose_values):.0f} mg/dL'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=10)
    
    # ============= Plot 2: Percentage Distribution =============
    ax2 = axes[0, 1]
    
    # Create percentage histogram
    weights = np.ones_like(glucose_values) / len(glucose_values) * 100
    n_pct, bins_pct, patches_pct = ax2.hist(glucose_values, bins=20, weights=weights,
                                            color='coral', edgecolor='black', alpha=0.7)
    
    # Add percentage labels
    for i, rect in enumerate(patches_pct):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height + 0.2,
                f'{n_pct[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Blood Glucose Level (mg/dL)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Samples (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Percentage Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ============= Plot 3: Cumulative Frequency =============
    ax3 = axes[1, 0]
    
    # Calculate cumulative frequency
    sorted_glucose = np.sort(glucose_values)
    cumulative_freq = np.arange(1, len(sorted_glucose) + 1)
    
    ax3.plot(sorted_glucose, cumulative_freq, 'b-', linewidth=2, label='Cumulative Frequency')
    ax3.fill_between(sorted_glucose, 0, cumulative_freq, alpha=0.3)
    
    # Add quartile lines
    q1, q2, q3 = np.percentile(glucose_values, [25, 50, 75])
    ax3.axvline(q1, color='red', linestyle=':', alpha=0.7, label=f'Q1: {q1:.1f}')
    ax3.axvline(q2, color='green', linestyle=':', alpha=0.7, label=f'Median: {q2:.1f}')
    ax3.axvline(q3, color='red', linestyle=':', alpha=0.7, label=f'Q3: {q3:.1f}')
    
    ax3.set_xlabel('Blood Glucose Level (mg/dL)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Frequency Distribution', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # ============= Plot 4: Density Plot with KDE =============
    ax4 = axes[1, 1]
    
    # Create histogram with density
    n_density, bins_density, patches_density = ax4.hist(glucose_values, bins=20, 
                                                        density=True, alpha=0.5, 
                                                        color='lightgreen', 
                                                        edgecolor='black')
    
    # Add KDE curve
    kde = stats.gaussian_kde(glucose_values)
    x_range = np.linspace(glucose_values.min(), glucose_values.max(), 200)
    kde_values = kde(x_range)
    ax4.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE (Kernel Density Estimate)')
    
    # Add normal distribution overlay
    mu, sigma = stats.norm.fit(glucose_values)
    normal_dist = stats.norm.pdf(x_range, mu, sigma)
    ax4.plot(x_range, normal_dist, 'b--', linewidth=2, alpha=0.7, 
            label=f'Normal Fit (μ={mu:.1f}, σ={sigma:.1f})')
    
    ax4.set_xlabel('Blood Glucose Level (mg/dL)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax4.set_title('Probability Density Distribution', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('glucose_frequency_histogram.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved glucose frequency histogram: glucose_frequency_histogram.png")
    
    # Create a simple single histogram as well
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Create the main histogram
    counts, bins, patches = ax.hist(glucose_values, bins=15, color='skyblue', 
                                    edgecolor='navy', alpha=0.8)
    
    # Add frequency labels on bars
    for count, bin_edge, patch in zip(counts, bins[:-1], patches):
        height = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2., height + 0.5,
               f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Color code by clinical range
    for i, patch in enumerate(patches):
        if bins[i] < 100:
            patch.set_facecolor('lightgreen')
            patch.set_label('Normal' if i == 0 else "")
        elif bins[i] < 126:
            patch.set_facecolor('yellow') 
            patch.set_label('Prediabetic' if bins[i-1] < 100 else "")
        else:
            patch.set_facecolor('lightcoral')
            patch.set_label('Diabetic' if bins[i-1] < 126 else "")
    
    # Add mean line
    mean_val = np.mean(glucose_values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.1f} mg/dL')
    
    ax.set_xlabel('Blood Glucose Level (mg/dL)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Samples)', fontsize=14, fontweight='bold')
    ax.set_title('Blood Glucose Level Frequency Distribution', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Add text box with key statistics
    textstr = f'N = {len(glucose_values)}\nMean = {np.mean(glucose_values):.1f} mg/dL\n'
    textstr += f'Median = {np.median(glucose_values):.1f} mg/dL\nStd = {np.std(glucose_values):.1f} mg/dL\n'
    textstr += f'Range = {min(glucose_values):.0f}-{max(glucose_values):.0f} mg/dL'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('glucose_simple_histogram.png', dpi=150, bbox_inches='tight')
    print("✅ Saved simple glucose histogram: glucose_simple_histogram.png")
    
    plt.show()
    
    # Print frequency table
    print("\n📊 GLUCOSE FREQUENCY TABLE")
    print("="*50)
    print(f"{'Range (mg/dL)':<20} {'Frequency':<12} {'Percentage':<10}")
    print("-"*50)
    
    # Create bins and count frequencies
    hist, bin_edges = np.histogram(glucose_values, bins=10)
    
    for i in range(len(hist)):
        range_str = f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}"
        freq = hist[i]
        pct = (freq / len(glucose_values)) * 100
        print(f"{range_str:<20} {freq:<12} {pct:.1f}%")
    
    print("-"*50)
    print(f"{'Total':<20} {len(glucose_values):<12} 100.0%")
    
    return glucose_values

if __name__ == "__main__":
    glucose_values = create_glucose_frequency_histogram()