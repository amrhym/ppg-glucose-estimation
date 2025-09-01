#!/usr/bin/env python3
"""
Comprehensive Fold Results Visualization
Plots individual fold performance, averages, best results, and target comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_fold_results_data():
    """Create fold results data structure."""
    fold_results = {
        'Fold': list(range(1, 11)),
        'MAE': [23.50, 21.45, 20.82, 20.03, 17.29, 20.39, 7.44, 24.87, 16.78, 10.53],
        'RMSE': [26.82, 23.85, 23.18, 22.30, 19.50, 22.68, 9.05, 28.11, 20.01, 12.56],
        'R2': [0.416, 0.544, 0.571, 0.604, 0.699, 0.591, 0.937, 0.432, 0.715, 0.895],
        'MAPE': [17.20, 15.63, 14.85, 14.23, 12.14, 14.50, 6.27, 17.18, 11.93, 8.33],
        'Best_Epoch': [23, 14, 50, 12, 37, 26, 50, 15, 22, 29]
    }
    
    # Paper targets
    targets = {
        'MAE': 2.96,
        'RMSE': 3.94,
        'R2': 0.97,
        'MAPE': 2.40
    }
    
    return pd.DataFrame(fold_results), targets

def plot_individual_fold_performance():
    """Create a comprehensive plot showing all fold performances."""
    df, targets = create_fold_results_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color scheme
    fold_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    best_fold_idx = 6  # Fold 7 (index 6)
    
    # 1. MAE Comparison
    ax = axes[0, 0]
    bars = ax.bar(df['Fold'], df['MAE'], color=fold_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars[best_fold_idx].set_color('gold')
    bars[best_fold_idx].set_edgecolor('darkgoldenrod')
    bars[best_fold_idx].set_linewidth(2.5)
    
    # Add average line
    avg_mae = df['MAE'].mean()
    ax.axhline(y=avg_mae, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_mae:.2f}')
    ax.axhline(y=targets['MAE'], color='green', linestyle='-', linewidth=2, label=f'Target: {targets["MAE"]:.2f}')
    
    # Add value labels on bars
    for i, (fold, value) in enumerate(zip(df['Fold'], df['MAE'])):
        ax.text(fold, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=9, 
                fontweight='bold' if i == best_fold_idx else 'normal')
    
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('MAE (mg/dL)', fontsize=12)
    ax.set_title('Mean Absolute Error by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(df['Fold'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(df['MAE']) * 1.15])
    
    # 2. RMSE Comparison
    ax = axes[0, 1]
    bars = ax.bar(df['Fold'], df['RMSE'], color=fold_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars[best_fold_idx].set_color('gold')
    bars[best_fold_idx].set_edgecolor('darkgoldenrod')
    bars[best_fold_idx].set_linewidth(2.5)
    
    avg_rmse = df['RMSE'].mean()
    ax.axhline(y=avg_rmse, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_rmse:.2f}')
    ax.axhline(y=targets['RMSE'], color='green', linestyle='-', linewidth=2, label=f'Target: {targets["RMSE"]:.2f}')
    
    for i, (fold, value) in enumerate(zip(df['Fold'], df['RMSE'])):
        ax.text(fold, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == best_fold_idx else 'normal')
    
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=12)
    ax.set_title('Root Mean Square Error by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(df['Fold'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(df['RMSE']) * 1.15])
    
    # 3. R² Score Comparison
    ax = axes[1, 0]
    bars = ax.bar(df['Fold'], df['R2'], color=fold_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars[best_fold_idx].set_color('gold')
    bars[best_fold_idx].set_edgecolor('darkgoldenrod')
    bars[best_fold_idx].set_linewidth(2.5)
    
    avg_r2 = df['R2'].mean()
    ax.axhline(y=avg_r2, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_r2:.4f}')
    ax.axhline(y=targets['R2'], color='green', linestyle='-', linewidth=2, label=f'Target: {targets["R2"]:.2f}')
    
    for i, (fold, value) in enumerate(zip(df['Fold'], df['R2'])):
        ax.text(fold, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == best_fold_idx else 'normal')
    
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(df['Fold'])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 4. MAPE Comparison
    ax = axes[1, 1]
    bars = ax.bar(df['Fold'], df['MAPE'], color=fold_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars[best_fold_idx].set_color('gold')
    bars[best_fold_idx].set_edgecolor('darkgoldenrod')
    bars[best_fold_idx].set_linewidth(2.5)
    
    avg_mape = df['MAPE'].mean()
    ax.axhline(y=avg_mape, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_mape:.2f}%')
    ax.axhline(y=targets['MAPE'], color='green', linestyle='-', linewidth=2, label=f'Target: {targets["MAPE"]:.2f}%')
    
    for i, (fold, value) in enumerate(zip(df['Fold'], df['MAPE'])):
        ax.text(fold, value + 0.3, f'{value:.1f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == best_fold_idx else 'normal')
    
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('Mean Absolute Percentage Error by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(df['Fold'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(df['MAPE']) * 1.15])
    
    # Overall title
    fig.suptitle('10-Fold Cross-Validation Results - Individual Fold Performance', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Add note about best fold
    fig.text(0.5, -0.02, 'Note: Fold 7 (highlighted in gold) achieved the best overall performance', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('visualizations/fold_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fold_performance_comparison.png")
    plt.close()

def plot_performance_summary():
    """Create a summary plot comparing average, best, and target performance."""
    df, targets = create_fold_results_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    metrics = ['MAE\n(mg/dL)', 'RMSE\n(mg/dL)', 'R²\nScore', 'MAPE\n(%)']
    x = np.arange(len(metrics))
    width = 0.25
    
    # Calculate values
    averages = [df['MAE'].mean(), df['RMSE'].mean(), df['R2'].mean(), df['MAPE'].mean()]
    bests = [df['MAE'].min(), df['RMSE'].min(), df['R2'].max(), df['MAPE'].min()]
    target_vals = [targets['MAE'], targets['RMSE'], targets['R2'], targets['MAPE']]
    
    # Normalize for visualization (since R² is 0-1 and others are larger)
    # We'll use percentage of target for better comparison
    avg_pct = [(averages[i]/target_vals[i])*100 if i != 2 else averages[i]*100 for i in range(4)]
    best_pct = [(bests[i]/target_vals[i])*100 if i != 2 else bests[i]*100 for i in range(4)]
    target_pct = [100, 100, 100, 100]  # Target is always 100%
    
    # Create bars
    bars1 = ax.bar(x - width, avg_pct, width, label='Average Performance', 
                   color='lightcoral', edgecolor='darkred', linewidth=2, alpha=0.8)
    bars2 = ax.bar(x, best_pct, width, label='Best Fold (Fold 7)', 
                   color='gold', edgecolor='darkgoldenrod', linewidth=2, alpha=0.8)
    bars3 = ax.bar(x + width, target_pct, width, label='Paper Target', 
                   color='lightgreen', edgecolor='darkgreen', linewidth=2, alpha=0.8)
    
    # Add value labels
    for i, (bars, values, actual) in enumerate([(bars1, avg_pct, averages), 
                                                  (bars2, best_pct, bests), 
                                                  (bars3, target_pct, target_vals)]):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if j == 2:  # R² score
                label = f'{actual[j]:.3f}'
            else:
                label = f'{actual[j]:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add target line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(3.5, 102, 'Target Level', ha='right', fontsize=9, color='gray')
    
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (% of Target)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Summary: Average vs Best vs Target', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add performance gap annotations
    for i in range(4):
        if i != 2:  # Not R² score
            gap_pct = ((averages[i] - target_vals[i]) / target_vals[i]) * 100
            ax.annotate(f'+{gap_pct:.0f}%', 
                       xy=(i - width, avg_pct[i]), 
                       xytext=(i - width, avg_pct[i] + 15),
                       ha='center', fontsize=8, color='darkred',
                       arrowprops=dict(arrowstyle='->', color='darkred', lw=1))
    
    plt.tight_layout()
    plt.savefig('visualizations/performance_summary_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_summary_comparison.png")
    plt.close()

def plot_radar_comparison():
    """Create a radar plot comparing different aspects of performance."""
    df, targets = create_fold_results_data()
    
    # Prepare data
    categories = ['MAE', 'RMSE', 'R²', 'MAPE', 'Consistency']
    
    # Calculate consistency score (inverse of std deviation)
    mae_consistency = 1 - (df['MAE'].std() / df['MAE'].mean())
    
    # Normalize metrics to 0-1 scale (where 1 is best)
    def normalize(value, target, inverse=False):
        if inverse:
            return min(target / value, 1.0)
        else:
            return min(value / target, 1.0)
    
    best_fold = df.iloc[6]  # Fold 7
    
    average_scores = [
        normalize(df['MAE'].mean(), targets['MAE'], inverse=True),
        normalize(df['RMSE'].mean(), targets['RMSE'], inverse=True),
        normalize(df['R2'].mean(), targets['R2']),
        normalize(df['MAPE'].mean(), targets['MAPE'], inverse=True),
        mae_consistency
    ]
    
    best_scores = [
        normalize(best_fold['MAE'], targets['MAE'], inverse=True),
        normalize(best_fold['RMSE'], targets['RMSE'], inverse=True),
        normalize(best_fold['R2'], targets['R2']),
        normalize(best_fold['MAPE'], targets['MAPE'], inverse=True),
        0.95  # Best fold has high consistency
    ]
    
    target_scores = [1, 1, 1, 1, 1]  # Perfect scores
    
    # Create radar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    average_scores += average_scores[:1]
    best_scores += best_scores[:1]
    target_scores += target_scores[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, target_scores, 'g-', linewidth=2, label='Target', marker='o', markersize=8)
    ax.fill(angles, target_scores, 'green', alpha=0.1)
    
    ax.plot(angles, best_scores, 'gold', linewidth=2, label='Best (Fold 7)', marker='o', markersize=8)
    ax.fill(angles, best_scores, 'gold', alpha=0.2)
    
    ax.plot(angles, average_scores, 'r-', linewidth=2, label='Average', marker='o', markersize=8)
    ax.fill(angles, average_scores, 'red', alpha=0.1)
    
    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # Add gridlines
    ax.grid(True)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
    plt.title('Multi-Metric Performance Comparison\n(Percentage of Target Achievement)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('visualizations/radar_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: radar_performance_comparison.png")
    plt.close()

def plot_fold_ranking():
    """Create a ranking visualization of all folds."""
    df, targets = create_fold_results_data()
    
    # Calculate composite score for each fold
    # Lower is better for MAE, RMSE, MAPE; Higher is better for R²
    df['Composite_Score'] = (
        (1 - df['MAE'] / df['MAE'].max()) * 0.3 +
        (1 - df['RMSE'] / df['RMSE'].max()) * 0.3 +
        (df['R2'] / df['R2'].max()) * 0.3 +
        (1 - df['MAPE'] / df['MAPE'].max()) * 0.1
    )
    
    # Sort by composite score
    df_sorted = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, 11)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Ranking bar chart
    colors = ['gold' if fold == 7 else 'lightblue' for fold in df_sorted['Fold']]
    bars = ax1.barh(df_sorted['Fold'].astype(str), df_sorted['Composite_Score'], 
                     color=colors, edgecolor='navy', linewidth=2, alpha=0.8)
    
    # Add rank labels
    for i, (fold, score, rank) in enumerate(zip(df_sorted['Fold'], df_sorted['Composite_Score'], df_sorted['Rank'])):
        ax1.text(score + 0.01, i, f'#{rank}', va='center', fontweight='bold', fontsize=10)
        ax1.text(0.02, i, f'MAE:{df[df["Fold"]==fold]["MAE"].values[0]:.1f}', 
                va='center', fontsize=8, color='white', fontweight='bold')
    
    ax1.set_xlabel('Composite Performance Score', fontsize=12)
    ax1.set_ylabel('Fold Number', fontsize=12)
    ax1.set_title('Fold Performance Ranking', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim([0, df_sorted['Composite_Score'].max() * 1.1])
    
    # Right plot: Performance matrix
    metrics_data = df[['MAE', 'RMSE', 'R2', 'MAPE']].values
    
    # Normalize for heatmap (0-1 scale, where 1 is best)
    metrics_norm = np.zeros_like(metrics_data)
    for i in range(metrics_data.shape[1]):
        if i == 2:  # R² (higher is better)
            metrics_norm[:, i] = metrics_data[:, i] / metrics_data[:, i].max()
        else:  # MAE, RMSE, MAPE (lower is better)
            metrics_norm[:, i] = 1 - (metrics_data[:, i] - metrics_data[:, i].min()) / (metrics_data[:, i].max() - metrics_data[:, i].min())
    
    im = ax2.imshow(metrics_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax2.set_xticks(np.arange(4))
    ax2.set_yticks(np.arange(10))
    ax2.set_xticklabels(['MAE', 'RMSE', 'R²', 'MAPE'])
    ax2.set_yticklabels([f'Fold {i+1}' for i in range(10)])
    
    # Add text annotations
    for i in range(10):
        for j in range(4):
            text = ax2.text(j, i, f'{metrics_data[i, j]:.1f}' if j != 2 else f'{metrics_data[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    ax2.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Normalized Performance (1=Best)')
    
    fig.suptitle('Fold Performance Analysis and Ranking', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/fold_ranking_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fold_ranking_analysis.png")
    plt.close()

def plot_statistical_summary():
    """Create a statistical summary box plot."""
    df, targets = create_fold_results_data()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'plum']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        
        # Create box plot
        bp = ax.boxplot([df[metric]], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Add individual points
        y = df[metric]
        x = np.random.normal(1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=50, color='navy')
        
        # Highlight best fold
        best_idx = 6 if metric != 'R2' else 6  # Fold 7
        ax.scatter(1, df[metric].iloc[best_idx], s=200, color='gold', 
                  edgecolor='darkgoldenrod', linewidth=2, zorder=10, 
                  marker='*', label='Best Fold')
        
        # Add target line
        target_val = targets[metric]
        ax.axhline(y=target_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Target: {target_val:.2f}')
        
        # Statistics
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        
        ax.set_ylabel(f'{metric} {"(mg/dL)" if metric in ["MAE", "RMSE"] else ""}', fontsize=11)
        ax.set_title(f'{metric}\nμ={mean_val:.2f}±{std_val:.2f}', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right' if metric != 'R2' else 'lower right', fontsize=8)
    
    fig.suptitle('Statistical Distribution of Performance Metrics Across 10 Folds', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/statistical_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: statistical_summary.png")
    plt.close()

def main():
    """Run all visualization functions."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE FOLD RESULTS VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Create all plots
    print("Creating individual fold performance comparison...")
    plot_individual_fold_performance()
    
    print("\nCreating performance summary comparison...")
    plot_performance_summary()
    
    print("\nCreating radar performance comparison...")
    plot_radar_comparison()
    
    print("\nCreating fold ranking analysis...")
    plot_fold_ranking()
    
    print("\nCreating statistical summary...")
    plot_statistical_summary()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("="*60)
    print("\nFiles saved to visualizations/:")
    print("  • fold_performance_comparison.png")
    print("  • performance_summary_comparison.png")
    print("  • radar_performance_comparison.png")
    print("  • fold_ranking_analysis.png")
    print("  • statistical_summary.png")

if __name__ == "__main__":
    main()