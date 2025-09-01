#!/usr/bin/env python3
"""
Comprehensive Training and Evaluation Pipeline for PPG Glucose Estimation
Training & Evaluation Specialist Implementation
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress, BarColumn, TimeElapsedColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.hybrid_model import HybridCNNGRU, ModelConfig
from src.data.dataloader import PPGDataset

console = Console()
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console.print(f"[bold green]Using device: {device}[/bold green]")

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    min_delta: float = 0.001
    weight_decay: float = 0.01
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    gradient_clip: float = 1.0
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter >= self.patience:
                console.print(f"[yellow]Early stopping triggered after {self.counter} epochs without improvement[/yellow]")
            self.early_stop = self.counter >= self.patience
        return self.early_stop


class ModelTrainer:
    """Handles model training with comprehensive logging and checkpointing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Training Epoch {epoch+1}", total=len(train_loader))
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                
                optimizer.step()
                
                total_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy().flatten())
                all_targets.extend(target.detach().cpu().numpy().flatten())
                
                progress.update(task, advance=1)
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mape = np.mean(np.abs((np.array(all_targets) - np.array(all_predictions)) / np.array(all_targets))) * 100
        r2 = r2_score(all_targets, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/MAE', mae, epoch)
        self.writer.add_scalar('Train/RMSE', rmse, epoch)
        self.writer.add_scalar('Train/MAPE', mape, epoch)
        self.writer.add_scalar('Train/R2', r2, epoch)
        
        return metrics
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """Validate model for one epoch."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"[magenta]Validation Epoch {epoch+1}", total=len(val_loader))
                
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    total_loss += loss.item()
                    all_predictions.extend(output.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy().flatten())
                    
                    progress.update(task, advance=1)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mape = np.mean(np.abs((np.array(all_targets) - np.array(all_predictions)) / np.array(all_targets))) * 100
        r2 = r2_score(all_targets, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/MAE', mae, epoch)
        self.writer.add_scalar('Val/RMSE', rmse, epoch)
        self.writer.add_scalar('Val/MAPE', mape, epoch)
        self.writer.add_scalar('Val/R2', r2, epoch)
        
        return metrics
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Complete training loop with early stopping and checkpointing."""
        console.print(f"\n[bold green]Starting training with configuration:[/bold green]")
        console.print(f"[cyan]Learning Rate:[/cyan] {self.config.learning_rate}")
        console.print(f"[cyan]Batch Size:[/cyan] {self.config.batch_size}")
        console.print(f"[cyan]Max Epochs:[/cyan] {self.config.epochs}")
        console.print(f"[cyan]Early Stopping Patience:[/cyan] {self.config.patience}")
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config.scheduler_factor, 
            patience=self.config.scheduler_patience,
            verbose=True
        )
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta)
        
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        for epoch in range(self.config.epochs):
            console.print(f"\n[bold blue]Epoch {epoch+1}/{self.config.epochs}[/bold blue]")
            
            # Training
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            training_history['train_losses'].append(train_metrics['loss'])
            training_history['train_metrics'].append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(model, val_loader, criterion, epoch)
            training_history['val_losses'].append(val_metrics['loss'])
            training_history['val_metrics'].append(val_metrics)
            
            # Scheduler step
            scheduler.step(val_metrics['loss'])
            
            # Print epoch summary
            console.print(f"[green]Train Loss: {train_metrics['loss']:.6f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}[/green]")
            console.print(f"[blue]Val Loss: {val_metrics['loss']:.6f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}[/blue]")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if self.config.save_best:
                    checkpoint_path = Path(self.config.checkpoint_dir) / f"best_model_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_metrics': val_metrics
                    }, checkpoint_path)
                    console.print(f"[yellow]Saved best model to {checkpoint_path}[/yellow]")
            
            # Early stopping
            if early_stopping(val_metrics['loss']):
                console.print(f"[red]Early stopping at epoch {epoch+1}[/red]")
                break
        
        self.writer.close()
        return training_history


def create_synthetic_dataset(n_samples: int = 1000, window_length: int = 300, 
                           glucose_range: Tuple[float, float] = (70.0, 200.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic PPG dataset for training."""
    console.print(f"[cyan]Creating synthetic dataset with {n_samples} samples...[/cyan]")
    
    ppg_data = []
    glucose_data = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in track(range(n_samples), description="Generating samples..."):
        # Generate random glucose value
        glucose = np.random.uniform(glucose_range[0], glucose_range[1])
        
        # Generate synthetic PPG signal (normalized)
        t = np.linspace(0, 10, window_length)  # 10 seconds at 30 Hz
        
        # Base PPG signal with heart rate variation based on glucose
        base_hr = 70 + (glucose - 100) * 0.1  # Slight correlation with glucose
        ppg_signal = np.sin(2 * np.pi * base_hr / 60 * t)
        
        # Add glucose-dependent amplitude and noise
        amplitude = 1.0 + (glucose - 100) * 0.002
        noise_level = 0.1 + (glucose - 100) * 0.0005
        
        ppg_signal = amplitude * ppg_signal + np.random.normal(0, noise_level, window_length)
        
        # Normalize signal
        ppg_signal = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-8)
        
        ppg_data.append(ppg_signal)
        glucose_data.append(glucose)
    
    return np.array(ppg_data), np.array(glucose_data)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in track(test_loader, description="Evaluating model..."):
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_predictions.extend(output.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mape = calculate_mape(np.array(all_targets), np.array(all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': all_predictions,
        'targets': all_targets
    }


def cross_validation_evaluation(ppg_data: np.ndarray, glucose_data: np.ndarray, 
                              model_config: ModelConfig, training_config: TrainingConfig,
                              k_folds: int = 10) -> Dict[str, Any]:
    """Perform k-fold cross-validation."""
    console.print(f"\n[bold green]Starting {k_folds}-fold cross-validation...[/bold green]")
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = {
        'mae_scores': [],
        'rmse_scores': [],
        'mape_scores': [],
        'r2_scores': [],
        'fold_details': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(ppg_data)):
        console.print(f"\n[cyan]Fold {fold + 1}/{k_folds}[/cyan]")
        
        # Split data
        X_train, X_val = ppg_data[train_idx], ppg_data[val_idx]
        y_train, y_val = glucose_data[train_idx], glucose_data[val_idx]
        
        # Create datasets and loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
        
        # Initialize model
        model = HybridCNNGRU(model_config).to(device)
        
        # Create trainer with reduced epochs for CV
        cv_training_config = TrainingConfig(
            learning_rate=training_config.learning_rate,
            batch_size=training_config.batch_size,
            epochs=min(50, training_config.epochs),  # Reduced for CV
            patience=10,
            checkpoint_dir=f"cv_checkpoints/fold_{fold+1}",
            log_dir=f"cv_logs/fold_{fold+1}"
        )
        
        trainer = ModelTrainer(cv_training_config)
        
        # Train model
        training_history = trainer.train(model, train_loader, val_loader)
        
        # Evaluate on validation set
        metrics = evaluate_model(model, val_loader)
        
        cv_results['mae_scores'].append(metrics['mae'])
        cv_results['rmse_scores'].append(metrics['rmse'])
        cv_results['mape_scores'].append(metrics['mape'])
        cv_results['r2_scores'].append(metrics['r2'])
        
        fold_detail = {
            'fold': fold + 1,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'r2': metrics['r2'],
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        cv_results['fold_details'].append(fold_detail)
        
        console.print(f"[green]Fold {fold+1} Results - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}[/green]")
    
    return cv_results


def create_visualizations(training_history: Dict[str, Any], test_metrics: Dict[str, float],
                         cv_results: Dict[str, Any], save_dir: str = "visualizations"):
    """Create comprehensive performance visualizations."""
    console.print("[cyan]Creating performance visualizations...[/cyan]")
    
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Training/Validation Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(training_history['train_losses']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves
    train_mae = [m['mae'] for m in training_history['train_metrics']]
    val_mae = [m['mae'] for m in training_history['val_metrics']]
    axes[0, 1].plot(epochs, train_mae, 'b-', label='Training MAE', linewidth=2)
    axes[0, 1].plot(epochs, val_mae, 'r-', label='Validation MAE', linewidth=2)
    axes[0, 1].set_title('MAE Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (mg/dL)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² curves
    train_r2 = [m['r2'] for m in training_history['train_metrics']]
    val_r2 = [m['r2'] for m in training_history['val_metrics']]
    axes[1, 0].plot(epochs, train_r2, 'b-', label='Training R²', linewidth=2)
    axes[1, 0].plot(epochs, val_r2, 'r-', label='Validation R²', linewidth=2)
    axes[1, 0].set_title('R² Score Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAPE curves
    train_mape = [m['mape'] for m in training_history['train_metrics']]
    val_mape = [m['mape'] for m in training_history['val_metrics']]
    axes[1, 1].plot(epochs, train_mape, 'b-', label='Training MAPE', linewidth=2)
    axes[1, 1].plot(epochs, val_mape, 'r-', label='Validation MAPE', linewidth=2)
    axes[1, 1].set_title('MAPE Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Predicted vs Actual Scatter Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    predictions = test_metrics['predictions']
    targets = test_metrics['targets']
    
    ax.scatter(targets, predictions, alpha=0.6, s=50)
    ax.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'Predicted vs Actual Glucose\nR² = {test_metrics["r2"]:.4f}, MAE = {test_metrics["mae"]:.2f} mg/dL', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(targets, predictions)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    residuals = np.array(targets) - np.array(predictions)
    
    # Residuals vs Predicted
    axes[0].scatter(predictions, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Glucose (mg/dL)')
    axes[0].set_ylabel('Residuals (mg/dL)')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (mg/dL)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/residual_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cross-validation results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('10-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
    
    metrics_names = ['mae_scores', 'rmse_scores', 'mape_scores', 'r2_scores']
    titles = ['MAE (mg/dL)', 'RMSE (mg/dL)', 'MAPE (%)', 'R² Score']
    
    for i, (metric, title) in enumerate(zip(metrics_names, titles)):
        row, col = i // 2, i % 2
        scores = cv_results[metric]
        
        axes[row, col].boxplot(scores, patch_artist=True)
        axes[row, col].set_title(f'{title}')
        axes[row, col].set_ylabel(title)
        axes[row, col].grid(True, alpha=0.3)
        
        # Add mean line
        mean_score = np.mean(scores)
        axes[row, col].axhline(y=mean_score, color='r', linestyle='--', linewidth=2, 
                              label=f'Mean: {mean_score:.4f}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cross_validation_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]All visualizations saved to {save_dir}/[/green]")


def generate_comparative_analysis(test_metrics: Dict[str, float], cv_results: Dict[str, Any]):
    """Generate comparative analysis with baseline methods."""
    console.print("\n[bold cyan]COMPARATIVE ANALYSIS WITH BASELINE METHODS[/bold cyan]")
    
    # Baseline methods from literature
    baseline_methods = {
        'Fu-Liang Yang': {'mae': 8.9, 'r2': 0.71},
        'Kim K.-D': {'mae': 7.05, 'r2': 0.92},
        'LRCN': {'mae': 4.7, 'r2': 0.88},
        'Our CNN-GRU': {
            'mae': test_metrics['mae'],
            'rmse': test_metrics['rmse'],
            'mape': test_metrics['mape'],
            'r2': test_metrics['r2']
        }
    }
    
    # Create comparison table
    table = Table(title="Performance Comparison with Baseline Methods")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("MAE (mg/dL)", style="magenta")
    table.add_column("RMSE (mg/dL)", style="magenta")
    table.add_column("MAPE (%)", style="magenta")
    table.add_column("R² Score", style="green")
    
    for method, metrics in baseline_methods.items():
        mae_str = f"{metrics.get('mae', 'N/A'):.2f}" if isinstance(metrics.get('mae'), (int, float)) else "N/A"
        rmse_str = f"{metrics.get('rmse', 'N/A'):.2f}" if isinstance(metrics.get('rmse'), (int, float)) else "N/A"
        mape_str = f"{metrics.get('mape', 'N/A'):.2f}" if isinstance(metrics.get('mape'), (int, float)) else "N/A"
        r2_str = f"{metrics.get('r2', 'N/A'):.4f}" if isinstance(metrics.get('r2'), (int, float)) else "N/A"
        
        table.add_row(method, mae_str, rmse_str, mape_str, r2_str)
    
    console.print(table)
    
    # Cross-validation summary
    console.print("\n[bold yellow]10-FOLD CROSS-VALIDATION SUMMARY[/bold yellow]")
    
    cv_table = Table(title="Cross-Validation Results Summary")
    cv_table.add_column("Metric", style="cyan")
    cv_table.add_column("Mean ± Std", style="magenta")
    cv_table.add_column("Range", style="green")
    cv_table.add_column("Target", style="yellow")
    
    metrics_info = [
        ("MAE (mg/dL)", cv_results['mae_scores'], "2.96"),
        ("RMSE (mg/dL)", cv_results['rmse_scores'], "3.94"),
        ("MAPE (%)", cv_results['mape_scores'], "2.40"),
        ("R² Score", cv_results['r2_scores'], "0.97")
    ]
    
    for metric_name, scores, target in metrics_info:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        cv_table.add_row(
            metric_name,
            f"{mean_score:.4f} ± {std_score:.4f}",
            f"{min_score:.4f} - {max_score:.4f}",
            target
        )
    
    console.print(cv_table)
    
    # Performance assessment
    mae_mean = np.mean(cv_results['mae_scores'])
    rmse_mean = np.mean(cv_results['rmse_scores'])
    mape_mean = np.mean(cv_results['mape_scores'])
    r2_mean = np.mean(cv_results['r2_scores'])
    
    console.print("\n[bold green]PERFORMANCE ASSESSMENT:[/bold green]")
    
    # Check against targets
    targets = {'mae': 2.96, 'rmse': 3.94, 'mape': 2.40, 'r2': 0.97}
    results = {'mae': mae_mean, 'rmse': rmse_mean, 'mape': mape_mean, 'r2': r2_mean}
    
    for metric, target in targets.items():
        actual = results[metric]
        if metric == 'r2':
            status = "✓" if actual >= target else "✗"
            diff = f"{((actual - target) / target * 100):+.1f}%"
        else:
            status = "✓" if actual <= target else "✗"
            diff = f"{((actual - target) / target * 100):+.1f}%"
        
        console.print(f"{status} {metric.upper()}: {actual:.4f} vs target {target:.4f} ({diff})")


def save_training_logs(training_history: Dict[str, Any], test_metrics: Dict[str, float], 
                      cv_results: Dict[str, Any], model_config: ModelConfig, 
                      training_config: TrainingConfig, save_path: str = "training_logs.json"):
    """Save comprehensive training logs."""
    logs = {
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'input_length': model_config.input_length,
            'cnn_small_kernels': model_config.cnn_small_kernels,
            'cnn_small_channels': model_config.cnn_small_channels,
            'cnn_large_kernels': model_config.cnn_large_kernels,
            'cnn_large_channels': model_config.cnn_large_channels,
            'gru_layers': model_config.gru_layers,
            'gru_hidden': model_config.gru_hidden,
            'gru_bidirectional': model_config.gru_bidirectional,
            'dense_dims': model_config.dense_dims,
            'dropout': model_config.dropout
        },
        'training_config': {
            'learning_rate': training_config.learning_rate,
            'batch_size': training_config.batch_size,
            'epochs': training_config.epochs,
            'patience': training_config.patience,
            'weight_decay': training_config.weight_decay
        },
        'training_history': {
            'train_losses': training_history['train_losses'],
            'val_losses': training_history['val_losses'],
            'epochs_trained': len(training_history['train_losses'])
        },
        'test_metrics': test_metrics,
        'cross_validation': {
            'n_folds': 10,
            'mae_scores': cv_results['mae_scores'],
            'rmse_scores': cv_results['rmse_scores'],
            'mape_scores': cv_results['mape_scores'],
            'r2_scores': cv_results['r2_scores'],
            'mae_mean': float(np.mean(cv_results['mae_scores'])),
            'mae_std': float(np.std(cv_results['mae_scores'])),
            'rmse_mean': float(np.mean(cv_results['rmse_scores'])),
            'rmse_std': float(np.std(cv_results['rmse_scores'])),
            'mape_mean': float(np.mean(cv_results['mape_scores'])),
            'mape_std': float(np.std(cv_results['mape_scores'])),
            'r2_mean': float(np.mean(cv_results['r2_scores'])),
            'r2_std': float(np.std(cv_results['r2_scores']))
        },
        'baseline_comparison': {
            'Fu-Liang Yang': {'mae': 8.9, 'r2': 0.71},
            'Kim K.-D': {'mae': 7.05, 'r2': 0.92},
            'LRCN': {'mae': 4.7, 'r2': 0.88}
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(logs, f, indent=2, default=str)
    
    console.print(f"[green]Training logs saved to {save_path}[/green]")


def main():
    """Main training and evaluation pipeline."""
    console.print(f"\n[bold green]{'='*80}[/bold green]")
    console.print(f"[bold cyan]{'PPG GLUCOSE ESTIMATION - TRAINING & EVALUATION PIPELINE'.center(80)}[/bold cyan]")
    console.print(f"[bold green]{'='*80}[/bold green]")
    
    # Configuration
    model_config = ModelConfig(
        input_length=300,
        cnn_small_kernels=[3, 5],
        cnn_small_channels=[64, 128],
        cnn_large_kernels=[11, 15],
        cnn_large_channels=[64, 128],
        gru_layers=2,
        gru_hidden=128,
        gru_bidirectional=True,
        dense_dims=[256, 128, 64],
        dropout=0.5
    )
    
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        patience=15,
        weight_decay=0.01
    )
    
    # Step 1: Create synthetic dataset
    console.print("\n[bold yellow]Step 1: Data Preparation[/bold yellow]")
    ppg_data, glucose_data = create_synthetic_dataset(n_samples=2000)
    console.print(f"[green]Created dataset with {len(ppg_data)} samples[/green]")
    console.print(f"[green]PPG data shape: {ppg_data.shape}[/green]")
    console.print(f"[green]Glucose range: {glucose_data.min():.1f} - {glucose_data.max():.1f} mg/dL[/green]")
    
    # Step 2: Split data
    console.print("\n[bold yellow]Step 2: Data Splitting[/bold yellow]")
    n_train = int(0.7 * len(ppg_data))
    n_val = int(0.15 * len(ppg_data))
    
    X_train, y_train = ppg_data[:n_train], glucose_data[:n_train]
    X_val, y_val = ppg_data[n_train:n_train+n_val], glucose_data[n_train:n_train+n_val]
    X_test, y_test = ppg_data[n_train+n_val:], glucose_data[n_train+n_val:]
    
    console.print(f"[green]Training samples: {len(X_train)}[/green]")
    console.print(f"[green]Validation samples: {len(X_val)}[/green]")
    console.print(f"[green]Test samples: {len(X_test)}[/green]")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)
    
    # Step 3: Initialize model
    console.print("\n[bold yellow]Step 3: Model Initialization[/bold yellow]")
    model = HybridCNNGRU(model_config).to(device)
    total_params = model.get_num_parameters()
    trainable_params = model.get_num_trainable_parameters()
    
    console.print(f"[green]Model initialized with {total_params:,} total parameters[/green]")
    console.print(f"[green]Trainable parameters: {trainable_params:,}[/green]")
    
    # Step 4: Training
    console.print("\n[bold yellow]Step 4: Model Training[/bold yellow]")
    trainer = ModelTrainer(training_config)
    training_history = trainer.train(model, train_loader, val_loader)
    
    # Step 5: Test evaluation
    console.print("\n[bold yellow]Step 5: Test Set Evaluation[/bold yellow]")
    test_metrics = evaluate_model(model, test_loader)
    
    console.print(f"[bold green]Test Results:[/bold green]")
    console.print(f"[green]MAE: {test_metrics['mae']:.4f} mg/dL[/green]")
    console.print(f"[green]RMSE: {test_metrics['rmse']:.4f} mg/dL[/green]")
    console.print(f"[green]MAPE: {test_metrics['mape']:.4f}%[/green]")
    console.print(f"[green]R² Score: {test_metrics['r2']:.4f}[/green]")
    
    # Step 6: Cross-validation
    console.print("\n[bold yellow]Step 6: 10-Fold Cross-Validation[/bold yellow]")
    cv_results = cross_validation_evaluation(ppg_data, glucose_data, model_config, training_config)
    
    # Step 7: Create visualizations
    console.print("\n[bold yellow]Step 7: Creating Visualizations[/bold yellow]")
    create_visualizations(training_history, test_metrics, cv_results)
    
    # Step 8: Generate comparative analysis
    console.print("\n[bold yellow]Step 8: Comparative Analysis[/bold yellow]")
    generate_comparative_analysis(test_metrics, cv_results)
    
    # Step 9: Save logs
    console.print("\n[bold yellow]Step 9: Saving Training Logs[/bold yellow]")
    save_training_logs(training_history, test_metrics, cv_results, model_config, training_config)
    
    console.print(f"\n[bold green]{'='*80}[/bold green]")
    console.print(f"[bold cyan]{'PIPELINE COMPLETED SUCCESSFULLY!'.center(80)}[/bold cyan]")
    console.print(f"[bold green]{'='*80}[/bold green]")


if __name__ == "__main__":
    main()