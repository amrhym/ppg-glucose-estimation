#!/usr/bin/env python3
"""
Improved training pipeline for PPG glucose estimation with all optimizations.
Implements hyperparameter tuning, advanced augmentation, and ensemble methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch available. Using device: {device}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using NumPy implementation.")

@dataclass
class ImprovedConfig:
    """Enhanced configuration with optimized hyperparameters."""
    
    # Data paths
    data_dir: str = "data/PPG_Dataset"
    checkpoint_dir: str = "improved_checkpoints"
    
    # Optimized hyperparameters
    learning_rate: float = 0.0001  # Reduced from 0.001
    batch_size: int = 16  # Reduced from 32
    max_epochs: int = 100
    early_stopping_patience: int = 20
    
    # Model architecture (enhanced)
    input_size: int = 300  # 10s at 30Hz
    cnn_channels: List[int] = None  # Will be [64, 128, 256]
    gru_hidden: int = 256  # Increased from 128
    gru_layers: int = 3  # Increased from 2
    dropout: float = 0.2  # Reduced from 0.3
    use_attention: bool = True
    use_residual: bool = True
    
    # Training strategy
    optimizer: str = 'AdamW'
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    scheduler: str = 'CosineAnnealingLR'
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    
    # Advanced data augmentation
    augmentation_factor: int = 10  # Increased from 4
    augmentation_methods: List[str] = None  # Will be comprehensive list
    mixup_alpha: float = 0.2
    cutmix_prob: float = 0.5
    
    # Feature engineering
    use_morphological_features: bool = True
    use_spectral_features: bool = True
    use_hrv_features: bool = True
    use_perfusion_index: bool = True
    
    # Ensemble configuration
    n_ensemble_models: int = 5
    ensemble_method: str = 'weighted_average'  # or 'voting', 'stacking'
    
    # Preprocessing (optimized)
    bandpass_low: float = 0.5
    bandpass_high: float = 8.0
    filter_order: int = 4
    target_fs: int = 30
    window_size: int = 10
    window_overlap: float = 0.5
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [64, 128, 256]
        if self.augmentation_methods is None:
            self.augmentation_methods = [
                'gaussian_noise',
                'baseline_wander',
                'amplitude_scaling',
                'time_warping',
                'frequency_masking',
                'time_masking',
                'mixup',
                'cutmix'
            ]

class AdvancedAugmentation:
    """Advanced data augmentation techniques."""
    
    @staticmethod
    def gaussian_noise(signal, noise_level=0.05):
        """Add Gaussian noise."""
        noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
        return signal + noise
    
    @staticmethod
    def baseline_wander(signal, fs=30, max_amplitude=0.1):
        """Simulate baseline wander."""
        duration = len(signal) / fs
        t = np.linspace(0, duration, len(signal))
        # Random low-frequency sine waves
        freq = np.random.uniform(0.15, 0.5)  # 0.15-0.5 Hz
        wander = max_amplitude * np.sin(2 * np.pi * freq * t)
        return signal + wander
    
    @staticmethod
    def amplitude_scaling(signal, scale_range=(0.8, 1.2)):
        """Random amplitude scaling."""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def time_warping(signal, warp_range=(0.95, 1.05)):
        """Time warping through interpolation."""
        warp_factor = np.random.uniform(*warp_range)
        old_length = len(signal)
        new_length = int(old_length * warp_factor)
        
        old_indices = np.arange(old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        
        f = interp1d(old_indices, signal, kind='cubic', fill_value='extrapolate')
        warped = f(new_indices)
        
        # Resample back to original length
        f2 = interp1d(np.arange(new_length), warped, kind='cubic', fill_value='extrapolate')
        return f2(np.linspace(0, new_length - 1, old_length))
    
    @staticmethod
    def frequency_masking(signal, fs=30, mask_param=2):
        """Mask random frequency bands."""
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        
        # Random frequency band to mask
        mask_f = np.random.uniform(0, fs/2)
        mask_width = np.random.uniform(0.5, mask_param)
        
        mask = (freqs >= mask_f - mask_width/2) & (freqs <= mask_f + mask_width/2)
        fft[mask] *= np.random.uniform(0, 0.1)
        
        return np.fft.irfft(fft, len(signal))
    
    @staticmethod
    def time_masking(signal, mask_param=20):
        """Mask random time segments."""
        signal_copy = signal.copy()
        mask_length = np.random.randint(1, min(mask_param, len(signal)))
        mask_start = np.random.randint(0, len(signal) - mask_length)
        
        # Replace with interpolation instead of zeros
        if mask_start > 0 and mask_start + mask_length < len(signal):
            signal_copy[mask_start:mask_start + mask_length] = np.linspace(
                signal_copy[mask_start - 1],
                signal_copy[mask_start + mask_length],
                mask_length
            )
        
        return signal_copy
    
    @staticmethod
    def apply_augmentation(signal, methods, fs=30):
        """Apply multiple augmentation methods."""
        augmented = signal.copy()
        
        for method in methods:
            if method == 'gaussian_noise':
                augmented = AdvancedAugmentation.gaussian_noise(augmented)
            elif method == 'baseline_wander':
                augmented = AdvancedAugmentation.baseline_wander(augmented, fs)
            elif method == 'amplitude_scaling':
                augmented = AdvancedAugmentation.amplitude_scaling(augmented)
            elif method == 'time_warping':
                augmented = AdvancedAugmentation.time_warping(augmented)
            elif method == 'frequency_masking':
                augmented = AdvancedAugmentation.frequency_masking(augmented, fs)
            elif method == 'time_masking':
                augmented = AdvancedAugmentation.time_masking(augmented)
        
        return augmented

class FeatureExtractor:
    """Extract advanced features from PPG signals."""
    
    @staticmethod
    def extract_morphological_features(signal, fs=30):
        """Extract morphological features from PPG."""
        features = {}
        
        # Find peaks
        from scipy import signal as scipy_signal
        peaks, properties = scipy_signal.find_peaks(signal, distance=fs*0.5, prominence=0.1)
        
        if len(peaks) > 1:
            # Peak intervals
            intervals = np.diff(peaks) / fs
            features['mean_interval'] = np.mean(intervals)
            features['std_interval'] = np.std(intervals)
            
            # Peak amplitudes
            amplitudes = signal[peaks]
            features['mean_amplitude'] = np.mean(amplitudes)
            features['std_amplitude'] = np.std(amplitudes)
            
            # Peak prominences
            features['mean_prominence'] = np.mean(properties.get('prominences', [0]))
            
            # Pulse width
            if 'widths' in properties:
                features['mean_width'] = np.mean(properties['widths']) / fs
        else:
            # Default values if not enough peaks
            features = {k: 0 for k in ['mean_interval', 'std_interval', 
                                       'mean_amplitude', 'std_amplitude',
                                       'mean_prominence', 'mean_width']}
        
        return features
    
    @staticmethod
    def extract_spectral_features(signal, fs=30):
        """Extract frequency domain features."""
        features = {}
        
        # Power spectral density
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        
        # Frequency bands
        vlf_band = (0.003, 0.04)  # Very low frequency
        lf_band = (0.04, 0.15)   # Low frequency
        hf_band = (0.15, 0.4)    # High frequency
        hr_band = (0.5, 3.0)     # Heart rate band
        
        # Band powers
        vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        hr_power = np.trapz(psd[(freqs >= hr_band[0]) & (freqs < hr_band[1])])
        
        features['vlf_power'] = vlf_power
        features['lf_power'] = lf_power
        features['hf_power'] = hf_power
        features['hr_power'] = hr_power
        features['lf_hf_ratio'] = lf_power / (hf_power + 1e-8)
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-8)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
        
        # Peak frequency
        features['peak_frequency'] = freqs[np.argmax(psd)]
        
        return features
    
    @staticmethod
    def extract_hrv_features(signal, fs=30):
        """Extract heart rate variability features."""
        features = {}
        
        # Find R-R intervals
        from scipy import signal as scipy_signal
        peaks, _ = scipy_signal.find_peaks(signal, distance=fs*0.5, prominence=0.1)
        
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
            
            # Time domain features
            features['mean_rr'] = np.mean(rr_intervals)
            features['std_rr'] = np.std(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
            
            # pNN50: percentage of successive RR intervals that differ by more than 50ms
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            features['pnn50'] = (nn50 / len(rr_intervals)) * 100
            
            # Heart rate
            features['mean_hr'] = 60000 / np.mean(rr_intervals)  # BPM
            features['std_hr'] = np.std(60000 / rr_intervals)
        else:
            # Default values
            features = {k: 0 for k in ['mean_rr', 'std_rr', 'rmssd', 
                                       'pnn50', 'mean_hr', 'std_hr']}
        
        return features
    
    @staticmethod
    def extract_perfusion_features(signal):
        """Extract perfusion-related features."""
        features = {}
        
        # DC and AC components
        dc_component = np.mean(signal)
        ac_component = np.max(signal) - np.min(signal)
        
        # Perfusion index
        features['perfusion_index'] = (ac_component / dc_component) * 100 if dc_component != 0 else 0
        
        # Signal quality index
        features['signal_quality'] = 1 - (np.std(signal) / (np.mean(np.abs(signal)) + 1e-8))
        
        return features
    
    @staticmethod
    def extract_all_features(signal, fs=30):
        """Extract all features."""
        all_features = {}
        
        # Morphological features
        morph_features = FeatureExtractor.extract_morphological_features(signal, fs)
        all_features.update({f'morph_{k}': v for k, v in morph_features.items()})
        
        # Spectral features
        spectral_features = FeatureExtractor.extract_spectral_features(signal, fs)
        all_features.update({f'spectral_{k}': v for k, v in spectral_features.items()})
        
        # HRV features
        hrv_features = FeatureExtractor.extract_hrv_features(signal, fs)
        all_features.update({f'hrv_{k}': v for k, v in hrv_features.items()})
        
        # Perfusion features
        perfusion_features = FeatureExtractor.extract_perfusion_features(signal)
        all_features.update({f'perfusion_{k}': v for k, v in perfusion_features.items()})
        
        return all_features

if TORCH_AVAILABLE:
    class ImprovedHybridModel(nn.Module):
        """Enhanced CNN-GRU model with attention and residual connections."""
        
        def __init__(self, config: ImprovedConfig):
            super().__init__()
            self.config = config
            
            # Enhanced CNN branches with different kernel sizes
            self.cnn_branch1 = self._create_cnn_branch([3, 5, 7], config.cnn_channels)
            self.cnn_branch2 = self._create_cnn_branch([7, 11, 15], config.cnn_channels)
            self.cnn_branch3 = self._create_cnn_branch([15, 21, 27], config.cnn_channels)
            
            # Calculate CNN output size
            cnn_out_size = len(config.cnn_channels) * 3 * (config.input_size // 8)
            
            # Enhanced bidirectional GRU
            self.gru = nn.GRU(
                config.input_size,
                config.gru_hidden,
                config.gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.gru_layers > 1 else 0
            )
            
            gru_out_size = config.gru_hidden * 2  # Bidirectional
            
            # Self-attention mechanism
            if config.use_attention:
                self.attention = nn.MultiheadAttention(
                    gru_out_size,
                    num_heads=8,
                    dropout=config.dropout,
                    batch_first=True
                )
                self.layer_norm = nn.LayerNorm(gru_out_size)
            
            # Feature fusion with residual connections
            fusion_input_size = cnn_out_size + gru_out_size
            
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_size, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(config.dropout),
                
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(config.dropout),
                
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(config.dropout),
                
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(config.dropout),
                
                nn.Linear(64, 1)
            )
            
            # Residual connections
            if config.use_residual:
                self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        def _create_cnn_branch(self, kernel_sizes, channels):
            """Create a CNN branch with specified kernel sizes."""
            layers = []
            in_channels = 1
            
            for i, (kernel_size, out_channels) in enumerate(zip(kernel_sizes, channels)):
                layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, padding_mode='replicate'),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.MaxPool1d(2)
                ])
                in_channels = out_channels
            
            return nn.Sequential(*layers)
        
        def forward(self, x):
            batch_size = x.shape[0]
            
            # Reshape for CNN (add channel dimension)
            x_cnn = x.unsqueeze(1)  # [batch, 1, seq_len]
            
            # CNN branches
            cnn1_out = self.cnn_branch1(x_cnn).flatten(1)
            cnn2_out = self.cnn_branch2(x_cnn).flatten(1)
            cnn3_out = self.cnn_branch3(x_cnn).flatten(1)
            cnn_features = torch.cat([cnn1_out, cnn2_out, cnn3_out], dim=1)
            
            # GRU processing
            x_gru = x.unsqueeze(1)  # [batch, 1, seq_len] for sequence
            gru_out, _ = self.gru(x_gru)
            gru_features = gru_out[:, -1, :]  # Take last output
            
            # Apply attention if enabled
            if self.config.use_attention:
                attn_out, _ = self.attention(gru_out, gru_out, gru_out)
                gru_features = self.layer_norm(gru_features + attn_out[:, -1, :])
            
            # Combine features
            combined = torch.cat([cnn_features, gru_features], dim=1)
            
            # Final prediction
            output = self.fusion(combined)
            
            # Apply residual connection if enabled
            if self.config.use_residual:
                # Simple residual from input mean
                input_mean = x.mean(dim=1, keepdim=True)
                output = output + self.residual_weight * input_mean
            
            return output
    
    class ImprovedDataset(Dataset):
        """Enhanced dataset with augmentation and feature extraction."""
        
        def __init__(self, data_dir: str, config: ImprovedConfig, mode='train'):
            self.data_dir = Path(data_dir)
            self.config = config
            self.mode = mode
            self.samples = []
            self.load_data()
        
        def load_data(self):
            """Load and preprocess PPG data."""
            raw_dir = self.data_dir / "RawData"
            label_dir = self.data_dir / "Labels"
            
            # Load all signal files
            signal_files = sorted(raw_dir.glob("signal_*.csv"))
            
            for signal_file in signal_files:
                # Find corresponding label file
                signal_id = signal_file.stem
                label_file = label_dir / f"label_{signal_id.split('_', 1)[1]}.csv"
                
                if label_file.exists():
                    # Load signal
                    signal_data = pd.read_csv(signal_file, header=None).values.flatten()
                    
                    # Load label
                    label_data = pd.read_csv(label_file)
                    glucose = label_data['Glucose'].values[0]
                    
                    # Preprocess
                    processed = self.preprocess_signal(signal_data)
                    
                    if processed is not None:
                        # Apply augmentation in training mode
                        if self.mode == 'train':
                            for _ in range(self.config.augmentation_factor):
                                augmented = AdvancedAugmentation.apply_augmentation(
                                    processed,
                                    np.random.choice(self.config.augmentation_methods, 
                                                   size=np.random.randint(1, 4), 
                                                   replace=False),
                                    self.config.target_fs
                                )
                                
                                # Extract features if enabled
                                if self.config.use_morphological_features or \
                                   self.config.use_spectral_features or \
                                   self.config.use_hrv_features:
                                    features = FeatureExtractor.extract_all_features(
                                        augmented, self.config.target_fs
                                    )
                                    self.samples.append((augmented, glucose, features))
                                else:
                                    self.samples.append((augmented, glucose, None))
                        else:
                            # No augmentation for validation/test
                            if self.config.use_morphological_features or \
                               self.config.use_spectral_features or \
                               self.config.use_hrv_features:
                                features = FeatureExtractor.extract_all_features(
                                    processed, self.config.target_fs
                                )
                                self.samples.append((processed, glucose, features))
                            else:
                                self.samples.append((processed, glucose, None))
            
            logger.info(f"Loaded {len(self.samples)} samples for {self.mode}")
        
        def preprocess_signal(self, signal):
            """Apply preprocessing pipeline."""
            try:
                from scipy import signal as scipy_signal
                
                # Original sampling rate
                orig_fs = 2175
                
                # Bandpass filter
                sos = scipy_signal.butter(self.config.filter_order, 
                                   [self.config.bandpass_low, self.config.bandpass_high],
                                   btype='band', fs=orig_fs, output='sos')
                filtered = scipy_signal.sosfilt(sos, signal)
                
                # Downsample
                downsample_factor = orig_fs // self.config.target_fs
                downsampled = scipy_signal.decimate(filtered, downsample_factor)
                
                # Create windows
                window_samples = int(self.config.window_size * self.config.target_fs)
                if len(downsampled) >= window_samples:
                    # Take center window
                    start = (len(downsampled) - window_samples) // 2
                    window = downsampled[start:start + window_samples]
                    
                    # Normalize
                    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    
                    return window
                
                return None
                
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}")
                return None
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            signal, glucose, features = self.samples[idx]
            
            # Convert to tensors
            signal_tensor = torch.FloatTensor(signal)
            glucose_tensor = torch.FloatTensor([glucose])
            
            if features is not None:
                features_tensor = torch.FloatTensor(list(features.values()))
                return signal_tensor, glucose_tensor, features_tensor
            
            return signal_tensor, glucose_tensor
    
    class ImprovedTrainer:
        """Enhanced trainer with advanced strategies."""
        
        def __init__(self, config: ImprovedConfig):
            self.config = config
            self.device = device
            self.best_models = []
        
        def train_model(self, train_loader, val_loader, model_idx=0):
            """Train a single model with improved strategies."""
            
            # Initialize model
            model = ImprovedHybridModel(self.config).to(self.device)
            
            # Optimizer with weight decay
            if self.config.optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate
                )
            
            # Learning rate scheduler
            if self.config.scheduler == 'CosineAnnealingLR':
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_epochs,
                    eta_min=1e-6
                )
            elif self.config.scheduler == 'OneCycleLR':
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.config.learning_rate * 10,
                    epochs=self.config.max_epochs,
                    steps_per_epoch=len(train_loader)
                )
            else:
                scheduler = None
            
            # Loss function
            criterion = nn.SmoothL1Loss()  # More robust than MSE
            
            # Training loop
            best_val_mae = float('inf')
            patience_counter = 0
            training_history = {'train_loss': [], 'val_mae': []}
            
            for epoch in range(self.config.max_epochs):
                # Training phase
                model.train()
                train_losses = []
                
                for batch in train_loader:
                    if len(batch) == 3:
                        inputs, targets, features = batch
                    else:
                        inputs, targets = batch
                        features = None
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Mixup augmentation
                    if self.config.mixup_alpha > 0 and np.random.random() < 0.5:
                        inputs, targets = self.mixup(inputs, targets, self.config.mixup_alpha)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss with label smoothing
                    loss = criterion(outputs, targets)
                    if self.config.label_smoothing > 0:
                        loss = (1 - self.config.label_smoothing) * loss + \
                               self.config.label_smoothing * loss.mean()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    
                    optimizer.step()
                    train_losses.append(loss.item())
                
                # Validation phase
                val_mae = self.evaluate(model, val_loader)
                
                # Learning rate scheduling
                if scheduler:
                    if self.config.scheduler == 'OneCycleLR':
                        scheduler.step()
                    else:
                        scheduler.step()
                
                # Logging
                avg_train_loss = np.mean(train_losses)
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_mae'].append(val_mae)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Model {model_idx} - Epoch {epoch+1}/{self.config.max_epochs}: "
                              f"Train Loss: {avg_train_loss:.4f}, Val MAE: {val_mae:.2f}")
                
                # Early stopping
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    patience_counter = 0
                    # Save best model
                    checkpoint_path = Path(self.config.checkpoint_dir) / f"model_{model_idx}_best.pth"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_mae': val_mae,
                        'config': self.config
                    }, checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            # Load best model
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            return model, best_val_mae, training_history
        
        def mixup(self, inputs, targets, alpha=0.2):
            """Mixup augmentation."""
            batch_size = inputs.size(0)
            
            # Generate mixup weights
            lam = np.random.beta(alpha, alpha)
            
            # Random shuffle for mixing
            index = torch.randperm(batch_size).to(self.device)
            
            # Mix inputs and targets
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            mixed_targets = lam * targets + (1 - lam) * targets[index]
            
            return mixed_inputs, mixed_targets
        
        def evaluate(self, model, data_loader):
            """Evaluate model performance."""
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in data_loader:
                    if len(batch) == 3:
                        inputs, targets, features = batch
                    else:
                        inputs, targets = batch
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
            
            predictions = np.concatenate(all_predictions)
            targets = np.concatenate(all_targets)
            
            mae = np.mean(np.abs(predictions - targets))
            return mae
        
        def train_ensemble(self, train_loader, val_loader):
            """Train ensemble of models."""
            logger.info(f"Training ensemble of {self.config.n_ensemble_models} models...")
            
            ensemble_models = []
            ensemble_scores = []
            
            for i in range(self.config.n_ensemble_models):
                logger.info(f"\nTraining model {i+1}/{self.config.n_ensemble_models}")
                
                # Train individual model
                model, val_mae, history = self.train_model(train_loader, val_loader, model_idx=i)
                
                ensemble_models.append(model)
                ensemble_scores.append(val_mae)
                
                logger.info(f"Model {i+1} achieved VAL MAE: {val_mae:.2f}")
            
            # Calculate ensemble weights based on performance
            if self.config.ensemble_method == 'weighted_average':
                # Weight inversely proportional to MAE
                weights = 1 / (np.array(ensemble_scores) + 1e-8)
                weights = weights / weights.sum()
                logger.info(f"Ensemble weights: {weights}")
            else:
                weights = np.ones(self.config.n_ensemble_models) / self.config.n_ensemble_models
            
            return ensemble_models, weights
        
        def ensemble_predict(self, models, weights, data_loader):
            """Make predictions with ensemble."""
            all_predictions = []
            all_targets = []
            
            for batch in data_loader:
                if len(batch) == 3:
                    inputs, targets, features = batch
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(self.device)
                
                # Get predictions from all models
                batch_predictions = []
                for model in models:
                    model.eval()
                    with torch.no_grad():
                        pred = model(inputs).cpu().numpy()
                        batch_predictions.append(pred)
                
                # Weighted average
                batch_predictions = np.array(batch_predictions)
                weighted_pred = np.sum(batch_predictions * weights.reshape(-1, 1, 1), axis=0)
                
                all_predictions.append(weighted_pred)
                all_targets.append(targets.numpy())
            
            predictions = np.concatenate(all_predictions)
            targets = np.concatenate(all_targets)
            
            return predictions, targets

def main():
    """Main training pipeline with all improvements."""
    
    logger.info("="*80)
    logger.info("IMPROVED PPG GLUCOSE ESTIMATION TRAINING PIPELINE")
    logger.info("="*80)
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for training. Please install PyTorch.")
        return
    
    # Initialize configuration
    config = ImprovedConfig()
    logger.info(f"Configuration: {config}")
    
    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = ImprovedDataset(config.data_dir, config, mode='train')
    val_dataset = ImprovedDataset(config.data_dir, config, mode='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = ImprovedTrainer(config)
    
    # Train ensemble
    ensemble_models, weights = trainer.train_ensemble(train_loader, val_loader)
    
    # Evaluate ensemble
    logger.info("\nEvaluating ensemble...")
    predictions, targets = trainer.ensemble_predict(ensemble_models, weights, val_loader)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    r2 = 1 - np.sum((targets - predictions)**2) / np.sum((targets - np.mean(targets))**2)
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100
    
    logger.info("\n" + "="*80)
    logger.info("FINAL ENSEMBLE PERFORMANCE")
    logger.info("="*80)
    logger.info(f"MAE:  {mae:.2f} mg/dL")
    logger.info(f"RMSE: {rmse:.2f} mg/dL")
    logger.info(f"R²:   {r2:.3f}")
    logger.info(f"MAPE: {mape:.1f}%")
    
    # Save results
    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'ensemble_weights': weights.tolist(),
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path(config.checkpoint_dir) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    return results

if __name__ == "__main__":
    results = main()