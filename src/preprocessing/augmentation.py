"""Data augmentation utilities for PPG signals."""

from typing import List, Optional, Tuple

import numpy as np


class DataAugmenter:
    """Augment PPG signals for training.
    
    Examples:
        >>> augmenter = DataAugmenter(gaussian_noise_std=[0.01, 0.02])
        >>> augmented = augmenter.augment(ppg_window)
    """
    
    def __init__(
        self,
        gaussian_noise_std: Optional[List[float]] = None,
        amplitude_scale: Optional[Tuple[float, float]] = None,
        baseline_wander_amp: Optional[float] = None,
        time_jitter_samples: Optional[int] = None,
    ):
        """Initialize data augmenter.
        
        Args:
            gaussian_noise_std: List of noise std values to apply
            amplitude_scale: (min, max) scaling factors
            baseline_wander_amp: Amplitude of baseline wander
            time_jitter_samples: Max samples to shift
        """
        self.gaussian_noise_std = gaussian_noise_std or [0.01, 0.02]
        self.amplitude_scale = amplitude_scale or (0.9, 1.1)
        self.baseline_wander_amp = baseline_wander_amp
        self.time_jitter_samples = time_jitter_samples
    
    def augment(
        self,
        signal_data: np.ndarray,
        n_augmentations: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Generate augmented versions of signal.
        
        Args:
            signal_data: Input signal window
            n_augmentations: Number of augmented versions (None = all)
            
        Returns:
            List of augmented signals
        """
        augmented = []
        
        # Gaussian noise augmentations
        for noise_std in self.gaussian_noise_std:
            if noise_std > 0:
                aug = self.add_gaussian_noise(signal_data, noise_std)
                augmented.append(aug)
        
        # Amplitude scaling
        if self.amplitude_scale:
            scale = np.random.uniform(*self.amplitude_scale)
            aug = self.scale_amplitude(signal_data, scale)
            augmented.append(aug)
        
        # Baseline wander
        if self.baseline_wander_amp:
            aug = self.add_baseline_wander(signal_data, self.baseline_wander_amp)
            augmented.append(aug)
        
        # Time jitter
        if self.time_jitter_samples:
            shift = np.random.randint(-self.time_jitter_samples, self.time_jitter_samples + 1)
            aug = self.apply_time_jitter(signal_data, shift)
            augmented.append(aug)
        
        # Limit number of augmentations if specified
        if n_augmentations and len(augmented) > n_augmentations:
            indices = np.random.choice(len(augmented), n_augmentations, replace=False)
            augmented = [augmented[i] for i in indices]
        
        return augmented
    
    def add_gaussian_noise(
        self,
        signal_data: np.ndarray,
        noise_std: float,
    ) -> np.ndarray:
        """Add Gaussian noise to signal.
        
        Args:
            signal_data: Input signal
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy signal
        """
        signal_std = np.std(signal_data)
        noise = np.random.normal(0, noise_std * signal_std, len(signal_data))
        return signal_data + noise
    
    def scale_amplitude(
        self,
        signal_data: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Scale signal amplitude.
        
        Args:
            signal_data: Input signal
            scale: Scaling factor
            
        Returns:
            Scaled signal
        """
        # Preserve mean
        mean = np.mean(signal_data)
        centered = signal_data - mean
        scaled = centered * scale
        return scaled + mean
    
    def add_baseline_wander(
        self,
        signal_data: np.ndarray,
        amplitude: float,
        frequency: float = 0.1,
    ) -> np.ndarray:
        """Add sinusoidal baseline wander.
        
        Args:
            signal_data: Input signal
            amplitude: Wander amplitude
            frequency: Wander frequency (Hz)
            
        Returns:
            Signal with baseline wander
        """
        n_samples = len(signal_data)
        t = np.arange(n_samples) / 30.0  # Assuming 30 Hz
        wander = amplitude * np.sin(2 * np.pi * frequency * t)
        return signal_data + wander
    
    def apply_time_jitter(
        self,
        signal_data: np.ndarray,
        shift: int,
    ) -> np.ndarray:
        """Apply time shift/jitter to signal.
        
        Args:
            signal_data: Input signal
            shift: Number of samples to shift (+/-)
            
        Returns:
            Shifted signal
        """
        if shift == 0:
            return signal_data
        
        if shift > 0:
            # Shift right (pad beginning)
            return np.pad(signal_data[:-shift], (shift, 0), mode='edge')
        else:
            # Shift left (pad end)
            return np.pad(signal_data[-shift:], (0, -shift), mode='edge')
    
    def add_motion_artifact(
        self,
        signal_data: np.ndarray,
        amplitude: float = 0.1,
        duration_fraction: float = 0.1,
    ) -> np.ndarray:
        """Add simulated motion artifact.
        
        Args:
            signal_data: Input signal
            amplitude: Artifact amplitude
            duration_fraction: Fraction of signal to affect
            
        Returns:
            Signal with motion artifact
        """
        n_samples = len(signal_data)
        artifact_len = int(n_samples * duration_fraction)
        
        # Random position for artifact
        start_idx = np.random.randint(0, n_samples - artifact_len)
        end_idx = start_idx + artifact_len
        
        # Create artifact (high frequency noise)
        artifact = amplitude * np.random.randn(artifact_len)
        
        # Apply artifact
        augmented = signal_data.copy()
        augmented[start_idx:end_idx] += artifact
        
        return augmented