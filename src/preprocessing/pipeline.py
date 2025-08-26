"""Complete preprocessing pipeline for PPG signals."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.augmentation import DataAugmenter
from src.preprocessing.filters import BandpassFilter, apply_bandpass
from src.preprocessing.resampler import Resampler
from src.preprocessing.windowing import WindowGenerator


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Sampling parameters
    raw_fs: float = 1000.0
    target_fs: float = 30.0
    
    # Filter parameters
    bandpass: Tuple[float, float] = (0.5, 8.0)
    filter_order: int = 4
    zero_phase: bool = True
    
    # Window parameters
    window_length_s: float = 10.0
    hop_length_s: Optional[float] = 5.0
    normalization: str = "zscore"
    
    # Augmentation parameters (training only)
    augment_training: bool = True
    gaussian_noise_std: List[float] = None
    amplitude_scale: Tuple[float, float] = (0.9, 1.1)
    
    # Quality parameters
    min_quality: Optional[float] = 0.5
    
    def __post_init__(self):
        if self.gaussian_noise_std is None:
            self.gaussian_noise_std = [0.0, 0.01, 0.02]


class PreprocessingPipeline:
    """Complete preprocessing pipeline for PPG signals.
    
    Examples:
        >>> config = PreprocessingConfig(raw_fs=1000, target_fs=30)
        >>> pipeline = PreprocessingPipeline(config)
        >>> windows = pipeline.process(ppg_signal)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.filter = BandpassFilter(
            band=self.config.bandpass,
            fs=self.config.raw_fs,
            order=self.config.filter_order,
            zero_phase=self.config.zero_phase,
        )
        
        self.resampler = Resampler(
            fs_original=self.config.raw_fs,
            fs_target=self.config.target_fs,
            method="decimate" if self.config.raw_fs / self.config.target_fs == int(self.config.raw_fs / self.config.target_fs) else "resample",
        )
        
        self.windower = WindowGenerator(
            window_length=self.config.window_length_s,
            hop_length=self.config.hop_length_s,
            fs=self.config.target_fs,
            normalize=self.config.normalization,
            min_quality=self.config.min_quality,
        )
        
        if self.config.augment_training:
            self.augmenter = DataAugmenter(
                gaussian_noise_std=self.config.gaussian_noise_std,
                amplitude_scale=self.config.amplitude_scale,
            )
        else:
            self.augmenter = None
    
    def process(
        self,
        signal_data: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> List[np.ndarray]:
        """Process PPG signal through complete pipeline.
        
        Args:
            signal_data: Raw PPG signal at original sampling rate
            quality_scores: Optional per-sample quality scores
            training: Enable training-specific processing (augmentation)
            
        Returns:
            List of processed windows
        """
        # Stage 1: Bandpass filter
        filtered = self.filter.apply(signal_data)
        
        # Stage 2: Downsample
        downsampled = self.resampler.resample(filtered)
        
        # Stage 3: Generate windows
        windows = []
        for window, quality in self.windower.generate(downsampled, quality_scores):
            if quality is None or quality >= (self.config.min_quality or 0):
                windows.append(window)
        
        # Stage 4: Augmentation (training only)
        if training and self.augmenter:
            augmented_windows = []
            for window in windows:
                # Add original
                augmented_windows.append(window)
                # Add augmented versions
                augmented = self.augmenter.augment(window)
                augmented_windows.extend(augmented)
            windows = augmented_windows
        
        return windows
    
    def process_batch(
        self,
        signals: List[np.ndarray],
        quality_scores: Optional[List[np.ndarray]] = None,
        training: bool = False,
    ) -> np.ndarray:
        """Process batch of PPG signals.
        
        Args:
            signals: List of raw PPG signals
            quality_scores: Optional list of quality scores
            training: Enable training-specific processing
            
        Returns:
            Stacked array of all windows
        """
        all_windows = []
        
        for i, signal in enumerate(signals):
            q_scores = quality_scores[i] if quality_scores else None
            windows = self.process(signal, q_scores, training)
            all_windows.extend(windows)
        
        if not all_windows:
            return np.array([])
        
        return np.stack(all_windows)
    
    def get_stage_outputs(
        self,
        signal_data: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Get intermediate outputs from each pipeline stage for debugging.
        
        Args:
            signal_data: Raw PPG signal
            
        Returns:
            Dictionary with stage names and outputs
        """
        outputs = {
            "raw": signal_data,
        }
        
        # Stage 1: Filter
        filtered = self.filter.apply(signal_data)
        outputs["filtered"] = filtered
        
        # Stage 2: Downsample
        downsampled = self.resampler.resample(filtered)
        outputs["downsampled"] = downsampled
        
        # Stage 3: Windows
        windows = list(self.windower.generate(downsampled))
        if windows:
            outputs["first_window"] = windows[0][0]
            outputs["n_windows"] = len(windows)
        
        return outputs