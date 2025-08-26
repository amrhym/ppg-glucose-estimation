"""Windowing and segmentation utilities for PPG signals."""

from typing import Generator, List, Optional, Tuple

import numpy as np


class WindowGenerator:
    """Generate fixed-length windows from continuous PPG signal.
    
    Examples:
        >>> gen = WindowGenerator(window_length=10.0, hop_length=5.0, fs=30)
        >>> windows = list(gen.generate(ppg_signal))
    """
    
    def __init__(
        self,
        window_length: float = 10.0,
        hop_length: Optional[float] = None,
        fs: float = 30.0,
        normalize: str = "zscore",
        min_quality: Optional[float] = None,
    ):
        """Initialize window generator.
        
        Args:
            window_length: Window duration in seconds
            hop_length: Hop duration in seconds (None = no overlap)
            fs: Sampling frequency
            normalize: Normalization method ('zscore', 'minmax', None)
            min_quality: Minimum quality score to accept window
        """
        self.window_length = window_length
        self.hop_length = hop_length or window_length
        self.fs = fs
        self.normalize = normalize
        self.min_quality = min_quality
        
        # Calculate sample counts
        self.window_samples = int(window_length * fs)
        self.hop_samples = int(self.hop_length * fs)
    
    def generate(
        self,
        signal_data: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, Optional[float]], None, None]:
        """Generate windows from signal.
        
        Args:
            signal_data: Input signal
            quality_scores: Per-sample quality scores (optional)
            
        Yields:
            (window, quality_score) tuples
        """
        n_samples = len(signal_data)
        
        # Generate windows
        for start_idx in range(0, n_samples - self.window_samples + 1, self.hop_samples):
            end_idx = start_idx + self.window_samples
            
            # Extract window
            window = signal_data[start_idx:end_idx].copy()
            
            # Calculate quality score if provided
            quality = None
            if quality_scores is not None:
                window_quality = quality_scores[start_idx:end_idx]
                quality = np.mean(window_quality)
                
                # Skip low quality windows if threshold set
                if self.min_quality and quality < self.min_quality:
                    continue
            
            # Normalize window
            window = self.normalize_window(window)
            
            yield window, quality
    
    def normalize_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize a window based on configured method.
        
        Args:
            window: Input window
            
        Returns:
            Normalized window
        """
        if self.normalize == "zscore":
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                return (window - mean) / std
            return window - mean
        
        elif self.normalize == "minmax":
            min_val = np.min(window)
            max_val = np.max(window)
            range_val = max_val - min_val
            if range_val > 0:
                return (window - min_val) / range_val
            return window - min_val
        
        elif self.normalize == "robust":
            # Robust normalization using median and MAD
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            if mad > 0:
                return (window - median) / (1.4826 * mad)
            return window - median
        
        else:
            return window
    
    def get_window_indices(self, signal_length: int) -> List[Tuple[int, int]]:
        """Get start and end indices for all windows.
        
        Args:
            signal_length: Length of signal in samples
            
        Returns:
            List of (start, end) index tuples
        """
        indices = []
        for start_idx in range(0, signal_length - self.window_samples + 1, self.hop_samples):
            end_idx = start_idx + self.window_samples
            indices.append((start_idx, end_idx))
        return indices


def sliding_window(
    signal_data: np.ndarray,
    window_size: int,
    stride: int,
    normalize: bool = True,
) -> np.ndarray:
    """Create sliding windows from signal (vectorized).
    
    Args:
        signal_data: Input signal
        window_size: Window size in samples
        stride: Stride in samples
        normalize: Apply z-score normalization
        
    Returns:
        Array of windows with shape (n_windows, window_size)
    """
    n_samples = len(signal_data)
    n_windows = (n_samples - window_size) // stride + 1
    
    # Create windows using stride tricks (memory efficient)
    shape = (n_windows, window_size)
    strides = (signal_data.strides[0] * stride, signal_data.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        signal_data, shape=shape, strides=strides, writeable=False
    )
    
    # Copy to avoid memory issues
    windows = np.array(windows)
    
    # Normalize if requested
    if normalize:
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        windows = (windows - mean) / (std + 1e-8)
    
    return windows