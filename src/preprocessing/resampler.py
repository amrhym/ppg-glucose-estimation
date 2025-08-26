"""Signal resampling utilities for PPG preprocessing."""

from typing import Optional, Union

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


class Resampler:
    """Resample PPG signals to target frequency.
    
    Examples:
        >>> resampler = Resampler(fs_original=1000, fs_target=30)
        >>> downsampled = resampler.resample(ppg_signal)
    """
    
    def __init__(
        self,
        fs_original: float,
        fs_target: float,
        method: str = "decimate",
        antialias: bool = True,
    ):
        """Initialize resampler.
        
        Args:
            fs_original: Original sampling frequency (Hz)
            fs_target: Target sampling frequency (Hz)
            method: 'decimate', 'resample', or 'interp'
            antialias: Apply antialiasing filter before downsampling
        """
        self.fs_original = fs_original
        self.fs_target = fs_target
        self.method = method
        self.antialias = antialias
        
        # Calculate resampling factor
        self.factor = fs_original / fs_target
        
        if fs_target > fs_original:
            raise ValueError(
                f"Upsampling not recommended for PPG. "
                f"Target fs ({fs_target}) > Original fs ({fs_original})"
            )
    
    def resample(self, signal_data: np.ndarray) -> np.ndarray:
        """Resample signal to target frequency.
        
        Args:
            signal_data: Input signal at original frequency
            
        Returns:
            Resampled signal at target frequency
        """
        if self.method == "decimate":
            return self._decimate(signal_data)
        elif self.method == "resample":
            return self._resample_fourier(signal_data)
        elif self.method == "interp":
            return self._resample_interp(signal_data)
        else:
            raise ValueError(f"Unknown resampling method: {self.method}")
    
    def _decimate(self, signal_data: np.ndarray) -> np.ndarray:
        """Downsample using decimation with antialiasing."""
        if self.factor != int(self.factor):
            # For non-integer factors, use Fourier method
            return self._resample_fourier(signal_data)
        
        factor = int(self.factor)
        
        # Apply antialiasing filter if requested
        if self.antialias:
            # Design lowpass filter
            nyquist = 0.5 * self.fs_original
            cutoff = 0.8 * (self.fs_target / 2) / nyquist
            b, a = signal.butter(8, cutoff, btype='low')
            signal_data = signal.filtfilt(b, a, signal_data)
        
        # Decimate
        if factor > 1:
            return signal_data[::factor]
        return signal_data
    
    def _resample_fourier(self, signal_data: np.ndarray) -> np.ndarray:
        """Resample using Fourier method."""
        n_original = len(signal_data)
        n_target = int(n_original * self.fs_target / self.fs_original)
        
        return signal.resample(signal_data, n_target)
    
    def _resample_interp(self, signal_data: np.ndarray) -> np.ndarray:
        """Resample using interpolation."""
        n_original = len(signal_data)
        n_target = int(n_original * self.fs_target / self.fs_original)
        
        # Original time vector
        t_original = np.arange(n_original) / self.fs_original
        
        # Target time vector
        t_target = np.arange(n_target) / self.fs_target
        
        # Interpolate
        f = interp1d(t_original, signal_data, kind='cubic', fill_value='extrapolate')
        
        return f(t_target)


def downsample_signal(
    signal_data: np.ndarray,
    fs_original: float,
    fs_target: float,
    method: str = "decimate",
) -> np.ndarray:
    """Convenience function to downsample signal.
    
    Args:
        signal_data: Input signal
        fs_original: Original sampling frequency
        fs_target: Target sampling frequency
        method: Resampling method
        
    Returns:
        Downsampled signal
    """
    resampler = Resampler(fs_original, fs_target, method=method)
    return resampler.resample(signal_data)