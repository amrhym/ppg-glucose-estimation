"""Signal filtering utilities for PPG preprocessing."""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, sosfiltfilt


class BandpassFilter:
    """Configurable bandpass filter for PPG signals.
    
    Examples:
        >>> filter = BandpassFilter(band=(0.5, 8.0), fs=1000, order=4)
        >>> filtered = filter.apply(ppg_signal)
    """
    
    def __init__(
        self,
        band: Tuple[float, float] = (0.5, 8.0),
        fs: float = 1000.0,
        order: int = 4,
        filter_type: str = "butter",
        zero_phase: bool = True,
    ):
        """Initialize bandpass filter.
        
        Args:
            band: (low_freq, high_freq) in Hz
            fs: Sampling frequency in Hz
            order: Filter order
            filter_type: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
            zero_phase: Use zero-phase filtering (filtfilt)
        """
        self.band = band
        self.fs = fs
        self.order = order
        self.filter_type = filter_type
        self.zero_phase = zero_phase
        
        # Design filter
        self.sos = self._design_filter()
    
    def _design_filter(self) -> np.ndarray:
        """Design the bandpass filter coefficients."""
        nyquist = 0.5 * self.fs
        low = self.band[0] / nyquist
        high = self.band[1] / nyquist
        
        if low <= 0 or high >= 1:
            raise ValueError(
                f"Band frequencies must be between 0 and Nyquist ({nyquist} Hz). "
                f"Got {self.band}"
            )
        
        # Design as second-order sections for numerical stability
        if self.filter_type == "butter":
            sos = signal.butter(self.order, [low, high], btype="band", output="sos")
        elif self.filter_type == "cheby1":
            sos = signal.cheby1(self.order, 0.5, [low, high], btype="band", output="sos")
        elif self.filter_type == "cheby2":
            sos = signal.cheby2(self.order, 40, [low, high], btype="band", output="sos")
        elif self.filter_type == "ellip":
            sos = signal.elliptic(self.order, 0.5, 40, [low, high], btype="band", output="sos")
        elif self.filter_type == "bessel":
            sos = signal.bessel(self.order, [low, high], btype="band", output="sos")
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        return sos
    
    def apply(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to signal.
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Filtered signal
        """
        if len(signal_data) < self.order * 3:
            raise ValueError(
                f"Signal too short ({len(signal_data)} samples) for order {self.order} filter"
            )
        
        # Remove NaN and Inf
        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.zero_phase:
            # Zero-phase filtering (forward-backward)
            filtered = sosfiltfilt(self.sos, signal_data)
        else:
            # Regular filtering
            filtered = signal.sosfilt(self.sos, signal_data)
        
        return filtered


def apply_bandpass(
    signal_data: np.ndarray,
    fs: float,
    band: Tuple[float, float] = (0.5, 8.0),
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Convenience function to apply bandpass filter.
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        band: (low_freq, high_freq) in Hz
        order: Filter order
        zero_phase: Use zero-phase filtering
        
    Returns:
        Filtered signal
    """
    filt = BandpassFilter(band=band, fs=fs, order=order, zero_phase=zero_phase)
    return filt.apply(signal_data)


def notch_filter(
    signal_data: np.ndarray,
    fs: float,
    freq: float = 60.0,
    Q: float = 30.0,
) -> np.ndarray:
    """Apply notch filter to remove powerline interference.
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        freq: Notch frequency (Hz), typically 50 or 60
        Q: Quality factor
        
    Returns:
        Filtered signal
    """
    w0 = freq / (fs / 2)
    b, a = signal.iirnotch(w0, Q)
    return filtfilt(b, a, signal_data)


def moving_average(signal_data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average filter.
    
    Args:
        signal_data: Input signal
        window: Window size for averaging
        
    Returns:
        Smoothed signal
    """
    kernel = np.ones(window) / window
    return np.convolve(signal_data, kernel, mode="same")