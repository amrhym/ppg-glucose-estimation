"""Signal quality metrics for PPG validation."""

from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch


def compute_snr(
    signal_data: np.ndarray,
    fs: float = 30.0,
    hr_range: Tuple[float, float] = (0.67, 3.0),  # 40-180 BPM in Hz
) -> float:
    """Compute signal-to-noise ratio around heart rate frequency.
    
    Args:
        signal_data: PPG signal
        fs: Sampling frequency
        hr_range: Expected HR frequency range in Hz
        
    Returns:
        SNR in dB
    """
    # Compute power spectral density
    freqs, psd = welch(signal_data, fs=fs, nperseg=min(len(signal_data), 256))
    
    # Find dominant frequency in HR range
    hr_mask = (freqs >= hr_range[0]) & (freqs <= hr_range[1])
    if not np.any(hr_mask):
        return 0.0
    
    hr_psd = psd[hr_mask]
    hr_freqs = freqs[hr_mask]
    
    # Find peak
    peak_idx = np.argmax(hr_psd)
    peak_freq = hr_freqs[peak_idx]
    peak_power = hr_psd[peak_idx]
    
    # Signal power: narrow band around peak
    signal_band = 0.1  # Hz
    signal_mask = np.abs(freqs - peak_freq) <= signal_band
    signal_power = np.sum(psd[signal_mask])
    
    # Noise power: everything else
    noise_mask = ~signal_mask & (freqs > 0.1)  # Exclude DC
    noise_power = np.sum(psd[noise_mask])
    
    if noise_power == 0:
        return 40.0  # Max SNR
    
    snr = 10 * np.log10(signal_power / noise_power)
    return np.clip(snr, -20, 40)


def compute_hr_plausibility(
    signal_data: np.ndarray,
    fs: float = 30.0,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
) -> Tuple[bool, float]:
    """Check if heart rate is within plausible range.
    
    Args:
        signal_data: PPG signal
        fs: Sampling frequency
        min_bpm: Minimum plausible heart rate
        max_bpm: Maximum plausible heart rate
        
    Returns:
        (is_plausible, estimated_bpm)
    """
    # Find peaks
    min_distance = int(fs * 60 / max_bpm)  # Min samples between peaks
    peaks, properties = find_peaks(
        signal_data,
        distance=min_distance,
        prominence=np.std(signal_data) * 0.3,
    )
    
    if len(peaks) < 2:
        return False, 0.0
    
    # Calculate heart rate from peak intervals
    intervals = np.diff(peaks) / fs  # In seconds
    hr_bpm = 60.0 / np.median(intervals)
    
    is_plausible = min_bpm <= hr_bpm <= max_bpm
    
    return is_plausible, hr_bpm


def compute_motion_score(
    signal_data: np.ndarray,
    fs: float = 30.0,
    high_freq_threshold: float = 10.0,
) -> float:
    """Compute motion artifact score based on high-frequency content.
    
    Args:
        signal_data: PPG signal
        fs: Sampling frequency
        high_freq_threshold: Frequency above which to measure power
        
    Returns:
        Motion score (0=no motion, 1=high motion)
    """
    # Compute power spectral density
    freqs, psd = welch(signal_data, fs=fs, nperseg=min(len(signal_data), 128))
    
    # Calculate high-frequency power ratio
    high_freq_mask = freqs > high_freq_threshold
    low_freq_mask = (freqs > 0.5) & (freqs <= high_freq_threshold)
    
    if not np.any(high_freq_mask) or not np.any(low_freq_mask):
        return 0.0
    
    high_power = np.sum(psd[high_freq_mask])
    low_power = np.sum(psd[low_freq_mask])
    total_power = high_power + low_power
    
    if total_power == 0:
        return 0.0
    
    motion_score = high_power / total_power
    return np.clip(motion_score, 0, 1)


def compute_signal_quality(
    signal_data: np.ndarray,
    fs: float = 30.0,
) -> float:
    """Compute overall signal quality score.
    
    Args:
        signal_data: PPG signal
        fs: Sampling frequency
        
    Returns:
        Quality score (0-1)
    """
    # Component scores
    snr = compute_snr(signal_data, fs)
    snr_score = np.clip((snr + 10) / 30, 0, 1)  # Map -10 to 20 dB -> 0 to 1
    
    hr_plausible, bpm = compute_hr_plausibility(signal_data, fs)
    hr_score = 1.0 if hr_plausible else 0.0
    
    motion = compute_motion_score(signal_data, fs)
    motion_score = 1.0 - motion  # Invert (low motion = high quality)
    
    # Check for flat segments
    diff = np.diff(signal_data)
    flat_ratio = np.sum(np.abs(diff) < 1e-10) / len(diff)
    flat_score = 1.0 - flat_ratio
    
    # Weighted average
    weights = [0.3, 0.3, 0.2, 0.2]
    scores = [snr_score, hr_score, motion_score, flat_score]
    
    quality = np.sum(np.array(weights) * np.array(scores))
    return np.clip(quality, 0, 1)


def compute_baseline_stability(
    signal_data: np.ndarray,
    window_size: int = 30,
) -> float:
    """Compute baseline stability score.
    
    Args:
        signal_data: PPG signal
        window_size: Window for baseline estimation
        
    Returns:
        Stability score (0-1)
    """
    # Estimate baseline using moving average
    if len(signal_data) < window_size:
        return 1.0
    
    kernel = np.ones(window_size) / window_size
    baseline = np.convolve(signal_data, kernel, mode='same')
    
    # Compute baseline variation
    baseline_std = np.std(baseline)
    signal_std = np.std(signal_data)
    
    if signal_std == 0:
        return 0.0
    
    stability = 1.0 - np.clip(baseline_std / signal_std, 0, 1)
    return stability


def compute_perfusion_index(
    signal_data: np.ndarray,
) -> float:
    """Compute perfusion index (pulse amplitude / DC level).
    
    Args:
        signal_data: PPG signal
        
    Returns:
        Perfusion index (0-100%)
    """
    # Find peaks and troughs
    peaks, _ = find_peaks(signal_data)
    troughs, _ = find_peaks(-signal_data)
    
    if len(peaks) == 0 or len(troughs) == 0:
        return 0.0
    
    # Calculate AC component (pulse amplitude)
    ac_component = np.median(signal_data[peaks]) - np.median(signal_data[troughs])
    
    # Calculate DC component
    dc_component = np.mean(signal_data)
    
    if dc_component == 0:
        return 0.0
    
    pi = (ac_component / abs(dc_component)) * 100
    return np.clip(pi, 0, 100)