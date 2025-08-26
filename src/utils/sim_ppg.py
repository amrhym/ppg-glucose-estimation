"""Synthetic PPG signal generator for testing."""

from typing import Optional, Tuple

import numpy as np


def generate_synthetic_ppg(
    duration: float = 10.0,
    fs: float = 30.0,
    heart_rate: float = 70.0,
    respiratory_rate: float = 15.0,
    snr_db: float = 20.0,
    glucose_level: float = 100.0,
    add_artifacts: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Generate synthetic PPG signal with physiological characteristics.
    
    This creates a realistic PPG waveform with:
    - Primary cardiac component
    - Dicrotic notch
    - Respiratory modulation
    - Baseline wander
    - Optional motion artifacts
    
    Args:
        duration: Signal duration in seconds
        fs: Sampling frequency in Hz
        heart_rate: Heart rate in BPM
        respiratory_rate: Respiratory rate in breaths/min
        snr_db: Signal-to-noise ratio in dB
        glucose_level: Simulated glucose level (affects morphology)
        add_artifacts: Whether to add motion artifacts
        seed: Random seed for reproducibility
        
    Returns:
        (ppg_signal, glucose_level)
    
    Examples:
        >>> ppg, glucose = generate_synthetic_ppg(duration=10, fs=30, heart_rate=75)
        >>> assert len(ppg) == 300  # 10s at 30Hz
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time vector
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Cardiac frequency
    f_cardiac = heart_rate / 60.0  # Hz
    
    # Generate cardiac component (main pulse)
    cardiac_phase = 2 * np.pi * f_cardiac * t
    
    # Primary pulse (systolic peak)
    pulse = (
        1.0 * np.sin(cardiac_phase) +
        0.3 * np.sin(2 * cardiac_phase) +  # Second harmonic
        0.1 * np.sin(3 * cardiac_phase)    # Third harmonic
    )
    
    # Add dicrotic notch (reflected wave)
    notch_delay = 0.15  # seconds after peak
    notch_phase = cardiac_phase - 2 * np.pi * f_cardiac * notch_delay
    dicrotic = 0.15 * np.sin(notch_phase) * np.exp(-3 * (t % (1/f_cardiac)))
    
    # Combine cardiac components
    ppg = pulse + dicrotic
    
    # Respiratory modulation
    f_resp = respiratory_rate / 60.0  # Hz
    resp_modulation = 0.05 * np.sin(2 * np.pi * f_resp * t)
    ppg = ppg * (1 + resp_modulation)
    
    # Baseline wander (very low frequency)
    f_baseline = 0.05  # Hz
    baseline = 0.02 * np.sin(2 * np.pi * f_baseline * t)
    ppg = ppg + baseline
    
    # Glucose-dependent morphology changes
    # Higher glucose -> reduced pulse amplitude, increased baseline
    glucose_factor = 1.0 - 0.002 * (glucose_level - 100)
    ppg = ppg * glucose_factor
    
    # Add physiological variability
    hr_variability = 0.02 * np.sin(2 * np.pi * 0.1 * t)  # HRV at 0.1 Hz
    ppg = ppg * (1 + hr_variability)
    
    # Add noise based on SNR
    signal_power = np.mean(ppg ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(n_samples)
    ppg = ppg + noise
    
    # Add motion artifacts if requested
    if add_artifacts:
        # Random motion events
        n_artifacts = np.random.randint(1, 4)
        for _ in range(n_artifacts):
            # Random position and duration
            start = np.random.randint(0, n_samples - int(fs))
            duration_samples = np.random.randint(int(0.5 * fs), int(2 * fs))
            end = min(start + duration_samples, n_samples)
            
            # High-frequency noise burst
            artifact = 0.5 * np.random.randn(end - start)
            ppg[start:end] += artifact
    
    # Normalize to realistic range
    ppg = ppg - np.mean(ppg)
    ppg = ppg / np.std(ppg)
    
    return ppg, glucose_level


def generate_ppg_dataset(
    n_samples: int = 100,
    duration: float = 10.0,
    fs: float = 30.0,
    glucose_range: Tuple[float, float] = (70, 180),
    hr_range: Tuple[float, float] = (50, 100),
    snr_range: Tuple[float, float] = (10, 30),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic PPG dataset with varied parameters.
    
    Args:
        n_samples: Number of samples to generate
        duration: Duration of each signal in seconds
        fs: Sampling frequency
        glucose_range: Range of glucose values
        hr_range: Range of heart rates
        snr_range: Range of SNR values
        seed: Random seed
        
    Returns:
        (ppg_signals, glucose_labels) with shapes (n_samples, n_points) and (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration * fs)
    ppg_signals = np.zeros((n_samples, n_points))
    glucose_labels = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Random parameters
        glucose = np.random.uniform(*glucose_range)
        heart_rate = np.random.uniform(*hr_range)
        snr = np.random.uniform(*snr_range)
        respiratory_rate = np.random.uniform(12, 20)
        
        # Generate signal
        ppg, _ = generate_synthetic_ppg(
            duration=duration,
            fs=fs,
            heart_rate=heart_rate,
            respiratory_rate=respiratory_rate,
            snr_db=snr,
            glucose_level=glucose,
            add_artifacts=np.random.random() < 0.1,  # 10% with artifacts
        )
        
        ppg_signals[i] = ppg
        glucose_labels[i] = glucose
    
    return ppg_signals, glucose_labels


def simulate_continuous_ppg(
    duration_minutes: float = 5.0,
    fs: float = 30.0,
    initial_glucose: float = 100.0,
    glucose_variation: float = 20.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate continuous PPG monitoring with varying glucose.
    
    Args:
        duration_minutes: Total duration in minutes
        fs: Sampling frequency
        initial_glucose: Starting glucose level
        glucose_variation: Maximum glucose variation
        seed: Random seed
        
    Returns:
        (ppg_signal, glucose_trajectory)
    """
    if seed is not None:
        np.random.seed(seed)
    
    duration_seconds = duration_minutes * 60
    t = np.arange(0, duration_seconds, 1/fs)
    n_samples = len(t)
    
    # Generate slowly varying glucose trajectory
    glucose_freq = 1 / 300  # Change every 5 minutes
    glucose_trajectory = (
        initial_glucose +
        glucose_variation * np.sin(2 * np.pi * glucose_freq * t) +
        0.1 * glucose_variation * np.random.randn(n_samples).cumsum() / np.sqrt(n_samples)
    )
    glucose_trajectory = np.clip(glucose_trajectory, 40, 400)
    
    # Generate PPG in segments
    segment_duration = 10.0  # seconds
    segment_samples = int(segment_duration * fs)
    n_segments = n_samples // segment_samples
    
    ppg_signal = np.zeros(n_samples)
    
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        
        # Get average glucose for this segment
        segment_glucose = np.mean(glucose_trajectory[start:end])
        
        # Varying heart rate (correlated with glucose)
        heart_rate = 70 + 0.2 * (segment_glucose - 100)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # Generate segment
        segment_ppg, _ = generate_synthetic_ppg(
            duration=segment_duration,
            fs=fs,
            heart_rate=heart_rate,
            glucose_level=segment_glucose,
            snr_db=20,
        )
        
        ppg_signal[start:end] = segment_ppg
    
    return ppg_signal, glucose_trajectory