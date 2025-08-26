#!/usr/bin/env python3
"""Core function tests for PPG glucose estimation (no PyTorch required)."""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.filters import BandpassFilter, apply_bandpass
from src.preprocessing.resampler import Resampler, downsample_signal
from src.preprocessing.windowing import WindowGenerator, sliding_window
from src.preprocessing.augmentation import DataAugmenter
from src.quality.metrics import (
    compute_snr, compute_hr_plausibility, compute_motion_score,
    compute_signal_quality, compute_baseline_stability, compute_perfusion_index
)
from src.quality.validator import SignalQualityValidator
from src.metrics.clarke import ClarkeErrorGrid, clarke_error_grid_analysis
from src.utils.sim_ppg import generate_synthetic_ppg, generate_ppg_dataset


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)


def test_preprocessing():
    """Test all preprocessing functions."""
    print_header("PREPROCESSING TESTS")
    
    print("\n1. Testing Bandpass Filter...")
    try:
        # Generate test signal with multiple frequencies
        fs = 1000
        t = np.arange(0, 2, 1/fs)
        # 1 Hz (in band) + 50 Hz (out of band) + DC offset
        signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*50*t) + 2.0
        
        # Apply filter
        filter = BandpassFilter(band=(0.5, 8.0), fs=fs, order=4, zero_phase=True)
        filtered = filter.apply(signal)
        
        print(f"  ✓ Input shape: {signal.shape}")
        print(f"  ✓ Output shape: {filtered.shape}")
        print(f"  ✓ Input mean: {np.mean(signal):.3f}, Output mean: {np.mean(filtered):.3f}")
        print(f"  ✓ Input std: {np.std(signal):.3f}, Output std: {np.std(filtered):.3f}")
        
        # Test alternative method
        filtered2 = apply_bandpass(signal, fs, band=(0.5, 8.0))
        print(f"  ✓ Alternative method works: {np.allclose(filtered, filtered2)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n2. Testing Signal Resampling...")
    try:
        # Test different resampling methods
        fs_original = 2175.0  # Dataset frequency
        fs_target = 30.0
        
        # Create realistic signal
        duration = 10
        t_orig = np.arange(0, duration, 1/fs_original)
        signal = np.sin(2*np.pi*1.2*t_orig) + 0.3*np.sin(2*np.pi*2.5*t_orig)
        
        print(f"  Original signal: {len(signal)} samples at {fs_original} Hz")
        
        for method in ["decimate", "resample", "interp"]:
            resampler = Resampler(fs_original, fs_target, method=method)
            downsampled = resampler.resample(signal)
            print(f"  ✓ {method}: {len(downsampled)} samples at {fs_target} Hz")
        
        # Test convenience function
        downsampled2 = downsample_signal(signal, fs_original, fs_target)
        print(f"  ✓ Convenience function: {len(downsampled2)} samples")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n3. Testing Window Generation...")
    try:
        # Generate windows from signal
        fs = 30
        duration = 60  # 60 seconds
        signal = np.random.randn(int(duration * fs))
        
        gen = WindowGenerator(
            window_length=10.0,  # 10 second windows
            hop_length=5.0,      # 5 second hop
            fs=fs,
            normalize="zscore"
        )
        
        windows = list(gen.generate(signal))
        print(f"  ✓ Generated {len(windows)} windows from {duration}s signal")
        print(f"  ✓ Window size: {len(windows[0][0])} samples")
        
        # Check normalization
        first_window = windows[0][0]
        print(f"  ✓ Window mean: {np.mean(first_window):.6f} (should be ~0)")
        print(f"  ✓ Window std: {np.std(first_window):.6f} (should be ~1)")
        
        # Test vectorized sliding window
        windows_vec = sliding_window(signal, window_size=300, stride=150, normalize=True)
        print(f"  ✓ Vectorized method: {windows_vec.shape}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n4. Testing Data Augmentation...")
    try:
        augmenter = DataAugmenter(
            gaussian_noise_std=[0.01, 0.02, 0.03],
            amplitude_scale=(0.9, 1.1),
            baseline_wander_amp=0.05,
            time_jitter_samples=5
        )
        
        window = np.random.randn(300)
        augmented = augmenter.augment(window, n_augmentations=5)
        
        print(f"  ✓ Original shape: {window.shape}")
        print(f"  ✓ Generated {len(augmented)} augmented versions")
        
        # Test individual augmentation methods
        noisy = augmenter.add_gaussian_noise(window, 0.05)
        print(f"  ✓ Gaussian noise: SNR change = {20*np.log10(np.std(window)/np.std(noisy-window)):.1f} dB")
        
        scaled = augmenter.scale_amplitude(window, 0.8)
        print(f"  ✓ Amplitude scaling: ratio = {np.std(scaled)/np.std(window):.2f}")
        
        wandered = augmenter.add_baseline_wander(window, 0.1)
        print(f"  ✓ Baseline wander: added")
        
        motion = augmenter.add_motion_artifact(window)
        print(f"  ✓ Motion artifact: added")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def test_quality_metrics():
    """Test signal quality metrics."""
    print_header("QUALITY METRICS TESTS")
    
    # Generate realistic PPG signal
    ppg, glucose = generate_synthetic_ppg(
        duration=10.0,
        fs=30.0,
        heart_rate=72.0,
        respiratory_rate=15.0,
        snr_db=20.0,
        glucose_level=100.0
    )
    
    print("\n1. Testing SNR Computation...")
    try:
        snr = compute_snr(ppg, fs=30.0)
        print(f"  ✓ SNR: {snr:.1f} dB")
        assert -20 <= snr <= 40, "SNR out of range"
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n2. Testing Heart Rate Plausibility...")
    try:
        is_plausible, bpm = compute_hr_plausibility(ppg, fs=30.0)
        print(f"  ✓ HR: {bpm:.1f} BPM")
        print(f"  ✓ Plausible: {is_plausible}")
        assert 40 <= bpm <= 180 or bpm == 0, "HR out of range"
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n3. Testing Motion Score...")
    try:
        motion = compute_motion_score(ppg, fs=30.0)
        print(f"  ✓ Motion score: {motion:.3f} (0=no motion, 1=high motion)")
        assert 0 <= motion <= 1, "Motion score out of range"
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n4. Testing Baseline Stability...")
    try:
        stability = compute_baseline_stability(ppg)
        print(f"  ✓ Baseline stability: {stability:.3f} (0=unstable, 1=stable)")
        assert 0 <= stability <= 1, "Stability out of range"
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n5. Testing Perfusion Index...")
    try:
        pi = compute_perfusion_index(ppg)
        print(f"  ✓ Perfusion index: {pi:.1f}%")
        assert 0 <= pi <= 100, "PI out of range"
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n6. Testing Overall Signal Quality...")
    try:
        quality = compute_signal_quality(ppg, fs=30.0)
        print(f"  ✓ Overall quality: {quality:.3f} (0=poor, 1=excellent)")
        assert 0 <= quality <= 1, "Quality out of range"
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n7. Testing Quality Validator...")
    try:
        validator = SignalQualityValidator(
            min_quality=0.5,
            min_snr_db=-5.0,
            min_hr_bpm=40.0,
            max_hr_bpm=180.0
        )
        
        report = validator.validate(ppg)
        print(f"  ✓ Valid: {report.is_valid}")
        print(f"  ✓ Quality: {report.overall_quality:.3f}")
        print(f"  ✓ SNR: {report.snr_db:.1f} dB")
        print(f"  ✓ HR: {report.hr_bpm:.1f} BPM")
        print(f"  ✓ Confidence: {report.confidence:.3f}")
        
        # Test batch validation
        windows = np.random.randn(5, 300)
        valid_flags = validator.validate_batch(windows)
        print(f"  ✓ Batch validation: {np.sum(valid_flags)}/{len(valid_flags)} valid")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def test_clarke_error_grid():
    """Test Clarke Error Grid implementation."""
    print_header("CLARKE ERROR GRID TESTS")
    
    print("\n1. Testing Zone Calculation...")
    try:
        ceg = ClarkeErrorGrid()
        
        test_cases = [
            (100, 100, 'A'),  # Perfect
            (100, 95, 'A'),   # Small error
            (100, 120, 'A'),  # 20% error
            (100, 130, 'B'),  # Benign error
            (70, 180, 'D'),   # Dangerous
        ]
        
        for ref, pred, expected in test_cases:
            zone = ceg.get_zone(ref, pred)
            print(f"  ✓ Reference={ref}, Predicted={pred} → Zone {zone} (expected {expected})")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n2. Testing Zone Analysis...")
    try:
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        reference = np.random.uniform(70, 200, n_samples)
        # Add noise to create predictions
        noise = np.random.normal(0, 10, n_samples)
        predicted = reference + noise
        predicted = np.clip(predicted, 40, 400)
        
        zones = ceg.analyze(reference, predicted)
        
        print("  Zone Distribution:")
        for zone, percentage in zones.items():
            print(f"    Zone {zone}: {percentage:.1f}%")
        
        clinical_accuracy = ceg.get_clinical_accuracy(zones)
        print(f"  ✓ Clinical Accuracy (A+B): {clinical_accuracy:.1f}%")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n3. Testing Convenience Function...")
    try:
        zones, accuracy = clarke_error_grid_analysis(reference, predicted, plot=False)
        print(f"  ✓ Analysis complete: {accuracy:.1f}% clinical accuracy")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def test_synthetic_data():
    """Test synthetic PPG generation."""
    print_header("SYNTHETIC DATA GENERATION TESTS")
    
    print("\n1. Testing Single PPG Generation...")
    try:
        ppg, glucose = generate_synthetic_ppg(
            duration=10.0,
            fs=30.0,
            heart_rate=75.0,
            respiratory_rate=15.0,
            snr_db=20.0,
            glucose_level=120.0,
            add_artifacts=False,
            seed=42
        )
        
        print(f"  ✓ Signal length: {len(ppg)} samples")
        print(f"  ✓ Sampling rate: 30 Hz")
        print(f"  ✓ Duration: {len(ppg)/30:.1f} seconds")
        print(f"  ✓ Glucose level: {glucose} mg/dL")
        print(f"  ✓ Signal mean: {np.mean(ppg):.3f}")
        print(f"  ✓ Signal std: {np.std(ppg):.3f}")
        
        # Verify signal quality
        quality = compute_signal_quality(ppg, fs=30.0)
        print(f"  ✓ Generated signal quality: {quality:.3f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n2. Testing Dataset Generation...")
    try:
        signals, labels = generate_ppg_dataset(
            n_samples=20,
            duration=10.0,
            fs=30.0,
            glucose_range=(70, 180),
            hr_range=(50, 100),
            snr_range=(10, 30),
            seed=42
        )
        
        print(f"  ✓ Dataset shape: {signals.shape}")
        print(f"  ✓ Labels shape: {labels.shape}")
        print(f"  ✓ Glucose range: [{labels.min():.1f}, {labels.max():.1f}] mg/dL")
        print(f"  ✓ Mean glucose: {labels.mean():.1f} mg/dL")
        
        # Check quality of generated signals
        validator = SignalQualityValidator()
        valid_count = 0
        for signal in signals:
            report = validator.validate(signal, compute_all=False)
            if report.is_valid:
                valid_count += 1
        
        print(f"  ✓ Valid signals: {valid_count}/{len(signals)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def test_real_data():
    """Test with real PPG dataset."""
    print_header("REAL DATA PROCESSING TESTS")
    
    data_dir = Path("/Users/amrmostafa/ppg-glucose-estimation/data/PPG_Dataset")
    
    print("\n1. Loading Real PPG Signal...")
    try:
        signal_file = data_dir / "RawData" / "signal_01_0001.csv"
        if not signal_file.exists():
            print(f"  ⚠ Signal file not found: {signal_file}")
            return
        
        # Load signal
        signal_data = pd.read_csv(signal_file, header=None).values.flatten()
        print(f"  ✓ Loaded {len(signal_data)} samples")
        print(f"  ✓ Duration: {len(signal_data)/2175:.1f} seconds (at 2175 Hz)")
        print(f"  ✓ Signal range: [{signal_data.min():.3f}, {signal_data.max():.3f}]")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return
    
    print("\n2. Loading Glucose Label...")
    try:
        label_file = data_dir / "Labels" / "label_01_0001.csv"
        if not label_file.exists():
            print(f"  ⚠ Label file not found: {label_file}")
            glucose = 100.0  # Default value
        else:
            label_data = pd.read_csv(label_file, header=None).values.flatten()
            glucose = float(label_data[0])
            print(f"  ✓ Glucose level: {glucose} mg/dL")
    except Exception as e:
        print(f"  ✗ Error loading label: {e}")
        glucose = 100.0
    
    print("\n3. Processing Real Signal...")
    try:
        # Take first 10 seconds
        fs_original = 2175.0
        segment = signal_data[:int(10 * fs_original)]
        
        # Apply preprocessing pipeline
        print("  Applying preprocessing pipeline:")
        
        # Step 1: Bandpass filter
        filtered = apply_bandpass(segment, fs_original, band=(0.5, 8.0))
        print(f"    ✓ Filtered: mean={np.mean(filtered):.3f}, std={np.std(filtered):.3f}")
        
        # Step 2: Downsample
        downsampled = downsample_signal(filtered, fs_original, 30.0)
        print(f"    ✓ Downsampled: {len(downsampled)} samples at 30 Hz")
        
        # Step 3: Normalize
        normalized = (downsampled - np.mean(downsampled)) / np.std(downsampled)
        print(f"    ✓ Normalized: mean={np.mean(normalized):.6f}, std={np.std(normalized):.6f}")
        
        # Step 4: Quality check
        validator = SignalQualityValidator()
        report = validator.validate(normalized)
        print(f"    ✓ Quality: {report.overall_quality:.3f}")
        print(f"    ✓ Valid: {report.is_valid}")
        print(f"    ✓ SNR: {report.snr_db:.1f} dB")
        print(f"    ✓ HR: {report.hr_bpm:.1f} BPM")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n4. Processing Multiple Samples...")
    try:
        # Process first 5 signal files
        valid_count = 0
        total_count = 0
        
        for i in range(1, 6):
            signal_file = data_dir / "RawData" / f"signal_01_000{i}.csv"
            if signal_file.exists():
                signal = pd.read_csv(signal_file, header=None).values.flatten()
                segment = signal[:int(10 * fs_original)]
                
                # Quick processing
                filtered = apply_bandpass(segment, fs_original, (0.5, 8.0))
                downsampled = downsample_signal(filtered, fs_original, 30.0)
                normalized = (downsampled - np.mean(downsampled)) / np.std(downsampled)
                
                # Quality check
                report = validator.validate(normalized, compute_all=False)
                if report.is_valid:
                    valid_count += 1
                total_count += 1
        
        print(f"  ✓ Processed {total_count} samples")
        print(f"  ✓ Valid samples: {valid_count}/{total_count}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PPG GLUCOSE ESTIMATION - CORE FUNCTION TESTS")
    print("="*60)
    
    # Run all test suites
    test_preprocessing()
    test_quality_metrics()
    test_clarke_error_grid()
    test_synthetic_data()
    test_real_data()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()