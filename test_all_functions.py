#!/usr/bin/env python3
"""Comprehensive test script for PPG glucose estimation system."""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.filters import BandpassFilter, apply_bandpass
from src.preprocessing.resampler import Resampler, downsample_signal
from src.preprocessing.windowing import WindowGenerator
from src.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from src.preprocessing.augmentation import DataAugmenter
from src.quality.metrics import (
    compute_snr, compute_hr_plausibility, compute_motion_score,
    compute_signal_quality, compute_baseline_stability
)
from src.quality.validator import SignalQualityValidator
from src.models.hybrid_model import HybridCNNGRU, ModelConfig
from src.metrics.clarke import ClarkeErrorGrid, clarke_error_grid_analysis
from src.utils.sim_ppg import generate_synthetic_ppg, generate_ppg_dataset
from src.data.dataloader import load_ppg_data

console = Console()


def test_preprocessing():
    """Test preprocessing functions."""
    console.print("\n[bold blue]Testing Preprocessing Functions[/bold blue]")
    
    results = []
    
    # Test 1: Bandpass Filter
    try:
        # Generate test signal
        fs = 1000
        t = np.arange(0, 2, 1/fs)
        signal = np.sin(2*np.pi*1*t) + np.sin(2*np.pi*50*t)  # 1Hz + 50Hz
        
        # Apply filter
        filter = BandpassFilter(band=(0.5, 8.0), fs=fs, order=4)
        filtered = filter.apply(signal)
        
        # Verify output
        assert filtered.shape == signal.shape
        assert np.std(filtered) > 0  # Not zero signal
        results.append(("Bandpass Filter", "✅ Pass", f"Filtered {len(signal)} samples"))
    except Exception as e:
        results.append(("Bandpass Filter", "❌ Fail", str(e)))
    
    # Test 2: Resampler
    try:
        # Downsample signal
        resampler = Resampler(fs_original=1000, fs_target=30)
        downsampled = resampler.resample(signal)
        
        expected_len = int(len(signal) * 30 / 1000)
        assert abs(len(downsampled) - expected_len) <= 1
        results.append(("Resampler", "✅ Pass", f"1000Hz → 30Hz ({len(downsampled)} samples)"))
    except Exception as e:
        results.append(("Resampler", "❌ Fail", str(e)))
    
    # Test 3: Window Generator
    try:
        # Generate windows
        gen = WindowGenerator(window_length=10.0, hop_length=5.0, fs=30)
        test_signal = np.random.randn(900)  # 30 seconds at 30Hz
        windows = list(gen.generate(test_signal))
        
        assert len(windows) == 5  # (30-10)/5 + 1
        assert all(len(w[0]) == 300 for w in windows)
        results.append(("Window Generator", "✅ Pass", f"Generated {len(windows)} windows"))
    except Exception as e:
        results.append(("Window Generator", "❌ Fail", str(e)))
    
    # Test 4: Data Augmenter
    try:
        augmenter = DataAugmenter(
            gaussian_noise_std=[0.01, 0.02],
            amplitude_scale=(0.9, 1.1)
        )
        window = np.random.randn(300)
        augmented = augmenter.augment(window)
        
        assert len(augmented) >= 2
        assert all(a.shape == window.shape for a in augmented)
        results.append(("Data Augmenter", "✅ Pass", f"Created {len(augmented)} augmentations"))
    except Exception as e:
        results.append(("Data Augmenter", "❌ Fail", str(e)))
    
    # Test 5: Full Pipeline
    try:
        config = PreprocessingConfig(
            raw_fs=1000.0,
            target_fs=30.0,
            window_length_s=10.0,
            hop_length_s=5.0
        )
        pipeline = PreprocessingPipeline(config)
        
        # Process test signal
        test_signal = np.random.randn(30000)  # 30s at 1000Hz
        windows = pipeline.process(test_signal, training=False)
        
        assert len(windows) > 0
        assert all(len(w) == 300 for w in windows)
        results.append(("Full Pipeline", "✅ Pass", f"Processed {len(windows)} windows"))
    except Exception as e:
        results.append(("Full Pipeline", "❌ Fail", str(e)))
    
    return results


def test_quality_validation():
    """Test quality validation functions."""
    console.print("\n[bold blue]Testing Quality Validation[/bold blue]")
    
    results = []
    
    # Test 1: SNR Computation
    try:
        # Generate clean PPG-like signal
        t = np.arange(0, 10, 1/30)
        signal = np.sin(2*np.pi*1.2*t)  # ~72 BPM
        
        snr = compute_snr(signal, fs=30)
        assert isinstance(snr, float)
        assert -20 <= snr <= 40
        results.append(("SNR Computation", "✅ Pass", f"SNR: {snr:.1f} dB"))
    except Exception as e:
        results.append(("SNR Computation", "❌ Fail", str(e)))
    
    # Test 2: HR Plausibility
    try:
        is_plausible, bpm = compute_hr_plausibility(signal, fs=30)
        assert isinstance(is_plausible, bool)
        assert 0 <= bpm <= 200
        results.append(("HR Plausibility", "✅ Pass", f"HR: {bpm:.1f} BPM"))
    except Exception as e:
        results.append(("HR Plausibility", "❌ Fail", str(e)))
    
    # Test 3: Motion Score
    try:
        motion = compute_motion_score(signal, fs=30)
        assert 0 <= motion <= 1
        results.append(("Motion Score", "✅ Pass", f"Motion: {motion:.2f}"))
    except Exception as e:
        results.append(("Motion Score", "❌ Fail", str(e)))
    
    # Test 4: Signal Quality
    try:
        quality = compute_signal_quality(signal, fs=30)
        assert 0 <= quality <= 1
        results.append(("Signal Quality", "✅ Pass", f"Quality: {quality:.2f}"))
    except Exception as e:
        results.append(("Signal Quality", "❌ Fail", str(e)))
    
    # Test 5: Quality Validator
    try:
        validator = SignalQualityValidator(min_quality=0.3)
        report = validator.validate(signal)
        
        assert hasattr(report, 'is_valid')
        assert hasattr(report, 'confidence')
        results.append(("Quality Validator", "✅ Pass", 
                       f"Valid: {report.is_valid}, Confidence: {report.confidence:.2f}"))
    except Exception as e:
        results.append(("Quality Validator", "❌ Fail", str(e)))
    
    return results


def test_model():
    """Test model architecture."""
    console.print("\n[bold blue]Testing Model Architecture[/bold blue]")
    
    results = []
    
    # Test 1: Model Creation
    try:
        config = ModelConfig(
            input_length=300,
            cnn_small_kernels=[3, 5],
            cnn_large_kernels=[11, 15],
            gru_layers=2,
            gru_hidden=64
        )
        model = HybridCNNGRU(config)
        
        params = model.get_num_parameters()
        assert params > 0
        results.append(("Model Creation", "✅ Pass", f"{params:,} parameters"))
    except Exception as e:
        results.append(("Model Creation", "❌ Fail", str(e)))
    
    # Test 2: Forward Pass
    try:
        batch_size = 4
        input_tensor = torch.randn(batch_size, 300)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 1)
        results.append(("Forward Pass", "✅ Pass", f"Output shape: {output.shape}"))
    except Exception as e:
        results.append(("Forward Pass", "❌ Fail", str(e)))
    
    # Test 3: Gradient Flow
    try:
        model.train()
        input_tensor = torch.randn(2, 300, requires_grad=True)
        target = torch.randn(2, 1)
        
        output = model(input_tensor)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()
        
        # Check if gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads
        results.append(("Gradient Flow", "✅ Pass", "Gradients computed"))
    except Exception as e:
        results.append(("Gradient Flow", "❌ Fail", str(e)))
    
    return results


def test_metrics():
    """Test evaluation metrics."""
    console.print("\n[bold blue]Testing Evaluation Metrics[/bold blue]")
    
    results = []
    
    # Test Clarke Error Grid
    try:
        # Generate test data
        reference = np.array([80, 120, 150, 180, 100])
        predicted = np.array([85, 115, 160, 170, 105])
        
        # Analyze
        zones, accuracy = clarke_error_grid_analysis(reference, predicted)
        
        assert all(z in zones for z in ['A', 'B', 'C', 'D', 'E'])
        assert 0 <= accuracy <= 100
        results.append(("Clarke Error Grid", "✅ Pass", 
                       f"Clinical accuracy: {accuracy:.1f}%"))
    except Exception as e:
        results.append(("Clarke Error Grid", "❌ Fail", str(e)))
    
    # Test zone calculation
    try:
        ceg = ClarkeErrorGrid()
        zone = ceg.get_zone(100, 95)
        assert zone in ['A', 'B', 'C', 'D', 'E']
        results.append(("Zone Calculation", "✅ Pass", f"100→95 mg/dL: Zone {zone}"))
    except Exception as e:
        results.append(("Zone Calculation", "❌ Fail", str(e)))
    
    return results


def test_synthetic_data():
    """Test synthetic data generation."""
    console.print("\n[bold blue]Testing Synthetic Data Generation[/bold blue]")
    
    results = []
    
    # Test 1: Single Signal Generation
    try:
        ppg, glucose = generate_synthetic_ppg(
            duration=10.0,
            fs=30.0,
            heart_rate=75.0,
            glucose_level=100.0
        )
        
        assert len(ppg) == 300  # 10s at 30Hz
        assert glucose == 100.0
        assert np.std(ppg) > 0
        results.append(("Single PPG Generation", "✅ Pass", f"Generated {len(ppg)} samples"))
    except Exception as e:
        results.append(("Single PPG Generation", "❌ Fail", str(e)))
    
    # Test 2: Dataset Generation
    try:
        signals, labels = generate_ppg_dataset(
            n_samples=10,
            duration=10.0,
            fs=30.0,
            glucose_range=(70, 180)
        )
        
        assert signals.shape == (10, 300)
        assert labels.shape == (10,)
        assert all(70 <= g <= 180 for g in labels)
        results.append(("Dataset Generation", "✅ Pass", f"Generated {len(labels)} samples"))
    except Exception as e:
        results.append(("Dataset Generation", "❌ Fail", str(e)))
    
    return results


def test_real_data():
    """Test with real PPG dataset."""
    console.print("\n[bold blue]Testing with Real PPG Dataset[/bold blue]")
    
    results = []
    data_dir = Path("/Users/amrmostafa/ppg-glucose-estimation/data/PPG_Dataset")
    
    # Test 1: Load CSV Signal
    try:
        signal_file = data_dir / "RawData" / "signal_01_0001.csv"
        if signal_file.exists():
            signal_data = pd.read_csv(signal_file, header=None).values.flatten()
            assert len(signal_data) > 0
            results.append(("Load CSV Signal", "✅ Pass", f"Loaded {len(signal_data)} samples"))
        else:
            results.append(("Load CSV Signal", "⚠️ Skip", "File not found"))
    except Exception as e:
        results.append(("Load CSV Signal", "❌ Fail", str(e)))
    
    # Test 2: Load Label
    try:
        label_file = data_dir / "Labels" / "label_01_0001.csv"
        if label_file.exists():
            label_data = pd.read_csv(label_file, header=None).values.flatten()
            glucose = float(label_data[0])
            assert 40 <= glucose <= 400
            results.append(("Load Label", "✅ Pass", f"Glucose: {glucose} mg/dL"))
        else:
            results.append(("Load Label", "⚠️ Skip", "File not found"))
    except Exception as e:
        results.append(("Load Label", "❌ Fail", str(e)))
    
    # Test 3: Process Real Signal
    try:
        if signal_file.exists():
            # Create pipeline
            config = PreprocessingConfig(
                raw_fs=2175.0,  # Original dataset frequency
                target_fs=30.0,
                window_length_s=10.0
            )
            pipeline = PreprocessingPipeline(config)
            
            # Process signal
            windows = pipeline.process(signal_data[:21750])  # 10 seconds
            
            assert len(windows) > 0
            results.append(("Process Real Signal", "✅ Pass", f"Created {len(windows)} windows"))
        else:
            results.append(("Process Real Signal", "⚠️ Skip", "No data"))
    except Exception as e:
        results.append(("Process Real Signal", "❌ Fail", str(e)))
    
    # Test 4: Quality Check Real Signal
    try:
        if signal_file.exists() and len(windows) > 0:
            validator = SignalQualityValidator()
            report = validator.validate(windows[0])
            
            results.append(("Quality Check", "✅ Pass", 
                          f"Quality: {report.overall_quality:.2f}, Valid: {report.is_valid}"))
        else:
            results.append(("Quality Check", "⚠️ Skip", "No windows"))
    except Exception as e:
        results.append(("Quality Check", "❌ Fail", str(e)))
    
    return results


def display_results(all_results):
    """Display test results in a table."""
    console.print("\n[bold green]Test Results Summary[/bold green]")
    
    table = Table(title="PPG Glucose Estimation System Tests")
    table.add_column("Module", style="cyan")
    table.add_column("Test", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="yellow")
    
    for module_name, module_results in all_results.items():
        for test_name, status, details in module_results:
            table.add_row(module_name, test_name, status, details)
    
    console.print(table)
    
    # Calculate statistics
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(
        1 for results in all_results.values() 
        for _, status, _ in results 
        if "Pass" in status
    )
    
    console.print(f"\n[bold]Total Tests: {total_tests}[/bold]")
    console.print(f"[green]Passed: {passed_tests}[/green]")
    console.print(f"[red]Failed: {total_tests - passed_tests}[/red]")
    console.print(f"[bold]Success Rate: {(passed_tests/total_tests)*100:.1f}%[/bold]")


def main():
    """Run all tests."""
    console.print("[bold magenta]PPG Glucose Estimation - Comprehensive Test Suite[/bold magenta]")
    console.print("=" * 60)
    
    all_results = {}
    
    # Run all test modules
    with console.status("[bold blue]Running tests..."):
        all_results["Preprocessing"] = test_preprocessing()
        all_results["Quality"] = test_quality_validation()
        all_results["Model"] = test_model()
        all_results["Metrics"] = test_metrics()
        all_results["Synthetic"] = test_synthetic_data()
        all_results["Real Data"] = test_real_data()
    
    # Display results
    display_results(all_results)
    
    # Save results to file
    results_file = Path("test_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved to {results_file}[/dim]")


if __name__ == "__main__":
    main()