"""Tests for preprocessing module."""

import numpy as np
import pytest

from src.preprocessing.filters import BandpassFilter, apply_bandpass
from src.preprocessing.resampler import Resampler
from src.preprocessing.windowing import WindowGenerator


class TestBandpassFilter:
    """Test bandpass filter functionality."""
    
    def test_filter_creation(self):
        """Test filter initialization."""
        filter = BandpassFilter(band=(0.5, 8.0), fs=1000, order=4)
        assert filter.band == (0.5, 8.0)
        assert filter.fs == 1000
        assert filter.order == 4
        assert filter.zero_phase is True
    
    def test_filter_apply(self):
        """Test filter application."""
        # Create test signal with known frequencies
        fs = 1000
        t = np.arange(0, 1, 1/fs)
        
        # Signal with 1 Hz (in band) and 20 Hz (out of band)
        signal = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 20 * t)
        
        # Apply filter
        filter = BandpassFilter(band=(0.5, 8.0), fs=fs)
        filtered = filter.apply(signal)
        
        # Check output shape
        assert filtered.shape == signal.shape
        
        # Check that high frequency is attenuated
        fft_orig = np.abs(np.fft.rfft(signal))
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        
        # Find power at 1 Hz and 20 Hz
        idx_1hz = np.argmin(np.abs(freqs - 1))
        idx_20hz = np.argmin(np.abs(freqs - 20))
        
        # 1 Hz should be preserved, 20 Hz should be attenuated
        assert fft_filt[idx_1hz] > 0.8 * fft_orig[idx_1hz]
        assert fft_filt[idx_20hz] < 0.1 * fft_orig[idx_20hz]
    
    def test_filter_edge_cases(self):
        """Test filter with edge cases."""
        filter = BandpassFilter()
        
        # Short signal
        with pytest.raises(ValueError):
            filter.apply(np.array([1, 2, 3]))
        
        # NaN handling
        signal = np.ones(1000)
        signal[500] = np.nan
        filtered = filter.apply(signal)
        assert not np.any(np.isnan(filtered))


class TestResampler:
    """Test signal resampling."""
    
    def test_downsample(self):
        """Test downsampling."""
        fs_orig = 1000
        fs_target = 30
        
        # Create test signal
        duration = 1  # second
        t_orig = np.arange(0, duration, 1/fs_orig)
        signal = np.sin(2 * np.pi * 1 * t_orig)
        
        # Resample
        resampler = Resampler(fs_orig, fs_target)
        downsampled = resampler.resample(signal)
        
        # Check output length
        expected_len = int(len(signal) * fs_target / fs_orig)
        assert abs(len(downsampled) - expected_len) <= 1
    
    def test_different_methods(self):
        """Test different resampling methods."""
        signal = np.random.randn(10000)
        
        for method in ["decimate", "resample", "interp"]:
            resampler = Resampler(1000, 30, method=method)
            output = resampler.resample(signal)
            assert len(output) == 300


class TestWindowGenerator:
    """Test windowing functionality."""
    
    def test_window_generation(self):
        """Test basic window generation."""
        # Create test signal
        fs = 30
        duration = 30  # seconds
        signal = np.random.randn(int(duration * fs))
        
        # Generate windows
        gen = WindowGenerator(window_length=10.0, hop_length=5.0, fs=fs)
        windows = list(gen.generate(signal))
        
        # Check number of windows
        expected_windows = int((duration - 10) / 5) + 1
        assert len(windows) == expected_windows
        
        # Check window size
        for window, _ in windows:
            assert len(window) == 300  # 10s at 30Hz
    
    def test_normalization(self):
        """Test window normalization."""
        signal = np.random.randn(600)
        
        # Test z-score normalization
        gen = WindowGenerator(window_length=10.0, fs=30, normalize="zscore")
        windows = list(gen.generate(signal))
        
        for window, _ in windows:
            assert abs(np.mean(window)) < 0.01
            assert abs(np.std(window) - 1.0) < 0.01
        
        # Test minmax normalization
        gen = WindowGenerator(window_length=10.0, fs=30, normalize="minmax")
        windows = list(gen.generate(signal))
        
        for window, _ in windows:
            assert np.min(window) >= -0.01
            assert np.max(window) <= 1.01
    
    def test_quality_filtering(self):
        """Test quality-based window filtering."""
        signal = np.random.randn(900)
        quality = np.ones(900) * 0.8
        quality[300:600] = 0.3  # Low quality region
        
        gen = WindowGenerator(window_length=10.0, fs=30, min_quality=0.5)
        windows = list(gen.generate(signal, quality))
        
        # Should skip middle window
        assert len(windows) == 2