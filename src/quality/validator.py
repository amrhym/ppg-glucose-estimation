"""Signal quality validation for PPG windows."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.quality.metrics import (
    compute_baseline_stability,
    compute_hr_plausibility,
    compute_motion_score,
    compute_perfusion_index,
    compute_signal_quality,
    compute_snr,
)


@dataclass
class QualityReport:
    """Quality assessment report for a PPG window."""
    
    overall_quality: float
    snr_db: float
    hr_plausible: bool
    hr_bpm: float
    motion_score: float
    baseline_stability: float
    perfusion_index: float
    is_valid: bool
    confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "overall_quality": self.overall_quality,
            "snr_db": self.snr_db,
            "hr_plausible": self.hr_plausible,
            "hr_bpm": self.hr_bpm,
            "motion_score": self.motion_score,
            "baseline_stability": self.baseline_stability,
            "perfusion_index": self.perfusion_index,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
        }


class SignalQualityValidator:
    """Validate PPG signal quality for glucose estimation.
    
    Examples:
        >>> validator = SignalQualityValidator(min_quality=0.5)
        >>> report = validator.validate(ppg_window)
        >>> if report.is_valid:
        ...     glucose = model.predict(ppg_window)
    """
    
    def __init__(
        self,
        min_quality: float = 0.5,
        min_snr_db: float = -5.0,
        min_hr_bpm: float = 40.0,
        max_hr_bpm: float = 180.0,
        max_motion_score: float = 0.5,
        min_baseline_stability: float = 0.3,
        min_perfusion_index: float = 0.1,
        fs: float = 30.0,
    ):
        """Initialize quality validator.
        
        Args:
            min_quality: Minimum overall quality score
            min_snr_db: Minimum SNR in dB
            min_hr_bpm: Minimum heart rate
            max_hr_bpm: Maximum heart rate
            max_motion_score: Maximum motion artifact score
            min_baseline_stability: Minimum baseline stability
            min_perfusion_index: Minimum perfusion index
            fs: Sampling frequency
        """
        self.min_quality = min_quality
        self.min_snr_db = min_snr_db
        self.min_hr_bpm = min_hr_bpm
        self.max_hr_bpm = max_hr_bpm
        self.max_motion_score = max_motion_score
        self.min_baseline_stability = min_baseline_stability
        self.min_perfusion_index = min_perfusion_index
        self.fs = fs
    
    def validate(
        self,
        signal_data: np.ndarray,
        compute_all: bool = True,
    ) -> QualityReport:
        """Validate signal quality.
        
        Args:
            signal_data: PPG signal window
            compute_all: Compute all metrics even if early failure
            
        Returns:
            Quality assessment report
        """
        # Initialize report values
        report_dict = {
            "overall_quality": 0.0,
            "snr_db": -np.inf,
            "hr_plausible": False,
            "hr_bpm": 0.0,
            "motion_score": 1.0,
            "baseline_stability": 0.0,
            "perfusion_index": 0.0,
            "is_valid": False,
            "confidence": 0.0,
        }
        
        # Check for basic validity
        if len(signal_data) == 0:
            return QualityReport(**report_dict)
        
        if np.all(signal_data == signal_data[0]):
            # Flat signal
            return QualityReport(**report_dict)
        
        # Compute metrics
        try:
            # Overall quality
            overall_quality = compute_signal_quality(signal_data, self.fs)
            report_dict["overall_quality"] = overall_quality
            
            if not compute_all and overall_quality < self.min_quality:
                return QualityReport(**report_dict)
            
            # SNR
            snr_db = compute_snr(signal_data, self.fs)
            report_dict["snr_db"] = snr_db
            
            # Heart rate plausibility
            hr_plausible, hr_bpm = compute_hr_plausibility(
                signal_data, self.fs, self.min_hr_bpm, self.max_hr_bpm
            )
            report_dict["hr_plausible"] = hr_plausible
            report_dict["hr_bpm"] = hr_bpm
            
            # Motion score
            motion_score = compute_motion_score(signal_data, self.fs)
            report_dict["motion_score"] = motion_score
            
            # Baseline stability
            baseline_stability = compute_baseline_stability(signal_data)
            report_dict["baseline_stability"] = baseline_stability
            
            # Perfusion index
            perfusion_index = compute_perfusion_index(signal_data)
            report_dict["perfusion_index"] = perfusion_index
            
        except Exception as e:
            # Return invalid report on error
            return QualityReport(**report_dict)
        
        # Determine validity
        is_valid = (
            overall_quality >= self.min_quality
            and snr_db >= self.min_snr_db
            and hr_plausible
            and motion_score <= self.max_motion_score
            and baseline_stability >= self.min_baseline_stability
            and perfusion_index >= self.min_perfusion_index
        )
        report_dict["is_valid"] = is_valid
        
        # Calculate confidence based on quality metrics
        confidence = self._calculate_confidence(report_dict)
        report_dict["confidence"] = confidence
        
        return QualityReport(**report_dict)
    
    def _calculate_confidence(self, metrics: Dict) -> float:
        """Calculate prediction confidence based on quality metrics.
        
        Args:
            metrics: Dictionary of quality metrics
            
        Returns:
            Confidence score (0-1)
        """
        if not metrics["is_valid"]:
            return 0.0
        
        # Weight different components
        components = [
            metrics["overall_quality"],
            np.clip((metrics["snr_db"] + 10) / 30, 0, 1),
            1.0 if metrics["hr_plausible"] else 0.0,
            1.0 - metrics["motion_score"],
            metrics["baseline_stability"],
            np.clip(metrics["perfusion_index"] / 10, 0, 1),
        ]
        
        weights = [0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
        confidence = np.sum(np.array(weights) * np.array(components))
        
        return np.clip(confidence, 0, 1)
    
    def validate_batch(
        self,
        windows: np.ndarray,
    ) -> np.ndarray:
        """Validate batch of windows.
        
        Args:
            windows: Array of shape (n_windows, window_size)
            
        Returns:
            Array of validity flags
        """
        valid = np.zeros(len(windows), dtype=bool)
        
        for i, window in enumerate(windows):
            report = self.validate(window, compute_all=False)
            valid[i] = report.is_valid
        
        return valid