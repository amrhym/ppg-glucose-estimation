"""
Clinical Validation Analysis for PPG Glucose Estimation System

This module provides comprehensive clinical validation tools specifically designed
for glucose estimation systems, with emphasis on patient safety and clinical
deployment readiness.

Author: Clinical Validation Analyst
Date: 2025-09-01
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Polygon
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    plt = sns = Polygon = LinearSegmentedColormap = None
    warnings.warn("Matplotlib/Seaborn not available. Plotting functionality disabled.")

# Import existing Clarke Error Grid
try:
    from .metrics.clarke import ClarkeErrorGrid, clarke_error_grid_analysis
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.metrics.clarke import ClarkeErrorGrid, clarke_error_grid_analysis
    except ImportError:
        warnings.warn("Clarke Error Grid module not found. Limited functionality.")
        ClarkeErrorGrid = None


@dataclass
class ClinicalMetrics:
    """Container for clinical validation metrics."""
    clarke_zones: Dict[str, float]
    clinical_accuracy: float
    hypo_detection_rate: float
    hyper_detection_rate: float
    false_hypo_rate: float
    false_hyper_rate: float
    sensitivity_hypo: float
    specificity_hypo: float
    sensitivity_hyper: float
    specificity_hyper: float
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    bias: float
    precision: float
    zone_a_percentage: float
    safe_zone_percentage: float  # Zone A + B


@dataclass 
class SafetyAssessment:
    """Container for clinical safety assessment."""
    deployment_ready: bool
    safety_score: float
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    critical_failures: List[str]
    warnings: List[str]
    recommendations: List[str]
    edge_case_analysis: Dict[str, Any]


class ClinicalValidator:
    """
    Comprehensive clinical validation system for glucose estimation models.
    
    This class implements medical device standards for glucose monitoring
    systems and provides detailed safety assessments for clinical deployment.
    """
    
    def __init__(self, target_zone_a: float = 100.0):
        """
        Initialize clinical validator.
        
        Args:
            target_zone_a: Target percentage for Zone A (clinically acceptable)
                          FDA recommends >95% for clinical devices
        """
        self.target_zone_a = target_zone_a
        self.hypo_threshold = 70.0  # mg/dL
        self.hyper_threshold = 180.0  # mg/dL
        self.severe_hypo_threshold = 54.0  # mg/dL
        self.severe_hyper_threshold = 250.0  # mg/dL
        
    def calculate_clinical_metrics(
        self, 
        reference: np.ndarray, 
        predicted: np.ndarray
    ) -> ClinicalMetrics:
        """
        Calculate comprehensive clinical validation metrics.
        
        Args:
            reference: True glucose values (mg/dL)
            predicted: Predicted glucose values (mg/dL)
            
        Returns:
            ClinicalMetrics object with all validation metrics
        """
        # Basic error metrics
        mae = np.mean(np.abs(reference - predicted))
        rmse = np.sqrt(np.mean((reference - predicted) ** 2))
        mape = np.mean(np.abs((reference - predicted) / reference)) * 100
        bias = np.mean(predicted - reference)
        
        # Precision (1/CV where CV is coefficient of variation)
        cv = np.std(predicted - reference) / np.mean(reference) * 100
        precision = 100 / cv if cv > 0 else 100
        
        # Clarke Error Grid Analysis
        if ClarkeErrorGrid:
            zones = ClarkeErrorGrid.analyze(reference, predicted)
            clinical_accuracy = ClarkeErrorGrid.get_clinical_accuracy(zones)
            zone_a_percentage = zones['A']
            safe_zone_percentage = zones['A'] + zones['B']
        else:
            zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
            clinical_accuracy = 0
            zone_a_percentage = 0
            safe_zone_percentage = 0
        
        # Hypoglycemia detection analysis
        hypo_ref = reference < self.hypo_threshold
        hypo_pred = predicted < self.hypo_threshold
        
        if np.any(hypo_ref):
            hypo_detection_rate = np.sum(hypo_pred & hypo_ref) / np.sum(hypo_ref) * 100
            sensitivity_hypo = hypo_detection_rate / 100
        else:
            hypo_detection_rate = 0
            sensitivity_hypo = 0
        
        if np.any(~hypo_ref):
            false_hypo_rate = np.sum(hypo_pred & ~hypo_ref) / np.sum(~hypo_ref) * 100
            specificity_hypo = 1 - (false_hypo_rate / 100)
        else:
            false_hypo_rate = 0
            specificity_hypo = 1
        
        # Hyperglycemia detection analysis
        hyper_ref = reference > self.hyper_threshold
        hyper_pred = predicted > self.hyper_threshold
        
        if np.any(hyper_ref):
            hyper_detection_rate = np.sum(hyper_pred & hyper_ref) / np.sum(hyper_ref) * 100
            sensitivity_hyper = hyper_detection_rate / 100
        else:
            hyper_detection_rate = 0
            sensitivity_hyper = 0
        
        if np.any(~hyper_ref):
            false_hyper_rate = np.sum(hyper_pred & ~hyper_ref) / np.sum(~hyper_ref) * 100
            specificity_hyper = 1 - (false_hyper_rate / 100)
        else:
            false_hyper_rate = 0
            specificity_hyper = 1
        
        return ClinicalMetrics(
            clarke_zones=zones,
            clinical_accuracy=clinical_accuracy,
            hypo_detection_rate=hypo_detection_rate,
            hyper_detection_rate=hyper_detection_rate,
            false_hypo_rate=false_hypo_rate,
            false_hyper_rate=false_hyper_rate,
            sensitivity_hypo=sensitivity_hypo,
            specificity_hypo=specificity_hypo,
            sensitivity_hyper=sensitivity_hyper,
            specificity_hyper=specificity_hyper,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            bias=bias,
            precision=precision,
            zone_a_percentage=zone_a_percentage,
            safe_zone_percentage=safe_zone_percentage
        )
    
    def assess_safety(
        self, 
        metrics: ClinicalMetrics, 
        reference: np.ndarray, 
        predicted: np.ndarray
    ) -> SafetyAssessment:
        """
        Perform comprehensive clinical safety assessment.
        
        Args:
            metrics: Clinical metrics from calculate_clinical_metrics
            reference: True glucose values
            predicted: Predicted glucose values
            
        Returns:
            SafetyAssessment with deployment readiness and safety analysis
        """
        critical_failures = []
        warnings = []
        recommendations = []
        
        # Zone A requirement check (target 100%, minimum 95% for FDA approval)
        if metrics.zone_a_percentage < 95.0:
            critical_failures.append(
                f"Zone A percentage ({metrics.zone_a_percentage:.1f}%) below FDA minimum (95%)"
            )
        elif metrics.zone_a_percentage < self.target_zone_a:
            warnings.append(
                f"Zone A percentage ({metrics.zone_a_percentage:.1f}%) below target ({self.target_zone_a:.1f}%)"
            )
        
        # Zone D and E analysis (dangerous zones)
        danger_zone_percentage = metrics.clarke_zones['D'] + metrics.clarke_zones['E']
        if danger_zone_percentage > 2.0:
            critical_failures.append(
                f"Dangerous zone percentage (D+E: {danger_zone_percentage:.1f}%) exceeds safety limit (2%)"
            )
        elif danger_zone_percentage > 0.5:
            warnings.append(
                f"Dangerous zone percentage (D+E: {danger_zone_percentage:.1f}%) above recommended limit (0.5%)"
            )
        
        # Hypoglycemia detection safety
        if metrics.sensitivity_hypo < 0.90:  # 90% sensitivity required
            critical_failures.append(
                f"Hypoglycemia detection sensitivity ({metrics.sensitivity_hypo:.2f}) below safety requirement (0.90)"
            )
        elif metrics.sensitivity_hypo < 0.95:
            warnings.append(
                f"Hypoglycemia detection sensitivity ({metrics.sensitivity_hypo:.2f}) below optimal level (0.95)"
            )
        
        # False hypoglycemia rate
        if metrics.false_hypo_rate > 5.0:
            warnings.append(
                f"False hypoglycemia rate ({metrics.false_hypo_rate:.1f}%) above recommended limit (5%)"
            )
        
        # Severe hypoglycemia edge cases
        severe_hypo_ref = reference < self.severe_hypo_threshold
        severe_hypo_pred = predicted < self.severe_hypo_threshold
        severe_hypo_missed = np.sum(severe_hypo_ref & ~severe_hypo_pred)
        
        if severe_hypo_missed > 0:
            critical_failures.append(
                f"Missed {severe_hypo_missed} severe hypoglycemic events (<{self.severe_hypo_threshold} mg/dL)"
            )
        
        # Clinical accuracy requirement
        if metrics.clinical_accuracy < 95.0:
            critical_failures.append(
                f"Clinical accuracy ({metrics.clinical_accuracy:.1f}%) below FDA requirement (95%)"
            )
        elif metrics.clinical_accuracy < 99.0:
            warnings.append(
                f"Clinical accuracy ({metrics.clinical_accuracy:.1f}%) below optimal level (99%)"
            )
        
        # MAPE requirement for clinical devices
        if metrics.mean_absolute_percentage_error > 15.0:
            critical_failures.append(
                f"MAPE ({metrics.mean_absolute_percentage_error:.1f}%) exceeds clinical limit (15%)"
            )
        elif metrics.mean_absolute_percentage_error > 10.0:
            warnings.append(
                f"MAPE ({metrics.mean_absolute_percentage_error:.1f}%) above optimal level (10%)"
            )
        
        # Bias assessment
        if abs(metrics.bias) > 10.0:
            warnings.append(
                f"System bias ({metrics.bias:.1f} mg/dL) exceeds recommended limit (±10 mg/dL)"
            )
        
        # Edge case analysis
        edge_cases = self._analyze_edge_cases(reference, predicted)
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(metrics, len(critical_failures), len(warnings))
        
        # Determine risk level
        if len(critical_failures) > 0:
            risk_level = "CRITICAL"
        elif safety_score < 70:
            risk_level = "HIGH"
        elif safety_score < 85:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, critical_failures, warnings)
        
        # Deployment readiness
        deployment_ready = (len(critical_failures) == 0 and 
                          safety_score >= 85 and 
                          metrics.zone_a_percentage >= 95.0)
        
        return SafetyAssessment(
            deployment_ready=deployment_ready,
            safety_score=safety_score,
            risk_level=risk_level,
            critical_failures=critical_failures,
            warnings=warnings,
            recommendations=recommendations,
            edge_case_analysis=edge_cases
        )
    
    def _analyze_edge_cases(
        self, 
        reference: np.ndarray, 
        predicted: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze edge cases and extreme glucose ranges."""
        edge_cases = {
            'severe_hypoglycemia': {},
            'severe_hyperglycemia': {},
            'rapid_changes': {},
            'extreme_values': {}
        }
        
        # Severe hypoglycemia analysis
        severe_hypo_mask = reference < self.severe_hypo_threshold
        if np.any(severe_hypo_mask):
            severe_hypo_ref = reference[severe_hypo_mask]
            severe_hypo_pred = predicted[severe_hypo_mask]
            edge_cases['severe_hypoglycemia'] = {
                'count': len(severe_hypo_ref),
                'mae': np.mean(np.abs(severe_hypo_ref - severe_hypo_pred)),
                'detection_rate': np.mean(severe_hypo_pred < self.severe_hypo_threshold) * 100,
                'max_error': np.max(np.abs(severe_hypo_ref - severe_hypo_pred))
            }
        
        # Severe hyperglycemia analysis
        severe_hyper_mask = reference > self.severe_hyper_threshold
        if np.any(severe_hyper_mask):
            severe_hyper_ref = reference[severe_hyper_mask]
            severe_hyper_pred = predicted[severe_hyper_mask]
            edge_cases['severe_hyperglycemia'] = {
                'count': len(severe_hyper_ref),
                'mae': np.mean(np.abs(severe_hyper_ref - severe_hyper_pred)),
                'detection_rate': np.mean(severe_hyper_pred > self.severe_hyper_threshold) * 100,
                'max_error': np.max(np.abs(severe_hyper_ref - severe_hyper_pred))
            }
        
        # Rapid glucose changes (if sequential data available)
        if len(reference) > 1:
            ref_changes = np.abs(np.diff(reference))
            pred_changes = np.abs(np.diff(predicted))
            rapid_change_mask = ref_changes > 20  # >20 mg/dL change
            
            if np.any(rapid_change_mask):
                edge_cases['rapid_changes'] = {
                    'count': np.sum(rapid_change_mask),
                    'tracking_accuracy': np.corrcoef(ref_changes[rapid_change_mask], 
                                                   pred_changes[rapid_change_mask])[0,1]
                }
        
        # Extreme value analysis
        extreme_low_mask = reference < 40
        extreme_high_mask = reference > 400
        
        edge_cases['extreme_values'] = {
            'extreme_low_count': np.sum(extreme_low_mask),
            'extreme_high_count': np.sum(extreme_high_mask),
            'range_coverage': f"{np.min(reference):.1f} - {np.max(reference):.1f} mg/dL"
        }
        
        return edge_cases
    
    def _calculate_safety_score(
        self, 
        metrics: ClinicalMetrics, 
        critical_count: int, 
        warning_count: int
    ) -> float:
        """Calculate overall safety score (0-100)."""
        base_score = 100
        
        # Deduct points for critical failures
        base_score -= critical_count * 25
        
        # Deduct points for warnings
        base_score -= warning_count * 5
        
        # Adjust based on Zone A percentage
        if metrics.zone_a_percentage < 95:
            base_score -= (95 - metrics.zone_a_percentage) * 2
        
        # Adjust based on dangerous zones
        danger_zones = metrics.clarke_zones['D'] + metrics.clarke_zones['E']
        if danger_zones > 0:
            base_score -= danger_zones * 10
        
        # Adjust based on hypoglycemia detection
        if metrics.sensitivity_hypo < 0.9:
            base_score -= (0.9 - metrics.sensitivity_hypo) * 100
        
        return max(0, min(100, base_score))
    
    def _generate_recommendations(
        self, 
        metrics: ClinicalMetrics, 
        critical_failures: List[str], 
        warnings: List[str]
    ) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        if metrics.zone_a_percentage < 95:
            recommendations.append(
                "Improve model accuracy to achieve >95% Zone A classification"
            )
        
        if metrics.sensitivity_hypo < 0.95:
            recommendations.append(
                "Enhance hypoglycemia detection algorithm with additional training data"
            )
        
        if metrics.mean_absolute_percentage_error > 10:
            recommendations.append(
                "Reduce systematic errors through calibration optimization"
            )
        
        if abs(metrics.bias) > 5:
            if metrics.bias > 0:
                recommendations.append(
                    "Address positive bias - model tends to overestimate glucose levels"
                )
            else:
                recommendations.append(
                    "Address negative bias - model tends to underestimate glucose levels"
                )
        
        if metrics.clarke_zones['D'] + metrics.clarke_zones['E'] > 0.5:
            recommendations.append(
                "Critical: Eliminate dangerous zone predictions through improved training"
            )
        
        if len(critical_failures) == 0 and len(warnings) == 0:
            recommendations.append(
                "System meets clinical safety standards - ready for regulatory submission"
            )
        
        return recommendations

    def plot_enhanced_clarke_grid(
        self, 
        reference: np.ndarray, 
        predicted: np.ndarray, 
        title: str = "Enhanced Clarke Error Grid Analysis",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create publication-ready enhanced Clarke Error Grid plot.
        
        Args:
            reference: True glucose values
            predicted: Predicted glucose values  
            title: Plot title
            save_path: Path to save high-resolution figure
            dpi: Resolution for saved figure
            
        Returns:
            Matplotlib figure object
        """
        if plt is None:
            print("Matplotlib not available")
            return None
            
        # Create figure with publication settings
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define zone boundaries more precisely
        x = np.linspace(0, 400, 1000)
        
        # Zone A (clinically accurate) - green background
        zone_a_x = [0, 70, 70, 290, 400, 400, 0]
        zone_a_y = [0, 70, 180, 400, 400, 0, 0]
        zone_a_poly = Polygon(list(zip(zone_a_x, zone_a_y)), 
                             alpha=0.1, facecolor='green', label='Zone A')
        ax.add_patch(zone_a_poly)
        
        # Plot zone boundary lines with medical device precision
        # Zone A boundaries (thick lines for clinical importance)
        ax.plot([0, 70], [70, 70], 'g-', linewidth=3, label='Zone A boundary')
        ax.plot([70, 70], [70, 180], 'g-', linewidth=3)
        ax.plot([70, 290], [180, 400], 'g-', linewidth=3)
        ax.plot([0, 70], [0, 70], 'g-', linewidth=3)  # Perfect prediction line segment
        ax.plot([70, 400], [84, 400], 'g-', linewidth=3)
        
        # Zone B boundaries (medium thickness)
        ax.plot([0, 175/3], [50, 50], 'orange', linewidth=2, linestyle='--', alpha=0.8)
        ax.plot([175/3, 320], [50, 320*0.6 + 10], 'orange', linewidth=2, linestyle='--', alpha=0.8)
        ax.plot([50, 50], [0, 400], 'orange', linewidth=2, linestyle='--', alpha=0.8)
        
        # Zone C, D, E boundaries (thinner lines for less critical zones)
        ax.plot([0, 400], [0, 400], 'k-', linewidth=1, alpha=0.5)  # Perfect prediction reference
        
        # Color-code data points by zone with enhanced visibility
        colors = []
        sizes = []
        alphas = []
        
        for ref, pred in zip(reference, predicted):
            if ClarkeErrorGrid:
                zone = ClarkeErrorGrid.get_zone(ref, pred)
            else:
                zone = 'A'  # Default if not available
                
            if zone == 'A':
                colors.append('#2E8B57')  # Sea green
                sizes.append(60)
                alphas.append(0.7)
            elif zone == 'B':
                colors.append('#FFA500')  # Orange
                sizes.append(50)
                alphas.append(0.6)
            elif zone == 'C':
                colors.append('#FF6347')  # Tomato
                sizes.append(80)
                alphas.append(0.8)
            elif zone == 'D':
                colors.append('#DC143C')  # Crimson
                sizes.append(100)
                alphas.append(0.9)
            else:  # Zone E
                colors.append('#8B0000')  # Dark red
                sizes.append(120)
                alphas.append(1.0)
        
        # Plot points with varying sizes based on clinical importance
        scatter = ax.scatter(reference, predicted, c=colors, s=sizes, 
                           alpha=0.7, edgecolors='black', linewidth=0.8)
        
        # Enhanced zone labels with medical context
        ax.text(35, 35, 'Zone A\n(Clinically\nAccurate)', fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        ax.text(320, 200, 'Zone B\n(Benign\nErrors)', fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.6))
        ax.text(200, 350, 'Zone B', fontsize=12, ha='center', va='center')
        ax.text(120, 350, 'Zone C\n(Over-\ncorrection)', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
        ax.text(30, 150, 'Zone D\n(Dangerous\nFailure)', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.6))
        ax.text(350, 50, 'Zone E\n(Erroneous\nTreatment)', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', alpha=0.6))
        
        # Medical device standard formatting
        ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Clinical range focus
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')
        
        # Add detailed statistics box
        if ClarkeErrorGrid:
            zones = ClarkeErrorGrid.analyze(reference, predicted)
            clinical_accuracy = ClarkeErrorGrid.get_clinical_accuracy(zones)
        else:
            zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
            clinical_accuracy = 0
            
        # Calculate additional metrics
        mae = np.mean(np.abs(reference - predicted))
        rmse = np.sqrt(np.mean((reference - predicted) ** 2))
        
        stats_text = (
            f"CLINICAL VALIDATION RESULTS\n"
            f"{'='*30}\n"
            f"Zone A (Clinically Accurate): {zones['A']:.1f}%\n"
            f"Zone B (Benign Errors): {zones['B']:.1f}%\n"
            f"Zone C (Overcorrection): {zones['C']:.1f}%\n"
            f"Zone D (Dangerous): {zones['D']:.1f}%\n"
            f"Zone E (Very Dangerous): {zones['E']:.1f}%\n"
            f"{'='*30}\n"
            f"Clinical Accuracy (A+B): {clinical_accuracy:.1f}%\n"
            f"MAE: {mae:.1f} mg/dL\n"
            f"RMSE: {rmse:.1f} mg/dL\n"
            f"Data Points: {len(reference)}"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         alpha=0.95, edgecolor='black', linewidth=1))
        
        # Add FDA compliance indicator
        compliance_color = 'green' if zones['A'] >= 95 and zones['D'] + zones['E'] <= 2 else 'red'
        compliance_text = "FDA COMPLIANT" if compliance_color == 'green' else "NON-COMPLIANT"
        
        ax.text(0.98, 0.02, compliance_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom',
                fontsize=12, fontweight='bold', color=compliance_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         alpha=0.9, edgecolor=compliance_color, linewidth=2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Enhanced Clarke Error Grid saved to: {save_path}")
        
        return fig

    def plot_clinical_metrics_table(
        self, 
        metrics: ClinicalMetrics, 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create publication-ready clinical metrics table."""
        if plt is None:
            print("Matplotlib not available")
            return None
            
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [
            ['METRIC', 'VALUE', 'STANDARD', 'STATUS'],
            ['Zone A Percentage', f'{metrics.zone_a_percentage:.1f}%', '≥95%', 
             '✓ PASS' if metrics.zone_a_percentage >= 95 else '✗ FAIL'],
            ['Clinical Accuracy (A+B)', f'{metrics.clinical_accuracy:.1f}%', '≥95%',
             '✓ PASS' if metrics.clinical_accuracy >= 95 else '✗ FAIL'],
            ['Dangerous Zones (D+E)', f'{metrics.clarke_zones["D"] + metrics.clarke_zones["E"]:.1f}%', '≤2%',
             '✓ PASS' if metrics.clarke_zones["D"] + metrics.clarke_zones["E"] <= 2 else '✗ FAIL'],
            ['Hypoglycemia Sensitivity', f'{metrics.sensitivity_hypo:.3f}', '≥0.90',
             '✓ PASS' if metrics.sensitivity_hypo >= 0.90 else '✗ FAIL'],
            ['Hypoglycemia Specificity', f'{metrics.specificity_hypo:.3f}', '≥0.95', 
             '✓ PASS' if metrics.specificity_hypo >= 0.95 else '✗ FAIL'],
            ['Mean Absolute Error', f'{metrics.mean_absolute_error:.1f} mg/dL', '≤15 mg/dL',
             '✓ PASS' if metrics.mean_absolute_error <= 15 else '✗ FAIL'],
            ['MAPE', f'{metrics.mean_absolute_percentage_error:.1f}%', '≤15%',
             '✓ PASS' if metrics.mean_absolute_percentage_error <= 15 else '✗ FAIL'],
            ['System Bias', f'{metrics.bias:.1f} mg/dL', '±10 mg/dL',
             '✓ PASS' if abs(metrics.bias) <= 10 else '✗ FAIL'],
            ['RMSE', f'{metrics.root_mean_squared_error:.1f} mg/dL', '≤20 mg/dL',
             '✓ PASS' if metrics.root_mean_squared_error <= 20 else '✗ FAIL'],
        ]
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Format table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Color code cells
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[i, j]
                
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 3:  # Status column
                    if '✓ PASS' in table_data[i][j]:
                        cell.set_facecolor('#90EE90')
                    elif '✗ FAIL' in table_data[i][j]:
                        cell.set_facecolor('#FFB6C1')
                else:
                    cell.set_facecolor('#F8F9FA')
                
                cell.set_edgecolor('#000000')
                cell.set_linewidth(1)
        
        ax.set_title('Clinical Validation Metrics Summary', 
                    fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Clinical metrics table saved to: {save_path}")
        
        return fig

    def plot_safety_zone_distribution(
        self, 
        metrics: ClinicalMetrics, 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create safety zone distribution chart."""
        if plt is None:
            print("Matplotlib not available")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart for zone distribution
        zones = list(metrics.clarke_zones.keys())
        percentages = list(metrics.clarke_zones.values())
        colors = ['#2E8B57', '#FFA500', '#FFFF00', '#FF4500', '#8B0000']
        
        wedges, texts, autotexts = ax1.pie(percentages, labels=zones, colors=colors, 
                                          autopct='%1.1f%%', startangle=90,
                                          explode=[0.05 if p > 0 else 0 for p in percentages])
        
        ax1.set_title('Clarke Error Grid Zone Distribution', 
                     fontsize=14, fontweight='bold')
        
        # Bar chart for clinical safety metrics
        safety_metrics = {
            'Zone A\n(Target: 100%)': metrics.zone_a_percentage,
            'Clinical Accuracy\n(A+B, Target: ≥95%)': metrics.clinical_accuracy,
            'Hypo Sensitivity\n(Target: ≥90%)': metrics.sensitivity_hypo * 100,
            'Hypo Specificity\n(Target: ≥95%)': metrics.specificity_hypo * 100,
            'Hyper Sensitivity\n(Target: ≥85%)': metrics.sensitivity_hyper * 100
        }
        
        targets = [100, 95, 90, 95, 85]
        bars = ax2.bar(range(len(safety_metrics)), list(safety_metrics.values()),
                      color=['green' if v >= t else 'red' 
                            for v, t in zip(safety_metrics.values(), targets)])
        
        # Add target lines
        for i, target in enumerate(targets):
            ax2.axhline(y=target, xmin=i/len(targets)-0.4/len(targets), 
                       xmax=i/len(targets)+0.4/len(targets), 
                       color='red', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Clinical Safety Metrics')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Clinical Safety Performance vs. Targets', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(safety_metrics)))
        ax2.set_xticklabels(list(safety_metrics.keys()), rotation=45, ha='right')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, safety_metrics.values())):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Safety zone distribution chart saved to: {save_path}")
        
        return fig

    def generate_clinical_report(
        self, 
        reference: np.ndarray, 
        predicted: np.ndarray, 
        save_dir: str = "./clinical_validation"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive clinical validation report.
        
        Args:
            reference: True glucose values
            predicted: Predicted glucose values
            save_dir: Directory to save report and figures
            
        Returns:
            Complete clinical validation report dictionary
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate metrics
        print("Calculating clinical validation metrics...")
        metrics = self.calculate_clinical_metrics(reference, predicted)
        
        # Perform safety assessment
        print("Performing clinical safety assessment...")
        safety = self.assess_safety(metrics, reference, predicted)
        
        # Generate plots
        print("Generating publication-ready figures...")
        
        # Enhanced Clarke Error Grid
        clarke_fig = self.plot_enhanced_clarke_grid(
            reference, predicted,
            title="Clinical Validation: Enhanced Clarke Error Grid Analysis",
            save_path=os.path.join(save_dir, "enhanced_clarke_error_grid.png")
        )
        
        # Clinical metrics table
        metrics_fig = self.plot_clinical_metrics_table(
            metrics,
            save_path=os.path.join(save_dir, "clinical_metrics_table.png")
        )
        
        # Safety distribution charts
        safety_fig = self.plot_safety_zone_distribution(
            metrics,
            save_path=os.path.join(save_dir, "safety_zone_distribution.png")
        )
        
        # Create comprehensive report
        report = {
            'executive_summary': {
                'deployment_ready': safety.deployment_ready,
                'safety_score': safety.safety_score,
                'risk_level': safety.risk_level,
                'zone_a_percentage': metrics.zone_a_percentage,
                'clinical_accuracy': metrics.clinical_accuracy,
                'total_predictions': len(reference)
            },
            'clinical_metrics': {
                'clarke_zones': metrics.clarke_zones,
                'clinical_accuracy': metrics.clinical_accuracy,
                'hypoglycemia_detection': {
                    'sensitivity': metrics.sensitivity_hypo,
                    'specificity': metrics.specificity_hypo,
                    'detection_rate': metrics.hypo_detection_rate,
                    'false_positive_rate': metrics.false_hypo_rate
                },
                'hyperglycemia_detection': {
                    'sensitivity': metrics.sensitivity_hyper,
                    'specificity': metrics.specificity_hyper,
                    'detection_rate': metrics.hyper_detection_rate,
                    'false_positive_rate': metrics.false_hyper_rate
                },
                'error_metrics': {
                    'mae': metrics.mean_absolute_error,
                    'rmse': metrics.root_mean_squared_error,
                    'mape': metrics.mean_absolute_percentage_error,
                    'bias': metrics.bias,
                    'precision': metrics.precision
                }
            },
            'safety_assessment': {
                'deployment_ready': safety.deployment_ready,
                'safety_score': safety.safety_score,
                'risk_level': safety.risk_level,
                'critical_failures': safety.critical_failures,
                'warnings': safety.warnings,
                'recommendations': safety.recommendations,
                'edge_case_analysis': safety.edge_case_analysis
            },
            'regulatory_compliance': {
                'fda_zone_a_requirement': metrics.zone_a_percentage >= 95.0,
                'fda_clinical_accuracy_requirement': metrics.clinical_accuracy >= 95.0,
                'dangerous_zone_limit': (metrics.clarke_zones['D'] + metrics.clarke_zones['E']) <= 2.0,
                'hypoglycemia_sensitivity_requirement': metrics.sensitivity_hypo >= 0.90,
                'mape_requirement': metrics.mean_absolute_percentage_error <= 15.0
            },
            'clinical_deployment_guidelines': {
                'patient_monitoring_recommendations': [
                    "Implement continuous glucose monitoring validation",
                    "Establish alert thresholds for dangerous predictions",
                    "Require clinical oversight for extreme values",
                    "Implement automatic safety shutoffs for Zone D/E predictions"
                ],
                'limitations': [
                    f"Edge case performance analysis shows limitations in extreme ranges",
                    f"Requires clinical validation in diverse patient populations",
                    f"Performance may vary with individual physiological differences"
                ],
                'contraindications': safety.critical_failures if safety.critical_failures else [
                    "None identified - system meets safety requirements"
                ]
            }
        }
        
        # Save detailed report to JSON
        import json
        with open(os.path.join(save_dir, "clinical_validation_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate executive summary text
        summary_text = self._generate_executive_summary(report)
        with open(os.path.join(save_dir, "executive_summary.txt"), 'w') as f:
            f.write(summary_text)
        
        print(f"\nClinical validation report generated in: {save_dir}")
        print(f"Safety Score: {safety.safety_score:.1f}/100")
        print(f"Risk Level: {safety.risk_level}")
        print(f"Deployment Ready: {'YES' if safety.deployment_ready else 'NO'}")
        
        return report
    
    def _generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """Generate executive summary text."""
        summary = f"""
CLINICAL VALIDATION EXECUTIVE SUMMARY
=====================================
Date: 2025-09-01
System: PPG Glucose Estimation Model

DEPLOYMENT READINESS: {"APPROVED" if report['executive_summary']['deployment_ready'] else "NOT APPROVED"}
Safety Score: {report['executive_summary']['safety_score']:.1f}/100
Risk Level: {report['executive_summary']['risk_level']}

KEY CLINICAL METRICS:
- Zone A (Clinically Accurate): {report['executive_summary']['zone_a_percentage']:.1f}%
- Clinical Accuracy (A+B): {report['executive_summary']['clinical_accuracy']:.1f}%
- Total Predictions Analyzed: {report['executive_summary']['total_predictions']:,}

FDA COMPLIANCE STATUS:
- Zone A Requirement (≥95%): {"✓ PASS" if report['regulatory_compliance']['fda_zone_a_requirement'] else "✗ FAIL"}
- Clinical Accuracy (≥95%): {"✓ PASS" if report['regulatory_compliance']['fda_clinical_accuracy_requirement'] else "✗ FAIL"}
- Dangerous Zone Limit (≤2%): {"✓ PASS" if report['regulatory_compliance']['dangerous_zone_limit'] else "✗ FAIL"}
- Hypoglycemia Detection (≥90%): {"✓ PASS" if report['regulatory_compliance']['hypoglycemia_sensitivity_requirement'] else "✗ FAIL"}

SAFETY ASSESSMENT:
Critical Failures: {len(report['safety_assessment']['critical_failures'])}
Warnings: {len(report['safety_assessment']['warnings'])}

CLINICAL DEPLOYMENT RECOMMENDATION:
{"This system meets clinical safety standards and is recommended for regulatory submission and clinical deployment with appropriate medical oversight." if report['executive_summary']['deployment_ready'] else "This system requires significant improvement before clinical deployment. Address critical failures before proceeding."}

For detailed analysis, refer to the complete clinical validation report.
        """
        return summary


def main():
    """Example usage of clinical validation system."""
    print("PPG Glucose Estimation - Clinical Validation Analysis")
    print("=" * 55)
    
    # Example with synthetic data (replace with real model predictions)
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate reference glucose values with realistic distribution
    reference = np.concatenate([
        np.random.normal(100, 20, int(0.7 * n_samples)),  # Normal range
        np.random.normal(60, 10, int(0.1 * n_samples)),   # Hypoglycemic
        np.random.normal(200, 30, int(0.2 * n_samples))   # Hyperglycemic
    ])
    reference = np.clip(reference, 40, 400)
    
    # Simulate predictions with some error
    predicted = reference + np.random.normal(0, 15, len(reference))
    predicted = np.clip(predicted, 40, 400)
    
    # Initialize clinical validator
    validator = ClinicalValidator(target_zone_a=100.0)
    
    # Generate comprehensive clinical validation report
    report = validator.generate_clinical_report(
        reference, predicted, 
        save_dir="./clinical_validation_results"
    )
    
    print("\nClinical validation analysis complete!")
    print("Check ./clinical_validation_results/ for detailed reports and figures.")


if __name__ == "__main__":
    main()