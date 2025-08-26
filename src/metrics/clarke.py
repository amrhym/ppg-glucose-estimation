"""Clarke Error Grid Analysis for glucose prediction evaluation."""

from typing import Dict, List, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class ClarkeErrorGrid:
    """Clarke Error Grid Analysis for clinical glucose evaluation.
    
    Zones:
        A: Clinically accurate (±20% or ±20 mg/dL)
        B: Benign errors (no clinical action)
        C: Overcorrection errors
        D: Failure to detect (dangerous)
        E: Erroneous treatment (very dangerous)
    """
    
    @staticmethod
    def get_zone(reference: float, predicted: float) -> str:
        """Determine Clarke Error Grid zone for a single prediction.
        
        Args:
            reference: True glucose value (mg/dL)
            predicted: Predicted glucose value (mg/dL)
            
        Returns:
            Zone letter (A-E)
        """
        if reference <= 70:
            if predicted <= 70:
                return 'A'
            elif predicted <= 180:
                if predicted <= (7/5) * reference:
                    return 'A'
                else:
                    return 'B'
            else:
                return 'B' if predicted <= (7/5) * reference else 'C'
        
        elif reference <= 180:
            if predicted >= 70 and predicted <= 180:
                if abs(predicted - reference) <= 0.2 * reference:
                    return 'A'
                else:
                    return 'B'
            elif predicted < 70:
                if reference <= 290 and predicted >= (7/5) * reference - 182:
                    return 'B'
                else:
                    return 'D'
            else:  # predicted > 180
                if predicted <= (7/5) * reference:
                    return 'B'
                else:
                    return 'C'
        
        else:  # reference > 180
            if predicted >= 70:
                if predicted > 180 and abs(predicted - reference) <= 0.2 * reference:
                    return 'A'
                elif predicted >= 130 and predicted <= (7/5) * reference:
                    return 'B'
                elif predicted < 130:
                    return 'C'
                else:
                    return 'B' if predicted <= (7/5) * reference else 'C'
            else:  # predicted < 70
                return 'E'
    
    @staticmethod
    def analyze(
        reference: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        """Analyze predictions using Clarke Error Grid.
        
        Args:
            reference: True glucose values
            predicted: Predicted glucose values
            
        Returns:
            Dictionary with zone percentages
        """
        zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        
        for ref, pred in zip(reference, predicted):
            zone = ClarkeErrorGrid.get_zone(ref, pred)
            zones[zone] += 1
        
        # Convert to percentages
        total = len(reference)
        for zone in zones:
            zones[zone] = (zones[zone] / total) * 100
        
        return zones
    
    @staticmethod
    def plot(
        reference: np.ndarray,
        predicted: np.ndarray,
        title: str = "Clarke Error Grid Analysis",
        save_path: str = None,
    ):
        """Plot Clarke Error Grid with predictions.
        
        Args:
            reference: True glucose values
            predicted: Predicted glucose values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if plt is None:
            print("Matplotlib not available, skipping plot")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot zones
        x = np.arange(0, 401, 1)
        
        # Zone A boundaries
        ax.plot([0, 400], [0, 400], 'k-', linewidth=1)  # Perfect prediction
        ax.plot([0, 175/3], [70, 70], 'k-', linewidth=1)
        ax.plot([175/3, 400], [70, 400*0.6 + 10], 'k-', linewidth=1)
        ax.plot([70, 70], [84, 400], 'k-', linewidth=1)
        ax.plot([0, 70], [180, 180], 'k-', linewidth=1)
        ax.plot([70, 290], [180, 400*0.6 + 10], 'k-', linewidth=1)
        
        # Zone B boundaries
        ax.plot([0, 175/3], [50, 50], 'k--', linewidth=0.5)
        ax.plot([175/3, 400], [50, 400*0.6 - 10], 'k--', linewidth=0.5)
        ax.plot([50, 50], [70, 400], 'k--', linewidth=0.5)
        ax.plot([0, 50], [150, 150], 'k--', linewidth=0.5)
        ax.plot([50, 250], [150, 400*0.6 - 10], 'k--', linewidth=0.5)
        
        # Zone labels
        ax.text(30, 20, 'A', fontsize=20, ha='center')
        ax.text(370, 260, 'B', fontsize=20, ha='center')
        ax.text(280, 370, 'B', fontsize=20, ha='center')
        ax.text(160, 370, 'C', fontsize=20, ha='center')
        ax.text(160, 20, 'C', fontsize=20, ha='center')
        ax.text(30, 140, 'D', fontsize=20, ha='center')
        ax.text(370, 120, 'D', fontsize=20, ha='center')
        ax.text(30, 370, 'E', fontsize=20, ha='center')
        ax.text(370, 20, 'E', fontsize=20, ha='center')
        
        # Plot data points
        colors = []
        for ref, pred in zip(reference, predicted):
            zone = ClarkeErrorGrid.get_zone(ref, pred)
            if zone == 'A':
                colors.append('green')
            elif zone == 'B':
                colors.append('yellow')
            elif zone == 'C':
                colors.append('orange')
            elif zone == 'D':
                colors.append('red')
            else:  # E
                colors.append('darkred')
        
        ax.scatter(reference, predicted, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
        ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add zone statistics
        zones = ClarkeErrorGrid.analyze(reference, predicted)
        stats_text = (
            f"Zone A: {zones['A']:.1f}%\n"
            f"Zone B: {zones['B']:.1f}%\n"
            f"Zone C: {zones['C']:.1f}%\n"
            f"Zone D: {zones['D']:.1f}%\n"
            f"Zone E: {zones['E']:.1f}%"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return fig
    
    @staticmethod
    def get_clinical_accuracy(zones: Dict[str, float]) -> float:
        """Calculate clinical accuracy (Zone A + B percentage).
        
        Args:
            zones: Dictionary with zone percentages
            
        Returns:
            Clinical accuracy percentage
        """
        return zones['A'] + zones['B']


def clarke_error_grid_analysis(
    reference: np.ndarray,
    predicted: np.ndarray,
    plot: bool = False,
    save_path: str = None,
) -> Tuple[Dict[str, float], float]:
    """Convenience function for Clarke Error Grid analysis.
    
    Args:
        reference: True glucose values
        predicted: Predicted glucose values
        plot: Whether to create plot
        save_path: Optional path to save plot
        
    Returns:
        (zone_percentages, clinical_accuracy)
    """
    ceg = ClarkeErrorGrid()
    zones = ceg.analyze(reference, predicted)
    accuracy = ceg.get_clinical_accuracy(zones)
    
    if plot:
        ceg.plot(reference, predicted, save_path=save_path)
    
    return zones, accuracy