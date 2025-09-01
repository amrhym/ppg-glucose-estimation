#!/usr/bin/env python3
"""
Clinical Validation Example for PPG Glucose Estimation System

This script demonstrates how to use the comprehensive clinical validation
system with real model predictions and reference glucose measurements.

Usage:
    python clinical_validation_example.py [--data-file path_to_data.csv]

Author: Clinical Validation Analyst
Date: 2025-09-01
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

try:
    from evaluation import ClinicalValidator
except ImportError:
    print("Error: Could not import clinical validation module")
    print("Make sure src/evaluation.py exists")
    sys.exit(1)


def load_prediction_data(file_path=None):
    """
    Load glucose prediction data from file or generate synthetic data.
    
    Expected CSV format:
    reference_glucose,predicted_glucose
    120.5,118.3
    85.2,87.1
    ...
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        if 'reference_glucose' in df.columns and 'predicted_glucose' in df.columns:
            reference = df['reference_glucose'].values
            predicted = df['predicted_glucose'].values
            print(f"Loaded {len(reference)} glucose measurements")
        else:
            print("Error: CSV must contain 'reference_glucose' and 'predicted_glucose' columns")
            return None, None
            
    else:
        print("Generating synthetic clinical validation data...")
        print("(Replace this with your actual model predictions)")
        
        # Generate realistic clinical validation dataset
        np.random.seed(2025)
        n_samples = 1200
        
        # Realistic glucose distribution based on clinical studies
        # Normal glucose: ~70% of readings
        normal_glucose = np.random.normal(105, 16, int(0.70 * n_samples))
        
        # Hypoglycemic episodes: ~15% of readings  
        hypo_glucose = np.random.normal(58, 12, int(0.15 * n_samples))
        
        # Hyperglycemic episodes: ~15% of readings
        hyper_glucose = np.random.normal(210, 35, int(0.15 * n_samples))
        
        reference = np.concatenate([normal_glucose, hypo_glucose, hyper_glucose])
        reference = np.clip(reference, 35, 400)
        
        # Simulate high-performance model predictions
        # Clinical-grade CGM typically has 8-12% MAPE
        model_error_std = 8.5  # mg/dL standard deviation
        systematic_bias = -1.2  # Small negative bias (common in CGM)
        
        predicted = reference + systematic_bias + np.random.normal(0, model_error_std, len(reference))
        
        # Simulate improved detection for critical ranges
        for i in range(len(reference)):
            # Enhanced hypoglycemia detection (critical for patient safety)
            if reference[i] < 70:
                if np.random.rand() < 0.92:  # 92% detection rate
                    if predicted[i] > 80:  # If prediction missed hypo
                        predicted[i] = reference[i] + np.random.normal(0, 5)
            
            # Enhanced severe hypoglycemia detection
            if reference[i] < 54:
                if np.random.rand() < 0.96:  # 96% detection rate for severe hypo
                    predicted[i] = min(predicted[i], 60)
            
            # Improved hyperglycemia detection
            if reference[i] > 180:
                if np.random.rand() < 0.89:  # 89% detection rate
                    if predicted[i] < 170:  # If prediction missed hyper
                        predicted[i] = reference[i] + np.random.normal(0, 8)
        
        predicted = np.clip(predicted, 35, 400)
        
        # Save synthetic data for reference
        synthetic_df = pd.DataFrame({
            'reference_glucose': reference,
            'predicted_glucose': predicted
        })
        synthetic_df.to_csv('synthetic_clinical_data.csv', index=False)
        print("Synthetic data saved to: synthetic_clinical_data.csv")
    
    return reference, predicted


def perform_clinical_validation(reference, predicted, output_dir="clinical_validation_results"):
    """
    Perform comprehensive clinical validation analysis.
    """
    print("\nPerforming Clinical Validation Analysis...")
    print("=" * 50)
    
    # Initialize clinical validator with FDA standards
    validator = ClinicalValidator(target_zone_a=95.0)  # FDA target
    
    # Generate comprehensive clinical report
    report = validator.generate_clinical_report(
        reference, 
        predicted, 
        save_dir=output_dir
    )
    
    # Display critical results
    print("\n" + "="*60)
    print("CLINICAL VALIDATION RESULTS")
    print("="*60)
    
    print(f"\nCLINICAL ACCURACY ASSESSMENT:")
    print(f"• Zone A (Clinically Accurate): {report['clinical_metrics']['clarke_zones']['A']:.1f}%")
    print(f"• Zone B (Benign Errors): {report['clinical_metrics']['clarke_zones']['B']:.1f}%")
    print(f"• Zone C (Overcorrection): {report['clinical_metrics']['clarke_zones']['C']:.1f}%")
    print(f"• Zone D (Dangerous Failure): {report['clinical_metrics']['clarke_zones']['D']:.1f}%")
    print(f"• Zone E (Erroneous Treatment): {report['clinical_metrics']['clarke_zones']['E']:.1f}%")
    print(f"• Overall Clinical Accuracy (A+B): {report['clinical_metrics']['clinical_accuracy']:.1f}%")
    
    print(f"\nSAFETY-CRITICAL METRICS:")
    print(f"• Hypoglycemia Detection Sensitivity: {report['clinical_metrics']['hypoglycemia_detection']['sensitivity']:.3f}")
    print(f"• Hypoglycemia Detection Specificity: {report['clinical_metrics']['hypoglycemia_detection']['specificity']:.3f}")
    print(f"• Hyperglycemia Detection Sensitivity: {report['clinical_metrics']['hyperglycemia_detection']['sensitivity']:.3f}")
    print(f"• False Hypoglycemia Rate: {report['clinical_metrics']['hypoglycemia_detection']['false_positive_rate']:.1f}%")
    
    print(f"\nERROR METRICS:")
    print(f"• Mean Absolute Error: {report['clinical_metrics']['error_metrics']['mae']:.1f} mg/dL")
    print(f"• Root Mean Square Error: {report['clinical_metrics']['error_metrics']['rmse']:.1f} mg/dL")
    print(f"• Mean Absolute Percentage Error: {report['clinical_metrics']['error_metrics']['mape']:.1f}%")
    print(f"• System Bias: {report['clinical_metrics']['error_metrics']['bias']:.1f} mg/dL")
    
    print(f"\nREGULATORY COMPLIANCE:")
    compliance_items = [
        ("FDA Zone A Requirement (≥95%)", report['regulatory_compliance']['fda_zone_a_requirement']),
        ("FDA Clinical Accuracy (≥95%)", report['regulatory_compliance']['fda_clinical_accuracy_requirement']),
        ("Dangerous Zone Limit (≤2%)", report['regulatory_compliance']['dangerous_zone_limit']),
        ("Hypoglycemia Detection (≥90%)", report['regulatory_compliance']['hypoglycemia_sensitivity_requirement']),
        ("MAPE Requirement (≤15%)", report['regulatory_compliance']['mape_requirement'])
    ]
    
    for item, status in compliance_items:
        print(f"• {item}: {'✓ PASS' if status else '✗ FAIL'}")
    
    print(f"\nDEPLOYMENT ASSESSMENT:")
    print(f"• Safety Score: {report['safety_assessment']['safety_score']:.1f}/100")
    print(f"• Risk Level: {report['safety_assessment']['risk_level']}")
    print(f"• Deployment Ready: {'YES' if report['safety_assessment']['deployment_ready'] else 'NO'}")
    
    if report['safety_assessment']['critical_failures']:
        print(f"\nCRITICAL FAILURES ({len(report['safety_assessment']['critical_failures'])}):")
        for i, failure in enumerate(report['safety_assessment']['critical_failures'], 1):
            print(f"  {i}. {failure}")
    
    if report['safety_assessment']['warnings']:
        print(f"\nWARNINGS ({len(report['safety_assessment']['warnings'])}):")
        for i, warning in enumerate(report['safety_assessment']['warnings'], 1):
            print(f"  {i}. {warning}")
    
    if report['safety_assessment']['recommendations']:
        print(f"\nCLINICAL RECOMMENDATIONS:")
        for i, rec in enumerate(report['safety_assessment']['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nEDGE CASE ANALYSIS:")
    edge_cases = report['safety_assessment']['edge_case_analysis']
    if edge_cases['severe_hypoglycemia']:
        severe_hypo = edge_cases['severe_hypoglycemia']
        print(f"• Severe Hypoglycemia (<54 mg/dL): {severe_hypo['count']} cases")
        print(f"  - Detection Rate: {severe_hypo['detection_rate']:.1f}%")
        print(f"  - MAE: {severe_hypo['mae']:.1f} mg/dL")
    
    if edge_cases['severe_hyperglycemia']:
        severe_hyper = edge_cases['severe_hyperglycemia']
        print(f"• Severe Hyperglycemia (>250 mg/dL): {severe_hyper['count']} cases")
        print(f"  - Detection Rate: {severe_hyper['detection_rate']:.1f}%")
        print(f"  - MAE: {severe_hyper['mae']:.1f} mg/dL")
    
    print(f"\n" + "="*60)
    print("CLINICAL VALIDATION COMPLETE")
    print("="*60)
    print(f"Detailed reports and publication-ready figures saved to: {output_dir}/")
    print(f"Files generated:")
    print(f"• enhanced_clarke_error_grid.png - High-resolution Clarke Error Grid")
    print(f"• clinical_metrics_table.png - Clinical validation metrics summary")
    print(f"• safety_zone_distribution.png - Safety zone analysis charts")  
    print(f"• clinical_validation_report.json - Complete validation data")
    print(f"• executive_summary.txt - Executive summary for stakeholders")
    
    return report


def main():
    """Main clinical validation workflow."""
    parser = argparse.ArgumentParser(
        description="Clinical Validation Analysis for PPG Glucose Estimation"
    )
    parser.add_argument(
        "--data-file", 
        type=str, 
        help="Path to CSV file with reference_glucose and predicted_glucose columns"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clinical_validation_results",
        help="Output directory for validation results"
    )
    
    args = parser.parse_args()
    
    print("PPG GLUCOSE ESTIMATION - CLINICAL VALIDATION ANALYSIS")
    print("=" * 60)
    print("Medical Device Safety Standards & FDA Compliance Assessment")
    print("=" * 60)
    
    # Load data
    reference, predicted = load_prediction_data(args.data_file)
    
    if reference is None or predicted is None:
        print("Error: Could not load glucose data")
        return 1
    
    # Perform clinical validation
    try:
        report = perform_clinical_validation(reference, predicted, args.output_dir)
        
        # Final recommendation
        print(f"\nFINAL CLINICAL RECOMMENDATION:")
        if report['safety_assessment']['deployment_ready']:
            print("✅ APPROVED FOR CLINICAL DEPLOYMENT")
            print("This system meets medical device safety standards.")
        else:
            print("❌ NOT APPROVED FOR CLINICAL DEPLOYMENT")
            print("Critical safety requirements not met. Address failures before deployment.")
        
        return 0 if report['safety_assessment']['deployment_ready'] else 1
        
    except Exception as e:
        print(f"Error during clinical validation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)