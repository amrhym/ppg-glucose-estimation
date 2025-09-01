#!/usr/bin/env python3
"""
Main Entry Point for PPG Glucose Estimation Pipeline
=====================================================
This is the main entry file that runs the complete pipeline including:
1. Data loading and preprocessing
2. Model training with 10-fold cross-validation
3. Evaluation and visualization
4. Clinical validation

Usage:
    python main.py                    # Run complete pipeline
    python main.py --mode train        # Train only
    python main.py --mode evaluate     # Evaluate existing models
    python main.py --mode visualize    # Generate visualizations only

Author: PPG Glucose Estimation Team
Date: 2025-09-01
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_complete_pipeline():
    """
    Run the complete training and evaluation pipeline.
    This is the main function that orchestrates everything.
    """
    logger = setup_logging()
    
    print("\n" + "="*80)
    print("PPG GLUCOSE ESTIMATION - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis pipeline implements the paper:")
    print('"Non-Invasive Glucose Level Monitoring from PPG using')
    print('a Hybrid CNN-GRU Deep Learning Network"\n')
    
    # Import the main pipeline module
    try:
        from training_evaluation_pipeline import main as run_pipeline
        logger.info("Starting complete pipeline execution...")
        
        # Run the pipeline
        run_pipeline()
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import pipeline module: {e}")
        print("\nError: Could not import training_evaluation_pipeline.py")
        print("Make sure all required files are present.")
        return 1
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\nError during pipeline execution: {e}")
        return 1

def run_training_only():
    """Run only the training phase."""
    logger = logging.getLogger(__name__)
    logger.info("Running training phase only...")
    
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    try:
        # Import necessary modules
        import torch
        import numpy as np
        from pathlib import Path
        
        # Add imports from src
        from models.hybrid_model import HybridCNNGRU, ModelConfig
        from data_preprocessing import PPGDataPreprocessor
        from torch.utils.data import DataLoader, TensorDataset
        
        print("\n1. Loading and preprocessing data...")
        preprocessor = PPGDataPreprocessor()
        X_train, X_val, y_train, y_val = preprocessor.prepare_data()
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        print("\n2. Initializing model...")
        config = ModelConfig()
        model = HybridCNNGRU(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        print("\n3. Starting training...")
        print("   This will run 10-fold cross-validation")
        print("   Expected time: ~15-20 minutes")
        
        # Run training via the pipeline
        from training_evaluation_pipeline import main
        main()
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1
    
    return 0

def run_evaluation_only():
    """Run evaluation on existing models."""
    logger = logging.getLogger(__name__)
    logger.info("Running evaluation phase only...")
    
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    try:
        # Check if models exist
        checkpoint_dir = Path('cv_checkpoints')
        if not checkpoint_dir.exists():
            print("❌ No trained models found. Please run training first.")
            return 1
        
        # Count available models
        model_files = list(checkpoint_dir.glob('*/best_model_*.pth'))
        print(f"\nFound {len(model_files)} trained models")
        
        # Run evaluation
        print("\n1. Loading models and generating predictions...")
        
        # Import evaluation module
        from final_evaluation import evaluate_all_folds
        
        # Run evaluation
        results = evaluate_all_folds()
        
        print("\n2. Results Summary:")
        print(f"   Average MAE: {results.get('avg_mae', 'N/A'):.2f} mg/dL")
        print(f"   Average R²: {results.get('avg_r2', 'N/A'):.4f}")
        print(f"   Best Fold MAE: {results.get('best_mae', 'N/A'):.2f} mg/dL")
        
        print("\n✅ Evaluation completed!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        return 1
    
    return 0

def run_visualization_only():
    """Generate all visualizations."""
    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations...")
    
    print("\n" + "="*60)
    print("VISUALIZATION MODE")
    print("="*60)
    
    try:
        print("\n1. Generating fold performance plots...")
        from plot_fold_results import main as plot_folds
        plot_folds()
        
        print("\n2. Generating training analysis plots...")
        from training_analysis_complete import main as plot_training
        plot_training()
        
        print("\n3. Generating clinical validation plots...")
        from clinical_validation_example import main as plot_clinical
        plot_clinical()
        
        print("\n✅ All visualizations generated!")
        print(f"   Check the 'visualizations/' folder for outputs")
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        print(f"\n❌ Visualization failed: {e}")
        return 1
    
    return 0

def print_system_info():
    """Print system and environment information."""
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib
    import scipy
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='PPG Glucose Estimation Pipeline - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --mode train        # Train models only
  python main.py --mode evaluate     # Evaluate existing models
  python main.py --mode visualize    # Generate visualizations
  python main.py --info             # Show system information
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'train', 'evaluate', 'visualize'],
        default='full',
        help='Pipeline mode to run (default: full)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Show system info if requested
    if args.info:
        print_system_info()
        return 0
    
    # Print header
    print("\n" + "="*80)
    print(" PPG GLUCOSE ESTIMATION SYSTEM ".center(80, "="))
    print("="*80)
    print("\nNon-Invasive Blood Glucose Monitoring using")
    print("Photoplethysmography (PPG) Signals")
    print("\nArchitecture: Hybrid CNN-GRU Deep Learning Network")
    print("Dataset: 67 PPG recordings from 23 subjects")
    print("Target: MAE < 3 mg/dL, R² > 0.97")
    print("="*80)
    
    # Run selected mode
    exit_code = 0
    
    try:
        if args.mode == 'full':
            print("\n🚀 Running COMPLETE PIPELINE...")
            print("This includes: Data preprocessing → Training → Evaluation → Visualization")
            print("Expected time: 15-20 minutes\n")
            exit_code = run_complete_pipeline()
            
        elif args.mode == 'train':
            print("\n🏋️ Running TRAINING ONLY...")
            exit_code = run_training_only()
            
        elif args.mode == 'evaluate':
            print("\n📊 Running EVALUATION ONLY...")
            exit_code = run_evaluation_only()
            
        elif args.mode == 'visualize':
            print("\n🎨 Running VISUALIZATION ONLY...")
            exit_code = run_visualization_only()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit_code = 1
    
    # Print footer
    if exit_code == 0:
        print("\n" + "="*80)
        print(" ✅ PIPELINE COMPLETED SUCCESSFULLY ".center(80, "="))
        print("="*80)
        print("\nOutputs:")
        print("  • Models: cv_checkpoints/")
        print("  • Visualizations: visualizations/")
        print("  • Logs: logs/")
        print("  • Reports: *.txt, *.md files")
    else:
        print("\n" + "="*80)
        print(" ❌ PIPELINE FAILED ".center(80, "="))
        print("="*80)
        print(f"Exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())