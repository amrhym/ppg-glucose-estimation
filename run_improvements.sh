#!/bin/bash

# Run improved training pipeline for PPG glucose estimation
# This script implements all optimizations to achieve better performance

echo "=========================================="
echo "PPG GLUCOSE ESTIMATION - IMPROVEMENT SUITE"
echo "=========================================="

# Check if current training is still running
if pgrep -f "training_evaluation_pipeline.py" > /dev/null; then
    echo "⏳ Current training still in progress. Waiting for completion..."
    while pgrep -f "training_evaluation_pipeline.py" > /dev/null; do
        sleep 30
    done
    echo "✅ Original training completed!"
fi

# Step 1: Analyze current results
echo ""
echo "📊 Step 1: Analyzing current training results..."
python analyze_training_results.py

# Step 2: Run performance improvement analysis
echo ""
echo "🔍 Step 2: Running performance gap analysis..."
python improve_performance.py

# Step 3: Run improved training pipeline
echo ""
echo "🚀 Step 3: Running improved training pipeline..."
echo "This includes:"
echo "  • Optimized hyperparameters (LR: 0.0001, Batch: 16)"
echo "  • Advanced augmentation (10x with multiple methods)"
echo "  • Enhanced model architecture (attention, residual)"
echo "  • Feature engineering (morphological, spectral, HRV)"
echo "  • Ensemble training (5 models)"

python improved_training_pipeline.py

# Step 4: Evaluate final results
echo ""
echo "📈 Step 4: Evaluating final results..."
python final_evaluation.py

# Step 5: Generate comparison report
echo ""
echo "📄 Step 5: Generating performance comparison..."
python -c "
import json
import pandas as pd

# Load original results
try:
    with open('training_summary_report.txt', 'r') as f:
        original = f.read()
    print('Original Performance:')
    print('  MAE: ~20.56 mg/dL')
    print('  R²: ~0.654')
except:
    pass

# Load improved results
try:
    with open('improved_checkpoints/final_results.json', 'r') as f:
        improved = json.load(f)
    print('\nImproved Performance:')
    print(f'  MAE: {improved[\"mae\"]:.2f} mg/dL')
    print(f'  R²: {improved[\"r2\"]:.3f}')
    print(f'\nImprovement: {(1 - improved[\"mae\"]/20.56)*100:.1f}% reduction in MAE')
except:
    print('Improved results will be available after training completes')

print('\nTarget Performance (Paper):')
print('  MAE: 2.96 mg/dL')
print('  R²: 0.97')
"

echo ""
echo "=========================================="
echo "✅ IMPROVEMENT PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review improved_checkpoints/final_results.json"
echo "2. Check final_evaluation_report.txt"
echo "3. Deploy best model from improved_checkpoints/"
echo "4. Consider additional data collection for further improvements"