# 🚀 How to Run the PPG Glucose Estimation Pipeline

## Quick Start

### 📌 Main Entry Points

There are **THREE** main entry files you can use:

1. **`main.py`** - User-friendly entry point with options (RECOMMENDED)
2. **`training_evaluation_pipeline.py`** - Complete pipeline script
3. **`run_complete_pipeline.py`** - Alternative complete pipeline

## 🎯 Option 1: Using main.py (RECOMMENDED)

This is the simplest and most flexible way to run the pipeline.

### Run Complete Pipeline (Training + Evaluation + Visualization)
```bash
python main.py
```
This runs everything: data preprocessing, model training, evaluation, and visualization generation.
- **Time**: ~15-20 minutes
- **Output**: Models, visualizations, reports

### Run Specific Modes

#### Training Only
```bash
python main.py --mode train
```
- Trains the model with 10-fold cross-validation
- Saves checkpoints to `cv_checkpoints/`

#### Evaluation Only
```bash
python main.py --mode evaluate
```
- Evaluates existing trained models
- Generates performance metrics

#### Visualization Only
```bash
python main.py --mode visualize
```
- Creates all visualization plots
- Saves to `visualizations/`

#### System Information
```bash
python main.py --info
```
- Shows Python, PyTorch, and package versions
- Checks CUDA availability

## 🎯 Option 2: Using training_evaluation_pipeline.py

This is the comprehensive pipeline that does everything in sequence:

```bash
python training_evaluation_pipeline.py
```

**What it does:**
1. ✅ Loads PPG dataset from `data/PPG_Dataset/`
2. ✅ Preprocesses signals (filtering, normalization, windowing)
3. ✅ Performs 10-fold cross-validation training
4. ✅ Evaluates on test set
5. ✅ Generates Clarke Error Grid
6. ✅ Creates performance visualizations
7. ✅ Saves results and models

**Expected Output:**
```
================================================================================
                    PPG GLUCOSE ESTIMATION PIPELINE
================================================================================
Step 1: Loading Data
Step 2: Preprocessing
Step 3: Creating Data Windows
Step 4: Splitting Data
Step 5: Model Architecture
Step 6: Training (10-Fold CV)
Step 7: Creating Visualizations
Step 8: Comparative Analysis
Step 9: Saving Training Logs
================================================================================
                    PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
```

## 🎯 Option 3: Using run_complete_pipeline.py

Alternative complete pipeline implementation:

```bash
python run_complete_pipeline.py
```

Similar to Option 2 but with slightly different implementation details.

## 📊 Other Useful Scripts

### Generate Fold Analysis Plots
```bash
python plot_fold_results.py
```
Creates detailed fold-by-fold performance visualizations.

### Clinical Validation
```bash
python clinical_validation_example.py
```
Runs clinical validation and generates Clarke Error Grid.

### Training Analysis
```bash
python training_analysis_complete.py
```
Analyzes training results and creates summary reports.

### Data Analysis
```bash
python data_analysis_simplified.py
```
Explores the dataset and creates data distribution plots.

## 📁 Expected Outputs

After running the complete pipeline, you should have:

```
ppg-glucose-estimation/
├── cv_checkpoints/          # Trained models
│   ├── fold_1/
│   │   └── best_model_epoch_*.pth
│   ├── fold_2/
│   └── ...fold_10/
├── visualizations/          # 28 visualization files
│   ├── fold_performance_comparison.png
│   ├── normalization_pipeline.png
│   ├── clarke_error_grid.png
│   └── ...
├── logs/                    # Training logs
│   └── pipeline_*.log
├── training_logs.json       # Detailed training metrics
├── TRAINING_RESULTS_SUMMARY.md
└── FINAL_TRAINING_RESULTS.md
```

## ⚙️ Requirements

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Key packages:
- torch >= 1.9.0
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- tensorboard

## 🎯 Performance Targets

The pipeline aims to achieve:
- **MAE**: < 3 mg/dL (Paper target: 2.96)
- **RMSE**: < 4 mg/dL (Paper target: 3.94)
- **R²**: > 0.97
- **Clarke Zone A**: > 95%

**Current Best Results (Fold 7):**
- MAE: 7.44 mg/dL
- RMSE: 9.05 mg/dL
- R²: 0.937
- MAPE: 6.27%

## 🔧 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the project root directory
2. **Memory Error**: Reduce batch size in training config
3. **CUDA Error**: Set device to 'cpu' if GPU not available
4. **Missing Data**: Ensure `data/PPG_Dataset/` exists with .mat files

### Debug Mode

Run with verbose output:
```bash
python main.py --verbose
```

Check logs:
```bash
tail -f logs/pipeline_*.log
```

## 📝 Notes

- **First Run**: Will take longer due to data loading and preprocessing
- **Subsequent Runs**: Faster as preprocessed data is cached
- **GPU**: Training is faster with CUDA-enabled GPU
- **Results**: May vary slightly between runs due to random initialization

## 🎓 Understanding the Pipeline

The pipeline implements the paper:
> "Non-Invasive Glucose Level Monitoring from PPG using a Hybrid CNN-GRU Deep Learning Network"

**Architecture:**
- Dual CNN branches (small and large kernels)
- Bidirectional GRU for temporal modeling
- Feature fusion and dense layers
- Output: Single glucose value prediction

**Data Flow:**
1. Raw PPG (2175 Hz) → Filtered (0.5-8 Hz) → Normalized → Downsampled (30 Hz)
2. Windowed into 10-second segments (300 samples)
3. Fed through hybrid CNN-GRU model
4. Outputs glucose prediction in mg/dL

---

**For more details, see:**
- `COMPLETE_GUIDE.md` - Comprehensive documentation
- `FUNCTION_REFERENCE.md` - API reference
- `FINAL_TRAINING_RESULTS.md` - Detailed results analysis