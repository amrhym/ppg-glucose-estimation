# 📚 Complete Function Reference - PPG Glucose Estimation System

## Table of Contents
1. [Preprocessing Functions](#preprocessing-functions)
2. [Quality Assessment Functions](#quality-assessment-functions)
3. [Model Functions](#model-functions)
4. [Data Loading Functions](#data-loading-functions)
5. [Evaluation Functions](#evaluation-functions)
6. [Utility Functions](#utility-functions)
7. [API Functions](#api-functions)

---

## 🔄 Preprocessing Functions

### `bandpass_filter(signal, low_freq, high_freq, fs, order=4)`
**Purpose**: Applies Butterworth bandpass filter to isolate physiological frequencies.

**Mathematical Foundation**:
```
H(s) = (s^2) / (s^2 + s(ω₀/Q) + ω₀^2)
where ω₀ = 2πf_center, Q = f_center/bandwidth
```

**Parameters**:
- `signal` (np.ndarray): Input PPG signal
- `low_freq` (float): Low cutoff frequency in Hz (typically 0.5)
- `high_freq` (float): High cutoff frequency in Hz (typically 8.0)
- `fs` (float): Sampling frequency in Hz
- `order` (int): Filter order (default=4)

**Returns**:
- `filtered` (np.ndarray): Bandpass filtered signal

**Example**:
```python
filtered = bandpass_filter(ppg_signal, 0.5, 8.0, fs=2175, order=4)
# Removes baseline wander (<0.5Hz) and high-frequency noise (>8Hz)
```

---

### `downsample_signal(signal, orig_fs, target_fs, method='decimate')`
**Purpose**: Reduces sampling rate while preventing aliasing.

**Methods**:
- `'decimate'`: Anti-aliasing filter + decimation
- `'resample'`: FFT-based resampling
- `'interp'`: Interpolation-based resampling

**Parameters**:
- `signal` (np.ndarray): Input signal
- `orig_fs` (int): Original sampling rate (Hz)
- `target_fs` (int): Target sampling rate (Hz)
- `method` (str): Downsampling method

**Returns**:
- `downsampled` (np.ndarray): Downsampled signal

**Example**:
```python
# Downsample from 2175Hz to 30Hz (72.5x reduction)
downsampled = downsample_signal(signal, 2175, 30, method='decimate')
```

---

### `create_sliding_windows(signal, window_size, overlap, fs)`
**Purpose**: Segments continuous signal into overlapping windows for analysis.

**Parameters**:
- `signal` (np.ndarray): Input signal
- `window_size` (float): Window duration in seconds
- `overlap` (float): Overlap fraction (0-1)
- `fs` (float): Sampling frequency

**Returns**:
- `windows` (list): List of signal windows

**Algorithm**:
```python
window_samples = int(window_size * fs)
hop_samples = int(window_samples * (1 - overlap))
windows = []
for i in range(0, len(signal) - window_samples, hop_samples):
    windows.append(signal[i:i + window_samples])
```

**Example**:
```python
# Create 10-second windows with 50% overlap
windows = create_sliding_windows(signal, 10, 0.5, fs=30)
```

---

### `normalize_signal(signal, method='zscore')`
**Purpose**: Normalizes signal to standard range for neural network input.

**Methods**:
- `'zscore'`: (x - μ) / σ
- `'minmax'`: (x - min) / (max - min)
- `'robust'`: (x - median) / IQR

**Parameters**:
- `signal` (np.ndarray): Input signal
- `method` (str): Normalization method

**Returns**:
- `normalized` (np.ndarray): Normalized signal

**Example**:
```python
normalized = normalize_signal(ppg_window, method='zscore')
# Output has mean=0, std=1
```

---

### `augment_signal(signal, noise_level=0.05, amplitude_scale=0.1, baseline_wander=0.02)`
**Purpose**: Generates synthetic variations for training data augmentation.

**Augmentation Types**:
1. **Gaussian Noise**: Simulates sensor noise
2. **Amplitude Scaling**: Simulates signal strength variations
3. **Baseline Wander**: Simulates low-frequency drift
4. **Time Jitter**: Simulates timing variations

**Parameters**:
- `signal` (np.ndarray): Input signal
- `noise_level` (float): Gaussian noise std as fraction of signal std
- `amplitude_scale` (float): Amplitude variation range
- `baseline_wander` (float): Baseline drift amplitude

**Returns**:
- `augmented` (np.ndarray): Augmented signal

**Example**:
```python
# Generate 4 augmented versions
for i in range(4):
    aug = augment_signal(original, noise_level=0.05)
    training_data.append(aug)
```

---

## 🔍 Quality Assessment Functions

### `calculate_snr(signal, fs, hr_band=(0.5, 3.0))`
**Purpose**: Calculates signal-to-noise ratio in heart rate frequency band.

**Algorithm**:
```python
# FFT to frequency domain
fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/fs)

# Signal power in HR band
signal_power = np.sum(|fft[hr_band]|²)

# Noise power outside HR band
noise_power = np.sum(|fft[~hr_band]|²)

# SNR in dB
snr_db = 10 * log10(signal_power / noise_power)
```

**Parameters**:
- `signal` (np.ndarray): Input signal
- `fs` (float): Sampling frequency
- `hr_band` (tuple): Heart rate frequency band

**Returns**:
- `snr` (float): Signal-to-noise ratio in dB

**Quality Thresholds**:
- Excellent: > 20 dB
- Good: 10-20 dB
- Acceptable: 0-10 dB
- Poor: < 0 dB

---

### `detect_heart_rate(signal, fs, min_hr=40, max_hr=180)`
**Purpose**: Extracts heart rate from PPG signal using peak detection.

**Algorithm**:
1. Find signal peaks using scipy.signal.find_peaks
2. Calculate inter-beat intervals (IBI)
3. Convert to heart rate: HR = 60 * fs / mean(IBI)
4. Validate against physiological limits

**Parameters**:
- `signal` (np.ndarray): PPG signal
- `fs` (float): Sampling frequency
- `min_hr` (int): Minimum valid HR (BPM)
- `max_hr` (int): Maximum valid HR (BPM)

**Returns**:
- `heart_rate` (float): Heart rate in BPM or None if invalid

**Example**:
```python
hr = detect_heart_rate(ppg_signal, fs=30)
if hr:
    print(f"Heart rate: {hr:.1f} BPM")
else:
    print("Could not detect valid heart rate")
```

---

### `check_motion_artifacts(signal, fs, threshold=0.3)`
**Purpose**: Detects motion artifacts using high-frequency content analysis.

**Parameters**:
- `signal` (np.ndarray): Input signal
- `fs` (float): Sampling frequency
- `threshold` (float): Artifact detection threshold

**Returns**:
- `has_artifacts` (bool): True if artifacts detected
- `artifact_score` (float): Artifact severity (0-1)

---

### `calculate_perfusion_index(signal)`
**Purpose**: Calculates perfusion index from PPG signal.

**Formula**:
```
PI = (AC_component / DC_component) × 100%
where:
- AC = pulsatile component (peak-to-peak)
- DC = non-pulsatile component (mean)
```

**Parameters**:
- `signal` (np.ndarray): PPG signal

**Returns**:
- `pi` (float): Perfusion index percentage

**Clinical Ranges**:
- Normal: 0.5-20%
- Low perfusion: < 0.5%
- High perfusion: > 20%

---

## 🧠 Model Functions

### `create_hybrid_model(input_size=300, cnn_channels=[32, 64], gru_hidden=128, gru_layers=2)`
**Purpose**: Instantiates the hybrid CNN-GRU architecture.

**Architecture Details**:
```
Total Parameters: 733,953
Model Size: 2.8 MB
Inference Time: ~10ms
```

**Parameters**:
- `input_size` (int): Input sequence length
- `cnn_channels` (list): CNN channel progression
- `gru_hidden` (int): GRU hidden units
- `gru_layers` (int): Number of GRU layers

**Returns**:
- `model` (HybridCNNGRU): Model instance

**Example**:
```python
model = create_hybrid_model(
    input_size=300,  # 10s at 30Hz
    cnn_channels=[32, 64, 128],
    gru_hidden=128,
    gru_layers=2
)
```

---

### `train_epoch(model, dataloader, optimizer, criterion)`
**Purpose**: Executes one training epoch.

**Parameters**:
- `model` (nn.Module): Neural network model
- `dataloader` (DataLoader): Training data loader
- `optimizer` (Optimizer): Optimization algorithm
- `criterion` (Loss): Loss function

**Returns**:
- `avg_loss` (float): Average epoch loss
- `metrics` (dict): Training metrics

**Training Loop**:
```python
for batch in dataloader:
    optimizer.zero_grad()
    predictions = model(batch.inputs)
    loss = criterion(predictions, batch.targets)
    loss.backward()
    optimizer.step()
```

---

### `evaluate_model(model, dataloader, criterion)`
**Purpose**: Evaluates model on validation/test set.

**Metrics Calculated**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- R²: Coefficient of Determination
- MAPE: Mean Absolute Percentage Error

**Parameters**:
- `model` (nn.Module): Trained model
- `dataloader` (DataLoader): Evaluation data
- `criterion` (Loss): Loss function

**Returns**:
- `metrics` (dict): Evaluation metrics

---

### `cross_validate(model_fn, dataset, n_folds=10)`
**Purpose**: Performs k-fold cross-validation for robust evaluation.

**Parameters**:
- `model_fn` (callable): Function to create model
- `dataset` (Dataset): Complete dataset
- `n_folds` (int): Number of CV folds

**Returns**:
- `cv_results` (dict): Per-fold and average metrics

**Example**:
```python
results = cross_validate(
    model_fn=lambda: create_hybrid_model(),
    dataset=ppg_dataset,
    n_folds=10
)
print(f"CV MAE: {results['mae_mean']:.2f} ± {results['mae_std']:.2f}")
```

---

## 📊 Data Loading Functions

### `load_ppg_signal(filepath, format='csv')`
**Purpose**: Loads PPG signal from various file formats.

**Supported Formats**:
- CSV: Single column or multi-column
- MAT: MATLAB format
- NPZ: NumPy compressed
- TXT: Plain text

**Parameters**:
- `filepath` (str): Path to signal file
- `format` (str): File format

**Returns**:
- `signal` (np.ndarray): PPG signal
- `metadata` (dict): Associated metadata

---

### `load_glucose_labels(filepath)`
**Purpose**: Loads glucose reference values and metadata.

**CSV Format**:
```
ID,Gender,Age,Glucose,Height,Weight
001,M,35,110,175,70
```

**Parameters**:
- `filepath` (str): Path to labels file

**Returns**:
- `labels` (dict): Glucose values and metadata

---

### `create_dataset(data_dir, preprocessing_config)`
**Purpose**: Creates PyTorch dataset from PPG files.

**Parameters**:
- `data_dir` (str): Data directory path
- `preprocessing_config` (PreprocessingConfig): Preprocessing settings

**Returns**:
- `dataset` (PPGDataset): PyTorch dataset

**Example**:
```python
dataset = create_dataset(
    data_dir='data/PPG_Dataset',
    preprocessing_config=PreprocessingConfig()
)
print(f"Dataset size: {len(dataset)} samples")
```

---

### `split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, patient_wise=True)`
**Purpose**: Splits dataset ensuring no data leakage.

**Parameters**:
- `dataset` (Dataset): Complete dataset
- `train_ratio` (float): Training set fraction
- `val_ratio` (float): Validation set fraction
- `patient_wise` (bool): Split by patient ID

**Returns**:
- `splits` (tuple): (train_set, val_set, test_set)

---

## 🏥 Evaluation Functions

### `clarke_error_grid_analysis(reference, predictions)`
**Purpose**: Performs Clarke Error Grid analysis for clinical safety assessment.

**Zone Definitions**:
- **Zone A**: Clinically accurate (±20% or ±20 mg/dL)
- **Zone B**: Benign errors (no treatment change)
- **Zone C**: Overcorrection errors
- **Zone D**: Dangerous failure to detect
- **Zone E**: Erroneous treatment

**Parameters**:
- `reference` (np.ndarray): True glucose values
- `predictions` (np.ndarray): Predicted values

**Returns**:
- `zones` (dict): Percentage in each zone
- `zone_labels` (np.ndarray): Zone for each point

**Example**:
```python
zones = clarke_error_grid_analysis(true_glucose, pred_glucose)
print(f"Zone A: {zones['A']:.1f}%")  # Should be >95%
print(f"Zone A+B: {zones['A'] + zones['B']:.1f}%")  # Should be >99%
```

---

### `calculate_clinical_metrics(reference, predictions)`
**Purpose**: Calculates comprehensive clinical performance metrics.

**Metrics Included**:
- Glucose-specific metrics (MAE, RMSE, MAPE)
- Clinical accuracy metrics
- Hypoglycemia detection sensitivity
- Hyperglycemia detection specificity
- Time in range calculations

**Parameters**:
- `reference` (np.ndarray): True values
- `predictions` (np.ndarray): Predicted values

**Returns**:
- `metrics` (dict): All clinical metrics

---

### `assess_hypoglycemia_detection(reference, predictions, threshold=70)`
**Purpose**: Evaluates hypoglycemia detection performance.

**Metrics**:
- Sensitivity (true positive rate)
- Specificity (true negative rate)
- Positive predictive value
- Negative predictive value

**Parameters**:
- `reference` (np.ndarray): True glucose
- `predictions` (np.ndarray): Predicted glucose
- `threshold` (float): Hypoglycemia threshold (mg/dL)

**Returns**:
- `metrics` (dict): Detection performance metrics

---

### `calculate_mard(reference, predictions)`
**Purpose**: Calculates Mean Absolute Relative Difference.

**Formula**:
```
MARD = (1/n) * Σ(|ref - pred| / ref) * 100%
```

**Parameters**:
- `reference` (np.ndarray): Reference values
- `predictions` (np.ndarray): Predicted values

**Returns**:
- `mard` (float): MARD percentage

**Clinical Standards**:
- Excellent: < 5%
- Good: 5-10%
- Acceptable: 10-15%
- Poor: > 15%

---

## 🛠️ Utility Functions

### `generate_synthetic_ppg(duration=10, fs=30, heart_rate=70, glucose=100, snr=20)`
**Purpose**: Generates synthetic PPG signal for testing.

**Parameters**:
- `duration` (float): Signal duration (seconds)
- `fs` (float): Sampling frequency
- `heart_rate` (float): Heart rate (BPM)
- `glucose` (float): Glucose level (mg/dL)
- `snr` (float): Signal-to-noise ratio (dB)

**Returns**:
- `signal` (np.ndarray): Synthetic PPG signal

**Algorithm**:
```python
# Base cardiac component
cardiac = sin(2π * hr_freq * t) + harmonics

# Glucose-dependent modulation
modulation = glucose_transfer_function(glucose)

# Add noise
noise = gaussian_noise(snr)

signal = cardiac * modulation + noise
```

---

### `visualize_preprocessing_stages(signal, fs=2175)`
**Purpose**: Creates visualization of all preprocessing stages.

**Stages Visualized**:
1. Raw signal
2. Filtered signal
3. Downsampled signal
4. Windowed segments
5. Normalized output

**Parameters**:
- `signal` (np.ndarray): Raw PPG signal
- `fs` (float): Original sampling frequency

**Returns**:
- `fig` (matplotlib.figure): Multi-panel figure

---

### `save_model_checkpoint(model, optimizer, epoch, metrics, filepath)`
**Purpose**: Saves complete model checkpoint.

**Checkpoint Contents**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics,
    'timestamp': datetime.now()
}
```

**Parameters**:
- `model` (nn.Module): Model to save
- `optimizer` (Optimizer): Optimizer state
- `epoch` (int): Current epoch
- `metrics` (dict): Performance metrics
- `filepath` (str): Save path

---

### `load_model_checkpoint(filepath, model, optimizer=None)`
**Purpose**: Loads model from checkpoint.

**Parameters**:
- `filepath` (str): Checkpoint path
- `model` (nn.Module): Model instance
- `optimizer` (Optimizer): Optional optimizer

**Returns**:
- `epoch` (int): Checkpoint epoch
- `metrics` (dict): Saved metrics

---

## 🌐 API Functions

### `process_ppg_request(signal, sampling_rate, quality_check=True)`
**Purpose**: Processes incoming PPG signal for API endpoint.

**Parameters**:
- `signal` (list): PPG signal values
- `sampling_rate` (float): Signal sampling rate
- `quality_check` (bool): Perform quality validation

**Returns**:
- `result` (dict): Glucose prediction and confidence

**Response Format**:
```json
{
    "glucose": 105.3,
    "confidence": 0.92,
    "quality_score": 0.85,
    "timestamp": "2024-09-01T12:00:00Z"
}
```

---

### `stream_predictions(websocket, model, buffer_size=300)`
**Purpose**: Handles WebSocket streaming for continuous monitoring.

**Parameters**:
- `websocket` (WebSocket): WebSocket connection
- `model` (nn.Module): Loaded model
- `buffer_size` (int): Signal buffer size

**Streaming Protocol**:
```python
async def stream_predictions(websocket, model):
    buffer = []
    async for message in websocket:
        data = json.loads(message)
        buffer.extend(data['samples'])
        
        if len(buffer) >= 300:  # 10s at 30Hz
            prediction = model.predict(buffer[-300:])
            await websocket.send(json.dumps({
                'glucose': prediction,
                'timestamp': time.time()
            }))
```

---

### `validate_api_input(data)`
**Purpose**: Validates incoming API request data.

**Validation Checks**:
1. Signal length (minimum 300 samples)
2. Sampling rate (supported: 30, 100, 1000, 2175 Hz)
3. Signal values (numeric, finite)
4. Optional metadata validation

**Parameters**:
- `data` (dict): Request data

**Returns**:
- `is_valid` (bool): Validation result
- `errors` (list): Validation errors if any

---

## 📈 Performance Optimization Functions

### `batch_process_signals(signals, model, batch_size=32)`
**Purpose**: Efficiently processes multiple signals in batches.

**Parameters**:
- `signals` (list): List of PPG signals
- `model` (nn.Module): Trained model
- `batch_size` (int): Batch size

**Returns**:
- `predictions` (np.ndarray): Glucose predictions

**Optimization**:
```python
# Process in batches for memory efficiency
predictions = []
for i in range(0, len(signals), batch_size):
    batch = signals[i:i+batch_size]
    with torch.no_grad():
        pred = model(batch)
    predictions.extend(pred)
```

---

### `optimize_model_for_deployment(model, quantize=True, prune=False)`
**Purpose**: Optimizes model for production deployment.

**Optimizations**:
1. **Quantization**: Reduces precision (32-bit → 8-bit)
2. **Pruning**: Removes redundant connections
3. **Fusion**: Combines operations
4. **ONNX Export**: Cross-platform deployment

**Parameters**:
- `model` (nn.Module): Original model
- `quantize` (bool): Apply quantization
- `prune` (bool): Apply pruning

**Returns**:
- `optimized_model`: Optimized model
- `stats` (dict): Optimization statistics

**Example**:
```python
optimized, stats = optimize_model_for_deployment(model)
print(f"Size reduction: {stats['size_reduction']:.1f}%")
print(f"Speed improvement: {stats['speed_gain']:.1f}x")
```

---

## 🔬 Advanced Analysis Functions

### `analyze_glucose_dynamics(predictions, timestamps)`
**Purpose**: Analyzes glucose trends and patterns.

**Metrics**:
- Rate of change
- Time in range (70-180 mg/dL)
- Glycemic variability
- Pattern detection

**Parameters**:
- `predictions` (np.ndarray): Glucose predictions
- `timestamps` (np.ndarray): Time points

**Returns**:
- `dynamics` (dict): Glucose dynamics analysis

---

### `detect_anomalies(signal, model, threshold=3.0)`
**Purpose**: Detects anomalous patterns in PPG signals.

**Parameters**:
- `signal` (np.ndarray): PPG signal
- `model`: Anomaly detection model
- `threshold` (float): Detection threshold (std deviations)

**Returns**:
- `anomalies` (list): Detected anomaly segments
- `scores` (np.ndarray): Anomaly scores

---

## 📝 Summary

This comprehensive function reference covers all major components of the PPG Glucose Estimation System. Each function is designed with clinical requirements in mind, ensuring accurate, reliable, and safe glucose predictions from non-invasive PPG signals.

**Key Design Principles**:
- **Modularity**: Each function performs a specific task
- **Robustness**: Extensive error handling and validation
- **Efficiency**: Optimized for real-time processing
- **Clinical Safety**: Adherence to medical device standards
- **Scalability**: Designed for production deployment

For implementation examples and complete workflows, refer to the main documentation and example scripts provided with the system.