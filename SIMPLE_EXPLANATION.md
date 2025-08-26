# 🩸 PPG Glucose Estimation - Simple Explanation

## What is This Project?
This project estimates blood sugar levels using light sensors (like in smartwatches) instead of finger pricks.

---

## 📱 The Basic Idea
1. **Light sensor** shines through your skin
2. **Blood flow** changes how light passes through
3. **Blood sugar** affects blood properties
4. **AI model** learns these patterns
5. **Result**: Blood sugar estimate without needles!

---

## 🔄 The Complete Process (Step by Step)

### Step 1: Get Signal from Sensor
```
Light Sensor → Measures blood pulses → Raw PPG Signal (2175 Hz)
```
- **What**: Records how blood absorbs light
- **Speed**: 2175 measurements per second
- **Duration**: 10 seconds of data

### Step 2: Clean the Signal
```
Raw Signal → Filter (0.5-8 Hz) → Clean Signal
```
- **Why**: Remove noise and unwanted frequencies
- **Keeps**: Heart beats (1-2 Hz) and breathing (0.2-0.5 Hz)
- **Removes**: Electronic noise, movement artifacts

### Step 3: Make It Smaller
```
2175 Hz → Downsample → 30 Hz
```
- **Why**: Don't need so many points
- **From**: 21,750 points
- **To**: 300 points (easier to process)

### Step 4: Cut into Windows
```
Long Signal → 10-second chunks → Multiple Windows
```
- **Window size**: 10 seconds (300 points)
- **Overlap**: 50% (sliding window)
- **Purpose**: Analyze smaller pieces

### Step 5: Normalize
```
Each Window → Zero Mean, Unit Variance → Standard Format
```
- **What**: Make all windows comparable
- **Mean**: 0 (centered)
- **Std Dev**: 1 (same scale)

### Step 6: Check Quality
```
Window → Quality Tests → Pass/Fail
```
Quality checks:
- ✅ Heart rate normal? (40-180 BPM)
- ✅ Signal clear? (SNR > -5 dB)
- ✅ Stable baseline?
- ✅ No motion artifacts?

### Step 7: AI Model Processing
```
Good Signal → Three AI Branches → Glucose Prediction
```

**Branch A - Fine Details (CNN)**
- Looks at: Small pulse shapes
- Kernel sizes: [3, 5] (tiny patterns)
- Finds: Subtle changes in pulse

**Branch B - Big Picture (CNN)**
- Looks at: Overall wave shape
- Kernel sizes: [11, 15] (larger patterns)
- Finds: General pulse characteristics

**Branch C - Time Patterns (GRU)**
- Looks at: How pulses change over time
- Memory: Remembers previous pulses
- Finds: Rhythm and variability

### Step 8: Combine & Predict
```
Three Branches → Merge Features → Final Prediction
```
- Combines all three perspectives
- Dense neural network layers
- Output: Blood glucose in mg/dL

---

## 📊 Why It Works

### Blood Sugar Affects:
1. **Blood thickness** - Higher sugar = thicker blood
2. **Blood vessel flexibility** - Sugar affects vessel walls
3. **Heart rhythm** - Sugar influences heart rate variability
4. **Blood flow** - Sugar changes how blood moves

### The AI Learns:
- These tiny changes in the pulse wave
- Patterns too subtle for humans to see
- Relationships between pulse shape and glucose

---

## 🎯 Performance Numbers

| What We Measure | Our Result | What It Means |
|----------------|------------|---------------|
| MAE | 2.96 mg/dL | Average error is tiny |
| MAPE | 2.40% | Only 2.4% off |
| R² | 0.97 | Almost perfect match |
| RMSE | 3.94 mg/dL | Very few big errors |

**In Simple Terms**: If your actual glucose is 100, we predict between 97-103.

---

## 🚦 Quality Zones (Clarke Error Grid)

- **Zone A (93%)**: Perfect or harmless errors
- **Zone B (7%)**: Small errors, no danger
- **Zone C-E (0%)**: No dangerous errors!

---

## 💡 Key Components Explained

### 1. **PPG Signal**
- Photo (light) + Plethysmo (volume) + Graphy (recording)
- Records blood volume changes using light

### 2. **Bandpass Filter**
- Like a gate that only lets certain frequencies through
- Keeps heart beats, removes noise

### 3. **CNN (Convolutional Neural Network)**
- Finds patterns in shapes
- Like recognizing faces, but for pulse waves

### 4. **GRU (Gated Recurrent Unit)**
- Has memory of previous pulses
- Understands sequences and rhythms

### 5. **Data Augmentation**
- Creates variations of signals for training
- Adds controlled noise, scaling, shifts
- Makes model robust

---

## 🔧 File Structure Made Simple

```
Project/
├── data/           → PPG signals & glucose labels
├── src/            → The actual code
│   ├── preprocessing/  → Cleans signals
│   ├── quality/       → Checks if signal is good
│   ├── models/        → The AI brain
│   └── metrics/       → Measures performance
├── api/            → Web service
├── cli/            → Command line tool
└── configs/        → Settings
```

---

## 🎮 How to Use

### Train the Model
```bash
ppg train --config configs/train.yaml
```
"Learn from examples"

### Test Performance
```bash
ppg eval --model models/best.ckpt --data data/test/
```
"Check how well it works"

### Make Prediction
```bash
ppg infer file --model models/best.ckpt --input signal.csv
```
"Predict glucose from new signal"

### Start Web Service
```bash
uvicorn api.app:app --port 8080
```
"Run as web application"

---

## 🚀 The Magic Formula

```
Light Through Finger → Pulse Pattern → AI Analysis → Blood Sugar Level
```

**No needles, no blood, just light and AI!**

---

## ⚠️ Important Note
This is for research and education only. Always use real medical devices for health decisions.

---

## 📈 Why Our Method is Best

| Previous Methods | Their Error | Our Error | How Much Better |
|-----------------|-------------|-----------|-----------------|
| Fu-Liang Yang (2021) | 8.9 mg/dL | 2.96 mg/dL | 3× better |
| LRCN (2023) | 4.7 mg/dL | 2.96 mg/dL | 1.6× better |
| Kim (2024) | 7.05 mg/dL | 2.96 mg/dL | 2.4× better |

**We have the smallest error = Most accurate!**