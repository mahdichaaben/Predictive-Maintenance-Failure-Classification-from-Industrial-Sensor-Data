# 🔧 Predictive Maintenance Failure Classification from Industrial Sensor Data

**Prepared by:** Mahdi Chaaben

## 📋 Project Overview

This project implements an intelligent predictive maintenance system for industrial milling machines using the AI4I 2020 Predictive Maintenance Dataset. Instead of traditional multi-class classification approaches, we developed **specialized binary Decision Tree classifiers** for each failure mode, achieving **90.56% accuracy** in correctly identifying specific failure types.

## 🎯 The Problem

Industrial machines fail in different ways, and each failure type has distinct physical causes:
- Dataset contains **10,000 machine operation cycles**
- Only **~3% result in actual failures** (severe class imbalance)
- **5 independent failure modes** with completely different root causes
- Traditional single-model approaches predict "No Failure" for everything → **unacceptable for safety-critical systems**

## 📊 Dataset Description

**AI4I 2020 Predictive Maintenance Dataset** - A synthetic simulation of a real-world CNC milling machine with 10,000 rows and 14 columns.

### Key Features
| Feature | Description |
|---------|-------------|
| **Type** | Quality variant (L/M/H - Low/Medium/High) |
| **Air temperature [K]** | ~300 K ±2 K |
| **Process temperature [K]** | Air temp + 10 K ±1 K |
| **Rotational speed [rpm]** | ~2860 W power calculation + noise |
| **Torque [Nm]** | Normal distribution around 40 Nm, σ=10 |
| **Tool wear [min]** | Cumulative wear over time |

### The 5 Failure Modes

| Mode | Name | Trigger Condition | Count |
|------|------|-------------------|-------|
| **TWF** | Tool Wear Failure | Tool wear hits 200–240 min | 120 |
| **HDF** | Heat Dissipation Failure | (Process − Air < 8.6 K) AND (Speed < 1380 rpm) | 115 |
| **PWF** | Power Failure | Power = Torque × ω not in [3500, 9000] W | 95 |
| **OSF** | Overstrain Failure | Tool wear × Torque > threshold (11k/12k/13k for L/M/H) | 98 |
| **RNF** | Random Failure | 0.1% chance per row (unpredictable) | 5 |

## 🔬 Our Solution: Physics-Driven Expert System

### Why Separate Binary Classifiers?

Traditional multi-class models struggle with:
- Extreme class imbalance (97% "No Failure")
- Different physical causes for each failure
- Rare events get ignored

**Our Approach:** One specialized Decision Tree expert per failure mode, each trained only on the sensors that physically cause that failure.

### Feature Engineering by Failure Mode

| Failure | Relevant Sensors | Root Cause |
|---------|-----------------|------------|
| **TWF** | Tool wear only | Pure wear-based threshold |
| **HDF** | Torque, Speed, Air Temp, Process Temp | Thermal runaway (high heat + low cooling) |
| **PWF** | Torque × Rotational Speed | Unsafe power levels |
| **OSF** | Torque, Tool wear, Speed | Mechanical overstrain |

### Key Correlations Discovered

![Strong Sensor Correlations](images/correlation_heatmap.png)

**Sensor-to-Failure Correlations (Point-Biserial):**
- **TWF:** Tool wear (+0.12) - primary driver
- **HDF:** Torque (+0.14), Air Temp (+0.14), Speed (−0.12)
- **PWF:** Speed (+0.12), Torque (+0.08)
- **OSF:** Torque (+0.18), Tool wear (+0.16)
- **RNF:** All correlations ≈ 0 (truly random, unpredictable)

## 🏗️ Model Architecture

```
Input: Sensor readings
    ↓
┌─────────────────────────────────┐
│  4 Binary Decision Trees        │
│  (max_depth=3, balanced weight) │
├─────────────────────────────────┤
│  • TWF Detector                 │
│  • HDF Detector                 │
│  • PWF Detector                 │
│  • OSF Detector                 │
└─────────────────────────────────┘
    ↓
Probability Fusion Layer
    ↓
P(No_Failure) = 1 - max(P_failures)
    ↓
Winner = argmax(all probabilities)
    ↓
Output: Failure mode + confidence
```

### Prediction Logic

```python
def predict_failure_mode_json(X):
    # 1. Get probabilities from each expert
    probs = {
        "TWF": model_TWF.predict_proba(X)[:, 1],
        "HDF": model_HDF.predict_proba(X)[:, 1],
        "PWF": model_PWF.predict_proba(X)[:, 1],
        "OSF": model_OSF.predict_proba(X)[:, 1]
    }
    
    # 2. Compute No_Failure probability
    max_fail_prob = max(probs.values())
    probs["No_Failure"] = 1 - max_fail_prob
    
    # 3. Select highest confidence prediction
    final_prediction = argmax(probs)
    
    return {
        "probabilities": probs,
        "prediction": final_prediction,
        "confidence": max(probs.values())
    }
```

## 📈 Results

### Individual Model Performance

| Model | Precision | Recall | F1-Score | Key Achievement |
|-------|-----------|--------|----------|-----------------|
| **TWF** | 0.06 | **1.00** | 0.11 | Zero missed failures |
| **HDF** | 0.43 | **0.91** | 0.58 | Caught 31/34 thermal failures |
| **PWF** | 0.49 | **1.00** | 0.66 | Perfect power failure detection |
| **OSF** | 0.21 | **0.90** | 0.34 | Only 3 missed overstrain events |

### Combined System Performance

**Test on ALL 339 Real Failures:**

```
Overall Accuracy: 90.56%
Correctly identified: 307/339 failures with exact failure type
Misclassifications: 32 (wrong type, but still flagged as failure)
Catastrophic misses: 0 (ZERO real failures predicted as "No_Failure")
```

**Confusion Matrix:**

![Confusion Matrix](images/confusion_matrix.png)

### Why This Matters

✅ **Perfect Safety:** No real breakdown ever gets classified as "No Failure"  
✅ **Interpretable:** Each tree has ≤3 levels, fully explainable to engineers  
✅ **Actionable:** Knowing the exact failure type enables targeted maintenance  
✅ **Physics-Aligned:** Models match known mechanical/thermal failure mechanisms  

## 🚀 Installation & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Quick Start

```python
import pandas as pd

# Load dataset
df = pd.read_csv('ai4i2020.csv')

# Train models (see notebook for full implementation)
from predictive_maintenance import train_models, predict_failure_mode_json

models = train_models(df)

# Predict on new data
sample = df[['Air temperature [K]', 'Process temperature [K]',
             'Rotational speed [rpm]', 'Torque [Nm]', 
             'Tool wear [min]']].iloc[[0]]

prediction = predict_failure_mode_json(sample)
print(prediction)
# Output: {
#   "TWF": 0.02, "HDF": 0.15, "PWF": 0.03, "OSF": 0.05,
#   "No_Failure": 0.85,
#   "final_prediction": "No_Failure",
#   "confidence": 0.85
# }
```

## 📁 Project Structure

```
.
├── dataset.csv                                    # AI4I 2020 dataset
├── Predictive Maintenance Classification.ipynb    # Main analysis notebook
├── images/                                        # Visualizations
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── failure_distribution.png
└── README.md                                      # This file
```

## 🔍 Key Insights

### 1. **Failure Mode Independence**
Correlation heatmap shows all failure modes are independent (correlations ≈ 0), confirming each has distinct root causes.

### 2. **Root Cause Analysis**

| Failure | Mechanism | Critical Threshold |
|---------|-----------|-------------------|
| **TWF** | Cumulative wear degradation | Tool wear > 200 min |
| **HDF** | Thermal runaway | Torque > 50 Nm + Speed < 1300 rpm + Temp > 303 K |
| **PWF** | Power envelope violation | Speed > 2400 & Torque < 20, OR Speed < 1400 & Torque > 60 |
| **OSF** | Mechanical stress overload | High torque × worn tool |

### 3. **Why Our Approach Wins**

❌ **Single Multi-Class Model:**
- Learns to predict "No Failure" for 97% accuracy
- Ignores rare but critical failure events
- Black-box decision making

✅ **Our Expert System:**
- Each model focuses on one physical failure mechanism
- Balanced training ensures rare events are learned
- Transparent, interpretable decisions
- **90.56% exact failure type identification**
- **Zero catastrophic misses**

## 🎓 Methodology

1. **Exploratory Data Analysis** - Correlation analysis, failure distribution, sensor behavior
2. **Root Cause Identification** - Physics-based feature selection per failure mode
3. **Individual Model Training** - Specialized Decision Trees with balanced class weights
4. **Probability Fusion** - Smart combination strategy for final prediction
5. **Rigorous Validation** - Tested on ALL 339 real failures from dataset

## 📊 Visualizations

The notebook includes:
- Sensor correlation heatmaps
- Failure mode co-occurrence analysis
- Scatterplot analysis for each failure type
- Decision tree visualizations
- Confusion matrices for each model
- Final system performance metrics

## 🏆 Achievements

- ✅ **90.56%** accuracy on real failure classification
- ✅ **100%** recall on critical failure modes (PWF, TWF)
- ✅ **Zero** catastrophic misses (no real failure classified as safe)
- ✅ Fully interpretable models (max 3-level decision trees)
- ✅ Physics-aligned predictions matching known failure mechanisms

## 📝 Future Improvements

- [ ] Implement ensemble methods for the combination layer
- [ ] Add time-series analysis for early failure prediction
- [ ] Deploy as real-time monitoring API
- [ ] Include RNF prediction using anomaly detection techniques
- [ ] Optimize false positive rates while maintaining 100% recall

## 📚 References

- AI4I 2020 Predictive Maintenance Dataset
- Scikit-learn Decision Tree Documentation
- Industrial Predictive Maintenance Best Practices

## 👤 Author

**Mahdi Chaaben**

---

**Built with 💡 by understanding physics first, then applying machine learning.**
