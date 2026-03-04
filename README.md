#  Identifying Exoplanetary Transits
### Deep Learning for Stellar Light Curve Analysis Using NASA Kepler Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KiHOwaAvt4cMpb0AvA8P-Js3s1QePVBn#scrollTo=ifPiQcGzbmJg)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20Kepler-blueviolet)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)
![License](https://img.shields.io/badge/License-CC0%201.0-lightgrey)

---

##  Overview

This project applies a **1D Convolutional Neural Network (CNN)** to automatically detect exoplanets from NASA's Kepler Space Telescope photometric data. By analyzing the brightness (flux) of stars over time — called a **light curve** — the model learns to identify the subtle, periodic dips in starlight caused by a planet passing in front of its host star.

The pipeline covers the complete machine learning lifecycle:

-  Data ingestion from Kaggle
-  Exploratory data analysis & visualization
-  Preprocessing, normalization & reshaping
-  1D-CNN architecture design
-  Class imbalance handling via weighted loss
-  Model training with validation diagnostics
-  Evaluation with confusion matrix, classification report & ROC-AUC

---

##  Scientific Background — The Transit Method

When an exoplanet passes directly between its host star and Earth, it **blocks a small fraction of the starlight**, producing a brief, measurable dip in brightness. This is called a **transit event**.

```
Star brightness ──────────────┐      ┌──────────────
                               \    /
                                \  /        ← Transit dip (planet blocking starlight)
                                 \/
```

These dips are:
- **Periodic** — they repeat every orbital period (the planet's "year")
- **Symmetric** — with a characteristic flat-bottomed U-shape profile
- **Shallow** — typically blocking only 0.01%–1% of stellar flux
- **Consistent** — identical depth and duration across multiple transits

The core challenge is that these signatures are often buried in stellar noise, instrument artifacts, and long-term variability — making automated detection essential at scale.

### Phase Folding

Since transits are periodic, **phase folding** stacks all transit events on top of one another (after identifying the orbital period via a Fourier Transform). Noise cancels out, the signal reinforces itself, and what was invisible in raw data becomes clearly detectable.

---

##  Dataset

| Property | Value |
|---|---|
| **Source** | NASA Kepler Mission via Kaggle |
| **Dataset** | `keplersmachines/kepler-labelled-time-series-data` |
| **License** | CC0 1.0 Public Domain |
| **Training Samples** | 5,087 stellar light curves |
| **Test Samples** | 570 stellar light curves |
| **Features per Star** | 3,197 flux measurements (time steps) |
| **Class 0 — No Planet** | 5,050 train / 565 test |
| **Class 1 — Planet** | 37 train / 5 test |
| **Class Imbalance Ratio** | ~136 : 1 |

Each row in the CSV represents a single star. Columns `FLUX.1` through `FLUX.3197` are sequential brightness measurements taken at a fixed cadence by the Kepler telescope.

###  The Class Imbalance Problem

Only **37 out of 5,087** training stars host confirmed planets — a ratio of ~136:1. A naive model that predicts "No Planet" for every star achieves 99.3% accuracy while being entirely useless. Addressing this imbalance is the most critical challenge in the entire pipeline.

---

## 🔧 Methodology

### 1. Label Encoding
Original labels (1 = No Planet, 2 = Planet) were remapped to binary format (0 / 1) for compatibility with sigmoid output.

### 2. Row-wise Standardization
Each star's 3,197-point flux sequence was independently standardized using `StandardScaler` applied **per row** (not per column). This removes differences in absolute brightness between stars, centering each light curve at zero with unit variance — critical for stable neural network training.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw.T).T  # Transpose for row-wise scaling
```

### 3. Tensor Reshaping for 1D-CNN
Reshaped from `(5087, 3197)` → `(5087, 3197, 1)`, adding the channel dimension required by Conv1D layers.

```python
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
# Output shape: (5087, 3197, 1)
```

### 4. Class-Weighted Loss
To force the model to prioritize rare planet detections, balanced class weights were computed and passed to `model.fit()`:

| Class | Label | Weight |
|---|---|---|
| 0 | No Planet | 0.504 |
| 1 | **Planet** | **68.74** — penalized ~137× more for a miss |

---

##  Model Architecture — 1D Convolutional Neural Network

A **1D-CNN** was chosen because it naturally encodes the inductive bias that transit events are **local, translatable patterns in time** — a sliding filter can detect a dip regardless of where it appears in the sequence. This is fundamentally different from Dense networks, which treat every time step independently.

```
Input: (batch, 3197, 1)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Block 1: Low-level feature extraction       │
│  Conv1D(16 filters, kernel=11) → BN → ReLU  │
│  MaxPooling1D(pool=4, stride=4)              │
│  Output: (batch, 800, 16)                    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Block 2: Mid-level feature extraction       │
│  Conv1D(32 filters, kernel=7)  → BN → ReLU  │
│  MaxPooling1D(pool=4, stride=4)              │
│  Output: (batch, 200, 32)                    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Block 3: High-level abstraction             │
│  Conv1D(64 filters, kernel=5)  → BN → ReLU  │
│  MaxPooling1D(pool=4, stride=4)              │
│  Output: (batch, 50, 64)                     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Classification Head                         │
│  Flatten → Dense(64, ReLU) → Dropout(0.5)   │
│  Dense(1, Sigmoid)                           │
│  Output: planet probability ∈ [0, 1]         │
└─────────────────────────────────────────────┘
```

**Total Trainable Parameters: 215,169**

### Design Decisions

| Choice | Reason |
|---|---|
| Growing filters (16 → 32 → 64) | Early layers detect primitive dips; deeper layers combine them into full transit profiles |
| Shrinking kernels (11 → 7 → 5) | After pooling, smaller kernels cover equivalent temporal spans of the original signal |
| BatchNorm before ReLU | Prevents internal covariate shift, stabilizes and accelerates training |
| Dropout(0.5) | With only 37 positive training examples, heavy regularization is essential to prevent memorization |
| Sigmoid output | Binary classification — outputs the probability that a star hosts a planet |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Cross-Entropy |
| Epochs | 30 |
| Batch Size | 64 |
| Validation Split | 10% |
| Framework | TensorFlow 2.19.0 / Keras |

---

##  Results & Model Performance

### Classification Report — Test Set (570 stars)

```
              precision    recall  f1-score   support

   No Planet       0.99      0.99      0.99       565
      Planet       0.29      0.40      0.33         5

    accuracy                           0.99       570
   macro avg       0.64      0.70      0.66       570
weighted avg       0.99      0.99      0.99       570
```

### Confusion Matrix

```
                   Predicted: No Planet    Predicted: Planet
Actual: No Planet       560  TN              5  FP
Actual: Planet            3  FN              2  TP
```

### Interpreting the Metrics

**Overall accuracy: 99%** — the model correctly classifies 99% of all stars.

However, because the dataset is so imbalanced, accuracy alone is misleading. The more meaningful metrics are per-class:

- **No Planet class** — Near-perfect (99% precision, 99% recall). The model reliably filters out non-planet stars.
- **Planet class** — 40% recall means the model successfully identifies 2 of the 5 confirmed planet stars in the test set. The 29% precision means some false positives are generated.

The low planet-class metrics are expected given only **5 test examples** exist — this makes statistical conclusions fragile. The weighted loss ensures the model at least attempts to find planets rather than ignoring them entirely.

---

##  How to Run

### Option A — Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KiHOwaAvt4cMpb0AvA8P-Js3s1QePVBn#scrollTo=ifPiQcGzbmJg)

You will need a **Kaggle API key** (`kaggle.json`) to download the dataset. Upload it when prompted in the first cell.

### Option B — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/exoplanetary-transits.git
cd exoplanetary-transits

# 2. Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn kaggle

# 3. Configure Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Download the dataset
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data
unzip kepler-labelled-time-series-data.zip

# 5. Launch the notebook
jupyter notebook Identifying_Exoplanetary_Transits.ipynb
```

---

##  Tools & Technologies

| Category | Library | Version |
|---|---|---|
| Deep Learning | TensorFlow / Keras | 2.19.0 |
| Preprocessing | Scikit-learn | Latest |
| Data Handling | Pandas | Latest |
| Numerics | NumPy | Latest |
| Visualization | Matplotlib | Latest |
| Visualization | Seaborn | Latest |
| Platform | Google Colab | — |
| Data Source | Kaggle API | — |

---

##  Future Work

| Enhancement | Expected Impact |
|---|---|
| **Phase Folding via FFT** | Dramatic noise reduction; transit signal becomes clearly visible after stacking |
| **SMOTE / Oversampling** | Synthetically balance training set with realistic planet examples |
| **1D ResNet** | Residual connections allow deeper networks without vanishing gradients |
| **LSTM / Bi-LSTM** | Recurrent layers to capture long-range periodic dependencies |
| **Transformer + Attention** | Attention heads to focus specifically on transit windows |
| **Threshold Optimization** | Lower decision threshold (e.g., 0.3) to boost planet recall |
| **Kepler Catalog Validation** | Cross-reference detections against the confirmed exoplanet archive |
| **TESS Dataset Integration** | Apply pipeline to NASA's newer TESS mission data |

---

##  Key Takeaways

- **Domain-driven preprocessing matters** — row-wise (per-star) normalization was the scientifically correct choice, not column-wise
- **Class imbalance requires deliberate intervention** — weighted loss penalizing missed planets 68× was essential for the model to be useful at all
- **Architecture should match data structure** — 1D-CNNs encode the right inductive bias for local temporal patterns in time-series data
- **Accuracy alone is misleading** — with a 99:1 class ratio, per-class precision and recall tell the real story
- **Regularization is non-negotiable with tiny positive class sizes** — Dropout(0.5) + BatchNorm help generalize from just 37 planet examples

---

##  License

Dataset: [CC0 1.0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) via NASA / Kaggle
