# 🧪 Neural Network for Theoretical Best Estimate (TBE) Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-success)
![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-important)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-blue)

## 📌 Script Summary: TBE Neural Network Predictor

### 🔬 **Purpose**
A PyTorch-based neural network that predicts TBE/AVTZ values for molecular excited states using quantum chemistry method results and molecular metadata.

### 🧠 **Core Features**
- **Advanced Architecture**: 3-layer MLP (256-128-64) with feature extraction + 2-layer regressor (32-1)
- **Key Technologies**:
  - PyTorch for deep learning
  - Scikit-learn for preprocessing
  - Rich for beautiful console output
  - MC Dropout for uncertainty estimation
- **Smart Training**:
  - Early stopping
  - Learning rate scheduling
  - Robust Huber loss
- **Visual Diagnostics**:
  - Scatter plots
  - Residual analysis
  - Q-Q plots

### ⚙️ **Workflow**
1. **Data Processing**:
   - Handles JSON input files
   - Automated cleaning of molecular data
   - One-hot encoding + standardization
2. **Training**:
   - Batch processing
   - Validation monitoring
   - Automatic model saving
3. **Prediction**:
   - Single/batch prediction modes
   - Optional uncertainty quantification
   - Rich-formatted output tables

### 📊 **Performance Metrics**
- MAE, RMSE, R² scores
- Explained variance
- Visual error analysis
  
## 🚀 Quick Start: Basic Usage

### Training:

```bash
python quest_nn.py --train --data-dir data/training/ --model model.pt
```

### Prediction:

```bash
python quest_nn.py --predict data/samples/ --model model.pt
```

## 🛠️ Advanced Options

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Maximum training epochs | 100 |
| `--batch-size` | Training batch size | 32 |
| `--lr` | Learning rate | 1e-3 |
| `--weight-decay` | L2 regularization | 1e-4 |

### Prediction Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-samples` | MC Dropout iterations | 100 |
| `--deterministic` | Disable MC Dropout | False |
| `--output-dir` | Prediction output directory | "predictions" |

## 📈 Sample Output

```bash
🔮 Processing file: ../../data/json/CHROM/Heptazine.json
                           Molecule: Heptazine                           
┏━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ State   ┃ Spin ┃ Actual TBE ┃ Predicted TBE ┃ ± Uncertainty ┃ Δ Error ┃
┡━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ ^1A_1"  │  1   │      3.999 │         4.108 │       ± 0.208 │  +0.109 │
│ ^1A_2'  │  1   │      2.717 │         2.965 │       ± 0.277 │  +0.248 │
│ ^1E"    │  1   │      4.108 │         4.146 │       ± 0.202 │  +0.038 │
│ ^1E'    │  1   │      4.478 │         4.498 │       ± 0.214 │  +0.020 │
│ ^3A_1"  │  3   │      3.992 │         4.071 │       ± 0.222 │  +0.079 │
│ ^3A_2'  │  3   │      2.936 │         3.297 │       ± 0.301 │  +0.361 │
│ ^3E"    │  3   │      4.080 │         4.126 │       ± 0.240 │  +0.046 │
│ ^3E'    │  3   │      3.649 │         3.876 │       ± 0.226 │  +0.227 │
│ --      │  --  │         -- │            -- │            -- │      -- │
│ Summary │      │ MAE: 0.141 │   RMSE: 0.182 │    MSE: 0.033 │         │
└─────────┴──────┴────────────┴───────────────┴───────────────┴─────────┘
✅ Saved predictions to: predictions/Heptazine_predictions.csv
```

