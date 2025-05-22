# 🧪 Neural Network for Theoretical Best Estimate (TBE) Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-success)
![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-important)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-blue)

## 📌 Script Summary: TBE Neural Network Predictor

### 🔬 **Purpose**
A PyTorch-based neural network that predicts Theoretical Best Estimate (TBE)/AVTZ values for molecular excited states using quantum chemistry method results and molecular metadata.

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

### 🚀 **Usage Scenarios**
- Quantum chemistry research
- Molecular excitation studies
- Method benchmarking
- Prediction with confidence intervals

### 🌟 **Key Advantages**
- Handles incomplete/messy chemical data
- Provides uncertainty estimates
- Production-ready CLI interface
- Comprehensive visualization outputs
  
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
🔮 Processing file: ../../data/json/CHROM/Anthraquinone.json
                         Molecule: Anthraquinone                          
┏━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ State    ┃ Spin ┃ Actual TBE ┃ Predicted TBE ┃ ± Uncertainty ┃ Δ Error ┃
┡━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ ^1A_g    │  1   │      4.104 │         4.128 │       ± 0.230 │  +0.024 │
│ ^1A_u    │  1   │      3.466 │         3.585 │       ± 0.251 │  +0.119 │
│ ^1B_{1g} │  1   │      3.226 │         3.432 │       ± 0.280 │  +0.206 │
│ ^1B_{1u} │  1   │      5.101 │         5.202 │       ± 0.182 │  +0.101 │
│ ^1B_{2u} │  1   │      4.219 │         4.331 │       ± 0.209 │  +0.112 │
│ ^1B_{2u} │  1   │      5.406 │         5.531 │       ± 0.204 │  +0.125 │
│ ^1B_{3g} │  1   │      4.321 │         4.455 │       ± 0.187 │  +0.134 │
│ ^3A_g    │  3   │      3.740 │         4.035 │       ± 0.218 │  +0.295 │
│ ^3A_u    │  3   │      3.262 │         3.341 │       ± 0.230 │  +0.079 │
│ ^3B_{1g} │  3   │      3.010 │         3.146 │       ± 0.297 │  +0.136 │
│ ^3B_{1u} │  3   │      3.443 │         3.689 │       ± 0.226 │  +0.246 │
│ ^3B_{3g} │  3   │      3.498 │         3.724 │       ± 0.223 │  +0.226 │
│ --       │  --  │         -- │            -- │            -- │      -- │
│ Summary  │      │ MAE: 0.150 │   RMSE: 0.168 │    MSE: 0.028 │         │
└──────────┴──────┴────────────┴───────────────┴───────────────┴─────────┘
```

