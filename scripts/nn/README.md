# 🧪 Neural Network for Theoretical Best Estimate (TBE) Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-success)
![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-important)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-blue)

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Advanced Architecture** | MLP with feature extraction layers (256-128-64) + regressor (32-1) |
| **Uncertainty Quantification** | MC Dropout for prediction confidence intervals |
| **Robust Training** | Early stopping, LR scheduling, and comprehensive metrics |
| **Beautiful Output** | Rich-formatted tables and visualizations |

## 🚀 Quick Start

## Basic Usage

### Training:

```bash
python quest_nn.py --train --data-dir data/training/ --model model.pt
```

### Prediction:

```bash
python quest_nn.py --predict data/samples/ --model model.pt
```

## 📊 Model Architecture
graph TD
    A[Input Features] --> B[Linear(256)]
    B --> C[BatchNorm + SiLU + Dropout]
    C --> D[Linear(128)]
    D --> E[BatchNorm + SiLU + Dropout]
    E --> F[Linear(64)]
    F --> G[BatchNorm + SiLU + Dropout]
    G --> H[Regressor: Linear(32)]
    H --> I[SiLU]
    I --> J[Linear(1)]
    J --> K[Output]

## 📂 File Structure

graph TD
    A[Input Features] --> B[Linear(256)]
    B --> C[BatchNorm + SiLU + Dropout]
    C --> D[Linear(128)]
    D --> E[BatchNorm + SiLU + Dropout]
    E --> F[Linear(64)]
    F --> G[BatchNorm + SiLU + Dropout]
    G --> H[Regressor: Linear(32)]
    H --> I[SiLU]
    I --> J[Linear(1)]
    J --> K[Output]

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

Molecule: Benzene
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ State  ┃ Spin    ┃ Actual TBE┃ Predicted TBE ┃ ± Uncertainty┃ Δ Error ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ 1B2u   │ Singlet │ 5.432     │ 5.401         │ ± 0.023      │ -0.031  │
│ 1B1u   │ Singlet │ 6.125     │ 6.098         │ ± 0.031      │ -0.027  │
├────────┼─────────┼───────────┼───────────────┼──────────────┼─────────┤
│ Summary         │ MAE: 0.029 │ RMSE: 0.031   │ MSE: 0.001   │         │
└────────────────┴───────────┴───────────────┴──────────────┴─────────┘

