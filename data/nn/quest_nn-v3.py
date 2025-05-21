#!/usr/bin/env python3
"""
Advanced Neural Network for Theoretical Best Estimate (TBE) Prediction

This script trains a neural network to predict TBE values for excited states
using various quantum chemistry methods as inputs. It supports both training
and prediction modes with uncertainty estimation via MC Dropout.
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, explained_variance_score)
from rich.console import Console
from rich.table import Table
from rich.progress import track
from typing import Dict, List, Tuple, Optional

# === Constants ===
TARGET_COL = "TBE/AVTZ"
CATEGORICAL_COLS = ["Molecule", "State", "Type", "Spin"]
NUMERICAL_COLS = [
    "CIS(D)", "CC2", "EOM-MP2", "CCSD",
    "SOS-ADC(2) [TM]", "SOS-CC2", "SCS-CC2",
    "SOS-ADC(2) [QC]", "ADC(2)"
]

# === Enhanced Model Architecture ===
class TBEPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use 'leaky_relu' as approximation for SiLU initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.regressor(features)

# === Data Loading and Processing ===
def load_json_directory(directory: str) -> pd.DataFrame:
    """Load all JSON files from a directory into a DataFrame."""
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        all_data.append(data)
                    elif isinstance(data, list):
                        all_data.extend(data)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error loading {filename}: {e}")
    return pd.DataFrame(all_data)

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Clean molecule names
    df['Molecule'] = df['Molecule'].str.strip()
    
    # Clean state names (remove special characters that might cause issues)
    df['State'] = df['State'].str.replace('^', '').str.replace('_', '')
    
    # Round numerical values to 6 decimal places
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].round(6)

    """Prepare data by cleaning and splitting into features and target."""
    df.columns = df.columns.str.strip()
    df = df[df[TARGET_COL].notna()]
    df = df.dropna(subset=NUMERICAL_COLS + CATEGORICAL_COLS)
    
    # Convert numerical columns to float
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any remaining NA values
    df = df.dropna(subset=NUMERICAL_COLS + CATEGORICAL_COLS)
    
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    
    return X, y

# === Training Utilities ===
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 50, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader, 
                  loss_fn: nn.Module) -> Dict[str, float]:
    """Evaluate model performance on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item()
            
            all_preds.extend(preds.squeeze().tolist())
            all_targets.extend(y_batch.squeeze().tolist())
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'mae': mean_absolute_error(all_targets, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'r2': r2_score(all_targets, all_preds),
        'explained_variance': explained_variance_score(all_targets, all_preds)
    }
    
    return metrics

# === Prediction Utilities ===
def predict_tbe(model: nn.Module, 
               pipeline: Pipeline, 
               input_dict: Dict,
               device: torch.device) -> float:
    """Make a single prediction."""
    df_input = pd.DataFrame([input_dict])
    X_input = pipeline.transform(df_input)
    X_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return model(X_tensor).item()

def predict_with_uncertainty(model: nn.Module,
                           pipeline: Pipeline,
                           input_dict: Dict,
                           device: torch.device,
                           n_iter: int = 100) -> Tuple[float, float]:
    """Make predictions with uncertainty estimation using MC Dropout."""
    # Clean input data
    input_dict = input_dict.copy()
    input_dict['Molecule'] = input_dict['Molecule'].strip()
    input_dict['State'] = input_dict['State'].replace('^', '').replace('_', '')
    
    df_input = pd.DataFrame([input_dict])
    X_input = pipeline.transform(df_input)
    X_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32).to(device)
    
    # Set model to eval mode but enable dropout
    model.eval()
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    model.apply(enable_dropout)
    
    # Disable batch norm
    def disable_batchnorm(m):
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
    model.apply(disable_batchnorm)
    
    with torch.no_grad():
        preds = [model(X_tensor).item() for _ in range(n_iter)]
    
    return np.mean(preds), np.std(preds)

# === Visualization ===
def plot_results(y_true: np.ndarray, 
                y_pred: np.ndarray, 
                save_path: str = "results.png") -> None:
    """Create comprehensive visualization of results."""
    plt.figure(figsize=(15, 10))
    
    # Scatter plot with regression line
    plt.subplot(2, 2, 1)
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.6})
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Reference TBE/AVTZ (eV)")
    plt.ylabel("Predicted TBE/AVTZ (eV)")
    plt.title("Prediction vs Reference")
    plt.grid(True)
    
    # Residual plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residuals (eV)")
    plt.title("Residual Distribution")
    plt.grid(True)
    
    # Error distribution
    plt.subplot(2, 2, 3)
    errors = np.abs(residuals)
    sns.histplot(errors, kde=True, bins=30)
    plt.xlabel("Absolute Error (eV)")
    plt.title("Error Distribution")
    plt.grid(True)
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === CLI Main Function ===
def cli():
    parser = argparse.ArgumentParser(
        description="Advanced Neural Network for Theoretical Best Estimate (TBE) Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data-dir", type=str, help="Path to directory with JSON files")
    parser.add_argument("--predict", type=str, help="Path to JSON file for prediction")
    parser.add_argument("--model", type=str, default="tbe_model.pt", help="Path to model weights")
    parser.add_argument("--pipeline", type=str, default="tbe_pipeline.pkl", help="Path to preprocessing pipeline")
    parser.add_argument("--n-samples", type=int, default=100, help="MC Dropout iterations for uncertainty")
    parser.add_argument("--deterministic", action="store_true", help="Disable MC Dropout for deterministic prediction")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [bold]{device}[/bold]")

    if args.train:
        if not args.data_dir:
            console.print("[bold red]❌ Please specify --data-dir for training[/bold red]")
            sys.exit(1)

        # Load and prepare data
        console.print("\n📊 [bold]Loading and preparing data...[/bold]")
        df = load_json_directory(args.data_dir)
        X, y = prepare_data(df)
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
        ])
        pipeline = Pipeline([("preprocessor", preprocessor)])
        
        # Process data
        X_processed = pipeline.fit_transform(X)
        input_dim = X_processed.shape[1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42
        )
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train.toarray(), dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val.toarray(), dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test.toarray(), dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Initialize model, loss, and optimizer
        model = TBEPredictor(input_dim).to(device)
        loss_fn = nn.HuberLoss()  # More robust than MSE
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.patience//2, factor=0.5
        )
        early_stopping = EarlyStopping(patience=args.patience)
 
        def print_lr(optimizer, epoch):
            for param_group in optimizer.param_groups:
                 console.print(f"Epoch {epoch}: Learning rate = {param_group['lr']:.2e}")
       
        # Save pipeline
        joblib.dump(pipeline, args.pipeline)
        console.print(f"✅ Saved pipeline to: [green]{args.pipeline}[/green]")
        
        # Training loop
        console.print("\n🚀 [bold]Starting training[/bold]...")
        best_val_loss = float('inf')
        
        for epoch in track(range(args.epochs), description="Training..."):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_metrics = evaluate_model(model, val_loader, loss_fn)
            scheduler.step(val_metrics['loss'])
            early_stopping(val_metrics['loss'])
            if epoch % 10 == 0:
                print_lr(optimizer, epoch)
                console.print(
                    f"Epoch {epoch:4d} | "
                    f"Train Loss: {train_loss/len(train_loader):.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f} | "
                    f"Val R²: {val_metrics['r2']:.4f}"
                )
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(), args.model)
            
            if early_stopping.early_stop:
                console.print(f"⏹ Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        console.print("\n📈 [bold]Final Evaluation[/bold]")
        model.load_state_dict(torch.load(args.model))  # Load best model
        test_metrics = evaluate_model(model, test_loader, loss_fn)
        
        console.print(f"Test Loss: {test_metrics['loss']:.4f}")
        console.print(f"Test MAE: {test_metrics['mae']:.4f} eV")
        console.print(f"Test RMSE: {test_metrics['rmse']:.4f} eV")
        console.print(f"Test R²: {test_metrics['r2']:.4f}")
        console.print(f"Explained Variance: {test_metrics['explained_variance']:.4f}")
        
        # Generate predictions for plotting
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = model(X_batch.to(device)).cpu()
                all_preds.extend(preds.squeeze().tolist())
                all_targets.extend(y_batch.squeeze().tolist())
        
        plot_results(np.array(all_targets), np.array(all_preds))
        console.print(f"\n✅ Saved model to: [bold green]{args.model}[/bold green]")
        console.print(f"📊 Saved results visualization to: [bold]results.png[/bold]")

    elif args.predict:
        if not os.path.exists(args.model) or not os.path.exists(args.pipeline):
            console.print("[bold red]❌ Missing model or pipeline file.[/bold red]")
            sys.exit(1)

        with open(args.predict, "r") as f:
            sample_data = json.load(f)
            if isinstance(sample_data, dict):
                sample_data = [sample_data]

        pipeline = joblib.load(args.pipeline)
        input_dim = pipeline.transform(
            pd.DataFrame([sample_data[0]])[NUMERICAL_COLS + CATEGORICAL_COLS]
        ).shape[1]
        
        model = TBEPredictor(input_dim).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        console.print(f"\n🔮 [bold]Predictions for:[/bold] [cyan]{args.predict}[/cyan]\n")
        results = []
        grouped_output = {}

        for entry in sample_data:
            mol = entry.get("Molecule", "?").strip()
            state = entry.get("State", "?")
            spin = entry.get("Spin", "?")
            
            try:
                if args.deterministic:
                    mean = predict_tbe(model, pipeline, entry, device)
                    std = 0.0
                else:
                    mean, std = predict_with_uncertainty(
                        model, pipeline, entry, device, n_iter=args.n_samples
                    )

                results.append({
                    "Molecule": mol,
                    "State": state,
                    "Spin": spin,
                    "Predicted_TBE": mean,
                    "Uncertainty": std,
                    "Prediction_Method": "Deterministic" if args.deterministic else "MC_Dropout"
                })
                grouped_output.setdefault(mol, []).append((state, spin, mean, std))
            except Exception as e:
                console.print(f"[yellow]⚠️ Skipped {mol} | {state} due to error: {e}[/yellow]")

        # Display results in rich tables
        for mol, states in grouped_output.items():
            table = Table(title=f"Molecule: {mol}", title_style="bold magenta")
            table.add_column("State", justify="left", style="cyan")
            table.add_column("TBE/AVTZ (eV)", justify="right", style="green")
            table.add_column("± Uncertainty", justify="right", style="yellow")
            
            for state, spin, mean, std in sorted(states, key=lambda x: x[0]):
                uncertainty_display = "N/A" if args.deterministic else f"± {std:.3f}"
                table.add_row(*map(str, [state, f"{mean:.3f}", uncertainty_display]))
            
            console.print(table)

        # Save results
        out_path = args.predict.replace(".json", "_predictions.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        console.print(f"\n✅ [green]Saved all predictions to:[/green] [bold]{out_path}[/bold]\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
