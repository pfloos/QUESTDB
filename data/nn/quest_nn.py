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
import matplotlib.pyplot as plt

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# === Constants ===
TARGET_COL = "TBE/AVTZ"
CATEGORICAL_COLS = ["Molecule", "State", "Type", "Spin"]
NUMERICAL_COLS = [
#   "Spin", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]",
#   "CIS(D)", "CC2", "EOM-MP2", "STEOM-CCSD", "CCSD",
#   "CCSD(T)(a)*", "CCSDR(3)", "CCSDT-3", "CC3", "CCSDT",
#   "SOS-ADC(2) [TM]", "SOS-CC2", "SCS-CC2",
#   "SOS-ADC(2) [QC]", "ADC(2)", "ADC(3)", "ADC(2.5)"
    "CIS(D)", "CC2", "EOM-MP2", "CCSD", 
    "SOS-ADC(2) [TM]", "SOS-CC2", "SCS-CC2",
    "SOS-ADC(2) [QC]", "ADC(2)"
]

# === Model ===
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Load all JSON files from a directory ===
def load_json_directory(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                try:
                    data = json.load(f)
                    all_data.extend(data if isinstance(data, list) else [data])
                except json.JSONDecodeError as e:
                    print(f"Error loading {filename}: {e}")
    return pd.DataFrame(all_data)

# === Load model ===
def load_model(path, input_dim):
    net = Net(input_dim)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

# === Prediction helpers ===
def predict_tbe(model, pipeline, input_dict):
    df_input = pd.DataFrame([input_dict])
    X_input = pipeline.transform(df_input)
    X_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(X_tensor).item()

def predict_with_uncertainty(model, pipeline, input_dict, n_iter=1000):
    df_input = pd.DataFrame([input_dict])
    X_input = pipeline.transform(df_input)
    X_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32)
    model.train()  # Enable dropout
    preds = [model(X_tensor).item() for _ in range(n_iter)]
    return np.mean(preds), np.std(preds)

def predict_deterministic(model, pipeline, input_dict):
    df_input = pd.DataFrame([input_dict])
    X_input = pipeline.transform(df_input)
    X_tensor = torch.tensor(X_input.toarray(), dtype=torch.float32)
    model.eval()  # turns off dropout
    with torch.no_grad():
        return model(X_tensor).item()

# === CLI main ===
def cli():
    parser = argparse.ArgumentParser(description="Train TBE predictor or predict from JSON.")
    parser.add_argument("--train", action="store_true", help="Train the model using JSON files from directory.")
    parser.add_argument("--data-dir", type=str, help="Path to directory with JSON files.")
    parser.add_argument("--predict", type=str, help="Path to a JSON file with one molecule's data.")
    parser.add_argument("--model", type=str, default="tbe_model.pt", help="Path to save/load model weights.")
    parser.add_argument("--pipeline", type=str, default="tbe_pipeline.pkl", help="Path to save/load preprocessing pipeline.")
    parser.add_argument("--n-samples", type=int, default=1000, help="MC Dropout iterations (for uncertainty).")
    args = parser.parse_args()

    if args.train:
        if not args.data_dir:
            print("❌ Please specify --data-dir.")
            sys.exit(1)

        df = load_json_directory(args.data_dir)
        df.columns = df.columns.str.strip()
        df = df[df[TARGET_COL].notna()]
        df = df.dropna(subset=NUMERICAL_COLS + CATEGORICAL_COLS)

        X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
        y = df[TARGET_COL]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
        ])
        pipeline = Pipeline([("preprocessor", preprocessor)])
        X_processed = pipeline.fit_transform(X)
        input_dim = X_processed.shape[1]

        # Save pipeline
        joblib.dump(pipeline, args.pipeline)
        print(f"✅ Saved pipeline to: {args.pipeline}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        model = Net(input_dim)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("🚀 Starting training...")
        for epoch in range(20000):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = loss_fn(pred, y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_test_tensor)
                    val_mae = mean_absolute_error(y_test_tensor.numpy(), val_pred.numpy())
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val MAE: {val_mae:.4f}")

        torch.save(model.state_dict(), args.model)
        print(f"✅ Model saved to: {args.model}")

        # === Plot
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).squeeze().numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Reference TBE/AVTZ")
        plt.ylabel("Predicted TBE/AVTZ")
        plt.title("Prediction vs. Reference")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("prediction_vs_reference.png")
        plt.show()

    elif args.predict:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if not os.path.exists(args.model) or not os.path.exists(args.pipeline):
            console.print("[bold red]❌ Missing model or pipeline file.[/bold red]")
            sys.exit(1)

        # Load input JSON
        with open(args.predict, "r") as f:
            sample_data = json.load(f)
            if isinstance(sample_data, dict):
                sample_data = [sample_data]

        pipeline = joblib.load(args.pipeline)
        input_dim = pipeline.transform(
            pd.DataFrame([sample_data[0]])[NUMERICAL_COLS + CATEGORICAL_COLS]
        ).shape[1]
        model = load_model(args.model, input_dim)

        console.print(f"\n🔮 [bold]Predictions for:[/bold] [cyan]{args.predict}[/cyan]\n")

        results = []
        grouped_output = {}

        # Loop through each state
        for entry in sample_data:
            mol = entry.get("Molecule", "?").strip()
            state = entry.get("State", "?")
            try:
                mean, std = predict_with_uncertainty(model, pipeline, entry, n_iter=args.n_samples)
                results.append({
                    "Molecule": mol,
                    "State": state,
                    "Predicted_TBE": mean,
                    "Uncertainty": std
                })
                grouped_output.setdefault(mol, []).append((state, mean, std))
            except Exception as e:
                console.print(f"[yellow]⚠️ Skipped {mol} | {state} due to error: {e}[/yellow]")

        # Display grouped results
        for mol, states in grouped_output.items():
            table = Table(title=f"Molecule: {mol}", title_style="bold magenta")
            table.add_column("State", justify="left", style="cyan")
            table.add_column("TBE/AVTZ (eV)", justify="right", style="green")
            table.add_column("± Uncertainty", justify="right", style="yellow")
            for state, mean, std in states:
                table.add_row(state, f"{mean:.3f}", f"± {std:.3f}")
            console.print(table)

        # Save to CSV
        out_path = args.predict.replace(".json", "_predictions.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        console.print(f"\n✅ [green]Saved all predictions to:[/green] [bold]{out_path}[/bold]\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
