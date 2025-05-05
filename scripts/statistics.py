import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()

SKIP_FIELDS = {
    "TBE/AVTZ", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]",
    "Size", "Group", "S/T", "V/R", "Type",
    "Safe ? (~50 meV)", "Special ?"
}

def compute_statistics(errors):
    errors = np.array(errors)
    return {
        "N": len(errors),
        "MSE": np.mean(errors),
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors**2)),
        "Min": np.min(errors),
        "Max": np.max(errors)
    }

def analyze_directory_with_plots(json_dir):
    method_errors = {}

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            for entry in data:
                ref_energy = entry.get("TBE/AVTZ")
                if ref_energy is None:
                    continue

                for method, energy in entry.items():
                    if method in SKIP_FIELDS or energy is None:
                        continue
                    try:
                        error = energy - ref_energy
                        method_errors.setdefault(method, []).append(error)
                    except TypeError:
                        continue

    table = Table(title="Excitation Energy Error Statistics by Method", header_style="bold magenta")
    table.add_column("Method", style="cyan")
    table.add_column("#", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for method, errors in method_errors.items():
        stats = compute_statistics(errors)
        table.add_row(
            method,
            str(stats["N"]),
            f"{stats['MSE']:.4f}",
            f"{stats['MAE']:.4f}",
            f"{stats['RMSE']:.4f}",
            f"{stats['Min']:.4f}",
            f"{stats['Max']:.4f}"
        )

        # Plot
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Error Distribution: {method}')
        plt.xlabel('Error (eV)')
        plt.ylabel('Count')
        plt.axvline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{method.replace('/', '_')}_error_histogram.png")
        plt.close()

    console.print(table)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_excitation_errors.py <json_directory>")
    else:
        analyze_directory_with_plots(sys.argv[1])
