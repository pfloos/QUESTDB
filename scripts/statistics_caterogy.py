import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()

SKIP_FIELDS = {
    "TBE/AVTZ", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]",
    "Size", "Group", "S/T", "V/R", "Type", 
    "Safe ? (~50 meV)", "Special ?"
}

CATEGORIES = {
    "S/T": [1.0, 3.0],
    "V/R": ["V", "R"],
    "Group": [12, 35, 69, 1016],
    "Type": None  # auto-detect
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

def analyze_by_category(json_dir):
    category_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            for entry in data:
                ref_energy = entry.get("TBE/AVTZ")
                if ref_energy is None:
                    continue

                exc_type = entry.get("Type")

                for method, energy in entry.items():
                    if method in SKIP_FIELDS or energy is None:
                        continue
                    try:
                        error = energy - ref_energy
                    except TypeError:
                        continue

                    for cat, valid in CATEGORIES.items():
                        val = entry.get(cat)
                        if val is None:
                            continue
                        if valid is None or val in valid:
                            category_data[cat][val][method].append(error)

                    if exc_type:
                        category_data["Type"][exc_type][method].append(error)

    for category, values in category_data.items():
        console.rule(f"[bold green]Category: {category}")
        for val, methods in values.items():
            table = Table(title=f"{category} = {val}", header_style="bold magenta")
            table.add_column("Method", style="cyan")
            table.add_column("#", justify="right")
            table.add_column("MSE (eV)", justify="right")
            table.add_column("MAE (eV)", justify="right")
            table.add_column("RMSE (eV)", justify="right")
            table.add_column("Min (eV)", justify="right")
            table.add_column("Max (eV)", justify="right")

            for method, errors in methods.items():
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
                plot_name = f"{category}_{val}_{method}".replace("/", "_").replace(" ", "_")
                plt.figure(figsize=(6, 4))
                plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
                plt.title(f'{method} errors ({category}={val})')
                plt.xlabel('Error (eV)')
                plt.ylabel('Count')
                plt.axvline(0, color='black', linestyle='--')
                plt.tight_layout()
                plt.savefig(f"{plot_name}_hist.png")
                plt.close()

            console.print(table)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_excitation_errors_by_category.py <json_directory>")
    else:
        analyze_by_category(sys.argv[1])
