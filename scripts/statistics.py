import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from rich.console import Console
from rich.table import Table
from rich import box

# === Config ===
SKIP_FIELDS = {
    "TBE/AVTZ", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]",
    "Size", "Group", "S/T", "V/R", "Type",
    "Safe ? (~50 meV)", "Special ?"
}

# === CLI Arguments ===
console = Console()

def load_data(json_dir: str, filters: dict):
    entries = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            for entry in data:
                ref_energy = entry.get("TBE/AVTZ")
                if ref_energy is None:
                    continue

                # Apply filters
                if filters.get("safe_only", True) and entry.get("Safe ? (~50 meV)") != "Y":
                    continue
                if filters.get("only_singlet") and entry.get("S/T") != 1:
                    continue
                if filters.get("only_triplet") and entry.get("S/T") != 3:
                    continue
                if filters.get("only_valence") and entry.get("V/R") != "V":
                    continue
                if filters.get("only_rydberg") and entry.get("V/R") != "R":
                    continue
                if filters.get("only_ppi") and entry.get("Type") != "ppi":
                    continue
                if filters.get("only_npi") and entry.get("Type") != "npi":
                    continue
                if "Size" in entry and (entry["Size"] < filters.get("min_size", 0) or entry["Size"] > filters.get("max_size", 1000)):
                    continue

                # Collect the valid entries
                method_errors = {method: energy - ref_energy for method, energy in entry.items() if method not in SKIP_FIELDS and isinstance(energy, (int, float))}
                entries.append({
                    "file": filename,
                    "index": entry.get("Index", None),
                    "data": method_errors,
                    "categories": {k: entry[k] for k in ["S/T", "V/R", "Type"] if k in entry},
                    "full": entry
                })
    return entries

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

def analyze_directory_with_plots(json_dir: str, filters: dict, print_graphs: bool):
    entries = load_data(json_dir, filters)
    method_errors = {}

    for entry in entries:
        for method, error in entry["data"].items():
            method_errors.setdefault(method, []).append(error)

    # Create the results table
    table = Table(title="Excitation Energy Error Statistics by Method", header_style="bold magenta", box=box.HEAVY)
    table.add_column("Method", style="cyan")
    table.add_column("#", justify="right")
    table.add_column("MSE (eV)", justify="right")
    table.add_column("MAE (eV)", justify="right")
    table.add_column("RMSE (eV)", justify="right")
    table.add_column("Min (eV)", justify="right")
    table.add_column("Max (eV)", justify="right")

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

        # Plot error distribution if the option is enabled
        if print_graphs:
            plt.figure(figsize=(6, 4))
            plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
            plt.title(f'Error Distribution: {method}')
            plt.xlabel('Error (eV)')
            plt.ylabel('Count')
            plt.axvline(0, color='black', linestyle='--')
            plt.tight_layout()
            plt.show()

    # Print table
    console.print(table)

def save_results(subset, output_json):
    with open(output_json, "w") as f:
        json.dump([entry["full"] for entry in subset], f, indent=2)
    console.print(f"\n✅ Saved results to [bold green]{output_json}[/]")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze excitation energy errors and create plots.")
    parser.add_argument("json_dir", help="Directory containing .json files")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum molecule size")
    parser.add_argument("--max-size", type=int, default=1000, help="Maximum molecule size")
    parser.add_argument("--only-singlet", action="store_true", help="Only include singlet transitions")
    parser.add_argument("--only-triplet", action="store_true", help="Only include triplet transitions")
    parser.add_argument("--only-valence", action="store_true", help="Only include valence transitions")
    parser.add_argument("--only-rydberg", action="store_true", help="Only include Rydberg transitions")
    parser.add_argument("--only-ppi", action="store_true", help="Only include π→π* transitions")
    parser.add_argument("--only-npi", action="store_true", help="Only include n→π* transitions")
    parser.add_argument("--allow-gd", action="store_true", help="Allow genuine double excitations")
    parser.add_argument("--allow-unsafe", dest="safe_only", action="store_false", help="Allow unsafe transitions")
    parser.add_argument("--save-results", action="store_true", help="Save filtered results to a JSON file")
    parser.add_argument("--print-graphs", action="store_true", help="Print error distribution graphs")

    args = parser.parse_args()

    filters = {
        "safe_only": args.safe_only,
        "only_singlet": args.only_singlet,
        "only_triplet": args.only_triplet,
        "only_valence": args.only_valence,
        "only_rydberg": args.only_rydberg,
        "only_ppi": args.only_ppi,
        "only_npi": args.only_npi,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "allow_gd": args.allow_gd,
    }

    if not os.path.isdir(args.json_dir):
        console.print(f"❌ Error: '{args.json_dir}' is not a valid directory.", style="bold red")
        sys.exit(1)

    analyze_directory_with_plots(args.json_dir, filters, args.print_graphs)

    if args.save_results:
        save_results([], "filtered_results.json")  # Specify appropriate subset if needed
