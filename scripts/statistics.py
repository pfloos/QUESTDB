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
    "Corr. Method", "%T1 [CC3/AVTZ]", "%T1 [CC3/AVDZ]", "f [LR-CC3/AVTZ]", "f [LR-CCSD/AVTZ]",
    "Size", "Group", "Spin", "V/R", "Type",
    "Safe ? (~50 meV)", "Special ?"
}

console = Console()

def load_data(json_input: str, filters: dict):
    entries = []

    # Determine if input is a file or a directory
    if os.path.isfile(json_input):
        filepaths = [json_input]
    elif os.path.isdir(json_input):
        filepaths = [os.path.join(json_input, f) for f in os.listdir(json_input) if f.endswith(".json")]
    else:
        console.print(f"❌ Error: '{json_input}' is not a valid file or directory.", style="bold red")
        sys.exit(1)

    for filepath in filepaths:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                console.print(f"[red]Failed to load {filepath}:[/red] {e}")
                continue

        for entry in data:
            ref_energy = entry.get("TBE/AVTZ")
            if ref_energy is None:
                continue

            # Apply filters
            if filters.get("safe_only", True) and entry.get("Safe ? (~50 meV)") != "Y":
                continue
            if filters.get("only_singlet") and entry.get("Spin") != 1:
                continue
            if filters.get("only_doublet") and entry.get("Spin") != 2:
                continue
            if filters.get("only_triplet") and entry.get("Spin") != 3:
                continue
            if filters.get("only_quartet") and entry.get("Spin") != 4:
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

            method_errors = {
                method: energy - ref_energy
                for method, energy in entry.items()
                if method not in SKIP_FIELDS and isinstance(energy, (int, float))
            }

            entries.append({
                "file": os.path.basename(filepath),
                "index": entry.get("Index", None),
                "data": method_errors,
                "categories": {k: entry[k] for k in ["Spin", "V/R", "Type"] if k in entry},
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

def analyze_with_plots(json_input: str, filters: dict, print_graphs: bool):
    entries = load_data(json_input, filters)
    method_errors = {}

    for entry in entries:
        for method, error in entry["data"].items():
            method_errors.setdefault(method, []).append(error)

    console.print(f"\n📊 Excitation Energy Error Statistics (in eV) by Method:\n")
    table = Table(show_header=True, header_style="bold magenta", box=box.HEAVY)
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

        if print_graphs:
            output_dir = "plots"
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
            plt.title(f'Error Distribution: {method}')
            plt.xlabel('Error (eV)')
            plt.ylabel('Count')
            plt.axvline(0, color='black', linestyle='--')
            plt.tight_layout()
            filename = os.path.join(output_dir, f"{method.replace('/', '_')}_error_distribution.png")
            plt.savefig(filename, dpi=150)
            plt.close()
            console.print(f"📈 Saved plot: [green]{filename}[/]")

    console.print(table)

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze excitation energy errors and create plots.")
    parser.add_argument("json_input", help="JSON file or directory containing .json files")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum molecule size")
    parser.add_argument("--max-size", type=int, default=1000, help="Maximum molecule size")
    parser.add_argument("--only-singlet", action="store_true", help="Only include singlet transitions")
    parser.add_argument("--only-doublet", action="store_true", help="Only include doublet transitions")
    parser.add_argument("--only-triplet", action="store_true", help="Only include triplet transitions")
    parser.add_argument("--only-quartet", action="store_true", help="Only include quartet transitions")
    parser.add_argument("--only-valence", action="store_true", help="Only include valence transitions")
    parser.add_argument("--only-rydberg", action="store_true", help="Only include Rydberg transitions")
    parser.add_argument("--only-ppi", action="store_true", help="Only include π→π* transitions")
    parser.add_argument("--only-npi", action="store_true", help="Only include n→π* transitions")
    parser.add_argument("--allow-unsafe", dest="safe_only", action="store_false", help="Allow unsafe transitions")
    parser.add_argument("--print-graphs", action="store_true", help="Print error distribution graphs")

    args = parser.parse_args()

    filters = {
        "safe_only": args.safe_only,
        "only_singlet": args.only_singlet,
        "only_doublet": args.only_doublet,
        "only_triplet": args.only_triplet,
        "only_quartet": args.only_quartet,
        "only_valence": args.only_valence,
        "only_rydberg": args.only_rydberg,
        "only_ppi": args.only_ppi,
        "only_npi": args.only_npi,
        "min_size": args.min_size,
        "max_size": args.max_size,
    }

    analyze_with_plots(args.json_input, filters, args.print_graphs)
