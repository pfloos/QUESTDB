import json
import os
import argparse
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def parse_list(arg):
    return arg.split(",") if arg else None

def parse_int_list(arg):
    return [int(x) for x in arg.split(",")] if arg else None

def load_json_files(paths):
    all_data = []
    for path in paths:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    with open(os.path.join(path, filename), "r") as f:
                        chunk = json.load(f)
                        if isinstance(chunk, list):
                            all_data.extend(chunk)
                        else:
                            raise ValueError(f"File {filename} does not contain a list.")
        elif os.path.isfile(path) and path.endswith(".json"):
            with open(path, "r") as f:
                chunk = json.load(f)
                if isinstance(chunk, list):
                    all_data.extend(chunk)
                else:
                    raise ValueError(f"File {path} does not contain a list.")
        else:
            raise ValueError(f"{path} is not a valid file or directory.")
    return all_data

def filter_excitations(
    input_files, output_file,
    spin=None, vr=None, safe=None,
    groups=None, types=None, exclude_gd=False,
    min_size=None, max_size=None
):
    data = load_json_files(input_files)
    total = len(data)

    def matches(entry):
        if spin is not None and entry.get("Spin") != spin:
            return False
        if vr is not None and entry.get("V/R") != vr:
            return False
        if safe is not None and entry.get("Safe ? (~50 meV)") != safe:
            return False
        if groups is not None and entry.get("Group") not in groups:
            return False
        if types is not None and entry.get("Type") not in types:
            return False
        if exclude_gd and entry.get("Special ?") == "gd":
            return False
        if min_size is not None and entry.get("Size") < min_size:
            return False
        if max_size is not None and entry.get("Size") > max_size:
            return False
        return True

    filtered = [entry for entry in data if matches(entry)]

    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    # ✅ Summary panel
    console.print(
        Panel.fit(
            f"[bold green]✔ Filtered excitations saved to[/] [yellow]{output_file}[/]",
            border_style="green",
            title="✅ [bold]Output",
        )
    )

    summary = Table(title="📊 Filtering Summary", box=box.HEAVY)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="bold magenta", justify="right")

    summary.add_row("🔢 Total entries", str(total))
    summary.add_row("🔍 Filtered entries", str(len(filtered)))
    summary.add_row("📉 Remaining (%)", f"{100 * len(filtered) / total:.1f}%" if total else "N/A")

    distinct_molecules = len(set(entry.get("Molecule", "unknown") for entry in filtered))
    summary.add_row("🧪 Distinct molecules", str(distinct_molecules))

    if spin:
        spin_names = {1: "Singlet", 2: "Doublet", 3: "Triplet", 4: "Quartet"}
        summary.add_row("🌀 Spin", spin_names.get(spin, f"Spin {spin}"))
    if vr: summary.add_row("🧭 V/R", "Valence" if vr == "V" else "Rydberg")
    if safe: summary.add_row("🔐 Safe", "Yes" if safe == "Y" else "No")
    if groups: summary.add_row("🧬 Groups", ", ".join(map(str, groups)))
    if types: summary.add_row("🎯 Types", ", ".join(types))
    if exclude_gd: summary.add_row("🚫 Exclude GD", "Yes")
    if min_size: summary.add_row("📏 Min Size", str(min_size))
    if max_size: summary.add_row("📏 Max Size", str(max_size))

    console.print(summary)

    spin_labels = {1: "Singlet", 2: "Doublet", 3: "Triplet", 4: "Quartet"}

    def print_breakdown(title, counter, icon):
        if counter:
            table = Table(title=f"{icon} {title}", box=box.HEAVY)
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right", style="magenta")
            for key, count in sorted(counter.items()):
                table.add_row(str(key), str(count))
            console.print(table)

    print_breakdown("Breakdown by Group", Counter(entry.get("Group") for entry in filtered), "🧬")
    print_breakdown("Breakdown by Type", Counter(entry.get("Type") for entry in filtered), "🎯")

    spin_counter = Counter()
    for entry in filtered:
        try:
            spin_value = int(entry.get("Spin"))
            label = spin_labels.get(spin_value, f"Spin {spin_value}")
            spin_counter[label] += 1
        except (ValueError, TypeError):
            spin_counter["Unknown"] += 1

    print_breakdown("Breakdown by Spin", spin_counter, "🌀")

    nature_labels = {"V": "Valence", "R": "Rydberg", "M": "Mixed"}

    nature_counter = Counter()
    for entry in filtered:
        key = entry.get("V/R", "").strip().upper()
        label = nature_labels.get(key, "Unknown")
        nature_counter[label] += 1
 
    print_breakdown("Breakdown by Nature", nature_counter, "🧭")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="✨ Fancy filter for QUEST excitations.")
    parser.add_argument("input_files", help="Path to directory or single .json file")
    parser.add_argument("output_file", help="Path to output JSON file")
    parser.add_argument("--spin", type=int, choices=[1, 2, 3, 4], help="1 for singlet, 2 for doublet, 3 for triplet, 4 for quartet")
    parser.add_argument("--nature", choices=["V", "R", "M"], help="'V' for valence, 'R' for Rydberg, 'M' for mixed")
    parser.add_argument("--safe", choices=["Y", "N"], help="'Y' = safe, 'N' = unsafe")
    parser.add_argument("--group", type=parse_int_list, help="Comma-separated list of Group numbers (12, 35, 69, 1016)")
    parser.add_argument("--type", type=parse_list, help="Comma-separated list of excitation types (e.g., npi,ppi,n3s)")
    parser.add_argument("--exclude-gd", action="store_true", help="Exclude genuine double excitations ('GD')")
    parser.add_argument("--min-size", type=int, help="Minimum molecule size to include")
    parser.add_argument("--max-size", type=int, help="Maximum molecule size to include")

    args = parser.parse_args()

    filter_excitations(
        input_files=args.input_files.split(','),
        output_file=args.output_file,
        spin=args.spin,
        vr=args.nature,
        safe=args.safe,
        groups=args.group,
        types=args.type,
        exclude_gd=args.exclude_gd,
        min_size=args.min_size,
        max_size=args.max_size
    )
