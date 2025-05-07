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

def filter_excitations(
    input_file, output_file,
    spin=None, vr=None, safe=None,
    groups=None, types=None, exclude_gd=False,
    min_size=None, max_size=None
):
    data = []
    if os.path.isdir(input_file):
        for filename in os.listdir(input_file):
            if filename.endswith(".json"):
                with open(os.path.join(input_file, filename), "r") as f:
                    chunk = json.load(f)
                    if isinstance(chunk, list):
                        data.extend(chunk)
                    else:
                        raise ValueError(f"File {filename} does not contain a list.")
    elif os.path.isfile(input_file) and input_file.endswith(".json"):
        with open(input_file, "r") as f:
            chunk = json.load(f)
            if isinstance(chunk, list):
                data = chunk
            else:
                raise ValueError("Expected a list of excitations in the JSON file.")
    else:
        raise ValueError(f"{input_file} is not a valid file or directory.")

    if not isinstance(data, list):
        raise ValueError("Expected a list of excitations in the JSON file.")

    total = len(data)

    def matches(entry):
        if spin is not None and entry.get("S/T") != spin:
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
    console.rule("[bold cyan]🎯 Filter Results Summary")
    console.print(
        Panel.fit(
            f"[bold green]✔ Filtered excitations saved to[/] [yellow]{output_file}[/]",
            border_style="green",
            title="✅ [bold]Output",
        )
    )

    summary = Table(title="📊 Filtering Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="bold magenta", justify="right")

    summary.add_row("🔢 Total entries", str(total))
    summary.add_row("🔍 Filtered entries", str(len(filtered)))
    summary.add_row("📉 Remaining (%)", f"{100 * len(filtered) / total:.1f}%" if total else "N/A")

    distinct_molecules = len(set(entry.get("Molecule", "unknown") for entry in filtered))
    summary.add_row("🧪 Distinct molecules", str(distinct_molecules))

    if spin: summary.add_row("🌀 Spin", f"{'Singlet' if spin == 1 else 'Triplet'}")
    if vr: summary.add_row("🧭 V/R", "Valence" if vr == "V" else "Rydberg")
    if safe: summary.add_row("🔐 Safe", "Yes" if safe == "Y" else "No")
    if groups: summary.add_row("🧬 Groups", ", ".join(map(str, groups)))
    if types: summary.add_row("🎯 Types", ", ".join(types))
    if exclude_gd: summary.add_row("🚫 Exclude GD", "Yes")
    if min_size: summary.add_row("📏 Min Size", str(min_size))
    if max_size: summary.add_row("📏 Max Size", str(max_size))

    console.print(summary)

    def print_breakdown(title, counter, icon):
        if counter:
            table = Table(title=f"{icon} {title}", box=box.MINIMAL_DOUBLE_HEAD)
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right", style="magenta")
            for key, count in sorted(counter.items()):
                table.add_row(str(key), str(count))
            console.print(table)

    print_breakdown("Breakdown by Group", Counter(entry.get("Group") for entry in filtered), "🧬")
    print_breakdown("Breakdown by Type", Counter(entry.get("Type") for entry in filtered), "🎯")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="✨ Fancy filter for QUEST excitations.")
    parser.add_argument("input_file", help="Path to directory containing .json files")
    parser.add_argument("output_file", help="Path to output JSON file")
    parser.add_argument("--spin", type=int, choices=[1, 3], help="1 for singlet, 3 for triplet")
    parser.add_argument("--nature", choices=["V", "R"], help="'V' for valence, 'R' for Rydberg")
    parser.add_argument("--safe", choices=["Y", "N"], help="'Y' = safe, 'N' = unsafe")
    parser.add_argument("--group", type=parse_int_list, help="Comma-separated list of Group numbers (12, 35, 69, 1016)")
    parser.add_argument("--type", type=parse_list, help="Comma-separated list of excitation types (e.g., npi,ppi,n3s)")
    parser.add_argument("--exclude-gd", action="store_true", help="Exclude genuine double excitations ('GD')")
    parser.add_argument("--min-size", type=int, help="Minimum molecule size to include")
    parser.add_argument("--max-size", type=int, help="Maximum molecule size to include")

    args = parser.parse_args()

    filter_excitations(
        input_file=args.input_file,
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
