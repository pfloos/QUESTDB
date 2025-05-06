import json
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
    with open(input_file, 'r') as f:
        data = json.load(f)

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
        if exclude_gd and entry.get("Type") == "GD":
            return False
        if min_size is not None and entry.get("Size") < min_size:
            return False
        if max_size is not None and entry.get("Size") > max_size:
            return False
        return True

    filtered = [entry for entry in data if matches(entry)]

    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    # Summary panel
    console.print(Panel.fit(f"[bold green]✔ Filtered excitations saved to[/] [yellow]{output_file}[/]", border_style="green"))

    summary = Table(title="Filtering Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan", no_wrap=True)
    summary.add_column("Value", style="magenta")

    summary.add_row("Total entries", str(total))
    summary.add_row("Filtered entries", str(len(filtered)))
    summary.add_row("Remaining (%)", f"{100 * len(filtered) / total:.1f}%" if total else "N/A")

    # ➕ New: Distinct molecule count
    unique_molecules = len(set(entry["Molecule"] for entry in filtered))
    summary.add_row("Distinct molecules", str(unique_molecules))

    if spin: summary.add_row("Spin", f"{'Singlet' if spin == 1 else 'Triplet'}")
    if vr: summary.add_row("V/R", "Valence" if vr == "V" else "Rydberg")
    if safe: summary.add_row("Safe", "Yes" if safe == "Y" else "No")
    if groups: summary.add_row("Groups", ", ".join(map(str, groups)))
    if types: summary.add_row("Types", ", ".join(types))
    if exclude_gd: summary.add_row("Exclude GD", "Yes")
    if min_size: summary.add_row("Min Size", str(min_size))
    if max_size: summary.add_row("Max Size", str(max_size))

    console.print(summary)

    # Breakdown by Group
    group_counts = Counter(entry.get("Group") for entry in filtered)
    if group_counts:
        table_group = Table(title="Breakdown by Group", box=box.MINIMAL_DOUBLE_HEAD)
        table_group.add_column("Group", style="cyan")
        table_group.add_column("Count", justify="right", style="magenta")
        for group, count in sorted(group_counts.items()):
            table_group.add_row(str(group), str(count))
        console.print(table_group)

    # Breakdown by Type
    type_counts = Counter(entry.get("Type") for entry in filtered)
    if type_counts:
        table_type = Table(title="Breakdown by Type", box=box.MINIMAL_DOUBLE_HEAD)
        table_type.add_column("Type", style="cyan")
        table_type.add_column("Count", justify="right", style="magenta")
        for typ, count in sorted(type_counts.items()):
            table_type.add_row(str(typ), str(count))
        console.print(table_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="✨ Fancy filter for QUEST excitations.")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", help="Path to output JSON file")
    parser.add_argument("--spin", type=int, choices=[1, 3], help="1=singlet, 3=triplet")
    parser.add_argument("--vr", choices=["V", "R"], help="'V' for valence, 'R' for Rydberg")
    parser.add_argument("--safe", choices=["Y", "N"], help="'Y' = safe, 'N' = unsafe")
    parser.add_argument("--group", type=parse_int_list, help="Comma-separated list of Group numbers")
    parser.add_argument("--type", type=parse_list, help="Comma-separated list of excitation types (e.g., npi,ppi,n3s)")
    parser.add_argument("--exclude-gd", action="store_true", help="Exclude genuine double excitations (Type == 'GD')")
    parser.add_argument("--min-size", type=int, help="Minimum molecule size to include")
    parser.add_argument("--max-size", type=int, help="Maximum molecule size to include")

    args = parser.parse_args()

    filter_excitations(
        input_file=args.input_file,
        output_file=args.output_file,
        spin=args.spin,
        vr=args.vr,
        safe=args.safe,
        groups=args.group,
        types=args.type,
        exclude_gd=args.exclude_gd,
        min_size=args.min_size,
        max_size=args.max_size
    )
