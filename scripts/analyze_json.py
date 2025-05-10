import argparse
import json
import os
from collections import Counter, defaultdict
from typing import List
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text

console = Console()


def load_json_files(paths: List[str]) -> List[dict]:
    all_data = []
    for path in paths:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    with open(os.path.join(path, filename)) as f:
                        all_data.extend(json.load(f))
        elif os.path.isfile(path) and path.endswith(".json"):
            with open(path) as f:
                all_data.extend(json.load(f))
    return all_data


def analyze_data(data: List[dict]) -> dict:
    stats = defaultdict(Counter)
    unique_molecules = set()

    for entry in data:
        multiplicity_value = entry.get("spin")
        transition_type = entry.get("Type", "").strip().lower()
        vr_value = entry.get("V/R", "").strip().upper()
        group = str(entry.get("Group", "unknown"))
        safe_flag = str(entry.get("Safe ? (~50 meV)", "")).strip().upper()
        molecule = entry.get("Molecule", "unknown")

        unique_molecules.add(molecule)

        if multiplicity_value == 1:
            multiplicity = "Singlet"
        elif multiplicity_value == 2:
            multiplicity = "Doublet"
        elif multiplicity_value == 3:
            multiplicity = "Triplet"
        elif multiplicity_value == 4:
            multiplicity = "Quartet"
        else:
            multiplicity = "Unknown"

        if vr_value == "V":
            vr_category = "Valence"
        elif vr_value == "R":
            vr_category = "Rydberg"
        elif vr_value == "M":
            vr_category = "Mixed"
        else:
            vr_category = "Unknown"

        stats["multiplicity"][multiplicity] += 1
        stats["type"][transition_type or "unknown"] += 1
        stats["v_r"][vr_category] += 1
        stats["group"][group] += 1
        stats["safe"]["Unsafe" if safe_flag == "N" else "Safe"] += 1

    stats["meta"]["distinct_molecules"] = len(unique_molecules)
    stats["meta"]["total_excitations"] = len(data)
    return stats


def print_stats(stats: dict):

    total = stats["meta"]["total_excitations"]
    safe = stats["safe"].get("Safe", 0)
    unsafe = stats["safe"].get("Unsafe", 0)
    molecules = stats["meta"]["distinct_molecules"]

    summary = Panel.fit(
        f"[bold white]Total excitations:[/] [cyan]{total}[/]\n"
        f"[bold green]✅ Safe:[/] {safe}    [bold red]⚠️  Unsafe:[/] {unsafe}\n"
        f"[bold yellow]🔬 Distinct molecules:[/] {molecules}",
        title="📊 [bold magenta]Summary",
        border_style="bright_blue",
        padding=(1, 3),
    )
    console.print(summary)

    def print_table(title: str, category: str, labels=None, icon="📌"):
        table = Table(title=f"{icon} {title}", header_style="bold white", title_style="bold magenta")
        table.add_column("Category", style="bold cyan")
        table.add_column("Count", justify="right", style="bold green")

        items = stats[category].items()
        if labels:
            items = [(label, stats[category].get(label, 0)) for label in labels]

        # Filter out zero rows
        items = [(k, v) for k, v in items if v > 0]
        if not items:
            return

        for key, count in sorted(items, key=lambda x: x[0].lower()):
            table.add_row(key.capitalize(), str(count))

        console.print(Padding(table, (1, 2)))

    print_table("Multiplicity Distribution", "multiplicity", labels=["Singlet", "Doublet", "Triplet", "Quartet", "Unknown"], icon="🌀")
    print_table("Transition Type", "type", icon="🎯")
    print_table("Transition Nature", "v_r", labels=["Valence", "Rydberg", "Unknown"], icon="🧭")
    print_table("Size Distribution", "group", icon="🧬")


def main():
    parser = argparse.ArgumentParser(description="Analyze QUEST excitation set JSON files.")
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more JSON files or directories containing QUEST-style excitation data."
    )
    args = parser.parse_args()

    data = load_json_files(args.files)
    stats = analyze_data(data)
    print_stats(stats)


if __name__ == "__main__":
    main()
