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
        multiplicity_value = entry.get("S/T")
        transition_type = entry.get("Type", "").strip().lower()
        vr_value = entry.get("V/R", "").strip().upper()
        group = str(entry.get("Group", "unknown"))
        safe_flag = str(entry.get("Safe ? (~50 meV)", "")).strip().upper()
        molecule = entry.get("Molecule", "unknown")

        unique_molecules.add(molecule)

        if multiplicity_value == 1:
            multiplicity = "singlet"
        elif multiplicity_value == 3:
            multiplicity = "triplet"
        else:
            multiplicity = "unknown"

        if vr_value == "V":
            vr_category = "valence"
        elif vr_value == "R":
            vr_category = "rydberg"
        else:
            vr_category = "unknown"

        stats["multiplicity"][multiplicity] += 1
        stats["type"][transition_type or "unknown"] += 1
        stats["v_r"][vr_category] += 1
        stats["group"][group] += 1
        stats["safe"]["unsafe" if safe_flag == "N" else "safe"] += 1

    stats["meta"]["distinct_molecules"] = len(unique_molecules)
    return stats


def print_stats(stats: dict):
    console.rule("[bold cyan]🔍 Excitation Set Analysis Summary")

    summary_panel = Panel.fit(
        f"[bold yellow]🧪 Distinct Molecules:[/bold yellow] {stats['meta']['distinct_molecules']}\n"
        f"[bold red]⚠️ Unsafe Excitations:[/bold red] {stats['safe'].get('unsafe', 0)}\n"
        f"[bold green]✅ Safe Excitations:[/bold green] {stats['safe'].get('safe', 0)}",
        title="📊 [bold]Summary",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(summary_panel)

    def print_table(title: str, category: str, labels=None, icon="📌"):
        table = Table(title=f"{icon} [bold]{title}[/bold]", header_style="bold magenta")
        table.add_column(category.capitalize(), style="bold white")
        table.add_column("Count", justify="right", style="bold green")

        items = stats[category].items()
        if labels:
            items = [(label, stats[category].get(label, 0)) for label in labels]
        for key, count in items:
            table.add_row(key.capitalize(), str(count))
        console.print(Padding(table, (1, 2)))

    print_table("By Multiplicity", "multiplicity", labels=["singlet", "triplet", "unknown"], icon="🌀")
    print_table("By Transition Type", "type", icon="🎯")
    print_table("By Valence/Rydberg", "v_r", labels=["valence", "rydberg", "unknown"], icon="🧭")
    print_table("By Group", "group", icon="🧬")


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
