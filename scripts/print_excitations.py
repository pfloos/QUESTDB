import os
import json
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich import box


def extract_states_from_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [
        {
            "State": entry.get("State", "N/A"),
            "Spin": entry.get("Spin", None),
            "V/R": entry.get("V/R", "N/A"),
            "Type": entry.get("Type", "N/A"),
            "TBE": entry.get("TBE/AVTZ", None),
            "Safe": entry.get("Safe ? (~50 meV)", "N/A"),
            "Molecule": entry.get("Molecule", os.path.basename(filepath)).strip()
        }
        for entry in data
        if "TBE/AVTZ" in entry and isinstance(entry["TBE/AVTZ"], (int, float))
    ]


def gather_states(input_path):
    molecule_dict = defaultdict(list)

    if os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):
            if filename.endswith(".json"):
                filepath = os.path.join(input_path, filename)
                try:
                    for state in extract_states_from_file(filepath):
                        molecule_dict[state["Molecule"]].append(state)
                except Exception as e:
                    Console().print(f"[red]Failed to process {filename}:[/red] {e}")
    elif input_path.endswith(".json"):
        try:
            for state in extract_states_from_file(input_path):
                molecule_dict[state["Molecule"]].append(state)
        except Exception as e:
            Console().print(f"[red]Failed to process {input_path}:[/red] {e}")
    else:
        raise ValueError("Input must be a JSON file or directory of JSON files.")

    return molecule_dict


def print_summary_table(input_path, filter_spin=None, filter_state=None, filter_type=None, filter_nature=None, safe_only=False):
    console = Console()
    table = Table(title="Excited State Summary", box=box.SIMPLE_HEAVY)

    table.add_column("Molecule", style="cyan", no_wrap=True)
    table.add_column("State", style="magenta")
    table.add_column("Spin", style="green")
    table.add_column("V/R", style="yellow")
    table.add_column("Type", style="blue")
    table.add_column("Safe?", style="bold red")
    table.add_column("TBE/AVTZ", justify="right", style="bold")

    molecule_dict = gather_states(input_path)

    for mol_name in sorted(molecule_dict.keys()):
        # Apply filtering
        filtered_states = []
        for state in molecule_dict[mol_name]:
            if filter_spin is not None and state["Spin"] != filter_spin:
                continue
            if filter_type is not None and state["Type"] != filter_type:
                continue
            if filter_state is not None and state["State"] != filter_state:
                continue
            if filter_nature is not None and state["V/R"] != filter_nature:
                continue
            if safe_only and state["Safe"] != "Y":
                continue
            filtered_states.append(state)

        # Skip molecule if no states remain
        if not filtered_states:
            continue

        sorted_states = sorted(filtered_states, key=lambda x: x["TBE"])
        first_row = True
        for state in sorted_states:
            table.add_row(
                mol_name if first_row else "",
                state["State"],
                str(state["Spin"]),
                state["V/R"],
                state["Type"],
                state["Safe"],
                f"{state['TBE']:.3f}"
            )
            first_row = False

    console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print a table of excited state characteristics from JSON file(s)."
    )
    parser.add_argument("input_path", help="Path to a JSON file or directory.")
    parser.add_argument("--spin", type=int, choices=[1, 2, 3, 4], help="Filter by spin (1, 2, 3, 4).")
    parser.add_argument("--state", type=str, help="Filter by exact state label (e.g., '^1A_1').")
    parser.add_argument("--type", type=str, help="Filter by excitation type (e.g., 'n3p').")
    parser.add_argument("--nature", type=str, choices=["V", "R", "M"], help="Filter by nature: Valence (V), Rydberg (R), Mixed (M).")
    parser.add_argument("--safe-only", action="store_true", help="Include only safe excitations.")

    args = parser.parse_args()

    print_summary_table(
        input_path=args.input_path,
        filter_spin=args.spin,
        filter_state=args.state,
        filter_type=args.type,
        filter_nature=args.nature,
        safe_only=args.safe_only
    )
