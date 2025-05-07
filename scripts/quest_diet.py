import os
import json
import random
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

import optuna

# === Config Defaults ===
DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 100
DEFAULT_MUTATION_RATE = 0.2
DEFAULT_TOURNAMENT_SIZE = 5

CATEGORY_KEYS = ["S/T", "V/R", "Type"]
REFERENCE_KEY = "TBE/AVTZ"
SKIP_KEYS = {
    "CASSCF", "CASPT2", "CASPT3", "SC-NEVPT2", "PC-NEVPT2",
    "Special ?", "Safe ? (~50 meV)", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]", "Size", "S/T", "V/R", "Group", "TBE/AVTZ"
}

console = Console()

def load_data(json_dir: str, filters: dict) -> List[Dict]:
    entries = []
    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file)) as f:
            data = json.load(f)
            for i, d in enumerate(data):
                if not filters.get("allow_gd", False) and d.get("Type") == "dou":
                    continue
                if filters.get("safe_only", True) and d.get("Safe ? (~50 meV)") != "Y":
                    continue
                if filters.get("only_singlet") and d.get("S/T") != 1:
                    continue
                if filters.get("only_triplet") and d.get("S/T") != 3:
                    continue
                if filters.get("only_valence") and d.get("V/R") != "V":
                    continue
                if filters.get("only_rydberg") and d.get("V/R") != "R":
                    continue
                if filters.get("only_ppi") and d.get("Type") != "ppi":
                    continue
                if filters.get("only_npi") and d.get("Type") != "npi":
                    continue
                if "Size" in d:
                    if d["Size"] < filters.get("min_size", 0) or d["Size"] > filters.get("max_size", 1000):
                        continue
                ref = d.get(REFERENCE_KEY)
                if ref is None:
                    continue
                values = {
                    k: v for k, v in d.items()
                    if k not in SKIP_KEYS and isinstance(v, (int, float))
                }
                if not values:
                    continue
                categories = {k: d[k] for k in CATEGORY_KEYS if k in d}
                entries.append({
                    "data": values,
                    "ref": ref,
                    "index": i,
                    "file": file,
                    "categories": categories,
                    "full": d
                })
    return entries

def compute_stats(entries: List[Dict]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for method in entries[0]["data"].keys():
        errors = [e["data"][method] - e["ref"] for e in entries if method in e["data"]]
        if errors:
            stats[method] = {
                "MAE": np.mean(np.abs(errors)),
                "MSE": np.mean(errors),
                "RMSE": np.sqrt(np.mean(np.square(errors))),
                "Count": len(errors)
            }
    return stats

def compute_fitness(subset: List[Dict], full_stats: Dict[str, Dict[str, float]]) -> float:
    subset_stats = compute_stats(subset)
    score = 0.0
    for method in full_stats:
        if method not in subset_stats:
            score += 100.0
            continue
        for metric in ["MAE", "MSE", "RMSE"]:
            diff = abs(subset_stats[method][metric] - full_stats[method][metric])
            score += diff
    return score

def tournament_selection(population, fitnesses, tournament_size):
    winners = random.sample(list(zip(population, fitnesses)), tournament_size)
    winners.sort(key=lambda x: x[1])
    return winners[0][0]


def crossover(parent1, parent2, size, max_molecules=None):
    combined = []
    seen = set()

    # If we have a molecule limit, we need to ensure we don't exceed it
    if max_molecules is not None:
        # Get all unique molecules from both parents
        parent1_molecules = {e["full"]["Molecule"] for e in parent1}
        parent2_molecules = {e["full"]["Molecule"] for e in parent2}
        all_molecules = parent1_molecules.union(parent2_molecules)

        # If combining would exceed the limit, we need to restrict
        if len(all_molecules) > max_molecules:
            # Choose which parent's molecules to keep
            if random.random() < 0.5:
                keep_molecules = parent1_molecules
            else:
                keep_molecules = parent2_molecules

            # Only include entries from the chosen molecules
            for item in parent1 + parent2:
                if item["full"]["Molecule"] in keep_molecules:
                    key = (item["file"], item["index"])
                    if key not in seen:
                        seen.add(key)
                        combined.append(item)
        else:
            # Normal crossover if we're under the limit
            for item in parent1 + parent2:
                key = (item["file"], item["index"])
                if key not in seen:
                    seen.add(key)
                    combined.append(item)
    else:
        # Original behavior if no molecule limit
        for item in parent1 + parent2:
            key = (item["file"], item["index"])
            if key not in seen:
                seen.add(key)
                combined.append(item)

    random.shuffle(combined)
    return combined[:size]

def mutate(subset, pool, size, mutation_rate, max_molecules=None):
    new_subset = subset.copy()
    if random.random() < mutation_rate:
        idx = random.randrange(len(new_subset))
        replacement = random.choice(pool)

        # Check if we're limiting molecules
        if max_molecules is not None:
            current_molecules = {e["full"]["Molecule"] for e in new_subset}
            new_molecule = replacement["full"]["Molecule"]

            # If adding this molecule would exceed our limit and it's a new molecule
            if new_molecule not in current_molecules and len(current_molecules) >= max_molecules:
                # Only allow replacements from existing molecules
                valid_replacements = [e for e in pool if e["full"]["Molecule"] in current_molecules]
                if valid_replacements:
                    replacement = random.choice(valid_replacements)

        replacement_key = (replacement["file"], replacement["index"])
        if replacement_key not in [(e["file"], e["index"]) for e in new_subset]:
            new_subset[idx] = replacement
    return new_subset

def genetic_algorithm(entries: List[Dict], target_size: int, filters: dict, generations, pop_size, mutation_rate, tournament_size, max_molecules=None):
    full_stats = compute_stats(entries)

    # Create initial population respecting molecule limit if specified
    if max_molecules is not None:
        population = []
        for _ in range(pop_size):
            # Get all unique molecules
            all_molecules = list({e["full"]["Molecule"] for e in entries})
            # Select random subset of molecules
            selected_molecules = random.sample(all_molecules, min(max_molecules, len(all_molecules)))
            # Get entries only from selected molecules
            molecule_entries = [e for e in entries if e["full"]["Molecule"] in selected_molecules]
            # Create individual from these entries
            population.append(random.sample(molecule_entries, min(target_size, len(molecule_entries))))
    else:
        population = [random.sample(entries, target_size) for _ in range(pop_size)]

    fitnesses = [compute_fitness(ind, full_stats) for ind in population]

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[green]{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console
    ) as progress:
        task = progress.add_task("[cyan]Running generations...", total=generations)

        for gen in range(generations):
            new_population = []
            for _ in range(pop_size):
                p1 = tournament_selection(population, fitnesses, tournament_size)
                p2 = tournament_selection(population, fitnesses, tournament_size)
                child = crossover(p1, p2, target_size, max_molecules)  # Updated call
                child = mutate(child, entries, target_size, mutation_rate, max_molecules)
                new_population.append(child)
            population = new_population
            fitnesses = [compute_fitness(ind, full_stats) for ind in population]
            best_fitness = min(fitnesses)
            progress.update(task, advance=1,
                          description=f"[cyan]Gen {gen+1}/{generations} [Best: {best_fitness:.6f}]")

    best_idx = np.argmin(fitnesses)
    return population[best_idx], compute_stats(population[best_idx]), full_stats, fitnesses[best_idx]

def save_results(subset: List[Dict], output_json):
    result = [e["full"] for e in subset]
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    molecules_used = len({e["Molecule"] for e in result})
    console.print(f"\n✅ Saved {len(subset)}-entry subset using {molecules_used} molecules to [bold green]{output_json}[/]")

def compare_stats(name: str, subset_stats, full_stats):
    console.print(f"\n📊 Statistics (in eV) for subset: [bold cyan]{name}[/]\n")
    table = Table(show_header=True, header_style="bold magenta", box=box.HEAVY)
    table.add_column("Method", justify="left")
    table.add_column("MAE (sub/full)", justify="center")
    table.add_column("MSE (sub/full)", justify="center")
    table.add_column("RMSE (sub/full)", justify="center")
    table.add_column("# (sub/full)", justify="center")

    for method in sorted(full_stats):
        fs = full_stats[method]
        ss = subset_stats.get(method, {"MAE": 0, "MSE": 0, "RMSE": 0, "Count": 0})

        def color(val, ref):
            if abs(val - ref) < 0.005:
                return "bold green"
            elif abs(val - ref) > 0.05:
                return "bold red"
            else:
                return "white"

        table.add_row(
            method,
            f"[{color(ss['MAE'], fs['MAE'])}]{ss['MAE']:.4f} / {fs['MAE']:.4f}[/{color(ss['MAE'], fs['MAE'])}]",
            f"[{color(ss['MSE'], fs['MSE'])}]{ss['MSE']:.4f} / {fs['MSE']:.4f}[/{color(ss['MSE'], fs['MSE'])}]",
            f"[{color(ss['RMSE'], fs['RMSE'])}]{ss['RMSE']:.4f} / {fs['RMSE']:.4f}[/{color(ss['RMSE'], fs['RMSE'])}]",
            f"{ss['Count']} / {fs['Count']}"
        )
    console.print(table)

    # Compute max absolute deviations and the corresponding methods
    max_diffs = {}
    for metric in ["MAE", "MSE", "RMSE"]:
        max_val = 0
        max_method = "N/A"
        for method in full_stats:
            subset_val = subset_stats.get(method, {}).get(metric, 0)
            full_val = full_stats[method][metric]
            diff = abs(subset_val - full_val)
            if diff > max_val:
                max_val = diff
                max_method = method
        max_diffs[metric] = (max_val, max_method)

    # Print max deviations with rich formatting
    console.print("\n📈 [bold]Maximum absolute deviations:[/]")
    for metric, (val, method) in max_diffs.items():
        color = "green" if val < 0.01 else "yellow" if val < 0.05 else "red"
        console.print(f"• {metric}: [{color}]{val:.4f} eV[/{color}]  ([bold]{method}[/])")
    console.print("")

def run_optimization(data, target_size, filters, n_trials=30):
    def objective(trial):
        pop_size = trial.suggest_int("population_size", 30, 150)
        mutation_rate = trial.suggest_float("mutation_rate", 0.05, 0.5)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)

        _, _, _, fitness = genetic_algorithm(
            entries=data,
            target_size=target_size,
            filters=filters,
            generations=30,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size
        )
        return fitness

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    console.print(f"\n🎯 Best hyperparameters: {study.best_params}", style="bold green")
    return study.best_params

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUEST diet: subsets of excitations with same statistics!")
    parser.add_argument("json_dir", help="Path to directory containing .json files")
    parser.add_argument("--size", type=int, required=True, help="Target subset size")
    parser.add_argument("--optimize-params", action="store_true", help="Use Optuna to optimize GA parameters")

    parser.add_argument("--only-singlet", action="store_true", help="Only include singlet transitions")
    parser.add_argument("--only-triplet", action="store_true", help="Only include triplet transitions")
    parser.add_argument("--only-valence", action="store_true", help="Only include valence transitions")
    parser.add_argument("--only-rydberg", action="store_true", help="Only include Rydberg transitions")
    parser.add_argument("--only-ppi", action="store_true", help="Only include π→π* transitions")
    parser.add_argument("--only-npi", action="store_true", help="Only include n→π* transitions")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum molecule size")
    parser.add_argument("--max-size", type=int, default=1_000, help="Maximum molecule size")
    parser.add_argument("--allow-gd", action="store_true", help="Allow genuine double excitations")
    parser.add_argument("--allow-unsafe", dest="safe_only", action="store_false", help="Allow unsafe transitions")
    parser.add_argument("--max-molecules", type=int, default=None, help="Maximum number of distinct molecules to include in subset")

    args = parser.parse_args()
    filters = {
        "only_singlet": args.only_singlet,
        "only_triplet": args.only_triplet,
        "only_valence": args.only_valence,
        "only_rydberg": args.only_rydberg,
        "only_ppi": args.only_ppi,
        "only_npi": args.only_npi,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "allow_gd": args.allow_gd,
        "safe_only": args.safe_only
    }

    if not os.path.isdir(args.json_dir):
        console.print(f"❌ Error: '{args.json_dir}' is not a valid directory.", style="bold red")
        exit(1)

    data = load_data(args.json_dir, filters)
    console.print(f"📂 Loaded {len(data)} total excitations from {args.json_dir}", style="green")

    if args.optimize_params:
        best = run_optimization(data, args.size, filters)
        pop_size = best["population_size"]
        mutation_rate = best["mutation_rate"]
        tournament_size = best["tournament_size"]
    else:
        pop_size = DEFAULT_POPULATION_SIZE
        mutation_rate = DEFAULT_MUTATION_RATE
        tournament_size = DEFAULT_TOURNAMENT_SIZE

    subset, subset_stats, full_stats, _ = genetic_algorithm(
        data,
        target_size=args.size,
        filters=filters,
        generations=DEFAULT_GENERATIONS,
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        max_molecules=args.max_molecules  
    )
    save_results(subset, f"diet_subset_{args.size}.json")
    compare_stats(f"{args.size} Excitations", subset_stats, full_stats)
