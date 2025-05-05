
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
    "TBE (unc.)", "Special ?", "Safe ? (~50 meV)", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]", "Size", "S/T", "Group", "TBE/AVTZ"
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
                if not filters.get("allow_gd", False) and d.get("Type") == "GD":
                    continue
                if filters.get("safe_only", True) and d.get("Safe ? (~50 meV)") != "Y":
                    continue
                if filters.get("only_singlet") and d.get("S/T") != "S":
                    continue
                if filters.get("only_triplet") and d.get("S/T") != "T":
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

def crossover(parent1, parent2, size):
    combined = []
    seen = set()
    for item in parent1 + parent2:
        key = (item["file"], item["index"])
        if key not in seen:
            seen.add(key)
            combined.append(item)
    random.shuffle(combined)
    return combined[:size]

def mutate(subset, pool, size, mutation_rate):
    new_subset = subset.copy()
    if random.random() < mutation_rate:
        idx = random.randrange(len(new_subset))
        replacement = random.choice(pool)
        replacement_key = (replacement["file"], replacement["index"])
        if replacement_key not in [(e["file"], e["index"]) for e in new_subset]:
            new_subset[idx] = replacement
    return new_subset

def genetic_algorithm(entries: List[Dict], target_size: int, filters: dict, generations, pop_size, mutation_rate, tournament_size):
    full_stats = compute_stats(entries)
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
                child = crossover(p1, p2, target_size)
                child = mutate(child, entries, target_size, mutation_rate)
                new_population.append(child)
            population = new_population
            fitnesses = [compute_fitness(ind, full_stats) for ind in population]
            best_fitness = min(fitnesses)
            progress.update(task, advance=1, description=f"[cyan]Gen {gen+1}/{generations} [Best fitness: {best_fitness:.6f}]")

    best_subset = population[np.argmin(fitnesses)]
    return best_subset, compute_stats(best_subset), full_stats, best_fitness

def save_results(subset: List[Dict], output_json):
    result = [e["full"] for e in subset]
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"\n✅ Saved {len(subset)}-entry subset to [bold green]{output_json}[/]")

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

    parser.add_argument("--only-singlet", action="store_true")
    parser.add_argument("--only-triplet", action="store_true")
    parser.add_argument("--only-valence", action="store_true")
    parser.add_argument("--only-rydberg", action="store_true")
    parser.add_argument("--only-ppi", action="store_true")
    parser.add_argument("--only-npi", action="store_true")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=1_000)
    parser.add_argument("--allow-gd", action="store_true")
    parser.add_argument("--no-safe-filter", dest="safe_only", action="store_false")

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
        tournament_size=tournament_size
    )
    save_results(subset, f"diet_subset_{args.size}.json")
    compare_stats(f"{args.size} Excitations", subset_stats, full_stats)
