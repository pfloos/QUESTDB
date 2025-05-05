import os
import json
import random
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict

# === Config ===
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5
CATEGORY_KEYS = ["S/T", "V/R", "Type"]
REFERENCE_KEY = "TBE/AVTZ"
SKIP_KEYS = {
    "TBE (unc.)", "Special ?", "Safe ? (~50 meV)", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]", "Size", "S/T", "Group", "TBE/AVTZ"
}

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
                if filters.get("only_ppi") and d.get("Type") != "ππ*":
                    continue
                if filters.get("only_npi") and d.get("Type") != "nπ*":
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

def tournament_selection(population, fitnesses):
    winners = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
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

def mutate(subset, pool, size):
    new_subset = subset.copy()
    if random.random() < MUTATION_RATE:
        idx = random.randrange(len(new_subset))
        replacement = random.choice(pool)
        replacement_key = (replacement["file"], replacement["index"])
        if replacement_key not in [(e["file"], e["index"]) for e in new_subset]:
            new_subset[idx] = replacement
    return new_subset

def genetic_algorithm(entries: List[Dict], target_size: int, filters: dict):
    print(f"\n🔧 Optimizing subset of {target_size} excitations...")
    if not entries:
        print("❌ No data available after filtering. Please adjust your filters.")
        exit(1)
    full_stats = compute_stats(entries)
    population = [random.sample(entries, target_size) for _ in range(POPULATION_SIZE)]
    fitnesses = [compute_fitness(ind, full_stats) for ind in population]

    for gen in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2, target_size)
            child = mutate(child, entries, target_size)
            new_population.append(child)
        population = new_population
        fitnesses = [compute_fitness(ind, full_stats) for ind in population]
        print(f"  Gen {gen+1:3d}/{GENERATIONS}, Best fitness: {min(fitnesses):.6f}")

    best_subset = population[np.argmin(fitnesses)]
    return best_subset, compute_stats(best_subset), full_stats

def save_results(subset: List[Dict], output_json):
    result = [e["full"] for e in subset]
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved {len(subset)}-entry subset to {output_json}")

def compare_stats(name: str, subset_stats, full_stats):
    print(f"\n📊 Statistics for subset: {name}")
    header = f"{'Method':15s} | {'MAE':^23s} | {'MSE':^23s} | {'RMSE':^23s} | {'N':^13s}"
    divider = "-" * len(header)
    print(header)
    print(divider)
    for method in sorted(full_stats):
        fs = full_stats[method]
        ss = subset_stats.get(method, {"MAE": 0, "MSE": 0, "RMSE": 0, "Count": 0})
        print(f"{method:15s} | "
              f"{ss['MAE']:.4f} / {fs['MAE']:.4f}".center(23) + " | "
              f"{ss['MSE']:.4f} / {fs['MSE']:.4f}".center(23) + " | "
              f"{ss['RMSE']:.4f} / {fs['RMSE']:.4f}".center(23) + " | "
              f"{ss['Count']:4d} / {fs['Count']:4d}".center(13))
    print(divider)

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize subsets of excitation energies.")
    parser.add_argument("json_dir", help="Path to directory containing .json files")
    parser.add_argument("--size", type=int, required=True, help="Target subset size")

    parser.add_argument("--only-singlet", action="store_true", help="Only include singlet transitions")
    parser.add_argument("--only-triplet", action="store_true", help="Only include triplet transitions")
    parser.add_argument("--only-valence", action="store_true", help="Only include valence transitions")
    parser.add_argument("--only-rydberg", action="store_true", help="Only include Rydberg transitions")
    parser.add_argument("--only-ppi", action="store_true", help="Only include π→π* transitions")
    parser.add_argument("--only-npi", action="store_true", help="Only include n→π* transitions")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum molecule size")
    parser.add_argument("--max-size", type=int, default=1_000, help="Maximum molecule size")
    parser.add_argument("--allow-gd", action="store_true", help="Allow genuine double excitations")
    parser.add_argument("--no-safe-filter", dest="safe_only", action="store_false", help="Do NOT restrict to safe transitions")

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
        print(f"❌ Error: '{args.json_dir}' is not a valid directory.")
        exit(1)

    data = load_data(args.json_dir, filters)
    print(f"📂 Loaded {len(data)} total excitations from {args.json_dir}")

    subset, subset_stats, full_stats = genetic_algorithm(data, target_size=args.size, filters=filters)
    save_results(subset, f"subset_{args.size}.json")
    compare_stats(f"{args.size} Excitations", subset_stats, full_stats)
