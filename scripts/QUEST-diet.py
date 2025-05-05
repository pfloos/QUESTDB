import os
import json
import numpy as np
import random
from collections import defaultdict

# === CONFIGURATION ===
JSON_DIR = "../data/json/MAIN"  # Folder containing all your JSON files
OUTPUT_DIR = "output_subsets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZES = [50, 100, 200]
REFERENCE_KEY = "TBE/AVTZ"
SKIP_KEYS = {
    "TBE (unc.)", "Special ?", "Safe ? (~50 meV)", "TBE/AVQZ", "Molecule", "State", "Method",
    "Corr. Method", "%T1 [CC3/AVTZ]", "f [LR-CC3/AVTZ]", "Size", "Group", "S/T", "V/R", "Type"
}

# === LOAD JSON FILES ===
def load_data(json_dir):
    all_entries = []
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname)) as f:
                data = json.load(f)
                for i, entry in enumerate(data):
                    ref = entry.get(REFERENCE_KEY)
                    if ref is None:
                        continue
                    filtered = {
                        k: v for k, v in entry.items()
                        if k not in SKIP_KEYS and isinstance(v, (float, int))
                    }
                    if filtered:
                        filtered["TBE/AVTZ"] = ref
                        filtered["molecule"] = fname.replace(".json", "")
                        filtered["index"] = i
                        all_entries.append(filtered)
    return all_entries

# === ERROR STATISTICS ===
def compute_error_stats(entries):
    errors = defaultdict(list)
    for entry in entries:
        ref = entry["TBE/AVTZ"]
        for method, value in entry.items():
            if method in {"TBE/AVTZ", "molecule", "index"}:
                continue
            errors[method].append(value - ref)

    stats = {}
    for method, err_list in errors.items():
        err_array = np.array(err_list)
        stats[method] = {
            "Count": len(err_array),
            "MAE": np.mean(np.abs(err_array)),
            "MSE": np.mean(err_array),
            "RMSE": np.sqrt(np.mean(err_array**2)),
            "MaxErr": np.max(np.abs(err_array)),
            "MinErr": np.min(err_array)
        }
    return stats

# === STRATIFIED SAMPLING ===
def create_subset(entries, target_size):
    # Sample a few transitions per molecule
    by_molecule = defaultdict(list)
    for e in entries:
        by_molecule[e["molecule"]].append(e)

    subset = []
    total = sum(len(v) for v in by_molecule.values())
    prop = target_size / total

    for mol, e_list in by_molecule.items():
        k = max(1, round(len(e_list) * prop))
        subset.extend(random.sample(e_list, min(k, len(e_list))))

    if len(subset) > target_size:
        subset = random.sample(subset, target_size)
    return subset

# === OUTPUT JSON FILE ===
def save_subset(subset, size):
    path = os.path.join(OUTPUT_DIR, f"subset_{size}.json")
    cleaned = []
    for entry in subset:
        new_entry = {
            k: v for k, v in entry.items()
            if k in {"TBE/AVTZ", "molecule", "index"} or isinstance(v, (float, int))
        }
        cleaned.append(new_entry)
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)

# === MAIN ROUTINE ===
def run_analysis():
    full_data = load_data(JSON_DIR)
    print(f"\nLoaded {len(full_data)} excitations from {JSON_DIR}")

    full_stats = compute_error_stats(full_data)
    print(f"\n[Full Dataset Statistics]")
    for method, s in full_stats.items():
        print(f"  {method:8}: MAE={s['MAE']:.3f}, MSE={s['MSE']:.3f}, RMSE={s['RMSE']:.3f}, Count={s['Count']}")

    for size in TARGET_SIZES:
        subset = create_subset(full_data, size)
        subset_stats = compute_error_stats(subset)

        print(f"\n[Subset of ~{size} excitations] Actual size: {len(subset)}")
        for method in subset_stats:
            fs = full_stats.get(method, {})
            ss = subset_stats[method]
            print(f"  {method:8}: MAE={ss['MAE']:.3f} (full: {fs.get('MAE', 0):.3f}), "
                  f"MSE={ss['MSE']:.3f} (full: {fs.get('MSE', 0):.3f}), "
                  f"n={ss['Count']}")
        save_subset(subset, size)
        print(f"  → Subset saved to: subset_{size}.json")

if __name__ == "__main__":
    run_analysis()
