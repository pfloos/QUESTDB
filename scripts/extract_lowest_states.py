import json
import argparse
from pathlib import Path

def find_lowest_energy_state(data, target_spin, vr_filter):
    """Find the lowest TBE/AVTZ energy state matching the given spin and V/R filter."""
    vr_map = {
        "valence": "V",
        "rydberg": "R",
        "mixed": "M"
    }

    filtered = [
        entry for entry in data
        if entry.get("Spin") == target_spin
        and "TBE/AVTZ" in entry
        and (vr_filter is None or entry.get("V/R") == vr_map[vr_filter])
    ]

    if not filtered:
        return None

    return min(filtered, key=lambda x: x["TBE/AVTZ"])

def main():
    parser = argparse.ArgumentParser(description="Extract lowest-energy state per JSON file for a given spin and excitation type.")
    parser.add_argument("directory", help="Directory containing JSON files")
    parser.add_argument("--spin", type=int, required=True, help="Target spin (e.g., 1 for singlet, 3 for triplet)")
    parser.add_argument("--vr", choices=["valence", "rydberg", "mixed"], help="Filter by excitation type: valence (V), rydberg (R), or mixed (M)")
    parser.add_argument("-o", "--output", default="lowest_states.json", help="Output JSON filename")
    args = parser.parse_args()

    input_dir = Path(args.directory)
    assert input_dir.is_dir(), f"Invalid directory: {input_dir}"

    results = []

    for json_file in sorted(input_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            lowest_state = find_lowest_energy_state(data, args.spin, args.vr)
            if lowest_state:
                lowest_state["Source File"] = json_file.name
                results.append(lowest_state)
        except Exception as e:
            print(f"⚠️ Error processing {json_file.name}: {e}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Extracted {len(results)} states. Saved to: {args.output}")

if __name__ == "__main__":
    main()
