import json
import math
import os
import sys

def is_empty(value):
    """Check if a value is empty, null, or NaN."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in ["", "nan", "null"]
    return False

def should_remove_entry(entry, keep_key="Molecule"):
    """Return True if only 'Molecule' is meaningful and all other fields are empty."""
    has_molecule = keep_key in entry and not is_empty(entry[keep_key])
    other_non_empty = any(k != keep_key and not is_empty(v) for k, v in entry.items())
    return has_molecule and not other_non_empty

def clean_entry(entry):
    """Remove individual empty fields."""
    return {k: v for k, v in entry.items() if not is_empty(k) and not is_empty(v)}

def clean_json_file(filepath, keep_key="Molecule"):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {filepath}")
            return

    if not isinstance(data, list):
        print(f"Skipping non-list JSON file: {filepath}")
        return

    cleaned_data = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if should_remove_entry(entry, keep_key):
            print(f"Removed entry from {filepath}:")
            print(json.dumps(entry, indent=2))
            continue  # Skip entry
        cleaned_data.append(clean_entry(entry))

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)

def clean_all_json_files_in_directory(directory, keep_key="Molecule"):
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            print(f"\nCleaning {filepath}...")
            clean_json_file(filepath, keep_key)

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cleanup_json.py /path/to/json_directory")
        sys.exit(1)

    target_directory = sys.argv[1]
    clean_all_json_files_in_directory(target_directory)
