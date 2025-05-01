
import pandas as pd
import sys
from pathlib import Path

def convert_json_to_csv(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for json_file in directory.glob("*.json"):
        try:
            df = pd.read_json(json_file)
            csv_file = json_file.with_suffix(".csv")
            df.to_csv(csv_file, index=False)
            print(f"Converted: {json_file.name} -> {csv_file.name}")
        except Exception as e:
            print(f"Failed to convert {json_file.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_json_to_csv.py <directory_path>")
    else:
        convert_json_to_csv(sys.argv[1])
