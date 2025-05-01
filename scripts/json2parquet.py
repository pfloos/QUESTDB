
import pandas as pd
import sys
from pathlib import Path

def convert_json_to_parquet(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for json_file in directory.glob("*.json"):
        try:
            df = pd.read_json(json_file)
            parquet_file = json_file.with_suffix(".parquet")
            df.to_parquet(parquet_file, engine="pyarrow")
            print(f"Converted: {json_file.name} -> {parquet_file.name}")
        except Exception as e:
            print(f"Failed to convert {json_file.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_json_to_parquet.py <directory_path>")
    else:
        convert_json_to_parquet(sys.argv[1])
