import pandas as pd
import os
import json
import re

# -------- CONFIGURATION --------
excel_file = "QUEST-All.xlsx"  # Your Excel file
output_base_dir = "json_by_molecule"  # Output folder
# --------------------------------

# Load all sheets
xls = pd.ExcelFile(excel_file)
sheet_names = xls.sheet_names

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Helper: Sanitize filenames
def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', str(name).strip())

# Process each sheet
for sheet in sheet_names:
    df = xls.parse(sheet)

    # Use the first row as the header
    df.columns = df.iloc[0]
    df = df[1:].copy()

    if "Molecule" not in df.columns:
        print(f"Skipping sheet '{sheet}' – no 'Molecule' column found.")
        continue

    df["Molecule"] = df["Molecule"].ffill()

    # Output folder for the sheet
    sheet_dir = os.path.join(output_base_dir, sheet)
    os.makedirs(sheet_dir, exist_ok=True)

    # Group and export JSON
    for molecule, group in df.groupby("Molecule"):
        molecule_name = sanitize_filename(molecule)
        records = group.dropna(axis=1, how='all').to_dict(orient="records")

        output_path = os.path.join(sheet_dir, f"{molecule_name}.json")
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

print("✅ Done! JSON files saved in:", output_base_dir)
