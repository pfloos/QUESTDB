import pandas as pd
import os
import json
import re
import unicodedata

# -------- CONFIGURATION --------
excel_file = "xlsx/QUEST-All.xlsx"  # Your Excel file
output_base_dir = "json"  # Output folder
# --------------------------------

# Load all sheets
xls = pd.ExcelFile(excel_file)
sheet_names = xls.sheet_names

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Helper: Replace special unicode characters with common equivalents
def clean_text(text):
    if not isinstance(text, str):
        return text
    replacements = {
        '\u2013': '-',    # en dash
        '\u2014': '-',    # em dash
        '\u2018': "'",    # left single quote
        '\u2019': "'",    # right single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\xa0': ' ',      # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return unicodedata.normalize('NFKC', text)

# Helper: Sanitize filenames
def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', str(name).strip())

# Helper: Drop duplicate column names, keeping only the first
def deduplicate_columns(df):
    seen = set()
    new_cols = []
    for col in df.columns:
        if col not in seen:
            new_cols.append(col)
            seen.add(col)
        else:
            new_cols.append(None)  # Mark duplicates for removal
    df.columns = new_cols
    df = df.loc[:, df.columns.notnull()]
    return df

# Main loop over all sheets
for sheet in sheet_names:
    df = xls.parse(sheet)

    # Clean header row
    df.columns = [clean_text(col) for col in df.iloc[0]]
    df = df[1:].copy()

    if "Molecule" not in df.columns:
        print(f"Skipping sheet '{sheet}' – no 'Molecule' column found.")
        continue

    # Deduplicate columns
    df = deduplicate_columns(df)

    # Clean molecule names and forward-fill
    df["Molecule"] = df["Molecule"].apply(clean_text).ffill()

    # Output folder per sheet
    sheet_dir = os.path.join(output_base_dir, sheet)
    os.makedirs(sheet_dir, exist_ok=True)

    # Group and write JSON
    for molecule, group in df.groupby("Molecule"):
        molecule_name = sanitize_filename(molecule)
        clean_group = group.dropna(axis=1, how='all').applymap(clean_text)
        records = clean_group.to_dict(orient="records")

        output_path = os.path.join(sheet_dir, f"{molecule_name}.json")
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

print("✅ Done! JSON files saved in:", output_base_dir)
