import pandas as pd
import os
import json
import re
import unicodedata
import argparse
from pathlib import Path

def clean_text(text):
    """Replace problematic Unicode characters and normalize."""
    if not isinstance(text, str):
        return text
    replacements = {
        '\u2013': '-', '\u2014': '-',
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\xa0': ' ',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return unicodedata.normalize('NFKC', text)

def sanitize_filename(name):
    """Make string safe for filenames."""
    return re.sub(r'[^\w\-_.]', '_', str(name).strip())

def deduplicate_columns(df):
    """Remove duplicate column names (keep only first)."""
    seen = set()
    new_cols = []
    for col in df.columns:
        if col not in seen:
            new_cols.append(col)
            seen.add(col)
        else:
            new_cols.append(None)
    df.columns = new_cols
    return df.loc[:, df.columns.notnull()]

def process_excel_file(excel_path, output_dir, skip_sheets=None, only_sheets=None):
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    os.makedirs(output_dir, exist_ok=True)
    total_written = 0

    for sheet in sheet_names:
        if skip_sheets and sheet in skip_sheets:
            print(f"⏭ Skipping sheet: {sheet}")
            continue
        if only_sheets and sheet not in only_sheets:
            continue

        print(f"📄 Processing sheet: {sheet}")
        df = xls.parse(sheet)

        # Clean and prepare header
        df.columns = [clean_text(col) for col in df.iloc[0]]
        df = df[1:].copy()

        if "Molecule" not in df.columns:
            print(f"⚠️  Skipping sheet '{sheet}' – no 'Molecule' column found.")
            continue

        df = deduplicate_columns(df)
        df["Molecule"] = df["Molecule"].apply(clean_text).ffill()

        sheet_dir = Path(output_dir) / sheet
        sheet_dir.mkdir(parents=True, exist_ok=True)

        for molecule, group in df.groupby("Molecule"):
            molecule_name = sanitize_filename(molecule)
            clean_group = group.dropna(axis=1, how='all').applymap(clean_text)
            records = clean_group.to_dict(orient="records")

            output_path = sheet_dir / f"{molecule_name}.json"
            with open(output_path, "w") as f:
                json.dump(records, f, indent=2)

            print(f"  ✅ Saved: {output_path.name}")
            total_written += 1

    print(f"\n🎉 Done! {total_written} JSON files written to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert Excel sheets to per-molecule JSON files.")
    parser.add_argument("--excel-file", required=True, help="Path to Excel file (e.g., QUEST-All.xlsx)")
    parser.add_argument("--output-dir", default="json", help="Directory to save JSON files (default: ./json)")
    parser.add_argument("--skip-sheets", nargs="+", help="Sheets to skip")
    parser.add_argument("--only-sheets", nargs="+", help="Only process these sheets (overrides skip)")

    args = parser.parse_args()

    process_excel_file(
        excel_path=args.excel_file,
        output_dir=args.output_dir,
        skip_sheets=args.skip_sheets,
        only_sheets=args.only_sheets
    )

if __name__ == "__main__":
    main()
