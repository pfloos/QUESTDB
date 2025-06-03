import os
import json
from docx import Document
import argparse

def extract_tables_from_docx(docx_path):
    """
    Extracts all tables from a DOCX file, assuming the first row of each table is the header.
    Returns a list of tables, each as a dictionary with headers and rows.
    """
    document = Document(docx_path)
    tables_json = []

    for table_index, table in enumerate(document.tables):
        rows = table.rows
        if not rows:
            continue

        # Extract header row
        headers = [cell.text.strip() for cell in rows[0].cells]
        data_rows = []

        # Extract data rows
        for row in rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            row_dict = {headers[i]: cells[i] if i < len(cells) else "" for i in range(len(headers))}
            data_rows.append(row_dict)

        tables_json.append({
            "table_index": table_index,
            "headers": headers,
            "data": data_rows
        })

    return tables_json

def process_directory(input_dir, output_dir):
    """
    Processes all .docx files in a directory and saves corresponding .json files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".docx"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base_name + ".json")

            try:
                tables = extract_tables_from_docx(input_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(tables, f, ensure_ascii=False, indent=2)
                print(f"✅ Converted: {filename} → {output_path}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all DOCX tables in a directory to JSON.")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing .docx files")
    parser.add_argument("--output-dir", "-o", default=".", help="Directory to save .json files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
