# 🧪 QUESTDB Utility Scripts

This directory contains a collection of Python scripts for processing and analyzing the data in this repository. They support tasks such as converting geometries, transforming coordinate units, manipulating structured datasets, and optimizing representative subsets of excitation energies.

---

## 📜 Available Scripts

### 🔹 `txt2xyz.py`
Converts molecular geometries written in a LaTeX `.tex` file to individual `.xyz` files.

---

### 🔹 `bohr2angstrom.py`
Converts atomic coordinates in `.xyz` files from **Bohr** (atomic units) to **Ångström**.

---

### 🔹 `xlsx2json.py`
Converts `.xlsx` Excel sheets (e.g. `QUEST-All.xlsx`) to per-molecule `.json` files.

**Usage:**
```bash
usage: xlsx2json.py --excel-file EXCEL_FILE [--output-dir OUTPUT_DIR]
                    [--skip-sheets SHEET1 SHEET2 ...] [--only-sheets SHEET1 SHEET2 ...]

options:
  -h, --help                    Show this help message and exit
  --excel-file EXCEL_FILE      Path to Excel file (e.g., QUEST-All.xlsx)
  --output-dir OUTPUT_DIR      Directory to save JSON files (default: ./json)
  --skip-sheets SHEET [...]    Names of sheets to skip
  --only-sheets SHEET [...]    Only process these sheets (overrides skip)
```

---

### 🔹 `cleanup_json.py`
Cleans the JSON files by removing unwanted characters or formatting inconsistencies.

---

### 🔹 `json2parquet.py`
Converts `.json` excitation data files to Apache Parquet format for efficient storage and fast processing.

---

### 🔹 `json2csv.py`
Converts `.json` excitation data files to flat `.csv` files.

---

### 🔹 `analyze_json.py`
Analyzes a set of `.json` files and extracts metadata about the types of excitations they contain.

**Usage:**
```bash
usage: analyze_json.py [-h] files [files ...]

Analyze QUEST excitation set JSON files.

positional arguments:
  files       One or more JSON files or directories containing QUEST-style excitation data.

options:
  -h, --help  show this help message and exit
```

---

### 🔹 `quest_diet.py`
The main script to create a “diet” subset of excitations from the QUEST database. The goal is to preserve the statistical properties of the full dataset using a genetic algorithm.

**Usage:**
```bash
usage: quest_diet.py [-h] --size SIZE [--optimize-params] [--only-singlet] [--only-doublet] [--only-triplet] [--only-quartet] [--only-valence]
                     [--only-rydberg] [--only-ppi] [--only-npi] [--min-size MIN_SIZE] [--max-size MAX_SIZE] [--allow-gd] [--allow-unsafe]
                     [--max-molecules MAX_MOLECULES]
                     json_dir

QUEST diet: subsets of excitations with same statistics!

positional arguments:
  json_dir              Path to directory containing .json files

options:
  -h, --help            show this help message and exit
  --size SIZE           Target subset size
  --optimize-params     Use Optuna to optimize GA parameters
  --only-singlet        Only include singlet transitions
  --only-doublet        Only include doublet transitions
  --only-triplet        Only include triplet transitions
  --only-quartet        Only include quartet transitions
  --only-valence        Only include valence transitions
  --only-rydberg        Only include Rydberg transitions
  --only-ppi            Only include π→π* transitions
  --only-npi            Only include n→π* transitions
  --min-size MIN_SIZE   Minimum molecule size
  --max-size MAX_SIZE   Maximum molecule size
  --allow-gd            Allow genuine double excitations
  --allow-unsafe        Allow unsafe transitions
  --max-molecules MAX_MOLECULES
                        Maximum number of distinct molecules to include in subset
```

---

### 🔹 `statistics.py`
Performs statistical analysis and generates error distribution plots for excitation energies across different methods.

**Usage:**
```bash
usage: statistics.py [-h] [--min-size MIN_SIZE] [--max-size MAX_SIZE] [--only-singlet] [--only-doublet] [--only-triplet] [--only-quartet]
                     [--only-valence] [--only-rydberg] [--only-ppi] [--only-npi] [--allow-unsafe] [--print-graphs]
                     json_input

Analyze excitation energy errors and create plots.

positional arguments:
  json_input           JSON file or directory containing .json files

options:
  -h, --help           show this help message and exit
  --min-size MIN_SIZE  Minimum molecule size
  --max-size MAX_SIZE  Maximum molecule size
  --only-singlet       Only include singlet transitions
  --only-doublet       Only include doublet transitions
  --only-triplet       Only include triplet transitions
  --only-quartet       Only include quartet transitions
  --only-valence       Only include valence transitions
  --only-rydberg       Only include Rydberg transitions
  --only-ppi           Only include π→π* transitions
  --only-npi           Only include n→π* transitions
  --allow-unsafe       Allow unsafe transitions
  --print-graphs       Print error distribution graphs
```

---

### 🔹 `filter_json.py`
Filters a subset of excitations based on user-specified criteria and outputs a new combined `.json` file.

**Usage:**
```bash
usage: filter_json.py [-h] [--spin {1,2,3,4}] [--nature {V,R,M}] [--safe {Y,N}] [--group GROUP] [--type TYPE] [--exclude-gd] [--min-size MIN_SIZE]
                      [--max-size MAX_SIZE]
                      input_files output_file

✨ Fancy filter for QUEST excitations.

positional arguments:
  input_files          Path to directory or single .json file
  output_file          Path to output JSON file

options:
  -h, --help           show this help message and exit
  --spin {1,2,3,4}     1 for singlet, 2 for doublet, 3 for triplet, 4 for quartet
  --nature {V,R,M}     'V' for valence, 'R' for Rydberg, 'M' for mixed
  --safe {Y,N}         'Y' = safe, 'N' = unsafe
  --group GROUP        Comma-separated list of Group numbers (12, 35, 69, 1016)
  --type TYPE          Comma-separated list of excitation types (e.g., npi,ppi,n3s)
  --exclude-gd         Exclude genuine double excitations ('GD')
  --min-size MIN_SIZE  Minimum molecule size to include
  --max-size MAX_SIZE  Maximum molecule size to include
```

---

### 🔹 `printing_excitations.py`
Print the various excited states and their corresponding characteristics gathered from a set of `.json` files or a single `.json` file.

```bash
usage: print_excitations.py [-h] [--spin {1,2,3,4}] [--state STATE] [--type TYPE] [--nature {V,R,M}] [--safe-only] input_path

Print a table of excited state characteristics from JSON file(s).

positional arguments:
  input_path        Path to a JSON file or directory.

options:
  -h, --help        show this help message and exit
  --spin {1,2,3,4}  Filter by spin (1, 2, 3, 4).
  --state STATE     Filter by exact state label (e.g., '^1A_1').
  --type TYPE       Filter by excitation type (e.g., 'n3p').
  --nature {V,R,M}  Filter by nature: Valence (V), Rydberg (R), Mixed (M).
  --safe-only       Include only safe excitations.
```

---

### 🔹 `extract_lowest_states.py`
Extract the lowest-energy states from a set of `.json` files (one per file).

```bash
usage: extract_lowest_states.py [-h] --spin SPIN [--vr {V,R,M}] [-o OUTPUT] directory

Extract lowest-energy state per JSON file for a given spin and excitation type.

positional arguments:
  directory            Directory containing JSON files

options:
  -h, --help           show this help message and exit
  --spin SPIN          Target spin (e.g., 1 for singlet, 2 for doublet, 3 for triplet, 4 for quartet)
  --vr {V,R,M}         Filter by excitation type: valence (V), rydberg (R), or mixed (M)
  -o, --output OUTPUT  Output JSON filename
```

---
