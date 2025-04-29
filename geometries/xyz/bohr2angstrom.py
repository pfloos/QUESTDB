import os

BOHR_TO_ANGSTROM = 0.529177

def convert_xyz_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        print(f"File {filepath} does not have enough lines to be a valid XYZ file.")
        return

    converted_lines = []
    for line in lines[2:]:  # Skip first two lines
        parts = line.strip().split()
        if len(parts) == 4:
            atom, x, y, z = parts
            try:
                x = float(x) * BOHR_TO_ANGSTROM
                y = float(y) * BOHR_TO_ANGSTROM
                z = float(z) * BOHR_TO_ANGSTROM
                converted_lines.append(f"{atom:2} {x:.8f} {y:.8f} {z:.8f}\n")
            except ValueError:
                print(f"Skipping line in {filepath}: {line}")
        else:
            print(f"Skipping malformed line in {filepath}: {line}")

    # Overwrite file
    with open(filepath, "w") as f:
        f.write(lines[0])  # Atom count
        f.write(lines[1])  # Comment line
        f.writelines(converted_lines)

def convert_all_xyz_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".xyz"):
            filepath = os.path.join(directory, filename)
            print(f"Converting {filepath}")
            convert_xyz_file(filepath)

# Change this to the correct directory
convert_all_xyz_in_directory("/Users/loos/Work/QUESTDB/geometries/xyz")
