def split_entries(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split on two or more newlines to separate entries
    blocks = [block.strip() for block in content.strip().split('\n\n') if block.strip()]
    return blocks

def write_xyz_files(blocks):
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        title_line = lines[0]
        filename = title_line.split()[0] + '.xyz'
        coordinates = lines[1:]

        with open(filename, 'w') as f:
            f.write(f"{len(coordinates)}\n")           # Line 1: number of atoms
            f.write(f"{title_line}\n")                # Line 2: full metadata line
            for line in coordinates:                  # Remaining lines: atom coordinates
                f.write(f"{line}\n")

if __name__ == "__main__":
    input_file = "formatted_geom.txt"  # Ensure the file is in the same directory
    entries = split_entries(input_file)
    write_xyz_files(entries)
