"""Normalize species, annotate transitions, and package benchmark inputs."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PAPERS = {
    "MAIN": "10.1021/acs.jctc.5c00975",
    "RAD": "10.1021/acs.jctc.5c00975",
    "BIO": "10.1021/acs.jctc.5c00975",
    "CHROM": "10.1021/acs.jctc.4c00906",
    "TM": "10.1021/acs.jctc.3c01080",
    "valence_IPs": "10.1021/acs.jctc.4c00216",
    "core_IPs": "10.1021/acs.jctc.6c00713",
    "DIPs": "10.1063/5.0250155",
}
NAMES = {
    "Ar": "Argon", "BeO": "Beryllium monoxide", "BF": "Boron monofluoride",
    "BH": "Boron monohydride", "BH2": "Boron dihydride radical", "BH3": "Borane", "BN": "Boron mononitride",
    "CH2CHF": "Vinyl fluoride", "CH3CN": "Acetonitrile", "CH3F": "Fluoromethane",
    "CH3NC": "Methyl isocyanide", "CH3OH": "Methanol", "CH4": "Methane",
    "Cl2": "Chlorine", "CS": "Carbon monosulfide", "F2": "Fluorine",
    "HCN": "Hydrogen cyanide", "HCOOH": "Formic acid", "HF": "Hydrogen fluoride",
    "HNCO": "Isocyanic acid", "HOF": "Hypofluorous acid", "LiCl": "Lithium chloride",
    "LiF": "Lithium fluoride", "Ne": "Neon", "NH2F": "Fluoramine",
    "OCS": "Carbonyl sulfide", "P2": "Diphosphorus", "PH3": "Phosphine",
    "PN": "Phosphorus mononitride", "SiH4": "Silane", "SiO": "Silicon monoxide",
    "NO": "Nitric oxide", "NO2": "Nitrogen dioxide", "OH": "Hydroxyl",
    "NH2": "Amino radical", "CH3": "Methyl radical", "HCO": "Formyl radical",
    "CN": "Cyano radical", "ClO2": "Chlorine dioxide",
}
DISPLAY_FORMULAS = {
    "hydrogen_chloride": "HCl", "HF": "HF", "LiCl": "LiCl", "LiF": "LiF",
    "SiH4": "SiH4", "PH3": "PH3", "SiO": "SiO", "CuCl": "CuCl",
}
EXCLUDED_LABELS = {"BH_ES": {"BH2"}}
FLUORESCENCE_GEOMETRIES = {"^1A_u[F]": "acetylene_ES1", "^1A_2[F]": "acetylene_ES2"}
TYPO_FIX = "20ada882a5d45c312a329a6b9a63fbc677630de1"
# The previous and corrected values and the repeated-symmetry occurrence
# identify these documented corrections without relying on a file line number.
REFERENCE_REVISIONS = {
    ("Borole.json", "^1B_1", 1): ((4.000, 4.863), None),
    ("HCO.json", "^2A'", 2): ((6.153, 6.149), (6.164, 6.160)),
    ("HCO.json", "^2A'", 3): ((6.365, 6.361), (6.375, 6.368)),
    ("HCO.json", "^4A''", 1): ((6.392, 6.374), (6.407, 6.389)),
    ("Nitromethyl.json", "^2B_2", 3): ((6.632, 6.509), None),
}


def molecular_formula(path, species):
    lines = path.read_text().splitlines()
    count = int(lines[0])
    elements = Counter(line.split()[0] for line in lines[2:2 + count])
    order = sorted(elements, key=lambda atom: ({"C": 0, "H": 1}.get(atom, 2), atom)
                   if "C" in elements else (0, atom))
    return "".join(atom + (str(elements[atom]) if elements[atom] > 1 else "") for atom in order) + (
        "+" if species.endswith("+") else ""
    )


def state_id(*parts):
    return sha256(repr(parts).encode("utf-8")).hexdigest()[:20]


def annotate(root, collections, geometries, registry):
    geometry_by_id = {g["id"]: g for g in geometries}
    by_label = defaultdict(list)
    species_index = {}
    for geometry in geometries:
        entry = registry["geometries"][geometry["id"]]
        species = entry["species"]
        # Keep the registry's species IDs so formula isomers never get merged.
        formula = molecular_formula(root / geometry["source"], species)
        molecule = species_index.setdefault(species, {
            "id": species, "name": NAMES.get(species, species[0].upper() + species[1:].replace("_", " ")),
            "formula": DISPLAY_FORMULAS.get(species, formula),
            "aliases": {formula}, "collections": set(), "geometries": [],
        })
        geometry["variant"] = entry.get("variant")
        geometry["moleculeId"] = species
        molecule["geometries"].append(geometry["id"])
        if entry.get("default") or "defaultGeometry" not in molecule:
            molecule["defaultGeometry"] = geometry["id"]
        molecule["aliases"].update([geometry["id"], *geometry["aliases"], *(label.strip() for label in geometry["labels"])])
        for group in geometry["subsets"]:
            for label in geometry["labels"]:
                by_label[group, label].append(geometry)

    for group, rows in collections.items():
        for row in rows:
            if "fields" in row:
                candidates = by_label[group, row["molecule"]]
                # An exact XYZ stem outranks a normalized collision (BH2/BH_2).
                exact = [g for g in candidates if Path(g["source"]).stem.casefold() == row["molecule"].strip().casefold()]
                candidate_species = {g["moleculeId"] for g in exact or candidates}
                if len(candidate_species) != 1:
                    raise ValueError(f"{row['source']}: ambiguous or missing species for {row['molecule']!r}")
                species = candidate_species.pop()
                variants = [g for g in candidates if g["moleculeId"] == species]
                if row["fluorescence"]:
                    variants = [g for g in variants if (g["variant"] or "").startswith("ES")]
                    if species == "acetylene":
                        gid = FLUORESCENCE_GEOMETRIES.get(re.sub(r"\s+", "", row["state"]))
                        variants = [g for g in variants if g["id"] == gid]
                else:
                    variants = [g for g in variants if not (g["variant"] or "").startswith("ES")]
                if len(variants) == 1:
                    row["geometryId"] = variants[0]["id"]
                for revised, pair in zip(("energy", "referenceQZ"), REFERENCE_REVISIONS.get(
                    (Path(row["source"]).name, row["state"].strip(), row["occurrence"]), ()
                )):
                    if pair and row[revised] == pair[1]:
                        row.setdefault("revisions", []).append({
                            "field": revised, "previous": pair[0], "value": pair[1],
                            "commit": TYPO_FIX, "reason": "Corrected a transcription error.",
                        })
            else:
                species = geometry_by_id[row["geometryId"]]["moleculeId"]
            molecule = species_index[species]
            row.update(collection=group, moleculeId=species, name=molecule["name"],
                       formula=molecule["formula"], paper=PAPERS[group])
            molecule["collections"].add(group)
            molecule["aliases"].add(row["molecule"].strip())

    for molecule in species_index.values():
        for field in ("aliases", "collections"):
            molecule[field] = sorted(molecule[field])
        for gid in molecule["geometries"]:
            geometry_by_id[gid].update(displayName=molecule["name"], formula=molecule["formula"],
                                       default=gid == molecule["defaultGeometry"],
                                       aliases=sorted(set(geometry_by_id[gid]["aliases"]) | set(molecule["aliases"])))
    return species_index


def write_pack(out, collection, title, rows, geometries, repository, revision):
    filename = f"downloads/QUESTDB-{collection}-{revision[:12]}.zip"
    reference_rows = [["state_id", "molecule", "formula", "state", "site", "root",
                       "geometry", "reference", "reference_eV", "source_json", "collection_reference"]]
    files = {}
    for row in rows:
        geometry = geometries[row["geometryId"]]
        source = row.get("benchmarkXyz", geometry["source"])
        name = geometry["id"] if source == geometry["source"] else Path(source).stem + "-benchmark"
        xyz = f"geometries/{name}.xyz"
        payload = (out / "xyz" / source).read_bytes()
        if files.setdefault(xyz, payload) != payload:
            raise ValueError(f"Conflicting benchmark geometries for {xyz}")
        reference_rows.append([
            row["stateId"], row["name"], row["formula"], row["state"].strip(), row.get("site", ""),
            row.get("root", row.get("occurrence")), xyz,
            f"sCI / {row['basis']}" if "basis" in row else "TBE / aug-cc-pVTZ", row["energy"],
            f"{repository}/blob/{revision}/{row['source']}", f"https://doi.org/{row['paper']}",
        ])

    def csv_bytes(records):
        buffer = StringIO(newline="")
        csv.writer(buffer).writerows(records)
        return buffer.getvalue().encode("utf-8")

    files["reference.csv"] = csv_bytes(reference_rows)
    files["results-template.csv"] = csv_bytes([
        ["state_id", "method", "basis", "calculated_eV"],
        *([r[0], "", "", ""] for r in reference_rows[1:]),
    ])
    files["README.txt"] = (
        f"QUESTDB · {title}\n\nSource commit: {revision}\n"
        f"{repository}/commit/{revision}\n\n"
        "reference.csv defines the benchmark cases, energies in eV, geometry paths and citations.\n"
        "Keep the state IDs when filling results-template.csv. For more than one method,\n"
        "use one row per state and method/basis combination.\n"
        "The XYZ files are in angstrom. Use the exact XYZ assigned to each reference.\n"
        "Data and geometries are distributed under CC BY-SA 4.0 (see LICENSE).\n"
    ).encode("utf-8")
    files["LICENSE"] = (out / "LICENSE").read_bytes()
    with ZipFile(out / filename, "w") as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(ZipInfo(name), payload, compress_type=ZIP_DEFLATED)
    return {"name": title, "file": filename, "count": len(rows)}


def benchmark_packs(root, out, rows, geometries, read_json, repository, revision):
    selection = read_json(root / "website/benchmark-cases.json")
    neutral = {r["stateId"]: r for r in rows["MAIN"]}
    core = {r["stateId"]: r for r in rows["core_IPs"]
            if r["method"] == "sCI" and r["basis"] == "aug-cc-pCVTZ"}
    by_id = {g["id"]: g for g in geometries}
    return [
        write_pack(out, "diet-50", "Neutral excitations · Diet-50",
                   [neutral[key] for key in selection["diet-50"]], by_id, repository, revision),
        write_pack(out, "core-ip-cvtz", "Core ionizations · sCI/aug-cc-pCVTZ",
                   [core[key] for key in selection["core-ip-cvtz"]], by_id, repository, revision),
    ]
