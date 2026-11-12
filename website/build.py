"""Build a static QUESTDB explorer from the current Git checkout."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/pfloos/QUESTDB"
NEUTRAL = {
    "MAIN": "Main set", "RAD": "Radicals", "CHROM": "Chromophores",
    "BIO": "Biomolecules", "TM": "Transition metals",
}
CHARGED = {
    "valence_IPs": "Valence ionizations",
    "core_IPs": "Core ionizations",
    "DIPs": "Double ionizations",
}
# This file has no method, molecule, or core-site identification.
CORE_SUPPLEMENT = "charged/core_IPs/output.json"
GEOMETRY_GROUPS = {
    "geometries/xyz": "Neutral",
    "charged/valence_IPs/geometries": "Valence IP",
    "charged/core_IPs/geometries": "Core IP",
}
# Source labels that differ from the XYZ filenames, including known variants.
GEOMETRY_ALIASES = {
    "Carbon dioxyde": "carbon_dioxide",
    "Criegee Interm.": "Criegee_intermediate",
    "Napthoquinone": "naphthoquinone",
    "Nitrodimethylaniline": "nitrodimethylaniniline",
    "Dimethylaminobenzonitrile": "dimethylaminobenzonitrile_1",
    "Dimethylaminobenzonitrile (twisted)": "dimethylaminobenzonitrile_2",
    "Indacene (C2h)": "indacene_1",
    "Indacene (D2h)": "indacene_2",
    "Pentalene": "pentalene_1",
    "Pentalene (D2h)": "pentalene_2",
    "Phenyl-pyrrole": "phenyl-pyrrole_1",
    "Phenyl-pyrrole (twisted)": "phenyl-pyrrole_2",
}
GEOMETRY_VARIANTS = {
    "dimethylaminobenzonitrile": ("planar", "twisted"),
    "phenyl-pyrrole": ("planar", "twisted"),
    "indacene": ("C2h", "D2h"),
    "pentalene": ("unconstrained", "D2h"),
}


def geometry_key(label):
    return "".join(c for c in unicodedata.normalize("NFKC", label).lower() if c.isalnum())


def geometry_names(paths):
    names = {path: path.stem for path in paths}
    excited = {}
    for path in paths:
        match = re.fullmatch(r"(.+)_(\d+)", path.stem)
        if not match:
            continue
        base, number = match[1], int(match[2])
        with path.open(encoding="utf-8") as xyz:
            next(xyz, "")
            comment = next(xyz, "").lower()
        if "ground state" in comment:
            names[path] = f"{base}_GS"
        elif "excited state" in comment:
            excited.setdefault(base, []).append((number, path))
        elif base in GEOMETRY_VARIANTS and 1 <= number <= len(GEOMETRY_VARIANTS[base]):
            names[path] = f"{base}_{GEOMETRY_VARIANTS[base][number - 1]}"
    for base, variants in excited.items():
        for index, (_, path) in enumerate(sorted(variants), 1):
            names[path] = f"{base}_ES{index if len(variants) > 1 else ''}"
    if len(set(names.values())) != len(names):
        raise ValueError("Geometry download filenames must be unique within each collection")
    return names


def geometry_assets(out, neutral_molecules):
    records = []
    for directory, group in GEOMETRY_GROUPS.items():
        paths = sorted((ROOT / directory).glob("*.xyz"))
        names = geometry_names(paths) if group == "Neutral" else {path: path.stem for path in paths}
        for path in paths:
            source = path.relative_to(ROOT).as_posix()
            archive_path = Path(source).with_name(f"{names[path]}.xyz").as_posix()
            destination = f"xyz/{archive_path}"
            (out / destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out / destination)
            if archive_path != source:
                shutil.copyfile(path, out / "xyz" / source)
            subsets = []
            labels = []
            if group == "Neutral":
                stem = geometry_key(path.stem)
                base = geometry_key(re.sub(r"_\d+$", "", path.stem))
                for subset, molecules in neutral_molecules.items():
                    matches = [
                        molecule for molecule in molecules
                        if (stem == geometry_key(GEOMETRY_ALIASES[molecule])
                            if molecule in GEOMETRY_ALIASES else geometry_key(molecule) in (stem, base))
                    ]
                    if matches:
                        subsets.append(subset)
                        labels.extend(matches)
            records.append({
                "name": names[path], "group": group, "source": source, "file": destination,
                "subsets": subsets, "labels": sorted(set(labels)),
            })

    groups = {"all": "All geometries", "Neutral": "All neutral sets",
              **{key: f"{key} · {label}" for key, label in NEUTRAL.items()},
              "Valence IP": "Valence IP", "Core IP": "Core IP"}
    downloads = []
    (out / "downloads").mkdir(exist_ok=True)
    for group, label in groups.items():
        members = [record for record in records if group in ("all", record["group"]) or group in record["subsets"]]
        if not members:
            continue
        filename = f"downloads/QUESTDB-geometries-{group.lower().replace(' ', '-')}.zip"
        with ZipFile(out / filename, "w") as archive:
            for record in members:
                entry = ZipInfo(record["file"].removeprefix("xyz/"))
                archive.writestr(entry, (out / record["file"]).read_bytes(), compress_type=ZIP_DEFLATED)
        downloads.append({"id": group, "name": label, "count": len(members), "file": filename})
    write_json(out / "data/geometries.json", records)
    return records, downloads


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def neutral_records(path):
    source = path.relative_to(ROOT).as_posix()
    for row in read_json(path):
        if not isinstance(row, dict) or not row.get("Molecule") or not row.get("State"):
            raise ValueError(f"{source}: expected a molecule and state for every record")
        yield {
            "molecule": row["Molecule"], "state": row["State"], "source": source,
            "energy": row.get("TBE/AVTZ"), "referenceQZ": row.get("TBE/AVQZ"),
            "safe": row.get("Safe ? (~50 meV)"), "spin": row.get("Spin"),
            "character": row.get("V/R"), "type": row.get("Type"), "fields": row,
        }


def charged_records(path, group):
    source = path.relative_to(ROOT).as_posix()
    data = read_json(path)
    if group == "DIPs":
        methods = ((spin, method, {basis: values})
                   for method, block in data["methods"].items()
                   for basis, sectors in block["basis_sets"].items()
                   for spin, values in sectors.items())
    elif group == "valence_IPs":
        methods = ((sector, method, block)
                   for sector, entries in data.items()
                   for entry in entries for method, block in entry.items())
    else:
        methods = (("Core IP", method, block) for method, block in data.items())
    for sector, method, block in methods:
        energies = block.get("Energy", block)
        for basis, values in energies.items():
            if values is None:
                continue
            series = values if isinstance(values, list) else [values]
            weights = block.get("Z", {}).get(basis)
            uncertainties = block.get("Incertitudes", {}).get(basis)
            for offset, energy in enumerate(series):
                if energy is None:
                    continue
                if not number(energy):
                    raise ValueError(f"{source}: invalid energy for {method}/{basis}")
                weight = weights[offset] if isinstance(weights, list) and offset < len(weights) else (
                    weights if not isinstance(weights, list) and offset == 0 else None
                )
                uncertainty = uncertainties[offset] if isinstance(uncertainties, list) and offset < len(uncertainties) else (
                    uncertainties if not isinstance(uncertainties, list) and offset == 0 else None
                )
                yield {
                    "molecule": path.stem, "state": sector, "source": source,
                    "method": method, "basis": basis, "root": offset + 1,
                    "energy": energy, "weight": weight, "uncertainty": uncertainty,
                }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    out = parser.parse_args().out.resolve()
    assets = out / "data"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js", "views.js"):
        shutil.copyfile(ROOT / "website" / name, out / name)
    shutil.copyfile(ROOT / "logo/TOC.jpg", out / "quest-art.jpg")
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    revision, date = subprocess.check_output(
        ["git", "show", "-s", "--format=%H%n%cI", "HEAD"], cwd=ROOT, text=True
    ).splitlines()
    collections = []
    neutral_molecules = {}
    for group, label in (NEUTRAL | CHARGED).items():
        neutral = group in NEUTRAL
        directory = ROOT / ("data/json" if neutral else "charged") / group
        paths = sorted(directory.glob("*.json"))
        rows = [
            row for path in paths if path.relative_to(ROOT).as_posix() != CORE_SUPPLEMENT
            for row in (neutral_records(path) if neutral else charged_records(path, group))
        ]
        if not rows:
            raise ValueError(f"{group}: no data found")
        rows.sort(key=lambda row: row["molecule"].casefold())
        write_json(assets / f"{group}.json", rows)
        if neutral:
            neutral_molecules[group] = {row["molecule"] for row in rows}
        collections.append({
            "id": group, "name": label, "kind": "neutral" if neutral else "charged",
            "rows": len(rows), "files": len({row["source"] for row in rows}),
        })
    geometries, downloads = geometry_assets(out, neutral_molecules)
    write_json(assets / "manifest.json", {
        "repository": REPO, "revision": revision, "date": date, "collections": collections,
        "geometries": len(geometries),
        "geometryDownloads": downloads,
        "neutralStates": sum(c["rows"] for c in collections if c["kind"] == "neutral"),
    })
    print(f"Built {len(collections)} collections from {revision[:12]} in {out}")


if __name__ == "__main__":
    main()
