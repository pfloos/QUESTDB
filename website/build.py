"""Build a static QUESTDB explorer from the current Git checkout."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

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
        collections.append({
            "id": group, "name": label, "kind": "neutral" if neutral else "charged",
            "rows": len(rows), "files": len({row["source"] for row in rows}),
        })
    geometries = []
    geometry_groups = {
        "geometries/xyz": "Neutral",
        "charged/valence_IPs/geometries": "Valence IP",
        "charged/core_IPs/geometries": "Core IP",
    }
    for directory, label in geometry_groups.items():
        for path in sorted((ROOT / directory).glob("*.xyz")):
            source = path.relative_to(ROOT).as_posix()
            destination = f"xyz/{source}"
            (out / destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out / destination)
            geometries.append({
                "name": path.stem, "group": label, "source": source, "file": destination,
            })
    write_json(assets / "geometries.json", geometries)
    write_json(assets / "manifest.json", {
        "repository": REPO, "revision": revision, "date": date, "collections": collections,
        "geometries": len(geometries),
        "neutralStates": sum(c["rows"] for c in collections if c["kind"] == "neutral"),
    })
    print(f"Built {len(collections)} collections from {revision[:12]} in {out}")


if __name__ == "__main__":
    main()
