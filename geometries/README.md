# 🧩 Geometries

## Canonical geometry registry

Use [`registry.json`](registry.json) to find the canonical XYZ and the original geometry used by each charged benchmark. XYZ coordinates are in ångströms.

- `geometries` is keyed by stable geometry ID. Each entry identifies a `species` and its canonical `xyz` path. A species with one geometry uses that entry as its default. Species with multiple geometries have named `variant` entries and one entry marked `default`. The default for an explicitly identified ground/excited pair is the ground-state geometry.
- `chargedBenchmarks` maps each valence-IP, core-IP, and DIP data file to a `geometryId` and its original `benchmarkXyz` path. DIP records use the valence-IP benchmark files. Use `benchmarkXyz` to reproduce the exact input coordinates and atom order, especially when checking a core-site assignment.
- When the same structure is present in the neutral and charged collections, the neutral XYZ is canonical. For structures unique to the charged collections, use the valence-IP XYZ where available, followed by the core-IP XYZ. Different charge states, isomers, electronic-state geometries, and documented conformers or symmetry variants have distinct records.
- `texLine` points to the coordinate block in [the neutral geometry TeX source](TeX/QUEST-geometries.tex). The CON XYZ was regenerated from that coordinate block with one conversion from bohr to ångströms.

The [geometry explorer](https://pfloos.github.io/QUESTDB/#geometries) provides the canonical archive and collection downloads.

Welcome to the **Geometries** section of the QUEST Database!

This directory provides the **molecular structures** used in QUESTDB in two convenient formats, designed for both **publication** and **computational use**:

---

## 📄 Contents

- 📚 **TeX Format** (`TeX/`)  
  Contains a `.tex` source file along with its compiled `.pdf`, presenting the **Cartesian coordinates** of all molecules in a **clean, citable format**.  
  Ideal for inclusion in scientific articles and reports.  
  All Cartesian coordinates are expressed in **atomic units (bohr)**.

- 🧪 **XYZ Format** (`xyz/`)  
  Includes individual `.xyz` files for **each molecule**, ready for **direct input** into computational chemistry software (e.g., Gaussian, ORCA, Q-Chem).  
All Cartesian coordinates are expressed in **Angstrom**.

---

## 📢 Important Notes

- All geometries have been **carefully curated** to ensure **consistency** and **reproducibility**.
- For benchmarking, validation, or method development, we **strongly recommend** using these geometries **without modification**.
