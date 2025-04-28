# 📊 Data Overview

This directory contains the core data from the **QUEST** database.  
It provides highly detailed information for each excited state, including both physical characteristics and computational results.

---

## 📋 Available Information

- **Molecular Size**: Number of non-hydrogen atoms.
- **Group**: Size group (e.g., `35` for systems with 3–5 non-hydrogen atoms).
- **Symmetry Label**: Symmetry of the excited state.
- **Spin Multiplicity**:
  - **S** — Singlet
  - **D** — Doublet
  - **T** — Triplet
  - **Q** — Quartet
- **Nature of the Excited State**:
  - **V** — Valence
  - **R** — Rydberg
  - **M** — Mixed (valence and Rydberg character)
- **Type of Electronic Transition**:
  - **ppi** — π → π\*  
  - **npi** — n → π\*  
  - **n3s** — n → 3s  
  - **p3s** — π → 3s  
  - **n3p** — n → 3p  
  - **p3p** — π → 3p  
  - **n4s** — n → 4s  
  - **n4p** — n → 4p  
  - **dou** — Double excitations  
  - **n.d.** — Not determined
- **Special Features of the Transition**:
  - **FL** — Fluorescence transition
  - **pd** — Partial double excitation
  - **gd** — Genuine double excitation
  - **wCT** — Weak charge-transfer excitation
  - **sCT** — Strong charge-transfer excitation
- **%T₁ (Single Excitation Character)**:  
  Percentage of single excitation computed at the **CC3/aug-cc-pVTZ** level.
- **Oscillator Strength (_f_)**:  
  Computed at the **LR-CC3/aug-cc-pVTZ** level.
- **Theoretical Best Estimates (TBEs)**:
  - TBEs computed using the aug-cc-pVTZ basis set.
  - Composite methods employed to obtain TBEs.
  - Chemical reliability of the transition (safe vs. unsafe).
- **TBEs in Larger Basis Sets**:
  - TBEs computed with the aug-cc-pVQZ basis.
  - Correction methods applied for basis set extrapolation.
- **Vertical Excitation Energies**:  
  Computed with a wide range of methods: 
  `CIS(D)`, `CC2`, `EOM-MP2`, `STEOM-CCSD`, `CCSD`, `CCSD(T)(a)*`, `CCSDR(3)`, `CCSDT-3`, `CC3`, `CCSDT`,  
  `SOS-ADC(2)[TM]`, `SOS-CC2`, `SCS-CC2`, `SOS-ADC(2)[QC]`, `ADC(2)`, `ADC(3)`, `ADC(2.5)`,  
  `CASSCF`, `CASPT2`, `CASPT2 (No IPEA)`, `CASPT3`, `CASPT3 (No IPEA)`, `SC-NEVPT2`, and `PC-NEVPT2`.

---

## 📂 Files in This Directory

| Filename             | Description |
|----------------------|-------------|
| **`QUEST-All.xlsx`**  | Full dataset including all molecular systems. |
| **`QUEST-Chromo.xlsx`** | Subset focused on chromophores. |
| **`QUEST-DNA.xlsx`**  | Subset focused on DNA building blocks. |
| **`QUEST-Radicals.xlsx`** | Subset focused on radical species. |
| **`QUEST-TM.xlsx`**   | Subset focused on transition metal complexes. |
| **`QUEST-Main.xlsx`** | Detailed per-system dataset with: <br> - Additional computational methods and basis sets <br> - Molecular orbitals involved in transitions <br> - Extra annotations for complex cases. |

---

## 🧠 Notes

- All data use **atomic units** unless otherwise specified.
- The database is continuously expanded and refined following the "mountaineering strategy" for reaching high-accuracy benchmarks.

---
