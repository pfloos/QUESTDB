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

Dedicated files for subsets (see below) also include:
- Additional computational methods and basis sets
- Molecular orbitals involved in transitions
- Extra annotations for complex cases.

---

## 📂 Files in This Directory

| Filename              | Description |
|-----------------------|-------------|
| **`QUEST-All.xlsx`**   | Contains all the information listed above for each transition. Each subset described below is presented in a dedicated tab. |
| **`QUEST-Main.xlsx`**  | Includes all results for relatively compact closed-shell molecules, typically containing 1 to 10 non-hydrogen atoms. |
| **`QUEST-Rad.xlsx`**   | A significant extension of our dataset for small organic and inorganic radicals, now including additional compounds, excited states, and a series of quartet excited states. |
| **`QUEST-Chrom.xlsx`** | Covers excited states of large closed-shell organic chromophores with 10 to 16 non-hydrogen atoms, such as azobenzene, BODIPY, and naphthalimide. |
| **`QUEST-DNA.xlsx`**   | Presents previously unpublished data for five nucleobases: adenine, cytosine, guanine, thymine, and uracil. |
| **`QUEST-TM.xlsx`**    | Contains results for 11 diatomic molecules featuring one transition metal (Cu, Sc, Ti, or Zn), covering both closed-shell and open-shell cases. |

---

## 🧠 Notes

- All data use **atomic units** unless otherwise specified. Notably, excitation energies are in eV.
- The database is continuously expanded and refined following the "mountaineering strategy" for reaching high-accuracy benchmarks.

---
