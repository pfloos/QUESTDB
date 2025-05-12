# 📊 Data Overview

This directory contains the core data from the **QUEST** database.  
It provides highly detailed information for each excited state, including both physical characteristics and computational results.

| Subset       | Compounds Total | Nature            | Transitions Total | Safe | 1–2 atoms | 3–5 atoms | 6–9 atoms | 10–16 atoms | S   | T   | D   | Q   | Valence | Rydberg | GD  | PD  | CT  | FL |
|--------------|------------------|--------------------|--------------------|--------|-----------|-----|-----|--------|-----|-----|-----|-----|-------------------------------|---------|-----|-----|-----|----|
| **Main**     | 117              | Organic & Inorganic      | 927                | 837    | 129       | 318 | 338 | 142    | 582 | 345 |     |     | 659       | 259     | 28  | 21  | 28  | 10 |
| **Rad**      | 33               | Open-shell         | 281                | xxx    | 201       | 80  |     |        |     |     | 217 | 64  | 162       | 82      | 13  | 21  |     |    |
| **Chrom**    | 18               | Organic Chromophore       | 158                | ~135  |           |     |     | 158    | 86  | 72  |     |     | 149    | 9       |     | 7   |     |    |
| **DNA**      | 5                | Nucleobases        | 56                 | ~51   |           |     | 33  | 23     | 35  | 21  |     |     | 40         | 16      |     |     |     |    |
| **TM**       | 11               | Transition metal diatomics | 67                 | 46     | 67        |     |     |        | 28  | 23  | 16  |     |   |         | 4   |     |     |    |
| **QUEST**    | 184              |                    | 1489               | ~xxx   | 397       | 398 | 371 | 323    | 731 | 461 | 233 | 64  | 1010      | 366     | 45  | 49  | 28  | 10 |

*S, T, D, and Q refer to singlet, triplet, doublet, and quartet states, respectively.  
GD, PD, CT, and FL denote genuine double, partial double, charge transfer, and fluorescence, respectively.*

---

## 📋 Available Information

To assist users in identifying excited states using their preferred methodology, the QUEST database provides the percentage of single excitation involved in the transition (%T₁), oscillator strength $f$, the patial extent of the electron cloud $\langle r^2 \rangle$, and dominant MO contributions for all excited states, computed with reasonably high levels of theory. Additionally, $\langle S^2 \rangle$ values are provided for doublet and quartet states, along with the dominant MO combinations.

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
  - **FL** — Fluorescence transition (computed at the S₁ optimized geometry).
  - **PD** — Partial double excitation, corresponds to a state where %T₁ is in the range 60%-80%.
  - **GD** — Genuine double excitation, characterized by %T₁ < 50% (often close to 0%).
  - **wCT** — Weak charge-transfer excitation.
  - **sCT** — Strong charge-transfer excitation.
- **%T₁ (Single Excitation Character)**:  
  Percentage of single excitations involved in the transition computed at the CC3/aug-cc-pVTZ level.
- **Oscillator Strength (_f_)**:  
  Computed at the LR-CC3/aug-cc-pVTZ level.
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
  In some cases, TD-DFT excitation energies obtained with various xc functionals are also available.

Dedicated files for subsets (see below) also include:
- Additional computational methods and basis sets.
- Molecular orbitals involved in each transition.
- Spatial extent of the electron cloud $\langle r^2 \rangle$.
- Expectation value of the spin operator $\langle S^2 \rangle$ for radicals.
- Extra annotations for complex cases.

---

## 📂 Files in This Directory

| Filename              | Description |
|-----------------------|-------------|
| **`QUEST-All.xlsx`**   | Contains all the information listed above for each transition. Each subset described below is presented in a dedicated sheet. |
| **`QUEST-Main.xlsx`**  | Includes all results for relatively compact closed-shell molecules, typically containing 1 to 10 non-hydrogen atoms. |
| **`QUEST-Rad.xlsx`**   | A significant extension of our dataset for small organic and inorganic radicals, now including additional compounds, excited states, and a series of quartet excited states. |
| **`QUEST-Chrom.xlsx`** | Covers excited states of large closed-shell organic chromophores with 10 to 16 non-hydrogen atoms, such as azobenzene, BODIPY, and naphthalimide. |
| **`QUEST-DNA.xlsx`**   | Presents previously unpublished data for five nucleobases: adenine, cytosine, guanine, thymine, and uracil. |
| **`QUEST-TM.xlsx`**    | Contains results for 11 diatomic molecules featuring one transition metal (Cu, Sc, Ti, or Zn), covering both closed-shell and open-shell cases. |

In addition, `.json` files, gathering the same information, are available for each molecule.

---

## 🥗 Diet Sets

This directory also includes an example of *diet* subset (located in the `diet` folder), which is carefully designed to reproduce the key statistical features of larger datasets while restricting data size. These size constraints make the subset well-suited for use with computationally demanding or inefficient methods, particularly during the early stages of method development.

The following subset is available:

- **50 excitations** across **20 molecules**: `diet_subset_50.json`  

This set is derived from the QUEST *Main* dataset, excluding unsafe and genuine double excitations (resulting in 824 transitions across 119 molecules). This filtered set of excitations can be found in `filtered_main_set.json`. The corresponding output file (`diet_subset_50.out`) generated using the `quest-diet.py` script is also provided.

> **Note 1:** This *diet* subset is **not unique**. It is generated using a genetic algorithm, which is inherently stochastic — meaning that operations like selection, crossover, and mutation involve random choices. As a result, running the `quest-diet.py` script multiple times (even with the same parameters) can produce different subsets of excitations. This variability allows for exploration of multiple near-optimal solutions, but it also means that results are not guaranteed to be reproducible unless a fixed random seed is explicitly set in the script or passed as a parameter.

> **Note 2:** This set was obtained using a genetic algorithm that minimizes a scoring function incorporating the MAE, MSE, and RMSE of single-reference wave function methods. Multireference methods (e.g., CASSCF, CASPT2, and NEVPT2) are excluded. The script can be easily modified to target a different set of methods.

## 🧠 Notes

- All data use **atomic units** unless otherwise specified. Notably, excitation energies are in eV.
- The database is continuously expanded and refined following the "mountaineering strategy" for reaching high-accuracy benchmarks.

---
