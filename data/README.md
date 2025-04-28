# Data

This directory contains the core data of the QUEST database. 
Specifically, it provides detailed information on each excited state, including:

- Molecular size (number of non-hydrogen atoms)
- Group (35 stands for systems from 3 to 5 non-H atoms)
- Symmetry label of the excited state
- Spin multiplicity:
  - **S** for singlet
  - **D** for doublet
  - **T** for triplet
  - **Q** for quartet
- Nature of the state:
  - **V** for valence
  - **R** for Rydberg
  - **M** for mixed character
- Type of electronic transition:
  - **ppi** for π→π\* transitions
  - **npi** for n→π\* transitions
  - **n3s** for n→3s transitions
  - **p3s** for π→3s transitions
  - **n3p** for n→3p transitions
  - **p3p** for π→3p transitions
  - **n4s** for n→4s transitions
  - **n4p** for n→4p transitions
  - **dou** for double excitations
  - **n.d.** for "not determined"
- Special feature of the transition:
  - **FL** for fluorescence transitions
  - **pd** for partial double excitations
  - **gd** for genuine double excitations
  - **wCT** for weak charge-transfer excitations
  - **sCT** for strong charge-transfer excitations
- Percentage of single excitation character %T1 (computed at the CC3/aug-cc-pVTZ level)
- Oscillator strength f (computed at the LR-CC3/aug-cc-pVTZ level)
- Theoretical Best Estimates (TBEs) in the aug-cc-pVTZ basis
- (Composite) computational methods used to obtain each TBE
- Chemical reliability of the transition (safe vs. unsafe)
- TBEs computed in the aug-cc-pVQZ basis
- Correction methods applied for aug-cc-pVQZ TBEs
- Vertical excitation energies computed with a broad set of methods, including:  
  `CIS(D)`, `CC2`, `EOM-MP2`, `STEOM-CCSD`, `CCSD`, `CCSD(T)(a)*`, `CCSDR(3)`, `CCSDT-3`, `CC3`, `CCSDT`,  
  `SOS-ADC(2)[TM]`, `SOS-CC2`, `SCS-CC2`, `SOS-ADC(2)[QC]`, `ADC(2)`, `ADC(3)`, `ADC(2.5)`,  
  `CASSCF`, `CASPT2`, `CASPT2 (No IPEA)`, `CASPT3`, `CASPT3 (No IPEA)`, `SC-NEVPT2`, and `PC-NEVPT2`.

## Available Files

- `QUEST-All.xlsx`:  
  Full dataset including all molecular systems.

- `QUEST-Chromo.xlsx`:  
  Subset focused on chromophores.

- `QUEST-DNA.xlsx`:  
  Subset focused on DNA building blocks.

- `QUEST-Radicals.xlsx`:  
  Subset focused on radical species.

- `QUEST-TM.xlsx`:  
  Subset focused on transition metal complexes.

- `QUEST-Main.xlsx`:  
  Detailed dataset organized with one tab per system, including additional information such as:
  - Results with more computational methods and basis sets
  - Molecular orbitals involved in each transition
  - Extra annotations for complex transitions
