# 🚀 QUESTDB: A Database of Highly-Accurate Excitation Energies

[![Funding](https://img.shields.io/badge/Funding-ERC%20PTEROSOR-orange)](https://lcpq.github.io/PTEROSOR/)
[![License](https://img.shields.io/badge/License-CC%20BY%20SA%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Last Update](https://img.shields.io/github/last-commit/pfloos/QUESTDB?label=last%20update)](https://github.com/pfloos/QUESTDB/commits/main)

---

## 📚 Table of Contents

- [✨ Key Features](#-key-features)
- [🧪 Why Use QUESTDB?](#-why-use-questdb)
- [📂 Repository Contents](#-repository-contents)
- [👥 Contributors](#-contributors)
- [📚 Main References](#-main-references)
- [📖 Other References](#-other-references)
- [🔋 Extension to Charged Excitations](#-extension-to-charged-excitations)
- [🗂️ Data Structure](#️-data-structure)
- [💰 Funding](#-funding)

---

## ✨ Key Features

- **🔬 High Accuracy:**  
  Data obtained using state-of-the-art methods (FCI, CC3, CCSDT, CCSDTQ, CC4, CASPT2/3, NEVPT2, etc.)

- **🌍 Wide Chemical Coverage:**  
  Includes small molecules, radicals, charged species, and transition metal complexes.

- **🎯 Challenging Excitations:**  
  Focus on **double excitations** and **intramolecular charge-transfer (CT) states**.

- **🛠️ Continuously Updated:**  
  Regularly improved with new high-level calculations and critical assessments.

- **📂 Easy-to-Use Format:**  
  Organized `.xlsx` spreadsheets and `.json` files for simple extraction and analysis.

---

## 🧪 Why Use QUESTDB?

QUESTDB supports researchers to:
- **Benchmark** TD-DFT, wavefunction-based, and emerging excited-state methods.
- **Guide** the development of new computational models.
- **Facilitate** interpretation of experimental spectra and photochemistry.

> **Note:** QUESTDB is a cornerstone of the **mountaineering strategy** — systematically climbing towards chemically-accurate excited-state data.

Our vision is to establish QUESTDB as a cornerstone resource for benchmarking and training the next generation of AI-driven models in excited-state science.

---

### ⚙️ Scripts for Subset Generation and Analysis

This repository includes Python scripts to help users generate representative *"diet"* subsets of QUEST excitation energies—for instance, sets of 50, 100, or 200 transitions that reproduce the statistical properties of the full database (e.g., MAE, RMSE) across different computational methods and excitation categories.

These tools are especially useful for benchmarking new methods quickly or for training machine learning models when computational cost is a limiting factor.

**Main functionalities include:**

- ✅ Generation of optimized subsets matching the full dataset’s distribution across:
  - Singlets vs triplets  
  - Valence vs Rydberg states  
  - Excitation types (e.g., \(n\pi^*\), \(\pi\pi^*\), etc.)  
  - Molecule sizes or other custom filters  
- ✅ Support for flexible user-defined filters (e.g., only valence, only singlets, exclude genuine doubles)  
- ✅ Preservation of full metadata in output JSON files  
- ✅ Styled statistical reports using the [`rich`](https://github.com/Textualize/rich) library  
- ✅ Optional optimization of subset selection using a genetic algorithm with Bayesian hyperparameter tuning (via [`optuna`](https://optuna.org/))  

To explore the tools, check out the [`tools/`](tools/) directory and use:

```bash
python tools/select_subset.py --help
```

---

## 📂 Repository Contents

This repository provides:
- **Molecular Structures**
- **Vertical Excitation Energies**
- **Oscillator Strengths**
- **Many Other Properties**

Data is structured in `.xlsx` and `.json` files for ease of use (see the `data` directory).

**📌 See the accompanying paper:**  
[**The QUEST database of highly-accurate excitation energies**]()  
P.F Loos, M. Boggio-Pasqua, A. Blondel, F. Lipparini, and D. Jacquemin,  
*J. Chem. Theory Comput.* (submitted).  

---

## 👥 Contributors

The QUESTDB project is maintained by a collaboration between:

- [Denis Jacquemin](https://www.univ-nantes.fr/denis-jacquemin-1) (Nantes)
- [Pierre-François Loos](https://pfloos.github.io/WEB_LOOS) (Toulouse)
- [Martial Boggio-Pasqua](https://www.lcpq.ups-tlse.fr/spip.php?rubrique313&lang=fr) (Toulouse)
- [Fábris Kossoski](https://kossoski.github.io) (Toulouse)
- [Filippo Lipparini](https://people.unipi.it/filippo_lipparini) (Pisa)
- [Anthony Scemama](https://scemama.github.io) (Toulouse)
- [Aymeric Blondel](https://www.univ-nantes.fr/aymeric-blondel) (Nantes)
- [Mickael Véril](https://mveril.github.io) (Toulouse)
- [Yann Damour](https://ydrnan.github.io/damour) (Toulouse)
- [Antoine Marie](https://antoine-marie.github.io) (Toulouse)

---

## 📚 Main References

Review articles on the QUEST database:

- [**The QUEST database of highly-accurate excitation energies**]()  
  P.F Loos, M. Boggio-Pasqua, A. Blondel, F. Lipparini, and D. Jacquemin,  
  *J. Chem. Theory Comput.* (submitted).  

- [**QUESTDB: a database of highly-accurate excitation energies for the electronic structure community**](https://doi.org/10.1002/wcms.1517)  
  M. Véril, A. Scemama, M. Caffarel, F. Lipparini, M. Boggio-Pasqua, D. Jacquemin, and P. F. Loos,  
  *WIREs Comput. Mol. Sci.* **11**, e1517 (2021).

- [**The quest for highly accurate excitation energies: a computational perspective**](https://dx.doi.org/10.1021/acs.jpclett.0c00014)  
  P. F. Loos, A. Scemama, and D. Jacquemin,  
  *J. Phys. Chem. Lett.* **11**, 2374 (2020).

Key QUESTDB publications:

- [**Reference energies for double excitations: improvement & extension**](https://doi.org/10.1021/acs.jctc.4c00410)  
  F. Kossoski, M. Boggio-Pasqua, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **20**, 5655 (2024).

- [**Reference vertical excitation energies for transition metal compounds**](https://doi.org/10.1021/acs.jctc.3c01080)  
  D. Jacquemin, F. Kossoski, F. Gam, M. Boggio-Pasqua, and P. F. Loos,  
  *J. Chem. Theory Comput.* **19**, 8782 (2023).

- [**A mountaineering strategy to excited states: revising reference values with EOM-CC4**](https://doi.org/10.1021/acs.jctc.2c00416)  
  P. F. Loos, F. Lipparini, D. A. Matthews, A. Blondel, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **18**, 4418 (2022).

- [**A mountaineering strategy to excited states: highly-accurate energies and benchmarks for bicyclic systems**](https://doi.org/10.1021/acs.jpca.1c08524)  
  P. F. Loos and D. Jacquemin,  
  *J. Phys. Chem. A* **125**, 10174 (2021).

- [**Reference energies for intramolecular charge-transfer excitations**](https://doi.org/10.1021/acs.jctc.1c00226)  
  P. F. Loos, M. Comin, X. Blase, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **17**, 3666 (2021).

- [**A mountaineering strategy to excited states: highly-accurate oscillator strengths and dipole moments of small molecules**](https://dx.doi.org/10.1021/acs.jctc.0c01111)  
  A. Chrayteh, A. Blondel, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **17**, 416 (2021).

- [**A mountaineering strategy to excited states: highly-accurate energies and benchmarks for exotic molecules and radicals**](https://dx.doi.org/10.1021/acs.jctc.0c00227)  
  P. F. Loos, A. Scemama, M. Boggio-Pasqua, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **16**, 3720 (2020).

- [**A mountaineering strategy to excited states: highly-accurate energies and benchmarks for medium size molecules**](https://dx.doi.org/10.1021/acs.jctc.9b01216)  
  P. F. Loos, F. Lipparini, M. Boggio-Pasqua, A. Scemama, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **16**, 1711 (2020).

- [**Reference energies for double excitations**](https://dx.doi.org/10.1021/acs.jctc.8b01205)  
  P. F. Loos, M. Boggio-Pasqua, A. Scemama, M. Caffarel, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **15**, 1939 (2019).

- [**A mountaineering strategy to excited states: highly-accurate reference energies and benchmarks**](https://dx.doi.org/10.1021/acs.jctc.8b00406)  
  P. F. Loos, A. Scemama, A. Blondel, Y. Garniron, M. Caffarel, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **14**, 4360 (2018).

---

## 📖 Other References

- [**Excited-state absorption: Reference oscillator strengths, wavefunction and TD-DFT benchmarks**](https://doi.org/10.1021/acs.jctc.5c00159)  
  J. Širůček, B. Le Guennic, Y. Damour, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* (in press).

- [**Reference CC3 excitation energies for organic chromophores: benchmarking TD-DFT, BSE/GW and wave function methods**](https://doi.org/10.1021/acs.jctc.4c00906)  
  I. Knysh, F. Lipparini, I. Duchemin, X. Blase, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **20**, 8152 (2024).

- [**Heptazine, cyclazine, and related compounds: chemically-accurate estimates of the inverted singlet-triplet gap**](https://doi.org/10.1021/acs.jpclett.3c03042)  
  P. F. Loos, F. Lipparini, and D. Jacquemin,  
  *J. Phys. Chem. Lett.* **14**, 11069 (2023).

- [**Ground- and excited-state dipole moments and oscillator strengths of full configuration interaction quality**](https://doi.org/10.1021/acs.jctc.2c01111)  
  Y. Damour, R. Quintero-Monsebaiz, M. Caffarel, D. Jacquemin, F. Kossoski, A. Scemama, and P. F. Loos,  
  *J. Chem. Theory Comput.* **19**, 221 (2023).

- [**Benchmarking CASPT3 vertical excitation energies**](https://doi.org/10.1063/5.0095887)  
  M. Boggio-Pasqua, D. Jacquemin, and P. F. Loos,  
  *J. Chem. Phys.* **157**, 014103 (2022).

- [**Reference energies for cyclobutadiene: automerization and excited states**](https://doi.org/10.1021/acs.jpca.2c02480)  
  E. Monino, M. Boggio-Pasqua, A. Scemama, D. Jacquemin, and P. F. Loos,  
  *J. Phys. Chem. A* **126**, 4664 (2022).

- [**Assessing the performances of CASPT2 and NEVPT2 for vertical excitation energies**](https://doi.org/10.1021/acs.jctc.1c01197)  
  R. Sarkar, P. F. Loos, M. Boggio-Pasqua, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **18**, 2418 (2022).

- [**Benchmarking TD-DFT and wave function methods for oscillator strengths and excited-state dipole moments**](https://dx.doi.org/10.1021/acs.jctc.0c01228)  
  R. Sarkar, M. Boggio-Pasqua, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **17**, 1106 (2021).

- [**How accurate are EOM-CC4 vertical excitation energies?**](https://doi.org/10.1063/5.0055994)  
  P. F. Loos, D. A. Matthews, F. Lipparini, and D. Jacquemin,  
  *J. Chem. Phys.* **154**, 221103 (2021).

- [**Is ADC(3) as accurate as CC3 for valence and Rydberg excitation energies?**](https://dx.doi.org/10.1021/acs.jpclett.9b03652)  
  P. F. Loos and D. Jacquemin,  
  *J. Phys. Chem. Lett.* **11**, 974 (2020).

- [**Cross comparisons between experiment, TD-DFT, CC and ADC for transition energies**](https://dx.doi.org/10.1021/acs.jctc.9b00446)  
  C. Suellen, R. Garcia Freitas, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **15**, 4581 (2019).

---

## 🔋 Extension to Charged Excitations

The QUEST database also contains charged excitations, mainly ionization potentials (IPs) at the moment.
Here is the short description of the **charged** excited states:

- Inner- and Outer-Valence IPs and Satellite Transitions:  
[**Reference energies for valence ionizations and satellite transitions**](https://doi.org/10.1021/acs.jctc.4c00216)  
A. Marie and P. F. Loos,  
*J. Chem. Theory Comput.* **20**, 4751 (2024).

- Valence Double IPs (DIPs) and Double Core Holes (DCHs):  
[**Anomalous propagators and the particle-particle channel: Bethe-Salpeter equation**](https://doi.org/10.1063/5.0250155)  
A. Marie, P. Romaniello, X. Blase, and P. F. Loos,  
*J. Chem. Phys.* **162**, 134105 (2025).

- Core IPs (coming soon)
  
---

## 🗂️ Data Structure

- **Molecular Structures:**  
  `.xyz` or `.TeX` formats

- **Excitation Energies, Oscillator Strengths and Other Properties:**  
  `.xls` spreadsheets and `.json` files

- **Scripts to Convert Data**  
  `.py` scripts to convert data from one format to another

- **Additional Metadata:**  
  *(Planned for future releases)*

---

## 💰 Funding

<p align="center">
  <a href="https://lcpq.github.io/PTEROSOR/">
    <img src="https://lcpq.github.io/PTEROSOR/img/ERC.png" width="200" alt="ERC Logo" />
  </a>
</p>

This database is supported by the **[PTEROSOR project](https://lcpq.github.io/PTEROSOR/)**, funded by the **European Research Council (ERC)** under the **EU Horizon 2020** research and innovation program (Grant Agreement No. **863481**).

---
