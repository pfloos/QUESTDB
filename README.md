# 🚀 QUESTDB: Highly-Accurate Excitation Energies Database

Welcome to the **QUEST Database** repository!

The QUEST (QUantum Excitation STandards) database is a **comprehensive and curated collection of highly-accurate excitation energies, oscillator strengths, dipole moments**, and related excited-state properties for a wide variety of molecules, radicals, and transition metal complexes.  
Its primary goal is to provide **reliable reference data** for benchmarking and validating theoretical methods in computational chemistry, such as time-dependent DFT (TD-DFT), coupled-cluster, and multireference methods.

**Key Features of QUESTDB:**
- **High accuracy:** Data are obtained from best-in-class electronic structure methods, including FCI, CC3, CCSDT, CCSDTQ, EOM-CC4, and selected multireference methods like CASPT2/3 and NEVPT2.
- **Wide coverage:** Small organic molecules, medium-sized compounds, radicals, charged species, and transition metal complexes are included.
- **Double and Charge-Transfer Excitations:** Special focus on notoriously challenging cases such as **double excitations** and **intramolecular charge-transfer states**.
- **Revised and updated:** The database is continuously improved and extended based on new high-level calculations and critical evaluations.
- **Usability:** All data are carefully organized in `.xls` spreadsheets to ease extraction, manipulation, and comparison.

By providing chemically-accurate reference values, QUESTDB helps researchers:
- Assess the performance of density-functional approximations, wavefunction-based methods, and emerging excited-state theories.
- Guide the development of new quantum chemical methods for excited states.
- Facilitate the interpretation of experimental spectra and photochemical processes.

QUESTDB plays a central role in the **mountaineering strategy**, a systematic and rigorous protocol to climb toward the most accurate excited-state data achievable with modern computational resources.

---

## 📂 Repository Contents

This repository contains:
- **Molecular structures**
- **Vertical excitation energies**
- **Oscillator strengths**
- **Dipole moments**  
organized in `.xls` files for easy handling and analysis.
---

## 👥 Contributors

- [Denis Jacquemin](https://www.univ-nantes.fr/denis-jacquemin-1) (Nantes)
- [Pierre-François Loos](https://pfloos.github.io/WEB_LOOS) (Toulouse)
- [Martial Boggio-Pasqua](https://www.lcpq.ups-tlse.fr/spip.php?rubrique313&lang=fr) (Toulouse)
- [Fábris Kossoski](https://kossoski.github.io) (Toulouse)
- [Filippo Lipparini](https://people.unipi.it/filippo_lipparini) (Pisa)
- [Anthony Scemama](https://scemama.github.io)
- [Aymeric Blondel](https://www.univ-nantes.fr/aymeric-blondel) (Nantes)
- [Mickael Véril](https://mveril.github.io) (Toulouse)
- [Yann Damour](https://ydrnan.github.io/damour) (Toulouse)
- [Antoine Marie](https://antoine-marie.github.io) (Toulouse)

---

## 📚 Main References

- [**Reference energies for double excitations: improvement & extension**](https://doi.org/10.1021/acs.jctc.4c00175)  
  F. Kossoski, M. Boggio-Pasqua, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **20**, 5655 (2024).

- [**Reference vertical excitation energies for transition metal compounds**](https://doi.org/10.1021/acs.jctc.3c00696)  
  D. Jacquemin, F. Kossoski, F. Gam, M. Boggio-Pasqua, and P. F. Loos,  
  *J. Chem. Theory Comput.* **19**, 8782 (2023).

- [**A mountaineering strategy to excited states: revising reference values with EOM-CC4**](https://doi.org/10.1021/acs.jctc.2c00251)  
  P. F. Loos, F. Lipparini, D. A. Matthews, A. Blondel, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **18**, 4418 (2022).

- [**A mountaineering strategy to excited states: highly-accurate energies and benchmarks for bicyclic systems**](https://doi.org/10.1021/acs.jpca.1c09692)  
  P. F. Loos and D. Jacquemin,  
  *J. Phys. Chem. A* **125**, 10174 (2021).

- [**Reference energies for intramolecular charge-transfer excitations**](https://doi.org/10.1021/acs.jctc.1c00277)  
  P. F. Loos, M. Comin, X. Blase, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **17**, 3666 (2021).

- [**QUESTDB: a database of highly-accurate excitation energies for the electronic structure community**](https://doi.org/10.1002/wcms.1517)  
  M. Véril, A. Scemama, M. Caffarel, F. Lipparini, M. Boggio-Pasqua, D. Jacquemin, and P. F. Loos,  
  *WIREs Comput. Mol. Sci.* **11**, e1517 (2021).

- [**A mountaineering strategy to excited states: highly-accurate oscillator strengths and dipole moments of small molecules**](https://doi.org/10.1021/acs.jctc.0c01184)  
  A. Chrayteh, A. Blondel, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **17**, 416 (2021).

- [**A mountaineering strategy to excited states: highly-accurate energies and benchmarks for exotic molecules and radicals**](https://doi.org/10.1021/acs.jctc.0c00257)  
  P. F. Loos, A. Scemama, M. Boggio-Pasqua, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **16**, 3720 (2020).

- [**The quest for highly accurate excitation energies: a computational perspective**](https://doi.org/10.1021/acs.jpclett.0c00459)  
  P. F. Loos, A. Scemama, and D. Jacquemin,  
  *J. Phys. Chem. Lett.* **11**, 2374 (2020).

- [**A mountaineering strategy to excited states: highly-accurate energies and benchmarks for medium size molecules**](https://doi.org/10.1021/acs.jctc.9b01216)  
  P. F. Loos, F. Lipparini, M. Boggio-Pasqua, A. Scemama, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **16**, 1711 (2020).

- [**Reference energies for double excitations**](https://doi.org/10.1021/acs.jctc.8b01205)  
  P. F. Loos, M. Boggio-Pasqua, A. Scemama, M. Caffarel, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **15**, 1939 (2019).

- [**A mountaineering strategy to excited states: highly-accurate reference energies and benchmarks**](https://doi.org/10.1021/acs.jctc.8b00548)  
  P. F. Loos, A. Scemama, A. Blondel, Y. Garniron, M. Caffarel, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **14**, 4360 (2018).

---

## 📖 Other References

- **Excited-state absorption: Reference oscillator strengths, wavefunction and TD-DFT benchmarks**  
  J. Širůček, B. Le Guennic, Y. Damour, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* (in press).

- [**Heptazine, cyclazine, and related compounds: chemically-accurate estimates of the inverted singlet-triplet gap**](https://doi.org/10.1021/acs.jpclett.3c03249)  
  P. F. Loos, F. Lipparini, and D. Jacquemin,  
  *J. Phys. Chem. Lett.* **14**, 11069 (2023).

- [**Ground- and excited-state dipole moments and oscillator strengths of full configuration interaction quality**](https://doi.org/10.1021/acs.jctc.2c01047)  
  Y. Damour, R. Quintero-Monsebaiz, M. Caffarel, D. Jacquemin, F. Kossoski, A. Scemama, and P. F. Loos,  
  *J. Chem. Theory Comput.* **19**, 221 (2023).

- [**Benchmarking CASPT3 vertical excitation energies**](https://doi.org/10.1063/5.0086134)  
  M. Boggio-Pasqua, D. Jacquemin, and P. F. Loos,  
  *J. Chem. Phys.* **157**, 014103 (2022).

- [**Reference energies for cyclobutadiene: automerization and excited states**](https://doi.org/10.1021/acs.jpca.2c03646)  
  E. Monino, M. Boggio-Pasqua, A. Scemama, D. Jacquemin, and P. F. Loos,  
  *J. Phys. Chem. A* **126**, 4664 (2022).

- [**Assessing the performances of CASPT2 and NEVPT2 for vertical excitation energies**](https://doi.org/10.1021/acs.jctc.2c00088)  
  R. Sarkar, P. F. Loos, M. Boggio-Pasqua, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **18**, 2418 (2022).

- [**Benchmarking TD-DFT and wave function methods for oscillator strengths and excited-state dipole moments**](https://doi.org/10.1021/acs.jctc.0c01289)  
  R. Sarkar, M. Boggio-Pasqua, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **17**, 1106 (2021).

- [**Accurate full configuration interaction correlation energy estimates for five- and six-membered rings**](https://doi.org/10.1063/5.0066362)  
  Y. Damour, M. Véril, F. Kossoski, M. Caffarel, D. Jacquemin, A. Scemama, and P. F. Loos,  
  *J. Chem. Phys.* **155**, 134104 (2021).

- [**How accurate are EOM-CC4 vertical excitation energies?**](https://doi.org/10.1063/5.0059440)  
  P. F. Loos, D. A. Matthews, F. Lipparini, and D. Jacquemin,  
  *J. Chem. Phys.* **154**, 221103 (2021).

- [**Is ADC(3) as accurate as CC3 for valence and Rydberg excitation energies?**](https://doi.org/10.1021/acs.jpclett.9b03839)  
  P. F. Loos and D. Jacquemin,  
  *J. Phys. Chem. Lett.* **11**, 974 (2020).

- [**Cross comparisons between experiment, TD-DFT, CC and ADC for transition energies**](https://doi.org/10.1021/acs.jctc.9b00376)  
  C. Suellen, R. Garcia Freitas, P. F. Loos, and D. Jacquemin,  
  *J. Chem. Theory Comput.* **15**, 4581 (2019).

---

## 📂 Data Structure

- Molecular structures: `.xyz` or `.mol` files
- Excitation energies and oscillator strengths: `.xls` spreadsheets
- Additional metadata: [future extensions]

---

## 💰 Funding

<img src="https://lcpq.github.io/PTEROSOR/img/ERC.png" width="200" />

This project is supported by the [PTEROSOR](https://lcpq.github.io/PTEROSOR/) project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 863481).

---
