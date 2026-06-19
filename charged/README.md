# ⚡ Charged Excitations of the QUEST Database

This directory gathers high-quality data on **charged excitations** from the **QUEST database**, offering a benchmark for state-of-the-art electronic structure methods.

📌 At this stage, the dataset includes **electron-removal processes** (i.e., ionization potentials). Future updates aim to incorporate **electron-addition transitions** (e.g., electron affinities).

---

## 📂 Contents

### 🔹 `IPs/`  
**Inner- and Outer-Valence Ionization Potentials & Satellite Transitions**  
Based on the study by [Marie and Loos (2024)](https://pubs.acs.org/10.1021/acs.jctc.4c00216), this dataset focuses on:
- **58 valence ionization potentials**
- **42 satellite transition energies**
  
Each molecule’s `.json` file contains:
- Energies computed with a variety of high-level **Green's function** and **wavefunction methods**
- **Spectral weights** for quasiparticle and satellite states
- **TBEs** and **FCI-based uncertainties**
- **Geometries** provided for reproducibility

> 🧪 Ideal for benchmarking **inner- and outer-valence IPs** and validating many-body methods like pp-RPA, pp-BSE, GW, DIP-EOM-CCSD, etc.

---

### 🔹 `Core_IPs/`  
**Core (inner-shell) Ionization Potentials**  
We have added a complementary dataset of high-quality **core-level (inner-shell) ionization potentials**. This new collection is intended to support benchmarking of methods that target deep core electrons and core spectroscopies (XPS-like transitions).

What to expect in `charged/Core_IPs/`:
- Per-molecule `.json` files containing core-ionization energies (vertical) for selected atoms/sites in each molecule
- Theoretical Best Estimates (TBEs) or carefully assessed reference values, where available
- Method-specific results from high-level approaches (ΔSCF/Δ coupled-cluster, EOM-CC variants, Green's function and related many-body approaches) and notes on convergence/uncertainty
- Geometries and computational details necessary for reproducibility

> 🧭 This dataset complements the valence IPs and DIPs sets and is especially useful to evaluate methods for core-level spectroscopy and to test basis-set and relativistic effects.

---

### 🔹 `DIPs/`  
**Valence Double Ionization Potentials (DIPs)**  
From [Marie *et al.* (2024)](https://doi.org/10.1063/5.0250155), this set includes:
- **Singlet and triplet DIPs** for 23 molecules
- All computed using the **aug-cc-pVTZ** basis

Each `.json` file contains excitation energies obtained with methods like **GW**, **CCSD**, or **CC4**, along with their theoretical rationale.

> 🧬 Complements the valence IP dataset with challenging **two-electron removal** benchmarks.

---

## 📈 Purpose

These datasets aim to support:
- Method development and comparison (Green’s function, EOM, CI, etc.)
- Assessment of **electron correlation**, **satellite states**, and **multielectron processes**
- Construction of **machine learning datasets** for charged excitations
