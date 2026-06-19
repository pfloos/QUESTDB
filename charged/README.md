# ⚡ Charged Excitations in the QUEST Database

This directory gathers high-quality benchmark data on **charged excitations** from the **QUEST database**, designed to assess and validate state-of-the-art electronic structure methods.

📌 At present, the dataset focuses exclusively on **electron-removal processes** (i.e., ionization potentials). Future releases will extend this collection to **electron-addition processes** (e.g., electron affinities).

---

## 📂 Contents

### 🔹 `valence_IPs/`
**Inner- and Outer-Valence Ionization Potentials & Satellite Transitions**

Based on the work of Marie and Loos (2024), this dataset provides reference data for:

- **58 valence ionization potentials**
- **42 satellite transition energies**

Each molecular `.json` file contains:

- Energies computed using a range of high-level **Green’s function** and **wavefunction-based methods**
- **Spectral weights** associated with quasiparticle and satellite states
- **Theoretical Best Estimates (TBEs)** and **FCI-based uncertainty estimates**
- Molecular **geometries** for full reproducibility

> 🧪 Particularly suited for benchmarking **inner- and outer-valence ionization energies** and assessing many-body approaches such as pp-RPA, pp-BSE, GW, DIP-EOM-CCSD, and related methods.

---

### 🔹 `core_IPs/`
**Core Ionization Potentials**

Based on the work of Marie, Burth, and Loos (2026), this dataset provides high-quality reference data for **core-level ionization potentials**.

This complementary collection is designed to benchmark methods targeting deep core electrons and core spectroscopies (e.g., X-ray photoelectron spectroscopy, XPS).

Each molecular `.json` file contains:

- Vertical **core-ionization energies** for selected atomic sites
- **Theoretical Best Estimates (TBEs)** or carefully assessed reference values
- **FCI reference energies** obtained within the **core-valence separation (CVS)** approximation
- Results from a variety of high-level methods, including ΔSCF, EOM-CC variants, Green’s function approaches, and related many-body methods
- Notes on convergence behavior and uncertainty estimates
- Molecular geometries and computational details for reproducibility

> 🧭 This dataset complements the valence IP and DIP collections and is particularly valuable for benchmarking core-level spectroscopies, basis-set effects, and relativistic corrections.

---

### 🔹 `DIPs/`
**Valence Double Ionization Potentials**

Based on the work of Marie *et al.* (2024), this dataset contains:

- **Singlet and triplet double ionization potentials (DIPs)** for **23 molecules**
- Reference calculations performed using the **aug-cc-pVTZ** basis set

Each `.json` file includes transition energies computed with methods such as **GW**, **CCSD**, and **CC4**, together with methodological details and theoretical justification.

> 🧬 This collection complements the valence IP dataset by providing challenging benchmarks for **two-electron removal processes**.

---

## 📈 Purpose

These datasets are intended to support:

- Development and benchmarking of electronic structure methods (Green’s function, EOM, CI, etc.)
- Assessment of **electron correlation effects**, **satellite structures**, and **multielectron processes**
- Construction of **machine-learning datasets** for charged excitations
- Systematic evaluation of accuracy across different ionization regimes (valence, core, and double ionization)
