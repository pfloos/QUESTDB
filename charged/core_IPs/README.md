# Core Ionization Potentials

## GF2 core-IP benchmark

The per-target JSON files include `GF2@HF`, `evGF2@HF`, and `qsGF2` core ionization energies at aug-cc-pCVTZ for all 84 CVS core targets. `GF2@HF` is the single-shot HF-based calculation. The eigenvalue-self-consistent and quasiparticle-self-consistent results use the SRG variants identified in the source data.

The JSON entries preserve the supplied energy precision and include the reported `Z` for `GF2@HF` and `evGF2@HF`. The qsGF2 self-energy derivative diagnostic is recorded only in the provenance table.

- [Per-target results and provenance](gf2_provenance.csv), including the CVS-FCI reference, orbital assignment, convergence status, and qsGF2 diagnostic.
- [Benchmark statistics](gf2_statistics.csv), reported against the same CVS-FCI references as the core-IP dataset.

On the [QUESTDB website](https://pfloos.github.io/QUESTDB/#benchmark), choose **Core ionizations** and **Select GF2 methods**.

<!-- GW residues added 2026-11-09 -->

## GW quasiparticle energies and renormalization factors

For each of the 84 CVS core-ionization targets, `G0W0@PBEh`,
`G0W0@HF`, and `evGW@HF` use the same `Energy`/`Z` representation used
in `charged/valence_IPs`, for example:

```json
"G0W0@HF": {
    "Energy": { "aug-cc-pCVTZ": 301.53 },
    "Z":      { "aug-cc-pCVTZ": 0.881860 }
}
```

`Energy` contains core IPs in eV, while `Z` is dimensionless. The PBEh
quasiparticle residues were transcribed from Tables S1--S2 of the SI for
A. Marie, L. Burth, and P.-F. Loos, *J. Chem. Theory Comput.* **22**,
6609 (2026), DOI: https://doi.org/10.1021/acs.jctc.6c00713 .
SI: https://antoine-marie.github.io/manuscript/15_Quest_coreIP_SI.pdf .
PBEh `Z` values have the two-decimal precision reported in the SI. Its
published IP energies and the corrected CH3NC site labels are preserved.

The `G0W0@HF` and `evGW@HF` residues come from the core-IP QuAcK
calculations used in the 2026 benchmark (84 matching states, aug-cc-pCVTZ).
The qsGW calculations also produced self-energy derivative diagnostics,
but these are not residues of the auxiliary static qsGW Green's function.
Therefore, qsGW `Z` values are **not reported in the per-target JSON files**;
the separately labeled `qsGW_Z_diagnostic` column is retained in
`gw_z_provenance.csv` for reproducibility. The qsGW IPs are unchanged.
The full computational settings and orbital-assignment caveats are given
in `charged/README.md`. In particular, the published PBEh starting point
uses 45% exchange for second-row and 70% for third-row core IPs.

The `gw_z_provenance.csv` file in this directory provides the 84 source
records, the higher-precision calculated GW energies, site-mapping
metadata, three GW residues, and the separate qsGW self-energy derivative
diagnostic. The update script and its source table
are also available in the corresponding update package.
