# Core Ionization Potentials

<!-- GW residues added 2026-11-09 -->

## GW quasiparticle energies and renormalization factors

For each of the 84 CVS core-ionization targets, the `G0W0@PBEh`,
`G0W0@HF`, `evGW@HF`, and `qsGW` entries use the same `Energy`/`Z`
representation used in `charged/valence_IPs`:

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

The `G0W0@HF` and `evGW@HF` residues and `qsGW` self-energy derivative
**diagnostics** come from the core-IP QuAcK calculations used in the
2026 benchmark (84 matching states, aug-cc-pCVTZ). For qsGW, `Z` is the
self-energy derivative evaluated for the quasiparticle solution, not an
independent spectral weight of the static qsGW effective Hamiltonian.
These methods' calculated IP energies are preserved as previously stored.
The full computational settings and orbital-assignment caveats are given
in `charged/README.md`. In particular, the published PBEh starting point
uses 45% exchange for second-row and 70% for third-row core IPs.

The `gw_z_provenance.csv` file in this directory provides the 84 source
records, the higher-precision calculated GW energies, site-mapping
metadata, and the four residues. The update script and its source table
are also available in the corresponding update package.
