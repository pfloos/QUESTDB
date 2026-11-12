# QUESTDB website

The explorer builds from the same Git checkout as the database. The build
requires Python 3.12 and Git. The published output contains only static files.

```bash
python3 website/build.py
python3 -m http.server 8000 --directory site
```

Open http://localhost:8000. Serve the directory over HTTP so the browser can
load the JSON collections. Each collection is loaded when selected. The
table displays 40 records at a time and CSV export includes the full filtered
result, the source JSON URL, and the source commit.

The five neutral collections retain the original metadata and computed
energies. Neutral method errors use the reported AVTZ reference for the
same row. In Benchmark, AVQZ is an optional reference for rows that report it. This changes the reference, while the method energies remain as recorded in the source. Core and valence benchmarks compare method values to sCI within the same source file, transition type, basis, and position in the source arrays. Valence IPs and satellites can be benchmarked separately. All summary metrics and plots use the intersection of states for the selected methods. The violin plot shows a smoothed density of signed errors and overlays the individual states.

The Geometries view and ZIP downloads use the [canonical geometry registry](../geometries/registry.json). The full archive contains each canonical geometry once. Individual collection archives include their shared canonical geometries, including a Double IP archive. Charged records link to the canonical preview and, when the source path differs, offer the original benchmark XYZ as a separate download. Existing direct XYZ URLs still serve their original files.

Canonical filenames use the geometry IDs. Those include `_GS`, `_ES`, or a descriptive suffix for documented variants. The ball-and-stick preview infers connectivity from interatomic distances and covalent radii.

Charged collections display the source's method, basis, sector,
and one-based root index. Root indices belong to the individual source
series. Use the original JSON and accompanying references to match states
between methods. Missing energies are omitted from the charged explorer,
and missing weights or uncertainties remain blank. The standalone core
file `charged/core_IPs/output.json` lacks a molecule, site, and method
identifier. It remains available in the source tree.

## Publish

Set **Settings → Pages → Build and deployment → Source** to **GitHub
Actions** in `pfloos/QUESTDB`. The **QUESTDB website** workflow builds on
pull requests and publishes on pushes to `main`. Run it manually for the
first deployment if the initial push happened before Pages was enabled.

Check the published explorer at https://pfloos.github.io/QUESTDB/. The
snapshot shown on the page should match the intended source commit.

The [legacy website](https://lcpq.github.io/QUESTDB_website/) displays a
move notice. Its source and history are in
[`LCPQ/QUESTDB_website`](https://github.com/LCPQ/QUESTDB_website).
