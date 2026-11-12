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

The Geometries view contains the bundled neutral, valence-IP, and core-IP XYZ files. Download individual files, the selected collection, or the full geometry archive. The five neutral subset archives are assembled from molecule labels in the current data, so a geometry can occur in more than one subset. The full archive includes each source XYZ file once and retains the source paths and original bytes.

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
