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
same row. In Benchmark, AVQZ is an optional reference for rows that report it. This changes the reference, while the method energies remain as recorded in the source. Core benchmarks match method values to sCI within the same core-site file, basis, and root. All summary metrics use the intersection of states for the selected methods.

The Geometries view contains the bundled neutral, valence-IP, and core-IP XYZ files. Downloaded bytes are identical to the source. Record-level links show matching labels and possible geometry variants.

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

After verifying the new site, the companion retirement patch can be
applied to `LCPQ/QUESTDB_website`. Commit it to the legacy repository's
default branch and run **Retire legacy QUEST website** from its Actions
tab. The retirement workflow checks the replacement before publishing a
move notice at the legacy URL. The old repository retains its source
and data history. The legacy Pages source should remain the root of
`gh-pages` for the retirement workflow.
