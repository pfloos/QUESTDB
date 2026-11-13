# QUESTDB website

The explorer builds from the same Git checkout as the database. The build
requires Python 3.12 and Git. The published output contains only static files.

```bash
python3 website/build.py
python3 -m http.server 8000 --directory site
```

Open http://localhost:8000. Serve the directory over HTTP so the browser can
load the JSON collections. The site is built from the repository's JSON data
and geometry registry. The Explorer displays 40 records per page and exports
the full filtered result as CSV with state IDs, source URLs, and the commit.

## Explore

Search accepts a molecule name, formula, source alias, or state text. Formula
matches preserve separate isomers. The formula index is generated from XYZ
element counts. Common names and display formulas that differ from those
counts are maintained in `catalog.py`.

The molecule page gathers states from all associated collections and puts
the canonical geometry first. It also lists separately documented structures
and provides direct XYZ downloads and a ball-and-stick preview with inferred
bonds. A charged state can also link to its original benchmark XYZ. Neutral
fluorescence entries use their documented excited-state structures.
All other geometry downloads use the [registry](../geometries/registry.json).

The state view displays the source JSON, collection citation, reference
method, basis correction and reliability annotation when recorded, plus a
method × basis table. Corrections with documented previous values and reasons
can be added to `REFERENCE_REVISIONS` in `catalog.py`. Where the neutral JSON
method column has no basis tag, the site marks the basis as unlabelled.

Molecule spectra use the TBE/AVTZ vertical energies and reported oscillator
strengths of the chosen collection. The broadening is a normalized Gaussian
or Lorentzian with an adjustable FWHM (default 0.20 eV). Each curve's line areas
are weighted by oscillator strength. Reported zeros remain clickable on the
baseline and unreported strengths appear on a separate row. Spectrum exports
include the sampled curve, transitions, line shape, FWHM and source revision.

## Benchmark

Options within a filter are combined with OR. Separate filters are combined
with AND. Select exact method/basis combinations to compare a particular
pair such as one method at one basis against a different method at another.
Neutral references are TBE/AVTZ and, where available, TBE/AVQZ. Charged
references are sCI at the selected basis, even when a method uses another
basis. Valence IPs, satellites and DIPs align by the one-based position in
their source arrays. Check the source and citations for state assignments.

The default sample is the intersection across selected method/basis
combinations. Coverage counts are measured before this intersection.
With **Shared states only** cleared, each method uses all eligible states
that it covers. The violin plot overlays the individual states on the
smoothed signed-error distribution.

Share links restore the filters, reference and plot, and report when the
site has moved to a different commit. The plotted-data and statistics CSVs
include the source commit and exact state IDs.

The downloadable Diet-50 and core-IP packs have fixed membership
(`benchmark-cases.json`). Each build uses the current checkout's energies
and exact corresponding XYZ files. Archives carry the source commit and
include reference values, citations and a blank results CSV. To reproduce
an older pack, check out its commit and rebuild the site.

Missing charged energies are omitted and missing uncertainties or weights
remain blank. The separate `charged/core_IPs/output.json` lacks a molecule,
site, and method identifier and remains in the source tree.

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
