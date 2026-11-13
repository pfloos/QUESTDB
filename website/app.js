"use strict";

const byId = id => document.getElementById(id);
const ui = Object.fromEntries(
    ["snapshot", "query", "safe", "export", "status", "error", "head", "rows", "prev",
        "page", "next", "detail", "detail-title", "detail-source", "detail-fields"]
        .map(id => [id, byId(id)])
);
const cache = new Map();
const pageSize = 40;
const metadata = /^(?:Size|Group|Spin|TBE(?:\/|$)|%|f\s*\[)/i;
const defaultNeutralBasis = "aug-cc-pVTZ";
const neutralBasisSuffix = /\s*\/\s*(AV[DTQ56]Z|aug-cc-pV[DTQ56]Z|cc-pV[DTQ56]Z)\s*$/i;
const characterNames = {V: "Valence", R: "Rydberg", M: "Mixed"};
const spinNames = {1: "Singlet", 2: "Doublet", 3: "Triplet", 4: "Quartet", 5: "Quintet"};
let manifest, moleculePromise, explorePromise, exploreReady = false;
let filtered = [], page = 0, exploreRows = [];
const alphabetical = new Intl.Collator();
const numerical = new Intl.Collator(undefined, {numeric: true});
const moleculeSearch = new Map();

function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
}

function option(label, value) {
    const node = element("option", label);
    node.value = value;
    return node;
}

function choices(select, label, values) {
    select.replaceChildren(option(label, ""), ...values.map(value => option(value, value)));
}

function sorted(values) {
    return [...new Set(values.filter(value => value != null && value !== ""))]
        .sort(numerical.compare);
}

async function json(path) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
        const response = await fetch(path, {signal: controller.signal});
        if (!response.ok) throw new Error(`The data request returned HTTP ${response.status}.`);
        return await response.json();
    } catch (error) {
        if (controller.signal.aborted) throw new Error("The data request timed out.");
        if (error instanceof TypeError) throw new Error("The data request could not connect.");
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

function loadMolecules() {
    if (!moleculePromise) {
        moleculePromise = json(snapshotUrl("data/molecules.json")).catch(error => {
            moleculePromise = null;
            throw error;
        });
    }
    return moleculePromise;
}

function showLoadError(status, message, retry) {
    const button = element("button", "Retry");
    button.type = "button";
    button.addEventListener("click", retry);
    status.replaceChildren(message + " ", button);
}

function snapshotUrl(path) {
    const encoded = path.split("/").map(encodeURIComponent).join("/");
    return `${encoded}?v=${encodeURIComponent(manifest.revision)}`;
}

function sourceLink(path, type = "blob") {
    return `${manifest.repository}/${type}/${manifest.revision}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function stateParts(value) {
    const match = String(value ?? "").trim().replaceAll('"', "''").match(/^\^\s*(\d+)\s*\\*(Sigma|Pi|Delta|Phi|Gamma|[A-Z])(?:_(?:\{([\da-z]+)\}|([\da-z]+)))?\^?([+'-]*)(\s*\[F\])?$/);
    if (!match) return null;
    const [, multiplicity, label, braced, bare, signs, note] = match;
    const symbol = {Sigma: "Σ", Pi: "Π", Delta: "Δ", Phi: "Φ", Gamma: "Γ"}[label] ?? label;
    return [multiplicity, symbol, braced ?? bare ?? "", signs.replace(/''/g, "″").replace(/'/g, "′").replace(/-/g, "−"), note ? " [F]" : ""];
}

function stateLabel(value) {
    const parts = stateParts(value);
    const label = element("span", parts ? "" : value ?? "—");
    if (parts) {
        const [multiplicity, symbol, subscript, superscript, note] = parts;
        label.className = "state-label";
        label.append(element("sup", multiplicity), symbol);
        if (subscript) label.append(element("sub", subscript));
        if (superscript) label.append(element("sup", superscript));
        if (note) label.append(note);
    }
    return label;
}

function stateOptionText(value) {
    const parts = stateParts(value);
    if (!parts) return value;
    const [multiplicity, symbol, subscript, superscript, note] = parts;
    const sub = {"0":"₀","1":"₁","2":"₂","3":"₃",a:"ₐ",e:"ₑ",i:"ᵢ",o:"ₒ",u:"ᵤ",v:"ᵥ"};
    const suffix = [...subscript].every(char => Object.hasOwn(sub, char))
        ? [...subscript].map(char => sub[char]).join("") : (subscript ? `(${subscript})` : "");
    return [...multiplicity].map(digit => "⁰¹²³⁴⁵⁶⁷⁸⁹"[digit]).join("") + symbol + suffix
        + superscript.replace(/\+/g, "⁺").replace(/−/g, "⁻") + note;
}

function normal(value) {
    return String(value ?? "").normalize("NFKC").toLowerCase()
        .replace(/[^\p{L}\p{N}]+/gu, "");
}

function moleculeSearchKey(value) {
    const text = value.trim();
    return normal(text) + (/[+-]$/.test(text) ? text.slice(-1) : "");
}

function fixed(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function systemText(row, formatted = false) {
    const formula = formatted ? String(row.formula).replace(/\d/g, digit => "₀₁₂₃₄₅₆₇₈₉"[digit]) : row.formula;
    return [row.name ?? row.molecule, formula === row.name ? null : formula, row.site].filter(Boolean).join(" · ");
}

function moleculeLabel(row, includeSite = true) {
    const label = element("span", row.name ?? row.molecule);
    if (row.formula && row.formula !== (row.name ?? row.molecule)) {
        label.append(" (", ...row.formula.split(/(\d+)/).map((part, i) =>
            i % 2 ? element("sub", part) : part), ")");
    }
    if (includeSite && row.site) label.append(` · ${row.site}`);
    return label;
}

function moleculeLink(row) {
    const link = element("a");
    link.href = `#molecule?id=${encodeURIComponent(row.moleculeId)}`;
    link.append(moleculeLabel(row));
    return link;
}

function collectionName(id) {
    return manifest.collections.find(collection => collection.id === id)?.name ?? id;
}

function loadCollection(id) {
    if (!cache.has(id)) {
        cache.set(id, json(snapshotUrl(`data/${id}.json`)).catch(error => {
            cache.delete(id);
            throw error;
        }));
    }
    return cache.get(id);
}

function normalizedBasis(basis) {
    return basis.replace(/^AV([DTQ56]Z)$/i, (_, tail) => "aug-cc-pV" + tail.toUpperCase())
        .replace(/^(aug-cc-p|cc-p)(C?V[DTQ56]Z)$/i, (_, prefix, tail) => prefix.toLowerCase() + tail.toUpperCase());
}

function referenceLabel(reference) {
    return reference.replace(/^TBE\/(AV[DTQ56]Z)$/, (_, basis) => `TBE / ${normalizedBasis(basis)}`);
}

function computations(row) {
    if (!row.fields) return [{method: row.method, basis: normalizedBasis(row.basis), energy: row.energy}];
    return Object.entries(row.fields)
        .filter(([key, value]) => Number.isFinite(value) && !metadata.test(key))
        .map(([label, energy]) => {
            const suffix = label.match(neutralBasisSuffix);
            const basis = row.methodBases?.[label] ?? suffix?.[1] ?? defaultNeutralBasis;
            return {method: suffix ? label.slice(0, suffix.index) : label, basis: normalizedBasis(basis), energy};
        });
}

function stateText(row) {
    const suffix = row.root ? ` #${row.root}` : row.occurrence > 1 ? ` (${row.occurrence})` : "";
    return stateOptionText(row.state.trim()) + suffix;
}

function stateButton(row) {
    const button = element("button");
    button.className = "row-link";
    button.type = "button";
    button.append(stateLabel(row.state.trim()));
    if (row.root) button.append(` #${row.root}`);
    if (row.occurrence > 1) button.append(` (${row.occurrence})`);
    button.addEventListener("click", () => showDetails(row));
    return button;
}

// Checkbox menus keep multiple selections usable on touchscreens and keyboards.
function createFilter(target, label, change, empty = "All") {
    const details = element("details");
    details.className = "filter-menu";
    const summary = element("summary");
    const options = element("div");
    options.className = "filter-options";
    const clear = element("button", empty === "All" ? "Clear selection" : "Clear");
    clear.type = "button";
    const list = element("div");
    options.append(clear, list);
    details.append(summary, options);
    target.append(details);
    let items = new Map();
    const filter = {
        selected: () => [...items].filter(([, item]) => item.input.checked).map(([value]) => value),
        set(values) {
            const wanted = new Set(values);
            for (const [value, item] of items) item.input.checked = wanted.has(value);
            update();
        },
        options(values, format = value => value) {
            const previous = filter.selected();
            items = new Map(values.map(value => {
                value = String(value);
                const item = element("label");
                const input = element("input");
                input.type = "checkbox";
                const text = element("span", format(value));
                const count = element("small");
                item.append(input, text, count);
                return [value, {input, item, count}];
            }));
            list.replaceChildren(...[...items.values()].map(item => item.item));
            target.hidden = !items.size;
            filter.set(previous);
        },
        counts(counts) {
            for (const [value, {input, count}] of items) {
                const total = counts.get(value) ?? 0;
                input.disabled = total === 0 && !input.checked;
                count.textContent = total.toLocaleString();
            }
        },
    };
    function update() {
        const count = filter.selected().length;
        summary.textContent = `${label} · ${count || empty}`;
        clear.disabled = !count;
    }
    details.addEventListener("toggle", () => {
        if (details.open) for (const other of document.querySelectorAll(".filter-menu[open]")) {
            if (other !== details) other.open = false;
        }
    });
    details.addEventListener("keydown", event => {
        if (event.key === "Escape" && details.open) {
            details.open = false;
            summary.focus();
        }
    });
    list.addEventListener("change", () => { update(); change(); });
    clear.addEventListener("click", () => { filter.set([]); change(); summary.focus(); });
    return filter;
}

document.addEventListener("click", event => {
    for (const menu of document.querySelectorAll(".filter-menu[open]")) {
        if (!menu.contains(event.target)) menu.open = false;
    }
});

function csvField(value) {
    if (value == null) return "";
    if (typeof value === "number") return String(value);
    const text = String(value);
    const safeText = /^[\s]*[=+@-]/.test(text) ? "'" + text : text;
    return '"' + safeText.replaceAll('"', '""') + '"';
}

function downloadCsv(name, records) {
    const csv = records.map(row => row.map(csvField).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const link = element("a");
    link.href = url;
    link.download = `QUESTDB-${name}-${manifest.revision.slice(0, 12)}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const facetValues = {
    collection: row => [row.collection],
    method: row => row.calculations.map(item => item.method),
    basis: row => row.calculations.map(item => item.basis),
    character: row => [row.character],
    spin: row => [row.spin],
    transition: row => [row.fields ? (row.fluorescence ? "Fluorescence" : "Vertical excitation") : row.state],
};
const facetLabels = {collection: "Collection", method: "Method", basis: "Basis set", character: "Character",
    spin: "Multiplicity", transition: "Transition"};
const facetFormats = {collection: collectionName, character: value => characterNames[value] ?? value,
    spin: value => spinNames[value] ?? value};
const exploreFilters = Object.fromEntries(Object.keys(facetValues).map(key =>
    [key, createFilter(byId(key), facetLabels[key], () => { page = 0; renderExplore(); })]));
const columns = ["Molecule", "State", "Collection", "Character", "Method", "Basis", "Energy (eV)",
    `${referenceLabel("TBE/AVTZ")} (eV)`, `${referenceLabel("TBE/AVQZ")} (eV)`, "Z", "Safe"];

function matchesCalculation(calculation, selections) {
    return (!selections.method.size || selections.method.has(calculation.method))
        && (!selections.basis.size || selections.basis.has(calculation.basis));
}

function recordValues(row, selections) {
    const selected = row.calculations.filter(calculation => matchesCalculation(calculation, selections));
    const showMethods = !row.fields || selections.method.size || selections.basis.size;
    const single = selected.length === 1 ? selected[0] : null;
    return [systemText(row), row.state.trim() + (row.root ? ` #${row.root}` : ""), collectionName(row.collection),
        characterNames[row.character] ?? row.character,
        showMethods ? single?.method ?? `${selected.length} selected` : "TBE",
        showMethods ? single?.basis ?? "See state details" : defaultNeutralBasis,
        showMethods ? single?.energy : row.energy, row.fields ? row.energy : null, row.referenceQZ, row.weight, row.safe];
}

function renderExplore() {
    if (!exploreReady) return;
    const query = ui.query.value.trim();
    const exact = moleculeSearch.get(moleculeSearchKey(query));
    const tokens = query.split(/\s+/).map(normal).filter(Boolean);
    const selection = Object.fromEntries(Object.entries(exploreFilters)
        .map(([key, filter]) => [key, new Set(filter.selected())]));
    const counts = Object.fromEntries(Object.keys(facetValues).map(key => [key, new Map()]));
    filtered = [];
    for (const row of exploreRows) {
        if (!(exact ? exact.has(row.moleculeId) : tokens.every(token => row.search.includes(token)))
            || (ui.safe.checked && (!row.fields || row.safe !== "Y"))) continue;
        const failures = Object.keys(facetValues).filter(key => key !== "method" && key !== "basis"
            && selection[key].size && !facetValues[key](row).some(value => selection[key].has(String(value))));
        const methods = row.calculations.filter(c => !selection.basis.size || selection.basis.has(c.basis));
        const bases = row.calculations.filter(c => !selection.method.size || selection.method.has(c.method));
        const calculated = (!selection.method.size && !selection.basis.size)
            || row.calculations.some(c => matchesCalculation(c, selection));
        if (!failures.length) {
            if (calculated) filtered.push(row);
            for (const [key, candidates] of [["method", methods.map(c => c.method)], ["basis", bases.map(c => c.basis)]]) {
                for (const value of new Set(candidates)) counts[key].set(value, (counts[key].get(value) ?? 0) + 1);
            }
        }
        if (!calculated || failures.length > 1) continue;
        for (const [key, values] of Object.entries(facetValues)) {
            if (key === "method" || key === "basis" || (failures.length && key !== failures[0])) continue;
            for (const value of values(row)) {
                if (value != null) counts[key].set(String(value), (counts[key].get(String(value)) ?? 0) + 1);
            }
        }
    }
    for (const [key, filter] of Object.entries(exploreFilters)) filter.counts(counts[key]);
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.min(page, pages - 1);
    ui.rows.replaceChildren(...filtered.slice(page * pageSize, (page + 1) * pageSize).map(row => {
        const tr = element("tr");
        recordValues(row, selection).forEach((value, index) => {
            const td = element("td");
            if (index === 0) td.append(moleculeLink(row));
            else if (index === 1) td.append(stateButton(row));
            else td.textContent = typeof value === "number" ? fixed(value) : value || "—";
            tr.append(td);
        });
        return tr;
    }));
    ui.status.textContent = `${filtered.length.toLocaleString()} of ${exploreRows.length.toLocaleString()} records`;
    ui.page.textContent = filtered.length ? `Page ${page + 1} of ${pages}` : "No matching records";
    ui.prev.disabled = page === 0;
    ui.next.disabled = page + 1 >= pages;
    ui.export.disabled = !filtered.length;
}

ui.export.addEventListener("click", () => {
    const selection = Object.fromEntries(Object.entries(exploreFilters)
        .map(([key, filter]) => [key, new Set(filter.selected())]));
    downloadCsv("explore", [[...columns, "State ID", "Source JSON", "QUESTDB commit"],
        ...filtered.flatMap(row => {
            const values = recordValues(row, selection);
            const extras = [row.stateId, sourceLink(row.source), manifest.revision];
            if (!row.fields || (!selection.method.size && !selection.basis.size)) return [[...values, ...extras]];
            return row.calculations.filter(c => matchesCalculation(c, selection))
                .map(c => [...values.slice(0, 4), c.method, c.basis, c.energy, ...values.slice(7), ...extras]);
        })]);
});

function energyMatrix(calculations) {
    const methods = sorted(calculations.map(entry => entry.method));
    const bases = sorted(calculations.map(entry => entry.basis));
    const table = element("table");
    const head = element("thead");
    const header = element("tr");
    for (const title of ["Method", ...bases]) {
        const th = element("th", title);
        th.scope = "col";
        header.append(th);
    }
    head.append(header);
    const body = element("tbody");
    for (const method of methods) {
        const tr = element("tr");
        tr.append(element("td", method));
        for (const basis of bases) {
            const energy = calculations.find(entry => entry.method === method && entry.basis === basis)?.energy;
            tr.append(element("td", fixed(energy)));
        }
        body.append(tr);
    }
    table.append(head, body);
    const wrapper = element("div");
    wrapper.className = "table-wrap compact-table";
    wrapper.tabIndex = 0;
    wrapper.append(table);
    return wrapper;
}

let detailGeneration = 0;
async function showDetails(row, benchmarkFields = null) {
    const generation = ++detailGeneration;
    ui["detail-title"].replaceChildren(moleculeLabel(row), " · ", stateLabel(row.state.trim()));
    if (row.root || row.occurrence > 1) ui["detail-title"].append(` #${row.root ?? row.occurrence}`);
    ui["detail-source"].href = sourceLink(row.source);
    byId("detail-paper").href = `https://doi.org/${row.paper}`;
    byId("detail-molecule").href = moleculeLink(row).href;
    const ref = byId("detail-reference");
    const matrix = byId("detail-matrix");
    const entries = row.fields ?? {
        "Source state": row.state, Method: row.method, Basis: normalizedBasis(row.basis), Root: row.root,
        "Energy (eV)": row.energy, Z: row.weight, "Uncertainty (eV)": row.uncertainty,
    };
    function fieldsList(fields) {
        return Object.entries(fields).flatMap(([key, value]) => [element("dt", key), element("dd", value ?? "—")]);
    }
    ui["detail-fields"].replaceChildren(...fieldsList(entries));
    ui["detail-fields"].parentElement.open = false;
    ref.replaceChildren();
    matrix.textContent = "Loading calculated energies…";
    showGeometryLinks(row, byId("detail-geometries"));
    if (!ui.detail.open) ui.detail.showModal();
    const section = element("dl");
    if (benchmarkFields) section.append(...fieldsList(benchmarkFields));
    if (row.fields) {
        section.append(...fieldsList({
            [`${referenceLabel("TBE/AVTZ")} (eV)`]: fixed(row.energy), "Reference method": row.fields.Method ?? row.fields["Method (all in RO)"],
            ...(Number.isFinite(row.referenceQZ) ? {[`${referenceLabel("TBE/AVQZ")} (eV)`]: fixed(row.referenceQZ),
                "Basis correction method": row.fields["Corr. Method"]} : {}),
            "Reliability": row.safe === "Y" ? "Safe (~50 meV)" : row.safe === "N" ? "Flagged by the dataset" : "Unreported",
            ...(row.fields["Special ?"] ? {"Source annotation": String(row.fields["Special ?"]).split(/\s+/)
                .map(code => ({FL: "Fluorescence", PD: "Partial double excitation", GD: "Genuine double excitation",
                    wCT: "Weak charge transfer", sCT: "Strong charge transfer"})[code] ?? code).join(" · ")} : {}),
            ...(row.strengthSource ? {[row.strengthSource]: row.strength} : {}),
        }));
    }
    if (section.children.length) ref.append(section);
    for (const revision of row.revisions ?? []) {
        const link = element("a", "Source correction");
        link.href = `${manifest.repository}/commit/${revision.commit}`;
        const name = referenceLabel(revision.field === "energy" ? "TBE/AVTZ" : "TBE/AVQZ");
        const note = element("p", `${name} changed from ${fixed(revision.previous)} to ${fixed(revision.value)} eV. ${revision.reason} `);
        note.append(link);
        ref.append(note);
    }
    try {
        const values = row.fields ? computations(row) : (await loadCollection(row.collection))
            .filter(entry => entry.stateId === row.stateId);
        if (generation !== detailGeneration) return;
        if (!row.fields) {
            const references = values.filter(entry => entry.method === "sCI");
            if (references.length) {
                const fields = element("dl");
                fields.append(...fieldsList(Object.fromEntries(references.map(entry => [
                    `sCI · ${entry.basis} (eV)`, fixed(entry.energy)
                    + (Number.isFinite(entry.uncertainty) ? ` ± ${fixed(entry.uncertainty)}` : ""),
                ]))));
                ref.append(fields);
            }
        }
        matrix.replaceChildren();
        if (values.length) {
            matrix.append(element("h3", "Calculated energies (eV)"), energyMatrix(values));
        }
    } catch (error) {
        if (generation === detailGeneration) matrix.textContent = error.message;
    }
}

byId("detail-molecule").addEventListener("click", () => ui.detail.close());
for (const [control, event] of [[ui.query, "input"], [ui.safe, "change"]]) {
    control.addEventListener(event, () => { page = 0; renderExplore(); });
}
ui.prev.addEventListener("click", () => { page--; renderExplore(); });
ui.next.addEventListener("click", () => { page++; renderExplore(); });

function loadExplore() {
    if (explorePromise) return explorePromise;
    ui.error.hidden = true;
    ui.status.textContent = "Loading data…";
    explorePromise = (async () => {
        const [molecules, ...data] = await Promise.all([
            loadMolecules(), ...manifest.collections.map(item => loadCollection(item.id)),
        ]);
        for (const molecule of Object.values(molecules)) {
            for (const alias of [molecule.name, molecule.formula, ...molecule.aliases]) {
                const key = moleculeSearchKey(alias);
                if (!key) continue;
                if (!moleculeSearch.has(key)) moleculeSearch.set(key, new Set());
                moleculeSearch.get(key).add(molecule.id);
            }
        }
        exploreRows = data.flat().sort((a, b) => alphabetical.compare(a.name, b.name)).map(row => ({
            ...row, calculations: computations(row),
            search: normal([systemText(row), row.state, row.type, row.source, row.method,
                ...molecules[row.moleculeId].aliases].join(" ")),
        }));
        for (const [key, values] of Object.entries(facetValues)) {
            const choices = key === "collection" ? manifest.collections.map(c => c.id)
                : sorted(exploreRows.flatMap(values));
            exploreFilters[key].options(choices, facetFormats[key]);
        }
        const header = element("tr");
        for (const column of columns) {
            const th = element("th", column);
            th.scope = "col";
            header.append(th);
        }
        ui.head.append(header);
        exploreReady = true;
        renderExplore();
    })().catch(error => {
        explorePromise = null;
        ui.error.hidden = false;
        ui.status.textContent = "";
        showLoadError(ui.error, `Could not load the Explorer. ${error.message}`, loadExplore);
    });
    return explorePromise;
}

function init() {
    try {
        manifest = JSON.parse(byId("site-manifest").textContent);
        ui.safe.checked = false;
        const revision = element("a", manifest.revision.slice(0, 7));
        revision.href = `${manifest.repository}/commit/${manifest.revision}`;
        ui.snapshot.replaceChildren(
            `${manifest.neutralStates.toLocaleString()} neutral states · ${manifest.collections.length} collections · ${manifest.date.slice(0, 10)} · `, revision
        );
        for (const link of document.querySelectorAll("[data-source]")) {
            link.href = sourceLink(link.dataset.source, "tree");
        }
        for (const pack of manifest.benchmarkPacks) {
            const li = element("li");
            const title = element("strong", pack.name);
            const details = element("small", `${pack.count} states · ${pack.geometries} XYZ · ${pack.reference}`);
            const link = element("a", "Download ZIP");
            link.href = snapshotUrl(pack.file);
            link.download = "";
            link.setAttribute("aria-label", `Download ${pack.name} benchmark ZIP`);
            const text = element("span");
            text.append(title, details);
            li.append(text, link);
            byId(`packs-${pack.kind}`).append(li);
        }
        document.dispatchEvent(new Event("quest-ready"));
    } catch (error) {
        const message = "Could not start QUESTDB. Reload the page to try again.";
        ui.error.hidden = false;
        ui.error.textContent = message;
        ui.status.textContent = "";
        byId("bench-status").textContent = message;
        byId("geometry-count").textContent = message;
        byId("molecule-states").textContent = message;
    }
}

document.addEventListener("DOMContentLoaded", init, {once: true});
