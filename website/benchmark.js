"use strict";

const get = suffix => byId(`bench-${suffix}`);
const palette = ["#30508e", "#a93820", "#167166", "#76528f", "#866500", "#2f749a"];
const coreGF2Methods = ["GF2@HF", "evGF2@HF", "qsGF2"];
let benchRows = [], benchData = [], benchMethods = [], benchGeneration = 0, benchReady = false;
let benchStats = [], plotPoints = [];
let methodOptions = [], pickedReference;

const filters = Object.fromEntries([
    ["sector", "Transition"], ["spin", "Multiplicity"], ["character", "Character"],
    ["method-list", "Method", "None"], ["basis", "Basis set"], ["spec", "Exact combinations"],
].map(([key, label, empty]) => [key, createFilter(get(key), label, renderBenchmark, empty)]));

function hasBenchmark() {
    return benchReady;
}

function selectedFilters() {
    return Object.fromEntries(Object.entries(filters).map(([key, filter]) => [key, new Set(filter.selected())]));
}

function specKey(method, basis) {
    return JSON.stringify([method, basis]);
}

function normalizedBasis(basis) {
    return /^aug-cc-p(cv|v)([dtq56])z$/i.test(basis)
        ? basis.replace(/aug-cc-p(\w+)$/i, (_, tail) => "aug-cc-p" + tail.toUpperCase()) : basis;
}

function benchmarkStates(data) {
    const states = new Map();
    for (const row of data) {
        if (!states.has(row.stateId)) {
            states.set(row.stateId, {row, calculations: new Map(), refs: new Map()});
        }
        const entry = states.get(row.stateId);
        for (const result of computations(row)) {
            const basis = normalizedBasis(result.basis);
            if (!row.fields && result.method === "sCI") entry.refs.set(basis, result.energy);
            else {
                const key = specKey(result.method, basis);
                if (entry.calculations.has(key)) throw new Error(`Duplicate result for ${stateText(row)} / ${key}`);
                entry.calculations.set(key, {...result, basis});
            }
        }
        if (row.fields) {
            entry.refs.set("TBE/AVTZ", row.energy);
            if (Number.isFinite(row.referenceQZ)) entry.refs.set("TBE/AVQZ", row.referenceQZ);
        }
    }
    return [...states.values()];
}

async function loadBenchmark(params = new URLSearchParams()) {
    const id = params.get("set");
    if (id && manifest.collections.some(c => c.id === id)) get("collection").value = id;
    const group = get("collection").value;
    const generation = ++benchGeneration;
    benchReady = false;
    benchRows = [];
    benchData = [];
    get("figure").hidden = true;
    get("gf2").hidden = true;
    for (const field of ["export", "export-stats", "share"]) get(field).disabled = true;
    get("stats").replaceChildren();
    get("inspect").replaceChildren(option("Select a state", ""));
    get("status").textContent = "Loading benchmark…";
    get("share-status").hidden = true;
    try {
        const data = await loadCollection(group);
        if (generation !== benchGeneration) return;
        benchRows = benchmarkStates(data);
        const neutral = Boolean(data[0]?.fields);
        const refs = sorted(benchRows.flatMap(item => [...item.refs.keys()]));
        const preferred = {core_IPs: "aug-cc-pCVTZ", valence_IPs: "aug-cc-pVTZ", DIPs: "aug-cc-pVTZ"}[group]
            ?? "TBE/AVTZ";
        get("reference").replaceChildren(...refs.map(ref => option(neutral ? ref : `sCI / ${ref}`, ref)));
        pickedReference = refs.includes(params.get("ref")) ? params.get("ref") : preferred;
        if (refs.includes(pickedReference)) get("reference").value = pickedReference;
        else pickedReference = get("reference").value;
        const unique = new Map();
        for (const {calculations} of benchRows) for (const [key, spec] of calculations) unique.set(key, spec);
        methodOptions = [...unique].map(([key, {method, basis}]) => ({key, method, basis}))
            .sort((a, b) => numerical.compare(a.method, b.method) || alphabetical.compare(a.basis, b.basis));
        const methods = sorted(methodOptions.map(spec => spec.method));
        const bases = sorted(methodOptions.map(spec => spec.basis));
        const sectors = neutral ? [] : sorted(benchRows.map(({row}) => row.state));
        filters.sector.options(sectors, value => value === "IP" ? "Valence IPs" : value);
        filters.spin.options(sorted(benchRows.map(({row}) => row.spin)), value => spinNames[value] ?? value);
        filters.character.options(sorted(benchRows.map(({row}) => row.character)), value => characterNames[value] ?? value);
        filters["method-list"].options(methods);
        filters.basis.options(bases);
        filters.spec.options(methodOptions.map(spec => spec.key), methodLabel);
        get("safe-label").hidden = !neutral || !benchRows.some(({row}) => row.safe != null);
        get("safe").checked = !get("safe-label").hidden && params.get("safe") === "1";
        get("matched").checked = params.get("all") !== "1";
        get("plot").value = ["errors", "violin", "parity", "paired"].includes(params.get("plot")) ? params.get("plot") : "errors";
        const desired = group === "core_IPs" ? ["G0W0@HF", "evGW@HF", "qsGW"]
            : group === "valence_IPs" ? ["CC3", "CCSDT", "GW"]
            : group === "DIPs" ? ["DIP-EOM-CCSD", "ppRPA@HF", "ppRPA@GW"]
            : ["ADC(2)", "CC3", "CCSD"];
        const sameBasis = methodOptions.filter(spec => spec.basis === preferred);
        const available = sorted((sameBasis.length ? sameBasis : methodOptions).map(spec => spec.method));
        const defaults = desired.filter(method => available.includes(method));
        const defaultMethods = defaults.length ? defaults : available.slice(0, 2);
        const shared = params.has("set");
        filters["method-list"].set(shared ? params.getAll("method") : defaultMethods);
        filters.basis.set(shared ? params.getAll("basis") : bases.includes(preferred) ? [preferred] : []);
        filters.sector.set(shared ? params.getAll("sector") : group === "valence_IPs" ? ["IP"] : []);
        for (const facet of ["spin", "character", "spec"]) filters[facet].set(params.getAll(facet));
        get("gf2").hidden = group !== "core_IPs" || !coreGF2Methods.every(method => methods.includes(method));
        benchReady = true;
        renderBenchmark();
        if (params.has("revision") && params.get("revision") !== manifest.revision) {
            showShareMessage("This link was created with a different QUESTDB commit. The comparison uses the current data.");
        }
    } catch (error) {
        if (generation !== benchGeneration) return;
        benchReady = false;
        showLoadError(get("status"), `Could not load the benchmark. ${error.message}`, () => loadBenchmark(params));
    }
}

function benchmarkStats(data, method) {
    const errors = data.filter(item => item.plotted.has(method))
        .map(item => item.calculations.get(method).energy - item.reference);
    const n = errors.length;
    const abs = errors.map(Math.abs).sort((a, b) => a - b);
    const mean = values => n ? values.reduce((sum, value) => sum + value, 0) / n : null;
    return {n, bias: mean(errors), mae: mean(abs),
        rmse: n ? Math.sqrt(mean(errors.map(error => error ** 2))) : null,
        median: n ? (abs[Math.floor(n / 2)] + abs[Math.floor((n - 1) / 2)]) / 2 : null};
}

function renderBenchmark() {
    if (!benchReady) return;
    const id = get("collection").value;
    const ref = get("reference").value;
    pickedReference = ref;
    const selections = selectedFilters();
    const exact = selections.spec;
    const allowsMethod = spec => selections["method-list"].has(spec.method);
    const allowsBasis = spec => !selections.basis.size || selections.basis.has(spec.basis);
    const allowsSpec = spec => !exact.size || exact.has(spec.key ?? specKey(spec.method, spec.basis));
    const facets = {sector: row => row.state, spin: row => row.spin, character: row => row.character};
    const counts = Object.fromEntries(Object.keys(filters).map(key => [key, new Map()]));
    const eligible = [];
    const coverage = new Map();
    for (const item of benchRows) {
        const reference = item.refs.get(ref);
        if (!Number.isFinite(reference) || (get("safe").checked && item.row.safe !== "Y")) continue;
        const failures = Object.keys(facets).filter(key => selections[key].size && !selections[key].has(String(facets[key](item.row))));
        if (failures.length > 1) continue;
        const specs = [...item.calculations.values()];
        const hasSelectedSpec = (!selections["method-list"].size && !exact.size)
            || specs.some(spec => exact.size ? allowsSpec(spec) : allowsMethod(spec) && allowsBasis(spec));
        for (const [key, value] of Object.entries(facets)) {
            if (hasSelectedSpec && (!failures.length || failures[0] === key)) addCounts(key, [value(item.row)]);
        }
        if (failures.length) continue;
        eligible.push({...item, reference});
        for (const key of item.calculations.keys()) coverage.set(key, (coverage.get(key) ?? 0) + 1);
        addCounts("method-list", specs.filter(spec => !selections.basis.size || selections.basis.has(spec.basis)).map(spec => spec.method));
        addCounts("basis", specs.filter(spec => exact.size ? allowsSpec(spec) : !selections["method-list"].size || allowsMethod(spec)).map(spec => spec.basis));
        addCounts("spec", specs.map(spec => specKey(spec.method, spec.basis)));
    }
    function addCounts(key, values) {
        for (const value of new Set(values.filter(value => value != null).map(String))) {
            counts[key].set(value, (counts[key].get(value) ?? 0) + 1);
        }
    }
    for (const [key, filter] of Object.entries(filters)) filter.counts(counts[key]);
    // Ignore method/basis pairs that do not exist anywhere in this collection.
    const selected = methodOptions.filter(spec => exact.size ? allowsSpec(spec) : allowsMethod(spec) && allowsBasis(spec));
    benchMethods = selected.map(spec => spec.key);
    const shared = get("matched").checked || get("plot").value === "paired";
    const sharedCount = benchMethods.length ? eligible.filter(item =>
        benchMethods.every(key => item.calculations.has(key))).length : 0;
    benchData = benchMethods.length ? eligible.flatMap(item => {
        const plotted = new Set(benchMethods.filter(key => item.calculations.has(key)));
        if (!plotted.size || (shared && plotted.size !== benchMethods.length)) return [];
        return [{...item, plotted}];
    }) : [];
    const plotAllowed = get("plot").value !== "paired" || benchMethods.length === 2;
    get("status").textContent = benchMethods.length
        ? `${sharedCount} shared states of ${eligible.length} eligible references · ${benchData.length} states in the ${shared ? "shared sample" : "combined sample"}`
            + (exact.size ? " · Using exact method/basis selections." : "")
            + (!plotAllowed ? " · Select exactly two method/basis combinations for the paired plot." : "")
            + (selected.some(spec => spec.basis !== ref) && !ref.startsWith("TBE/") ? ` · All errors use sCI / ${ref}.` : "")
            + (id === "valence_IPs" || id === "DIPs" ? " · Roots align by position in the source arrays." : "")
        : "Select methods with available basis sets.";
    benchStats = selected.map(spec => ({
        ...spec, ...benchmarkStats(benchData, spec.key), coverage: coverage.get(spec.key) ?? 0, eligible: eligible.length,
    }));
    get("stats").replaceChildren(...benchStats.map(item => {
        const tr = element("tr");
        for (const value of [`${item.method} · ${item.basis}`, `${item.coverage} / ${item.eligible}`, item.n,
            fixed(item.bias), fixed(item.mae), fixed(item.rmse), fixed(item.median)]) tr.append(element("td", value));
        return tr;
    }));
    get("inspect").replaceChildren(option("Select a state", ""), ...benchData.map(({row, reference}, i) =>
        option(`${systemText(row, true)} · ${stateText(row)} · ${fixed(reference)} eV`, i)));
    get("export").disabled = !benchData.length || !plotAllowed;
    get("export-stats").disabled = !benchStats.length;
    get("share").disabled = !benchMethods.length;
    get("figure").hidden = !benchData.length || !plotAllowed;
    if (get("figure").hidden) plotPoints = [];
    else plotBenchmark();
}

function methodLabel(key) {
    const [method, basis] = JSON.parse(key);
    return `${method} · ${basis}`;
}

function inspectBenchmark(index) {
    const item = benchData[index];
    if (!item) return;
    const {row, reference, calculations, plotted} = item;
    const fields = {[`${get("reference").selectedOptions[0].textContent} (eV)`]: fixed(reference)};
    for (const key of plotted) {
        const energy = calculations.get(key).energy;
        fields[`${methodLabel(key)} (eV)`] = fixed(energy);
        fields[`${methodLabel(key)} error (eV)`] = fixed(energy - reference);
    }
    showDetails(row, fields);
}

function violinProfile(errors, min, max) {
    const mean = errors.reduce((sum, error) => sum + error, 0) / errors.length;
    const deviation = Math.sqrt(errors.reduce((sum, error) => sum + (error - mean) ** 2, 0) / errors.length);
    if (!deviation || errors.length < 2) return [];
    const bandwidth = Math.max((max - min) / 80, 1.06 * deviation * errors.length ** -.2);
    const start = Math.max(min, Math.min(...errors) - 3 * bandwidth);
    const end = Math.min(max, Math.max(...errors) + 3 * bandwidth);
    return Array.from({length: 81}, (_, index) => {
        const value = start + index * (end - start) / 80;
        const density = errors.reduce((sum, error) =>
            sum + Math.exp(-.5 * ((value - error) / bandwidth) ** 2), 0) / (errors.length * bandwidth);
        return {value, density};
    });
}

function drawViolins(ctx, x, min, max, top, band) {
    const profiles = benchMethods.map(method =>
        violinProfile(benchData.filter(item => item.plotted.has(method))
            .map(item => item.calculations.get(method).energy - item.reference), min, max));
    const peak = Math.max(0, ...profiles.flatMap(profile => profile.map(point => point.density)));
    profiles.forEach((profile, index) => {
        if (!profile.length) return;
        const center = top + (index + .5) * band;
        const halfWidth = density => .42 * band * density / peak;
        ctx.beginPath();
        ctx.moveTo(x(profile[0].value), center);
        for (const point of profile) ctx.lineTo(x(point.value), center - halfWidth(point.density));
        for (const point of [...profile].reverse()) ctx.lineTo(x(point.value), center + halfWidth(point.density));
        ctx.closePath();
        ctx.fillStyle = ctx.strokeStyle = palette[index % palette.length];
        ctx.globalAlpha = .22;
        ctx.fill();
        ctx.globalAlpha = .8;
        ctx.stroke();
    });
    ctx.globalAlpha = 1;
}

function plotBenchmark() {
    const canvas = get("chart");
    const width = canvas.clientWidth;
    if (!benchData.length || !width) return;
    const parity = get("plot").value === "parity";
    const paired = get("plot").value === "paired";
    const violin = get("plot").value === "violin";
    const bands = paired ? 1 : benchMethods.length;
    const height = parity ? 360 : Math.max(220, 65 + (violin ? 47 : 28) * bands);
    const ctx = setupCanvas(canvas, width, height);
    const left = parity ? 60 : paired ? 35 : Math.min(200, width * .34);
    const right = width - 20, top = 18, bottom = height - 38;
    const data = paired ? benchData.map((item, i) => {
        const [a, b] = benchMethods.map(key => Math.abs(item.calculations.get(key).energy - item.reference));
        return {i, j: 0, error: a - b, color: Math.abs(a - b) < 1e-12 ? "#506079" : palette[a < b ? 0 : 1]};
    }) : benchMethods.flatMap((method, j) => benchData.flatMap((item, i) => {
        if (!item.plotted.has(method)) return [];
        const value = item.calculations.get(method).energy;
        return [{i, j, ref: item.reference, value, error: value - item.reference}];
    }));
    let min = parity ? Infinity : -.01;
    let max = parity ? -Infinity : .01;
    for (const point of data) {
        min = Math.min(min, ...(parity ? [point.ref, point.value] : [-Math.abs(point.error)]));
        max = Math.max(max, ...(parity ? [point.ref, point.value] : [Math.abs(point.error)]));
    }
    if (violin) {
        min *= 1.2;
        max *= 1.2;
    }
    max = Math.max(min + .01, max);
    const size = Math.min(right - left, bottom - top);
    const xRight = parity ? left + size : right;
    const yTop = parity ? bottom - size : top;
    const x = value => left + (value - min) / (max - min) * (xRight - left);
    const y = value => bottom - (value - min) / (max - min) * (bottom - yTop);
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#354662";
    ctx.strokeStyle = "#a2acc0";
    ctx.beginPath();
    ctx.moveTo(left, yTop); ctx.lineTo(left, bottom); ctx.lineTo(xRight, bottom);
    ctx.stroke();
    ctx.beginPath();
    if (parity) { ctx.moveTo(left, bottom); ctx.lineTo(xRight, yTop); }
    else { ctx.moveTo(x(0), top); ctx.lineTo(x(0), bottom); }
    ctx.stroke();
    ctx.textAlign = "center";
    for (let tick = 0; tick <= 4; tick++) {
        const value = min + tick * (max - min) / 4;
        ctx.fillText(fixed(value, max - min >= 10 ? 0 : 2), x(value), bottom + 15);
        if (parity) {
            ctx.textAlign = "right";
            ctx.fillText(fixed(value, max - min >= 10 ? 0 : 2), left - 5, y(value) + 4);
            ctx.textAlign = "center";
        }
    }
    ctx.fillText(parity ? "Reference energy (eV)" : paired ? "|Error A| − |Error B| (eV)" : "Signed error (eV)", (left + xRight) / 2, height - 6);
    if (parity) {
        ctx.save();
        ctx.translate(12, (bottom + yTop) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Calculated energy (eV)", 0, 0);
        ctx.restore();
    }
    const band = (bottom - top) / bands;
    if (violin) drawViolins(ctx, x, min, max, top, band);
    if (!parity && !paired) {
        ctx.textAlign = "right";
        benchMethods.forEach((method, j) => {
            ctx.fillText(methodLabel(method), left - 7, top + (j + .5) * band + 4, left - 12);
        });
    }
    plotPoints = data.map(p => ({
        i: p.i, x: x(parity ? p.ref : p.error),
        y: parity ? y(p.value) : top + (p.j + .5) * band + ((p.i % 9) - 4) * Math.min(1.35, band / 14),
        color: p.color ?? palette[p.j % palette.length],
    }));
    ctx.globalAlpha = .65;
    for (const point of plotPoints) {
        ctx.beginPath(); ctx.arc(point.x, point.y, 3.2, 0, 2 * Math.PI);
        ctx.fillStyle = point.color; ctx.fill();
    }
    ctx.globalAlpha = 1;
    const caption = get("caption");
    caption.replaceChildren(element("span", (violin ? "Violin width shows estimated density. " : "")
        + "Select a point to inspect its source state. "));
    if (paired) {
        const winsA = data.filter(point => point.error < -1e-12).length;
        const winsB = data.filter(point => point.error > 1e-12).length;
        caption.append(element("span", `A smaller: ${winsA} · B smaller: ${winsB} · Tied: ${data.length - winsA - winsB}. `));
    }
    if (parity || paired) {
        benchMethods.forEach((method, j) => {
            const legend = element("span", `● ${paired ? (j === 0 ? "A: " : "B: ") : ""}${methodLabel(method)}  `);
            legend.style.color = palette[j % palette.length];
            caption.append(legend);
        });
    }
}

function benchmarkParams() {
    const params = new URLSearchParams({
        set: get("collection").value, ref: pickedReference,
        plot: get("plot").value, revision: manifest.revision,
    });
    if (get("safe").checked) params.set("safe", "1");
    if (!get("matched").checked) params.set("all", "1");
    for (const [field, filter] of Object.entries(filters)) {
        for (const value of filter.selected()) params.append(field === "method-list" ? "method" : field, value);
    }
    return params;
}

function showShareMessage(message) {
    get("share-status").hidden = false;
    get("share-status").textContent = message;
}

document.addEventListener("quest-ready", () => {
    get("collection").replaceChildren(...manifest.collections.map(c => option(c.name, c.id)));
    navigate();
});
get("collection").addEventListener("change", () => loadBenchmark());
get("gf2").addEventListener("click", () => {
    get("reference").value = "aug-cc-pCVTZ";
    filters.basis.set(["aug-cc-pCVTZ"]);
    filters["method-list"].set(coreGF2Methods);
    filters.spec.set([]);
    renderBenchmark();
});
for (const field of ["reference", "safe", "matched", "plot"]) get(field).addEventListener("change", renderBenchmark);
get("inspect").addEventListener("change", () => {
    if (get("inspect").value !== "") inspectBenchmark(Number(get("inspect").value));
    get("inspect").value = "";
});
get("chart").addEventListener("click", event => {
    const box = get("chart").getBoundingClientRect();
    let closest, distance = 9;
    for (const point of plotPoints) {
        const d = Math.hypot(point.x - event.clientX + box.left, point.y - event.clientY + box.top);
        if (d < distance) { distance = d; closest = point; }
    }
    if (closest) inspectBenchmark(closest.i);
});
get("share").addEventListener("click", async () => {
    const url = new URL(location.href);
    url.hash = `benchmark?${benchmarkParams()}`;
    try {
        await navigator.clipboard.writeText(url.href);
        showShareMessage("Copied comparison link.");
    } catch {
        showShareMessage("");
        const input = element("input");
        input.type = "url";
        input.readOnly = true;
        input.value = url.href;
        input.setAttribute("aria-label", "Comparison link");
        get("share-status").append(input);
        input.focus();
        input.select();
    }
});

function benchmarkExportMeta() {
    return [get("collection").value, get("reference").selectedOptions[0].textContent, get("plot").value,
        get("matched").checked || get("plot").value === "paired" ? "Shared" : "Per method", manifest.revision,
        benchmarkParams().toString()];
}
const exportMetaColumns = ["Collection", "Reference", "Plot", "Sample", "QUESTDB commit", "Comparison parameters"];

get("export").addEventListener("click", () => {
    const meta = benchmarkExportMeta();
    const paired = get("plot").value === "paired";
    const records = [[...exportMetaColumns, "State ID", "Molecule", "Formula", "Site", "State", "Root",
        "Reference (eV)", "Method", "Basis", "Calculated (eV)", "Signed error (eV)",
        "Paired |error A| - |error B| (eV)", "Source JSON"]];
    for (const {row, calculations, reference, plotted} of benchData) {
        const delta = paired ? [...plotted].reduce((difference, key, index) =>
            difference + (index ? -1 : 1) * Math.abs(calculations.get(key).energy - reference), 0) : null;
        for (const key of plotted) {
            const {method, basis, energy} = calculations.get(key);
            records.push([...meta, row.stateId, row.name, row.formula, row.site, row.state.trim(),
                row.root ?? row.occurrence, reference, method, basis, energy, energy - reference, delta, sourceLink(row.source)]);
        }
    }
    downloadCsv("benchmark", records);
});
get("export-stats").addEventListener("click", () => {
    const meta = benchmarkExportMeta();
    downloadCsv("benchmark-statistics", [[...exportMetaColumns, "Method", "Basis", "Coverage", "Eligible states",
        "States plotted", "Bias (eV)", "MAE (eV)", "RMSE (eV)", "Median |error| (eV)", "State IDs"],
    ...benchStats.map(({key, method, basis, coverage, eligible, n, bias, mae, rmse, median}) => [
        ...meta, method, basis, coverage, eligible, n, bias, mae, rmse, median,
        benchData.filter(item => item.plotted.has(key)).map(item => item.row.stateId).join(" "),
    ])]);
});
window.addEventListener("resize", () => {
    if (!byId("benchmark").hidden && !get("figure").hidden) plotBenchmark();
});
