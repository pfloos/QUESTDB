"use strict";

const byId = id => document.getElementById(id);
const get = suffix => byId(`bench-${suffix}`);
const palette = ["#30508e", "#a93820", "#167166", "#76528f", "#866500", "#2f749a"];
let geometryIndex = [], geometryPromise, benchRows = [], benchData = [], benchMethods = [], benchGeneration = 0;
let plotPoints = [], structureAtoms = [], structureGeneration = 0, rotation = [0.3, 0.5], dragStart;
const benchCache = new Map();

function navigate() {
    const panels = [...document.querySelectorAll("[data-panel]")];
    const hash = location.hash.slice(1);
    const id = panels.some(panel => panel.id === hash) ? hash : "explore";
    for (const panel of panels) panel.hidden = panel.id !== id;
    for (const link of document.querySelectorAll("[data-view]")) {
        if (link.hash === `#${id}`) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
    }
    if (hash === id) requestAnimationFrame(() => byId(id).scrollIntoView());
    if (id === "benchmark" && manifest) {
        if (!benchRows.length) loadBenchmark();
        else plotBenchmark();
    }
    if (id === "geometries" && manifest) {
        loadGeometries().then(renderGeometries).catch(error => { byId("geometry-count").textContent = error.message; });
    }
}
window.addEventListener("hashchange", navigate);
document.addEventListener("quest-ready", () => {
    get("collection").replaceChildren(...manifest.collections
        .filter(collection => collection.kind === "neutral" || collection.id === "core_IPs")
        .map(collection => option(collection.name, collection.id)));
    navigate();
});

function loadGeometries() {
    if (!geometryPromise) {
        geometryPromise = json("data/geometries.json").then(data => {
            geometryIndex = data;
            return data;
        }).catch(error => {
            geometryPromise = null;
            throw error;
        });
    }
    return geometryPromise;
}

function geometryUrl(record) {
    return record.file.split("/").map(encodeURIComponent).join("/");
}

function geometryItem(record, tag) {
    const item = element(tag);
    item.className = "geometry-item";
    const actions = element("div");
    actions.className = "geometry-actions";
    const download = element("a", "Download XYZ");
    download.href = geometryUrl(record);
    download.download = record.file.split("/").pop();
    const preview = element("button", "3D preview");
    preview.type = "button";
    preview.addEventListener("click", () => openStructure(record));
    actions.append(download, preview);
    item.append(element("strong", record.name), element("small", record.group), actions);
    return item;
}

function showGeometryLinks(row, target) {
    target.textContent = "Looking for geometries…";
    loadGeometries().then(records => {
        const name = normal(row.molecule);
        const site = row.molecule.toLowerCase();
        const group = row.source.startsWith("charged/core_IPs/") ? "Core IP"
            : row.source.startsWith("charged/valence_IPs/") ? "Valence IP" : "Neutral";
        const candidates = records.filter(record => {
            if (record.group !== group) return false;
            const stem = record.name.toLowerCase();
            return normal(stem) === name || (group === "Core IP" && site.startsWith(stem + "_"))
                || (group === "Neutral" && normal(stem.replace(/_\d+$/, "")) === name);
        });
        if (candidates.length) {
            target.replaceChildren(element("p", "Geometries with matching source labels"),
                ...candidates.map(record => geometryItem(record, "div")));
        } else {
            const browse = element("a", "Browse all geometries");
            browse.href = "#geometries";
            browse.addEventListener("click", () => ui.detail.close());
            target.replaceChildren(element("p", "No geometry matched this source label."), browse);
        }
    }).catch(error => { target.textContent = error.message; });
}

function renderGeometries() {
    const query = normal(byId("geometry-query").value);
    const group = byId("geometry-group").value;
    const matches = geometryIndex.filter(record => normal(record.name).includes(query)
        && (!group || record.group === group));
    byId("geometry-count").textContent = `${matches.length} of ${geometryIndex.length} XYZ files`;
    byId("geometry-list").replaceChildren(...matches.map(record => geometryItem(record, "li")));
}

async function openStructure(record) {
    const dialog = byId("structure");
    const id = ++structureGeneration;
    byId("structure-title").textContent = record.name;
    byId("structure-download").href = geometryUrl(record);
    byId("structure-source").href = sourceLink(record.source);
    const status = byId("structure-status");
    status.textContent = "Loading structure…";
    structureAtoms = [];
    dialog.showModal();
    drawStructure();
    try {
        const response = await fetch(geometryUrl(record));
        if (!response.ok) throw new Error("Could not load this XYZ file.");
        const lines = (await response.text()).trimStart().split(/\r?\n/);
        const count = Number(lines[0]);
        const atoms = lines.slice(2, 2 + count).map(line => {
            const [symbol, x, y, z] = line.trim().split(/\s+/);
            return {symbol, xyz: [Number(x), Number(y), Number(z)]};
        });
        if (!Number.isInteger(count) || count < 1 || atoms.length !== count
            || atoms.some(atom => !atom.symbol || !atom.xyz.every(Number.isFinite))) {
            throw new Error("This XYZ file could not be rendered. The original is available above.");
        }
        if (id !== structureGeneration || !dialog.open) return;
        const center = [0, 1, 2].map(axis => atoms.reduce((sum, atom) => sum + atom.xyz[axis], 0) / atoms.length);
        structureAtoms = atoms.map(atom => ({...atom, xyz: atom.xyz.map((value, axis) => value - center[axis])}));
        rotation = [0.3, 0.5];
        status.textContent = `${count} atoms · drag or use the arrow keys to rotate`;
        drawStructure();
    } catch (error) {
        if (id === structureGeneration && dialog.open) status.textContent = error.message;
    }
}

function setupCanvas(canvas, width, height) {
    const scale = Math.min(2, window.devicePixelRatio || 1);
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    return ctx;
}

function drawStructure() {
    const canvas = byId("structure-canvas");
    const width = canvas.clientWidth, height = 300;
    if (!width) return;
    const ctx = setupCanvas(canvas, width, height);
    const extent = Math.max(1, ...structureAtoms.map(atom => Math.hypot(...atom.xyz)));
    const scale = Math.min(width, height) * 0.39 / extent;
    const [a, b] = rotation;
    const colors = {H: "#53647a", C: "#273752", O: "#a93820", N: "#30508e", F: "#167166", S: "#876600"};
    const atoms = structureAtoms.map(atom => {
        const [x, y, z] = atom.xyz;
        const u = Math.cos(a) * x + Math.sin(a) * z;
        const v = -Math.sin(a) * x + Math.cos(a) * z;
        return {...atom, px: width / 2 + u * scale,
            py: height / 2 - (Math.cos(b) * y - Math.sin(b) * v) * scale,
            depth: Math.sin(b) * y + Math.cos(b) * v};
    });
    for (const atom of atoms.sort((left, right) => left.depth - right.depth)) {
        const radius = atom.symbol === "H" ? 8 : 12;
        ctx.beginPath();
        ctx.arc(atom.px, atom.py, radius, 0, 2 * Math.PI);
        ctx.fillStyle = colors[atom.symbol] ?? "#6d4d87";
        ctx.fill();
        ctx.fillStyle = "white";
        ctx.font = "bold 11px system-ui";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(atom.symbol, atom.px, atom.py);
    }
}

const structureCanvas = byId("structure-canvas");
structureCanvas.addEventListener("pointerdown", event => {
    dragStart = [event.clientX, event.clientY];
    structureCanvas.setPointerCapture(event.pointerId);
});
structureCanvas.addEventListener("pointermove", event => {
    if (!dragStart) return;
    rotation[0] += (event.clientX - dragStart[0]) * 0.015;
    rotation[1] += (event.clientY - dragStart[1]) * 0.015;
    dragStart = [event.clientX, event.clientY];
    drawStructure();
});
for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
    structureCanvas.addEventListener(event, () => { dragStart = null; });
}
structureCanvas.addEventListener("keydown", event => {
    const steps = {ArrowLeft: [0, -0.13], ArrowRight: [0, 0.13], ArrowUp: [1, -0.13], ArrowDown: [1, 0.13]};
    if (!Object.hasOwn(steps, event.key)) return;
    event.preventDefault();
    const [axis, step] = steps[event.key];
    rotation[axis] += step;
    drawStructure();
});
byId("geometry-query").addEventListener("input", renderGeometries);
byId("geometry-group").addEventListener("change", renderGeometries);

function coreStates(data) {
    const states = new Map();
    for (const row of data) {
        const key = JSON.stringify([row.source, row.basis, row.root]);
        if (!states.has(key)) states.set(key, {...row, fields: Object.create(null)});
        const state = states.get(key);
        if (Object.hasOwn(state.fields, row.method)) throw new Error(`Duplicate core result: ${row.source} / ${row.basis} / ${row.method}`);
        state.fields[row.method] = row.energy;
    }
    return [...states.values()];
}

async function loadBenchmark() {
    const id = get("collection").value;
    const generation = ++benchGeneration;
    const status = get("status");
    benchRows = [];
    benchData = [];
    get("figure").hidden = true;
    get("export").disabled = true;
    get("stats").replaceChildren();
    get("method-list").replaceChildren();
    get("inspect").replaceChildren(option("Select a state", ""));
    status.textContent = "Loading benchmark…";
    try {
        if (!benchCache.has(id)) benchCache.set(id, await json(`data/${id}.json`));
        if (generation !== benchGeneration) return;
        const core = id === "core_IPs";
        const data = benchCache.get(id);
        benchRows = core ? coreStates(data) : data;
        const refs = core ? sorted(benchRows.filter(row => Number.isFinite(row.fields.sCI)).map(row => row.basis))
            : ["TBE/AVTZ", ...(benchRows.some(row => Number.isFinite(row.referenceQZ)) ? ["TBE/AVQZ"] : [])];
        get("reference").replaceChildren(...refs.map(ref => option(core ? `sCI / ${ref}` : ref, ref)));
        if (core && refs.includes("aug-cc-pCVTZ")) get("reference").value = "aug-cc-pCVTZ";
        choices(get("spin"), "All multiplicities", sorted(benchRows.map(row => row.spin)));
        choices(get("character"), "All characters", sorted(benchRows.map(row => row.character)));
        for (const field of ["spin", "character"]) get(`${field}-label`).hidden = get(field).options.length === 1;
        get("safe-label").hidden = core || !benchRows.some(row => row.safe != null);
        get("safe").checked = false;
        const methods = sorted(benchRows.flatMap(row => Object.entries(row.fields)
            .filter(([key, value]) => Number.isFinite(value) && (core ? key !== "sCI" : !metadata.test(key)))
            .map(([key]) => key)));
        const defaults = core ? ["G0W0@HF", "evGW@HF", "qsGW"] : ["ADC(2)", "CC3", "CCSD"];
        get("method-list").replaceChildren(...methods.map((method, index) => {
            const label = element("label");
            const input = element("input");
            input.type = "checkbox";
            input.value = method;
            input.checked = methods.some(name => defaults.includes(name)) ? defaults.includes(method) : index < 2;
            input.addEventListener("change", renderBenchmark);
            label.append(input, method);
            return label;
        }));
        renderBenchmark();
    } catch (error) {
        if (generation === benchGeneration) status.textContent = error.message;
    }
}

function renderBenchmark() {
    if (!benchRows.length) return;
    const core = get("collection").value === "core_IPs";
    const ref = get("reference").value;
    const spin = get("spin").value, character = get("character").value;
    benchMethods = [...get("method-list").querySelectorAll(":checked")].map(input => input.value);
    const eligible = benchRows
        .filter(row => (core ? row.basis === ref
            : (!spin || String(row.spin) === spin) && (!character || row.character === character)
                && (!get("safe").checked || row.safe === "Y")))
        .map(row => ({row, reference: core ? row.fields.sCI : ref === "TBE/AVQZ" ? row.referenceQZ : row.energy}))
        .filter(item => Number.isFinite(item.reference));
    benchData = benchMethods.length
        ? eligible.filter(({row}) => benchMethods.every(method => Number.isFinite(row.fields[method]))) : [];
    get("status").textContent = benchMethods.length
        ? `${benchData.length} shared states from ${eligible.length} eligible references. Coverage is shown before matching.`
        : "Select at least one method.";
    get("stats").replaceChildren(...benchMethods.map(method => {
        const errors = benchData.map(item => item.row.fields[method] - item.reference);
        const abs = errors.map(Math.abs).sort((a, b) => a - b);
        const n = errors.length;
        const mean = values => values.reduce((a, b) => a + b, 0) / n;
        const median = (abs[Math.floor(n / 2)] + abs[Math.floor((n - 1) / 2)]) / 2;
        const tr = element("tr");
        for (const value of [method, `${eligible.filter(({row}) => Number.isFinite(row.fields[method])).length} / ${eligible.length}`, n,
            fixed(mean(errors)), fixed(mean(abs)), fixed(Math.sqrt(mean(errors.map(error => error ** 2)))), fixed(median)]) {
            tr.append(element("td", value));
        }
        return tr;
    }));
    get("inspect").replaceChildren(option("Select a state", ""), ...benchData.map(({row, reference}, index) =>
        option(`${row.molecule} · ${row.state} · ${fixed(reference)} eV`, index)));
    get("export").disabled = !benchData.length;
    get("figure").hidden = !benchData.length;
    plotBenchmark();
}

function inspectBenchmark(index) {
    const item = benchData[index];
    if (!item) return;
    const {row, reference} = item;
    const fields = {"Reference (eV)": reference};
    for (const method of benchMethods) {
        fields[`${method} (eV)`] = row.fields[method];
        fields[`${method} error (eV)`] = fixed(row.fields[method] - reference);
    }
    showDetails({...row, fields});
}

function plotBenchmark() {
    const canvas = get("chart");
    const width = canvas.clientWidth;
    if (!benchData.length || !width) return;
    const parity = get("plot").value === "parity";
    const height = parity ? 360 : Math.max(220, 65 + 28 * benchMethods.length);
    const ctx = setupCanvas(canvas, width, height);
    const left = parity ? 60 : Math.min(145, width * .34);
    const right = width - 20, top = 18, bottom = height - 38;
    const data = benchMethods.flatMap((method, j) => benchData.map((item, i) => ({
        i, j, ref: item.reference, value: item.row.fields[method], error: item.row.fields[method] - item.reference,
    })));
    let min = parity ? Infinity : -.01;
    let max = parity ? -Infinity : .01;
    for (const point of data) {
        min = Math.min(min, ...(parity ? [point.ref, point.value] : [-Math.abs(point.error)]));
        max = Math.max(max, ...(parity ? [point.ref, point.value] : [Math.abs(point.error)]));
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
    ctx.fillText(parity ? "Reference energy (eV)" : "Signed error (eV)", (left + xRight) / 2, height - 6);
    if (parity) {
        ctx.save();
        ctx.translate(12, (bottom + yTop) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Calculated energy (eV)", 0, 0);
        ctx.restore();
    }
    const band = (bottom - top) / benchMethods.length;
    if (!parity) {
        ctx.textAlign = "right";
        benchMethods.forEach((method, j) => {
            ctx.fillText(method, left - 7, top + (j + .5) * band + 4, left - 12);
        });
    }
    plotPoints = data.map(p => ({
        i: p.i, x: x(parity ? p.ref : p.error),
        y: parity ? y(p.value) : top + (p.j + .5) * band + ((p.i % 9) - 4) * Math.min(1.35, band / 14),
        color: palette[p.j % palette.length],
    }));
    ctx.globalAlpha = .65;
    for (const point of plotPoints) {
        ctx.beginPath(); ctx.arc(point.x, point.y, 3.2, 0, 2 * Math.PI);
        ctx.fillStyle = point.color; ctx.fill();
    }
    ctx.globalAlpha = 1;
    const caption = get("caption");
    caption.replaceChildren(element("span", "Select a point to inspect its source state. "));
    if (parity) {
        benchMethods.forEach((method, j) => {
            const legend = element("span", `● ${method}  `);
            legend.style.color = palette[j % palette.length];
            caption.append(legend);
        });
    }
}

get("collection").addEventListener("change", loadBenchmark);
for (const field of ["reference", "spin", "character", "safe"]) get(field).addEventListener("change", renderBenchmark);
get("plot").addEventListener("change", plotBenchmark);
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
get("export").addEventListener("click", () => {
    const records = [["Molecule/site", "State", "Reference", "Reference (eV)", "Method", "Energy (eV)", "Error (eV)", "Source JSON", "QUESTDB commit"],
        ...benchData.flatMap(({row, reference}) => benchMethods.map(method => [row.molecule, row.state,
            get("reference").selectedOptions[0].textContent, reference, method, row.fields[method],
            row.fields[method] - reference, sourceLink(row.source), manifest.revision]))];
    const csv = records.map(row => row.map(csvField).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const link = element("a");
    link.href = url;
    link.download = `QUESTDB-benchmark-${get("collection").value}-${manifest.revision.slice(0, 7)}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("resize", () => {
    if (!byId("benchmark").hidden) plotBenchmark();
    if (byId("structure").open) drawStructure();
});
navigate();
