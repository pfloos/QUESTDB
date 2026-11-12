"use strict";

const byId = id => document.getElementById(id);
const get = suffix => byId(`bench-${suffix}`);
const palette = ["#30508e", "#a93820", "#167166", "#76528f", "#866500", "#2f749a"];
let geometryIndex = [], geometryPromise, benchRows = [], benchData = [], benchMethods = [], benchGeneration = 0;
let plotPoints = [], structureAtoms = [], structureBonds = [], structureGeneration = 0;
let rotation = [0.3, 0.5], structureZoom = 1, dragStart;
const covalentRadii = {
    H: .31, Li: 1.28, Be: .96, B: .84, C: .76, N: .71, O: .66, F: .57,
    Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Sc: 1.70, Ti: 1.60, Cu: 1.32, Zn: 1.22,
};
const atomColors = {
    H: "#ecf1f7", C: "#43546b", N: "#3867c5", O: "#d34536", F: "#48a157",
    Cl: "#398a42", S: "#e2b43a", P: "#db8333", B: "#c07863", Cu: "#a96535",
};
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
        .filter(collection => collection.kind === "neutral" || ["core_IPs", "valence_IPs"].includes(collection.id))
        .map(collection => option(collection.name, collection.id)));
    byId("geometry-group").replaceChildren(...manifest.geometryDownloads
        .map(group => option(group.name, group.id)));
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

function geometryUrl(path) {
    return path.split("/").map(encodeURIComponent).join("/");
}

function geometryItem(record, tag) {
    const item = element(tag);
    item.className = "geometry-item";
    const actions = element("div");
    actions.className = "geometry-actions";
    const download = element("a", "Download XYZ");
    download.href = geometryUrl(record.file);
    download.download = record.file.split("/").pop();
    const preview = element("button", "3D preview");
    preview.type = "button";
    preview.addEventListener("click", () => openStructure(record));
    actions.append(download, preview);
    item.append(element("strong", record.name), element("small", record.groups.join(" · ")), actions);
    if (record.note) item.append(element("p", record.note));
    return item;
}

function showGeometryLinks(row, target) {
    target.textContent = "Looking for geometries…";
    loadGeometries().then(records => {
        const subset = row.source.split("/")[2];
        const candidates = row.geometryId
            ? records.filter(record => record.id === row.geometryId)
            : records.filter(record => record.subsets.includes(subset)
                && record.labels.some(label => normal(label) === normal(row.molecule)));
        if (candidates.length) {
            target.replaceChildren(element("p", row.geometryId ? "Canonical geometry" : "Geometries associated with this system"),
                ...candidates.map(record => geometryItem(record, "div")));
            if (row.benchmarkXyz && row.benchmarkXyz !== candidates[0].source) {
                const original = element("a", "Download original benchmark XYZ");
                original.href = geometryUrl(`xyz/${row.benchmarkXyz}`);
                original.download = row.benchmarkXyz.split("/").pop();
                target.append(original);
            }
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
    const archives = manifest?.geometryDownloads;
    if (!archives) return;
    const matches = geometryIndex.filter(record => [record.name, record.species, record.source.split("/").pop(),
        ...record.labels, ...record.aliases]
        .some(label => normal(label).includes(query))
        && (group === "all" || record.groups.includes(group) || record.subsets.includes(group)));
    const all = archives.find(archive => archive.id === "all");
    const selected = archives.find(archive => archive.id === group);
    const allLink = byId("geometry-download-all");
    const setLink = byId("geometry-download-set");
    allLink.href = all.file;
    allLink.textContent = `Download all geometries (ZIP, ${all.count} XYZ)`;
    setLink.hidden = !selected || selected === all;
    if (!setLink.hidden) {
        setLink.href = selected.file;
        setLink.textContent = `Download ${selected.name} (ZIP, ${selected.count} XYZ)`;
    }
    byId("geometry-downloads").hidden = false;
    byId("geometry-count").textContent = `${matches.length} of ${selected?.count ?? 0} XYZ files`;
    byId("geometry-list").replaceChildren(...matches.map(record => geometryItem(record, "li")));
}

function inferBonds(atoms) {
    const bonds = [];
    atoms.forEach((atom, i) => {
        for (let j = 0; j < i; j++) {
            const other = atoms[j];
            const radius = covalentRadii[atom.symbol] + covalentRadii[other.symbol];
            const distance = Math.hypot(...atom.xyz.map((value, axis) => value - other.xyz[axis]));
            if (distance > .35 && distance <= Math.max(1.2 * radius, radius + .30)) bonds.push([j, i]);
        }
    });
    return bonds;
}

async function openStructure(record) {
    const dialog = byId("structure");
    const id = ++structureGeneration;
    byId("structure-title").textContent = record.name;
    byId("structure-download").href = geometryUrl(record.file);
    byId("structure-download").download = `${record.name}.xyz`;
    byId("structure-source").href = sourceLink(record.source);
    const status = byId("structure-status");
    status.textContent = "Loading structure…";
    structureAtoms = [];
    structureBonds = [];
    dialog.showModal();
    drawStructure();
    try {
        const response = await fetch(geometryUrl(record.file));
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
        structureBonds = inferBonds(structureAtoms);
        status.textContent = `${count} atoms · ${structureBonds.length} inferred bonds · drag or use the arrow keys to rotate`;
        resetStructure();
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
    const width = canvas.clientWidth, height = 340;
    if (!width) return;
    const ctx = setupCanvas(canvas, width, height);
    const extent = Math.max(1, ...structureAtoms.map(atom => Math.hypot(...atom.xyz)));
    const scale = Math.min(width, height) * .38 / extent * structureZoom;
    const [a, b] = rotation;
    function project([x, y, z]) {
        const u = Math.cos(a) * x + Math.sin(a) * z;
        const v = -Math.sin(a) * x + Math.cos(a) * z;
        const depth = Math.sin(b) * y + Math.cos(b) * v;
        const perspective = 1 / (1 - depth / (5 * extent));
        return {px: width / 2 + u * scale * perspective,
            py: height / 2 - (Math.cos(b) * y - Math.sin(b) * v) * scale * perspective,
            depth, perspective};
    }
    const color = atom => atomColors[atom.symbol] ?? "#8963a1";
    const shapes = structureAtoms.map(atom => ({atom, ...project(atom.xyz)}));
    for (const [i, j] of structureBonds) {
        const first = structureAtoms[i], last = structureAtoms[j];
        const point = fraction => project(first.xyz.map((value, axis) => value + fraction * (last.xyz[axis] - value)));
        const ends = [point(0), point(.5), point(1)];
        for (let k = 0; k < 2; k++) {
            const from = ends[k], to = ends[k + 1];
            shapes.push({from, to, depth: (from.depth + to.depth) / 2, color: color(k ? last : first)});
        }
    }
    for (const shape of shapes.sort((left, right) => left.depth - right.depth)) {
        if (!shape.atom) {
            const {from, to} = shape;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(from.px, from.py);
            ctx.lineTo(to.px, to.py);
            const thickness = Math.max(3, Math.min(9, scale * .18)) * from.perspective;
            ctx.lineWidth = thickness + 1.5;
            ctx.strokeStyle = "#485366";
            ctx.stroke();
            ctx.lineWidth = thickness;
            ctx.strokeStyle = shape.color;
            ctx.stroke();
            continue;
        }
        const {atom, px, py, perspective} = shape;
        const radius = Math.max(5, Math.min(18, scale * (atom.symbol === "H" ? .23 : .36))) * perspective;
        const shading = ctx.createRadialGradient(px - radius * .32, py - radius * .4, 0, px, py, radius);
        shading.addColorStop(0, "#fff");
        shading.addColorStop(.48, color(atom));
        shading.addColorStop(1, "#354154");
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, 2 * Math.PI);
        ctx.fillStyle = shading;
        ctx.fill();
        if (radius >= 8) {
            ctx.fillStyle = atom.symbol === "H" || atom.symbol === "S" ? "#1c2635" : "#fff";
            ctx.font = "bold 11px system-ui";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(atom.symbol, px, py);
        }
    }
}

function resetStructure() {
    rotation = [0.3, 0.5];
    structureZoom = 1;
    drawStructure();
}

for (const [id, factor] of [["structure-zoom-out", 1 / 1.25], ["structure-zoom-in", 1.25]]) {
    byId(id).addEventListener("click", () => {
        structureZoom = Math.max(.5, Math.min(3, structureZoom * factor));
        drawStructure();
    });
}
byId("structure-reset").addEventListener("click", resetStructure);

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

function chargedStates(data) {
    const states = new Map();
    for (const row of data) {
        const key = JSON.stringify([row.source, row.state, row.basis, row.root]);
        if (!states.has(key)) states.set(key, {...row, fields: Object.create(null)});
        const state = states.get(key);
        if (Object.hasOwn(state.fields, row.method)) throw new Error(`Duplicate result: ${row.source} / ${row.state} / ${row.basis} / ${row.method}`);
        state.fields[row.method] = row.energy;
    }
    return [...states.values()];
}

function benchmarkOptions() {
    const id = get("collection").value;
    const charged = id === "core_IPs" || id === "valence_IPs";
    const valence = id === "valence_IPs";
    const rows = valence ? benchRows.filter(row => row.state === get("sector").value) : benchRows;
    const refs = charged ? sorted(rows.filter(row => Number.isFinite(row.fields.sCI)).map(row => row.basis))
        : ["TBE/AVTZ", ...(rows.some(row => Number.isFinite(row.referenceQZ)) ? ["TBE/AVQZ"] : [])];
    get("reference").replaceChildren(...refs.map(ref => option(charged ? `sCI / ${ref}` : ref, ref)));
    const preferredBasis = valence ? "aug-cc-pVTZ" : "aug-cc-pCVTZ";
    if (charged && refs.includes(preferredBasis)) get("reference").value = preferredBasis;
    choices(get("spin"), "All multiplicities", sorted(rows.map(row => row.spin)));
    choices(get("character"), "All characters", sorted(rows.map(row => row.character)));
    for (const field of ["spin", "character"]) get(`${field}-label`).hidden = get(field).options.length === 1;
    get("safe-label").hidden = charged || !rows.some(row => row.safe != null);
    get("safe").checked = false;
    const methods = sorted(rows.flatMap(row => Object.entries(row.fields)
        .filter(([key, value]) => Number.isFinite(value) && (charged ? key !== "sCI" : !metadata.test(key)))
        .map(([key]) => key)));
    const defaults = valence ? ["CC3", "CCSDT", "GW"]
        : charged ? ["G0W0@HF", "evGW@HF", "qsGW"] : ["ADC(2)", "CC3", "CCSD"];
    const hasDefaults = methods.some(name => defaults.includes(name));
    get("method-list").replaceChildren(...methods.map((method, index) => {
        const label = element("label");
        const input = element("input");
        input.type = "checkbox";
        input.value = method;
        input.checked = hasDefaults ? defaults.includes(method) : index < 2;
        input.addEventListener("change", renderBenchmark);
        label.append(input, method);
        return label;
    }));
    renderBenchmark();
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
        const charged = id === "core_IPs" || id === "valence_IPs";
        const data = benchCache.get(id);
        benchRows = charged ? chargedStates(data) : data;
        get("sector-label").hidden = id !== "valence_IPs";
        if (id === "valence_IPs") {
            get("sector").replaceChildren(...sorted(benchRows.map(row => row.state))
                .map(sector => option(sector === "IP" ? "Valence IPs" : sector, sector)));
        }
        benchmarkOptions();
    } catch (error) {
        if (generation === benchGeneration) status.textContent = error.message;
    }
}

function renderBenchmark() {
    if (!benchRows.length) return;
    const id = get("collection").value;
    const valence = id === "valence_IPs";
    const charged = valence || id === "core_IPs";
    const ref = get("reference").value;
    const spin = get("spin").value, character = get("character").value;
    benchMethods = [...get("method-list").querySelectorAll(":checked")].map(input => input.value);
    const eligible = benchRows
        .filter(row => (charged ? row.basis === ref && (!valence || row.state === get("sector").value)
            : (!spin || String(row.spin) === spin) && (!character || row.character === character)
                && (!get("safe").checked || row.safe === "Y")))
        .map(row => ({row, reference: charged ? row.fields.sCI : ref === "TBE/AVQZ" ? row.referenceQZ : row.energy}))
        .filter(item => Number.isFinite(item.reference));
    benchData = benchMethods.length
        ? eligible.filter(({row}) => benchMethods.every(method => Number.isFinite(row.fields[method]))) : [];
    get("status").textContent = benchMethods.length
        ? `${benchData.length} shared states from ${eligible.length} eligible references. Coverage is shown before matching.`
            + (valence ? " States are paired by their position in the source arrays." : "")
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
        option(`${systemText(row, true)} · ${stateOptionText(row.state)}${row.root ? ` #${row.root}` : ""} · ${fixed(reference)} eV`, index)));
    get("export").disabled = !benchData.length;
    get("figure").hidden = !benchData.length;
    plotBenchmark();
}

function inspectBenchmark(index) {
    const item = benchData[index];
    if (!item) return;
    const {row, reference} = item;
    const fields = {"Reference (eV)": reference};
    if (row.root) fields["Source root"] = row.root;
    for (const method of benchMethods) {
        fields[`${method} (eV)`] = row.fields[method];
        fields[`${method} error (eV)`] = fixed(row.fields[method] - reference);
    }
    showDetails({...row, fields});
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
        violinProfile(benchData.map(({row, reference}) => row.fields[method] - reference), min, max));
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
    const violin = get("plot").value === "violin";
    const height = parity ? 360 : Math.max(220, 65 + (violin ? 47 : 28) * benchMethods.length);
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
    ctx.fillText(parity ? "Reference energy (eV)" : "Signed error (eV)", (left + xRight) / 2, height - 6);
    if (parity) {
        ctx.save();
        ctx.translate(12, (bottom + yTop) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Calculated energy (eV)", 0, 0);
        ctx.restore();
    }
    const band = (bottom - top) / benchMethods.length;
    if (violin) drawViolins(ctx, x, min, max, top, band);
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
    caption.replaceChildren(element("span", (violin ? "Violin width shows estimated density. " : "")
        + "Select a point to inspect its source state. "));
    if (parity) {
        benchMethods.forEach((method, j) => {
            const legend = element("span", `● ${method}  `);
            legend.style.color = palette[j % palette.length];
            caption.append(legend);
        });
    }
}

get("collection").addEventListener("change", loadBenchmark);
get("sector").addEventListener("change", benchmarkOptions);
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
    const records = [["Molecule/site", "State", "Root", "Reference", "Reference (eV)", "Method", "Energy (eV)", "Error (eV)", "Source JSON", "QUESTDB commit"],
        ...benchData.flatMap(({row, reference}) => benchMethods.map(method => [systemText(row), row.state,
            row.root ?? "", get("reference").selectedOptions[0].textContent, reference, method, row.fields[method],
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
