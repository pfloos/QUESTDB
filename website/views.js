"use strict";

let geometryIndex = [], geometryPromise;
let structureAtoms = [], structureBonds = [], structureGeneration = 0;
let rotation = [0.3, 0.5], structureZoom = 1, dragStart;
const covalentRadii = {
    H: .31, Li: 1.28, Be: .96, B: .84, C: .76, N: .71, O: .66, F: .57,
    Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Sc: 1.70, Ti: 1.60, Cu: 1.32, Zn: 1.22,
};
const atomColors = {
    H: "#ecf1f7", C: "#43546b", N: "#3867c5", O: "#d34536", F: "#48a157",
    Cl: "#398a42", S: "#e2b43a", P: "#db8333", B: "#c07863", Cu: "#a96535",
};

function navigate() {
    const panels = [...document.querySelectorAll("[data-panel]")];
    const [hash, query] = location.hash.slice(1).split("?");
    const params = new URLSearchParams(query);
    const id = panels.some(panel => panel.id === hash) ? hash : "explore";
    for (const panel of panels) panel.hidden = panel.id !== id;
    for (const link of document.querySelectorAll("[data-view]")) {
        if (link.hash === `#${id}`) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
    }
    const showPacks = id === "benchmark" && params.get("packs") === "1";
    if (showPacks) byId("benchmark-packs").open = true;
    if (hash === id) requestAnimationFrame(() => byId(showPacks ? "benchmark-packs" : id).scrollIntoView());
    if (!manifest) return;
    if (id === "explore") loadExplore();
    if (id === "benchmark") {
        if (params.has("set") || !hasBenchmark()) loadBenchmark(params);
        else if (!byId("bench-figure").hidden) plotBenchmark();
    }
    if (id === "molecule") openMolecule(params.get("id"));
    if (id === "geometries") {
        loadGeometries().then(renderGeometries).catch(error => { byId("geometry-count").textContent = error.message; });
    }
}
window.addEventListener("hashchange", navigate);
document.addEventListener("quest-ready", () => {
    byId("geometry-group").replaceChildren(...manifest.geometryDownloads
        .map(group => option(group.name, group.id)));
});

function loadGeometries() {
    if (!geometryPromise) {
        geometryPromise = json(snapshotUrl("data/geometries.json")).then(data => {
            geometryIndex = data;
            return data;
        }).catch(error => {
            geometryPromise = null;
            throw error;
        });
    }
    return geometryPromise;
}

function geometryItem(record, tag) {
    const item = element(tag);
    item.className = "geometry-item";
    const actions = element("div");
    actions.className = "geometry-actions";
    const download = element("a", "Download XYZ");
    download.href = snapshotUrl(record.file);
    download.download = record.file.split("/").pop();
    const preview = element("button", "3D preview");
    preview.type = "button";
    preview.addEventListener("click", () => openStructure(record));
    actions.append(download, preview);
    const title = element("strong");
    title.append(moleculeLabel({name: record.displayName, formula: record.formula}, false));
    if (record.variant) title.append(` · ${record.variant}`);
    item.append(title, element("small", record.groups.join(" · ")), actions);
    if (record.note) item.append(element("p", record.note));
    return item;
}

function showGeometryLinks(row, target) {
    const state = row.stateId;
    target.dataset.state = state;
    target.textContent = "Loading geometry…";
    loadGeometries().then(records => {
        if (target.dataset.state !== state) return;
        const geometry = records.find(record => record.id === row.geometryId);
        if (!geometry) {
            const link = moleculeLink(row);
            link.textContent = "Browse the molecule’s geometries";
            target.replaceChildren(element("p", "No calculation geometry is assigned in the registry."), link);
            return;
        }
        const original = row.benchmarkXyz && row.benchmarkXyz !== geometry.source;
        target.replaceChildren(element("h3", original ? "Canonical structure" : "Calculation geometry"),
            geometryItem(geometry, "div"));
        if (original) {
            const download = element("a", "Download calculation XYZ");
            download.href = snapshotUrl(`xyz/${row.benchmarkXyz}`);
            download.download = row.benchmarkXyz.split("/").pop();
            target.append(download);
        }
    }).catch(error => { if (target.dataset.state === state) target.textContent = error.message; });
}

function renderGeometries() {
    const query = normal(byId("geometry-query").value);
    const group = byId("geometry-group").value;
    const archives = manifest?.geometryDownloads;
    if (!archives) return;
    const matches = geometryIndex.filter(record => [record.name, record.species, record.displayName, record.formula, record.source.split("/").pop(),
        ...record.labels, ...record.aliases]
        .some(label => normal(label).includes(query))
        && (group === "all" || record.groups.includes(group) || record.subsets.includes(group)));
    const all = archives.find(archive => archive.id === "all");
    const selected = archives.find(archive => archive.id === group);
    const allLink = byId("geometry-download-all");
    const setLink = byId("geometry-download-set");
    allLink.href = snapshotUrl(all.file);
    byId("geometry-download-registry").href = snapshotUrl("data/geometry-registry.json");
    allLink.textContent = `Download all geometries (ZIP, ${all.count} XYZ)`;
    setLink.hidden = !selected || selected === all;
    if (!setLink.hidden) {
        setLink.href = snapshotUrl(selected.file);
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
    byId("structure-title").replaceChildren(moleculeLabel({name: record.displayName, formula: record.formula}));
    if (record.variant) byId("structure-title").append(` · ${record.variant}`);
    byId("structure-download").href = snapshotUrl(record.file);
    byId("structure-download").download = `${record.name}.xyz`;
    byId("structure-source").href = sourceLink(record.source);
    const status = byId("structure-status");
    status.textContent = "Loading structure…";
    structureAtoms = [];
    structureBonds = [];
    dialog.showModal();
    drawStructure();
    try {
        const response = await fetch(snapshotUrl(record.file));
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


window.addEventListener("resize", () => {
    if (byId("structure").open) drawStructure();
});
