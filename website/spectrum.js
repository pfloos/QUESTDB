"use strict";

const spectrum = id => byId(`spectrum-${id}`);
let moleculeGeneration = 0, moleculeRows = [], spectrumData = null, spectrumOwner = null;

async function openMolecule(id) {
    const generation = ++moleculeGeneration;
    const molecule = Object.hasOwn(molecules, id) ? molecules[id] : null;
    const title = byId("molecule-title");
    const collections = byId("molecule-collections");
    const geometries = byId("molecule-geometries");
    const states = byId("molecule-states");
    const previous = spectrumOwner === id ? spectrum("collection").value : "";
    spectrumOwner = id;
    moleculeRows = [];
    spectrumData = null;
    byId("spectrum").hidden = true;
    title.replaceChildren(molecule ? moleculeLabel(molecule, false) : "Molecule not found");
    collections.textContent = molecule?.collections.map(collectionName).join(" · ") || "";
    geometries.replaceChildren();
    states.replaceChildren();
    if (!molecule) return;
    states.textContent = "Loading states…";
    try {
        const [index, ...collections] = await Promise.all([
            loadGeometries(), ...molecule.collections.map(loadCollection),
        ]);
        if (generation !== moleculeGeneration) return;
        const records = index.filter(record => record.moleculeId === id).sort((a, b) => Number(b.default) - Number(a.default));
        geometries.replaceChildren(element("h3", "Molecular geometries"), ...records.map(record => {
            const item = geometryItem(record, "div");
            if (record.default) {
                const title = record.source.startsWith("geometries/xyz/") ? "Canonical neutral geometry" : "Canonical geometry";
                item.prepend(element("small", title));
            }
            return item;
        }));
        moleculeRows = collections.flat().filter(row => row.moleculeId === id);
        const neutral = sorted(moleculeRows.filter(row => row.fields && !row.fluorescence && Number.isFinite(row.energy))
            .map(row => row.collection));
        spectrum("collection").replaceChildren(...neutral.map(id => option(collectionName(id), id)));
        if (neutral.includes(previous)) spectrum("collection").value = previous;
        else if (neutral.includes("MAIN")) spectrum("collection").value = "MAIN";
        byId("spectrum").hidden = !neutral.length;
        if (neutral.length) drawSpectrum();
        states.replaceChildren(element("h3", "States and reference energies"));
        for (const id of molecule.collections) {
            const rows = moleculeRows.filter(row => row.collection === id && (row.fields || row.method === "sCI"));
            // A charged state can have several reference bases. Its detail view shows all of them.
            const unique = new Map(rows.map(row => [row.stateId, row]));
            if (!unique.size) continue;
            const group = element("details");
            group.className = "state-group";
            group.open = true;
            group.append(element("summary", `${collectionName(id)} · ${unique.size} states`));
            const list = element("ul");
            for (const row of unique.values()) {
                const li = element("li");
                if (row.site) li.append(`${row.site} · `);
                li.append(stateButton(row));
                if (row.fields && Number.isFinite(row.energy)) li.append(` · ${fixed(row.energy)} eV`);
                list.append(li);
            }
            group.append(list);
            states.append(group);
        }
    } catch (error) {
        if (generation === moleculeGeneration) states.textContent = error.message;
    }
}

function lineDensity(delta, fwhm, shape) {
    if (shape === "Lorentzian") {
        const gamma = fwhm / 2;
        return gamma / (Math.PI * (delta ** 2 + gamma ** 2));
    }
    const sigma = fwhm / (2 * Math.sqrt(2 * Math.log(2)));
    return Math.exp(-0.5 * (delta / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
}

function makeSpectrum(rows, width, shape) {
    const min = Math.max(0, rows[0].energy - 4 * width);
    const max = rows.at(-1).energy + 4 * width;
    const count = Math.max(600, Math.ceil((max - min) * 8 / width));
    const weighted = rows.filter(row => row.strength > 0);
    const samples = Array.from({length: count + 1}, (_, i) => {
        const energy = min + i * (max - min) / count;
        return [energy, weighted.reduce((sum, row) => sum + row.strength * lineDensity(energy - row.energy, width, shape), 0)];
    });
    return {min, max, samples, width, shape, rows};
}

function drawSpectrum() {
    if (byId("molecule").hidden || byId("spectrum").hidden) return;
    const widthInput = spectrum("width");
    if (!widthInput.validity.valid || !widthInput.value) {
        spectrum("export").disabled = true;
        spectrum("caption").textContent = "Enter a FWHM between 0.01 and 2.00 eV.";
        return;
    }
    const width = Number(widthInput.value);
    const shape = spectrum("shape").value;
    const rows = moleculeRows.filter(row => row.collection === spectrum("collection").value && row.fields
        && !row.fluorescence && Number.isFinite(row.energy)).sort((a, b) => a.energy - b.energy);
    if (!rows.length) return;
    spectrumData = makeSpectrum(rows, width, shape);
    const {min, max, samples} = spectrumData;
    const canvas = spectrum("canvas");
    const w = canvas.clientWidth;
    if (!w) return;
    const height = 305, top = 22, left = 60, right = w - 24, baseline = height - 74;
    const ctx = setupCanvas(canvas, w, height);
    const peak = samples.reduce((highest, [, intensity]) => Math.max(highest, intensity), 0);
    const x = value => left + (value - min) * (right - left) / (max - min);
    const y = value => baseline - value * (baseline - top) / (peak || 1);
    ctx.fillStyle = "#354662";
    ctx.strokeStyle = "#a2acc0";
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, baseline);
    ctx.lineTo(right, baseline);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.font = "12px system-ui";
    for (let i = 0; i <= 4; i++) {
        const energy = min + i * (max - min) / 4;
        ctx.fillText(fixed(energy, 2), x(energy), baseline + 18);
    }
    ctx.fillText("Vertical energy (eV)", (left + right) / 2, height - 10);
    ctx.textAlign = "right";
    if (peak) for (let i = 0; i <= 3; i++) {
        const intensity = i * peak / 3;
        ctx.fillText(Number(intensity.toPrecision(2)), left - 5, y(intensity) + 4, 38);
    }
    ctx.save();
    ctx.translate(14, (baseline + top) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Oscillator strength density (1/eV)", 0, 0);
    ctx.restore();
    if (peak) {
        ctx.beginPath();
        ctx.moveTo(x(min), baseline);
        for (const [energy, intensity] of samples) ctx.lineTo(x(energy), y(intensity));
        ctx.lineTo(x(max), baseline);
        ctx.closePath();
        ctx.fillStyle = "#dcefe6";
        ctx.fill();
        ctx.beginPath();
        for (const [energy, intensity] of samples) ctx.lineTo(x(energy), y(intensity));
        ctx.strokeStyle = "#23775c";
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    const markers = spectrum("markers").checked;
    const unreported = rows.filter(row => row.strength == null).length;
    const zero = rows.filter(row => row.strength === 0).length;
    if (unreported) {
        ctx.textAlign = "right";
        ctx.fillStyle = "#506079";
        ctx.fillText("f unreported", left - 5, baseline + 41, 55);
        ctx.strokeStyle = "#8b96aa";
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(left, baseline + 37);
        ctx.lineTo(right, baseline + 37);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    const fText = row => row.strength == null ? "f unreported" : `f = ${row.strength}`;
    const pointLabel = row => `${stateText(row)} · ${fixed(row.energy)} eV · ${fText(row)}`;
    spectrum("points").replaceChildren(...rows.flatMap(row => {
        if (!markers && row.strength != null) return [];
        const button = element("button");
        button.type = "button";
        button.className = `spectrum-point ${row.strength == null ? "unreported" : row.strength === 0 ? "zero" : ""}`;
        button.style.left = `${x(row.energy)}px`;
        const height = row.strength ? Math.max(5, (baseline - top) * lineDensity(0, width, shape) * row.strength / peak) : 0;
        if (height) {
            ctx.strokeStyle = "#58746a";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x(row.energy), baseline);
            ctx.lineTo(x(row.energy), baseline - height);
            ctx.stroke();
        }
        button.style.top = `${row.strength == null ? baseline + 37 : baseline - height}px`;
        button.title = pointLabel(row);
        button.setAttribute("aria-label", button.title);
        button.addEventListener("click", () => showDetails(row));
        return [button];
    }));
    spectrum("inspect").replaceChildren(option("Select a state", ""),
        ...rows.map((row, i) => option(pointLabel(row), i)));
    spectrum("caption").textContent = `${shape} · FWHM ${fixed(width, 2)} eV · ${rows.length} transitions · ${zero} reported zero · ${unreported} unreported`
        + (peak ? "" : " · No positive oscillator strength reported");
    spectrum("export").disabled = false;
}

for (const id of ["collection", "shape", "width", "markers"]) spectrum(id).addEventListener("change", drawSpectrum);
spectrum("width").addEventListener("input", drawSpectrum);
spectrum("inspect").addEventListener("change", event => {
    if (event.target.value !== "") showDetails(spectrumData.rows[Number(event.target.value)]);
    event.target.value = "";
});
spectrum("export").addEventListener("click", () => {
    const {width, shape, rows, samples} = spectrumData;
    const prefix = [shape, width, spectrum("collection").value, manifest.revision];
    downloadCsv("spectrum", [["Line shape", "FWHM (eV)", "Collection", "QUESTDB commit", "Record",
        "Energy (eV)", "Oscillator strength density (1/eV)", "State ID", "State", "Oscillator strength", "Strength source"],
    ...samples.map(([energy, intensity]) => [...prefix, "Curve", energy, intensity]),
    ...rows.map(row => [...prefix, "Transition", row.energy, "", row.stateId, row.state.trim(), row.strength, row.strengthSource])]);
});
window.addEventListener("resize", drawSpectrum);
