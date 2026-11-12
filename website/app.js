"use strict";

const ui = Object.fromEntries(
    ["snapshot", "collection", "query", "method", "method-title", "facet",
        "facet-title", "safe", "safe-label", "export", "status", "error", "head",
        "rows", "prev", "page", "next", "detail", "detail-title", "detail-source",
        "detail-fields"].map(id => [id, document.getElementById(id)])
);
const cache = new Map();
const pageSize = 40;
const metadata = /^(?:Size|Group|Spin|TBE(?:\/|$)|%|f\s*\[)/i;
let manifest, active, rows = [], filtered = [], page = 0, request = 0;

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
        .sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
}

async function json(path, options) {
    const response = await fetch(path, options);
    if (!response.ok) throw new Error(`Could not read ${path} (${response.status}).`);
    return response.json();
}

function snapshotUrl(path) {
    const encoded = path.split("/").map(encodeURIComponent).join("/");
    return `${encoded}?v=${encodeURIComponent(manifest.revision)}`;
}

function sourceLink(path, type = "blob") {
    return `${manifest.repository}/${type}/${manifest.revision}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function controls(disabled) {
    for (const key of ["query", "method", "facet", "safe"]) ui[key].disabled = disabled;
    ui.export.disabled = disabled || !filtered.length;
}

function report(error) {
    ui.error.hidden = false;
    ui.error.textContent = error.message;
    ui.status.textContent = "";
}

async function changeCollection() {
    const id = ui.collection.value;
    const generation = ++request;
    const collection = manifest.collections.find(item => item.id === id);
    ui.error.hidden = true;
    controls(true);
    ui.prev.disabled = true;
    ui.next.disabled = true;
    ui.status.textContent = "Loading collection…";
    ui.head.replaceChildren();
    ui.rows.replaceChildren();
    ui.page.textContent = "";
    try {
        if (!cache.has(id)) cache.set(id, await json(snapshotUrl(`data/${id}.json`)));
        if (generation !== request) return;
        rows = cache.get(id);
        active = collection;
        const neutral = active.kind === "neutral";
        ui["method-title"].textContent = neutral ? "Compare method" : "Method";
        ui["facet-title"].textContent = neutral ? "Character" : "Basis set";
        ui["safe-label"].hidden = !neutral || !rows.some(row => row.safe != null);
        ui.safe.checked = false;
        const methods = neutral
            ? rows.flatMap(row => Object.entries(row.fields)
                .filter(([key, value]) => Number.isFinite(value) && !metadata.test(key))
                .map(([key]) => key))
            : rows.map(row => row.method);
        choices(ui.method, neutral ? "Reference only" : "All methods", sorted(methods));
        choices(ui.facet, neutral ? "All characters" : "All basis sets",
            sorted(rows.map(row => neutral ? row.character : row.basis)));
        page = 0;
        render();
        controls(false);
    } catch (error) {
        if (generation === request) report(error);
    }
}

function normal(value) {
    return String(value ?? "").normalize("NFKC").toLowerCase()
        .replace(/[^\p{L}\p{N}]+/gu, "");
}

function fixed(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function headers() {
    if (active.kind === "charged") return ["Molecule / site", "Sector", "Method", "Basis", "Root", "Energy (eV)", "Z"];
    const columns = ["Molecule", "State", "Character", "TBE/AVTZ (eV)", "TBE/AVQZ (eV)", "Safe"];
    if (ui.method.value) columns.push(`${ui.method.value} (eV)`, "Δ vs AVTZ (eV)");
    return columns;
}

function values(row) {
    if (active.kind === "charged") {
        return [systemText(row), row.state, row.method, row.basis, row.root, row.energy, row.weight];
    }
    const result = [row.molecule, row.state, row.character, row.energy, row.referenceQZ, row.safe];
    if (ui.method.value) {
        const energy = row.fields[ui.method.value];
        result.push(energy, Number.isFinite(energy) && Number.isFinite(row.energy) ? energy - row.energy : null);
    }
    return result;
}

function stateParts(value) {
    const match = String(value ?? "").trim().match(/^\^\s*(\d+)\s*\\*(Sigma|Pi|Delta|Phi|Gamma|[A-Z])(?:_(?:\{([\da-z]+)\}|([\da-z]+)))?\^?([+'-]*)(\s*\[F\])?$/);
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

function formulaParts(row) {
    return row.source.startsWith("charged/") && /^(?:[A-Z][a-z]?\d*)+$/.test(row.molecule)
        ? row.molecule.split(/(\d+)/) : [row.molecule];
}

function systemText(row, formatted = false) {
    const formula = formatted ? formulaParts(row).map((part, index) =>
        index % 2 ? part.replace(/\d/g, digit => "₀₁₂₃₄₅₆₇₈₉"[digit]) : part).join("") : row.molecule;
    return row.site ? `${formula} · ${row.site}` : formula;
}

function moleculeLabel(row, includeSite = true) {
    const label = element("span", row.molecule);
    label.replaceChildren(...formulaParts(row).map((part, index) => index % 2 ? element("sub", part) : part));
    if (includeSite && row.site) label.append(` · ${row.site}`);
    return label;
}

function showDetails(row, focusGeometry = false) {
    ui["detail-title"].replaceChildren(moleculeLabel(row), " · ", stateLabel(row.state));
    ui["detail-source"].href = sourceLink(row.source);
    const entries = row.fields ?? {
        Molecule: row.molecule, ...(row.site ? {"Core site": row.site} : {}),
        Sector: row.state, Method: row.method, Basis: row.basis,
        Root: row.root, "Energy (eV)": row.energy, Z: row.weight, "Uncertainty (eV)": row.uncertainty,
    };
    ui["detail-fields"].replaceChildren(...Object.entries(entries).flatMap(([key, value]) => {
        const field = element("dd");
        field.append(key === "State" ? stateLabel(value) : key === "Molecule" ? moleculeLabel(row, false)
            : value == null ? "—" : String(value));
        return [element("dt", key), field];
    }));
    showGeometryLinks(row, document.getElementById("detail-geometries"));
    ui.detail.showModal();
    if (focusGeometry) document.getElementById("detail-geometries").focus();
}

function render() {
    const tokens = ui.query.value.trim().split(/\s+/).map(normal).filter(Boolean);
    const neutral = active.kind === "neutral";
    const method = ui.method.value;
    const facet = ui.facet.value;
    filtered = rows.filter(row => {
        const terms = normal([systemText(row), row.state, row.type, row.source, row.method].join(" "));
        return tokens.every(token => terms.includes(token))
            && (!ui.safe.checked || !neutral || row.safe === "Y")
            && (!facet || (neutral ? row.character : row.basis) === facet)
            && (!method || (neutral ? Number.isFinite(row.fields[method]) : row.method === method));
    });
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.min(page, pages - 1);
    ui.head.replaceChildren();
    const heading = element("tr");
    for (const name of headers()) {
        const th = element("th", name);
        th.scope = "col";
        heading.append(th);
    }
    ui.head.append(heading);
    const content = document.createDocumentFragment();
    for (const row of filtered.slice(page * pageSize, (page + 1) * pageSize)) {
        const tr = element("tr");
        for (const [i, value] of values(row).entries()) {
            const td = element("td");
            if (i === 0) {
                const button = element("button");
                button.append(moleculeLabel(row));
                button.type = "button";
                button.className = "row-link";
                button.addEventListener("click", () => showDetails(row));
                const geometry = element("button", "Geometry");
                geometry.type = "button";
                geometry.className = "geometry-shortcut";
                geometry.setAttribute("aria-label", `Geometry for ${systemText(row)}`);
                geometry.addEventListener("click", () => showDetails(row, true));
                td.append(button, geometry);
            } else if (i === 1 && neutral) {
                td.append(stateLabel(value));
            } else {
                td.textContent = i === 4 && !neutral ? value : typeof value === "number" ? fixed(value) : value ?? "—";
            }
            tr.append(td);
        }
        content.append(tr);
    }
    ui.rows.replaceChildren(content);
    let summary = `${filtered.length.toLocaleString()} of ${rows.length.toLocaleString()} ${neutral ? "states" : "method results"}`;
    if (neutral && method) {
        const errors = filtered.filter(row => Number.isFinite(row.energy))
            .map(row => Math.abs(row.fields[method] - row.energy));
        if (errors.length) summary += ` · MAE ${fixed(errors.reduce((a, b) => a + b, 0) / errors.length)} eV vs AVTZ (${errors.length} matched)`;
    }
    ui.status.textContent = summary;
    ui.page.textContent = filtered.length ? `Page ${page + 1} of ${pages}` : "No matching records";
    ui.prev.disabled = page === 0;
    ui.next.disabled = page + 1 >= pages;
    ui.export.disabled = !filtered.length;
}

function csvField(value) {
    if (value == null) return "";
    if (typeof value === "number") return String(value);
    const text = String(value);
    const safeText = /^[\s]*[=+@-]/.test(text) ? "'" + text : text;
    return '"' + safeText.replaceAll('"', '""') + '"';
}

ui.export.addEventListener("click", () => {
    const extra = active.kind === "charged" ? ["Uncertainty (eV)"] : ["Spin", "Type"];
    const csv = [[...headers(), ...extra, "Source JSON", "QUESTDB commit"],
        ...filtered.map(row => [...values(row),
            ...(active.kind === "charged" ? [row.uncertainty] : [row.spin, row.type]),
            sourceLink(row.source), manifest.revision])]
        .map(row => row.map(csvField).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const link = element("a");
    link.href = url;
    link.download = `QUESTDB-${active.id}-${manifest.revision.slice(0, 7)}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
ui.collection.addEventListener("change", changeCollection);
for (const [key, event] of [["query", "input"], ["method", "change"], ["facet", "change"], ["safe", "change"]]) {
    ui[key].addEventListener(event, () => { page = 0; render(); });
}
ui.prev.addEventListener("click", () => { page--; render(); });
ui.next.addEventListener("click", () => { page++; render(); });

async function init() {
    try {
        manifest = await json("data/manifest.json", {cache: "no-cache"});
        const revision = element("a", manifest.revision.slice(0, 7));
        revision.href = `${manifest.repository}/commit/${manifest.revision}`;
        ui.snapshot.replaceChildren(
            `${manifest.neutralStates.toLocaleString()} neutral states · ${manifest.collections.length} collections · ${manifest.date.slice(0, 10)} · `, revision
        );
        for (const link of document.querySelectorAll("[data-source]")) {
            link.href = sourceLink(link.dataset.source, "tree");
        }
        ui.collection.replaceChildren(...manifest.collections.map(item => option(item.name, item.id)));
        ui.collection.disabled = false;
        document.dispatchEvent(new Event("quest-ready"));
        await changeCollection();
    } catch (error) {
        report(error);
    }
}

init();
