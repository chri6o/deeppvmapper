// PV-Mapper web UI — plain JS, Leaflet for the map.
const CHECK_TXT = { bestaetigt: "bestätigt", teilweise: "teilweise bestätigt", nicht_bestaetigt: "nicht bestätigt – verworfen",
  nur_aktuell: "im Vergleichsbild keine PV (evtl. neue Anlage)", nur_zweitbild: "nur im Vergleichsbild", keine_pv: "keine PV" };
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (n, d = 0) => (n == null ? "–" : Number(n).toLocaleString("de-DE", { maximumFractionDigits: d, minimumFractionDigits: d }));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) { o.method = o.method || "POST"; o.headers = { "Content-Type": "application/json" }; o.body = JSON.stringify(o.json); delete o.json; }
  const r = await fetch(path, o);
  const t = await r.text();
  let d; try { d = JSON.parse(t); } catch { d = t; }
  if (!r.ok) throw new Error((d && d.detail) || t || r.statusText);
  return d;
}

// ------------------------------------------------------------------ state
const LAYOUT_TXT = { flush: "flächig", tilted_mono: "aufgeständert (Süd)", east_west: "Ost-West", vertical_bifacial: "senkrecht bifazial", mixed: "gemischt", unknown: "unbestimmt" };
const S = { info: null, sources: [], input: null, jobs: [], job: null, poll: null, drawn: null, srcLayer: null };

// ------------------------------------------------------------------ map
const map = L.map("map", { zoomControl: true }).setView([51.2, 10.4], 6);
const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 21, maxNativeZoom: 19, attribution: "© OpenStreetMap" }).addTo(map);
const layerCtl = L.control.layers({ "OpenStreetMap": osm }, {}, { collapsed: true }).addTo(map);
const dataLayer = L.geoJSON(null, {
  style: (f) => f.properties.kind === "pv" ? { color: "#e11d48", weight: 1, fillOpacity: .55 }
    : f.properties.status === "pv" ? { color: "#f2b705", weight: 2, fillOpacity: .15 }
    : f.properties.status === "fehler" ? { color: "#c62828", weight: 1, dashArray: "3", fillOpacity: .05 }
    : { color: "#2563eb", weight: 1, fillOpacity: .05 },
  pointToLayer: (f, ll) => L.circleMarker(ll, { radius: 5, color: f.properties.status === "pv" ? "#f2b705" : "#2563eb" }),
  onEachFeature: (f, l) => { if (f.properties.id) l.on("click", () => showBuilding(f.properties.id)); },
}).addTo(map);
// installation outlines from the typology, coloured by layout (own overlay, can be switched off)
const LAYOUT_COL = { flush: "#7c3aed", tilted_mono: "#ea580c", east_west: "#0891b2", vertical_bifacial: "#16a34a", mixed: "#64748b", unknown: "#94a3b8" };
const instLayer = L.geoJSON(null, {
  style: (f) => ({ color: LAYOUT_COL[f.properties.layout] || "#94a3b8", weight: 2, dashArray: "5 3", fillOpacity: .12 }),
  onEachFeature: (f, l) => {
    l.bindPopup(() => instPopup(f.properties));
    l.on("click", () => showBuilding(f.properties.id));
  },
}).addTo(map);
layerCtl.addOverlay(instLayer, "Anlagen nach Bauform");
const instLegend = L.control({ position: "bottomright" });
instLegend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend hidden");
  d.innerHTML = "<b>Anlagen</b><br>" + Object.entries(LAYOUT_COL).map(([k, c]) => `<i style="border-color:${c}"></i>${esc(LAYOUT_TXT[k])}`).join("<br>");
  return d;
};
instLegend.addTo(map);
function showInstLegend(on) { const el = document.querySelector(".legend"); if (el) el.classList.toggle("hidden", !on || !map.hasLayer(instLayer)); }
map.on("overlayadd overlayremove", () => showInstLegend(instLayer.getLayers().length > 0));
function instPopup(p) {
  const az = p.azimuth_deg == null ? "" : ` · ${Math.round(p.azimuth_deg)}°${p.azimuth2_deg != null ? "/" + Math.round(p.azimuth2_deg) + "°" : ""}`;
  return `<b>${esc(p.id)} · Anlage ${p.system || p.n}${p.system ? " · Teilfläche " + p.n : ""}</b><br>${esc(LAYOUT_TXT[p.layout] || p.layout)}${az}${p.tilt_class ? " · Neigung " + esc(p.tilt_class) + "°" : ""}<br>` +
    `${fmt(p.outline_area_m2, 0)} m² × Belegung ${fmt(p.fill_factor, 2)}${p.fill_factor_source === "derived_pitch" ? " (gemessen)" : ""}<br>` +
    `<b>${fmt(p.kwp_est, 1)} kWp</b> (${fmt(p.kwp_low, 0)}–${fmt(p.kwp_high, 0)})${yearText(p) ? "<br>" + yearText(p).slice(3) : ""}`;
}
// free roof area (green) and obstacles (orange) of the roof potential (roofs.py)
const roofLayer = L.geoJSON(null, {
  style: (f) => f.properties.kind === "free" ? { color: "#15803d", weight: 1, fillColor: "#22c55e", fillOpacity: .35 }
    : { color: "#c2410c", weight: 1, fillColor: "#f97316", fillOpacity: .45 },
  onEachFeature: (f, l) => {
    if (f.properties.kind === "free") l.bindPopup(() => `<b>${esc(f.properties.id)}</b><br>frei ${fmt(f.properties.usable_area_m2, 0)} m² → <b>${fmt(f.properties.kwp_potential, 0)} kWp</b> (${fmt(f.properties.kwp_pot_low, 0)}–${fmt(f.properties.kwp_pot_high, 0)})`);
    l.on("click", () => showBuilding(f.properties.id));
  },
}).addTo(map);
layerCtl.addOverlay(roofLayer, "Freie Dachflächen und Aufbauten");
// ground-mounted PV (docs/freiflaechen.md): own layer, coloured by origin
const G_COL = { bestand_osm: "#2563eb", bestand_tzsam: "#7c3aed", suche: "#ea580c" };
const G_TXT = { bestand_osm: "OSM", bestand_tzsam: "TZ-SAM", suche: "neu gefunden" };
const groundLayer = L.geoJSON(null, {
  style: (f) => f.properties.pruefstatus === "verworfen" ? { color: "#64748b", weight: 1, dashArray: "4 3", fillOpacity: .05 }
    : { color: G_COL[f.properties.herkunft] || "#64748b", weight: 2, fillOpacity: f.properties.herkunft === "suche" ? .3 : .12 },
  onEachFeature: (f, l) => l.bindPopup(() => groundPopup(f.properties)),
});
layerCtl.addOverlay(groundLayer, "Freiflächen");
function groundPopup(p) {
  const q = p.quellen || {}, z = p.zeitreihe || {};
  const osmLinks = (q.osm || []).map((k) => `<a href="https://www.openstreetmap.org/${{ w: "way", r: "relation", n: "node" }[k[0]] || "way"}/${k.slice(1)}" target="_blank">${esc(k)}</a>`).join(", ");
  const months = Object.keys(z).sort().map((m) => `${m}: ${z[m].s2 == null ? "–" : Math.round(z[m].s2 * 100) + " %"}`).join(" · ");
  const wc = q.worldcover_2021 ? Object.entries(q.worldcover_2021).map(([k, v]) => `${esc(k)} ${Math.round(v * 100)} %`).join(", ") : "";
  return `<b>${esc(q.osm_name || p.id)}</b><br>${esc(G_TXT[p.herkunft] || p.herkunft)} · ${fmt(p.flaeche_zaun_m2 / 1e4, 1)} ha` +
    ` · ${esc(G_STATUS[p.pruefstatus] || p.pruefstatus)}${p.verwurf_grund ? " (" + esc(G_REASONS[p.verwurf_grund.replace("manuell:", "")] || p.verwurf_grund) + ")" : ""}` +
    `${p.typ && p.typ !== "unbekannt" ? "<br>Typ: " + esc(p.typ) : ""}${p.baujahr_schaetzung ? "<br>Baujahr: " + p.baujahr_schaetzung : ""}` +
    `${p.first_seen ? "<br>zuerst gesehen: " + esc(p.first_seen.slice(0, 7)) : ""}` +
    `${p.kwp_min != null ? `<br><b>Leistung (Schätzung): ${fmt(p.kwp_min / 1000, 1)}–${fmt(p.kwp_max / 1000, 1)} MWp</b> DC · AC ${fmt(p.ac_kw_min / 1000, 1)}–${fmt(p.ac_kw_max / 1000, 1)} MW<br><span class="muted">Methode: ${esc(G_METHOD[p.methode] || p.methode)}${p.ertrag_kwh_kwp ? ` · Ertrag ${fmt(p.ertrag_kwh_kwp)} kWh/kWp (PVGIS, fest)` : ""}</span>` : ""}` +
    `${q.osm_leistung_kwp ? "<br>Leistung laut OSM: " + fmt(q.osm_leistung_kwp / 1000, 1) + " MWp (nur Gegencheck)" : ""}` +
    `${q.tzsam_kwp_schaetzung ? "<br>TZ-SAM-Schätzung: " + fmt(q.tzsam_kwp_schaetzung / 1000, 1) + " MW" : ""}` +
    `${months ? "<br>als PV erkannt (Sentinel-2): " + months : ""}${wc ? "<br>WorldCover 2021: " + wc : ""}` +
    `${osmLinks ? "<br>OSM: " + osmLinks : ""}${(q.tzsam || []).length ? "<br>TZ-SAM: " + esc(q.tzsam.join(", ")) + " (CC BY-NC)" : ""}` +
    (q.osm_name ? `<br><span class="muted">${esc(p.id)}</span>` : "") + groundReviewHtml(p);
}
const G_METHOD = { modulflaeche: "Modulfläche × Wirkungsgrad nach Baujahr", zaunflaeche: "Zaunfläche × Leistungsdichte nach Typ und Baujahr",
  zaunflaeche_kalibriert: "Zaunfläche × Leistungsdichte (mit MaStR kalibriert)", s2_polygon: "Sentinel-2-Fläche × Leistungsdichte (grob)" };
const G_TYPES = { fest_sued: "fest, Süd", fest_ost_west: "fest, Ost-West", tracker: "Tracker", agri_vertikal: "Agri-PV vertikal", floating: "Floating", unbekannt: "unbekannt" };
const G_REASONS = { keine_pv: "keine PV", gewaechshaus: "Gewächshaus/Folie", baustelle: "Baustelle ohne Module", dach: "Dachanlage", doppelt: "doppelt erfasst", sonstiges: "sonstiges" };
const G_STATUS = { kandidat: "zu prüfen", manuell_geprueft: "bestätigt", gegenprobe_bestaetigt: "bestätigt (Gegenprobe)", verworfen: "verworfen" };
function groundReviewHtml(p) {
  const last = (p.historie || []).slice(-1)[0];
  return `<div class="g-review" data-id="${esc(p.id)}">` +
    `<div class="row"><button class="primary small" data-act="confirm">✓ Bestätigen</button>` +
    `<select data-act="discard"><option value="">✗ Verwerfen …</option>${Object.entries(G_REASONS).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("")}</select></div>` +
    `<div class="row"><select data-act="type">${Object.entries(G_TYPES).map(([k, v]) => `<option value="${k}"${k === p.typ ? " selected" : ""}>Typ: ${esc(v)}</option>`).join("")}</select>` +
    `${p.pruefstatus !== "kandidat" ? `<button class="secondary small" data-act="reset" title="zurück auf „zu prüfen“">↺</button>` : ""}</div>` +
    `${last ? `<span class="muted">zuletzt: ${esc(last.aktion)} ${esc((last.zeit || "").replace("T", " ").slice(0, 16))}</span>` : ""}</div>`;
}
async function groundReview(id, action, extra = {}) {
  try { await api(`/api/ground/${encodeURIComponent(id)}/review`, { json: { action, ...extra } }); }
  catch (e) { toast("Fehler: " + e.message); return; }
  map.closePopup();
  toast(action === "confirm" ? "Bestätigt" : action === "discard" ? "Verworfen" : action === "reset" ? "Zurückgesetzt" : "Typ gespeichert");
  S.gLast = id;
  await loadGround(false, true);
  if (action !== "type") groundNext();
}
map.on("popupopen", (e) => {
  const box = e.popup.getElement()?.querySelector(".g-review");
  if (!box) return;
  const id = box.dataset.id;
  box.querySelectorAll("[data-act]").forEach((el) => {
    const act = el.dataset.act;
    if (el.tagName === "BUTTON") el.onclick = () => groundReview(id, act);
    else el.onchange = () => { if (!el.value) return; groundReview(id, act, act === "discard" ? { reason: el.value } : { typ: el.value }); };
  });
});
function groundNext() {
  const rows = $$("#g-list tr.click");
  if (!rows.length) { toast("Keine weiteren Flächen in dieser Auswahl"); return; }
  const i = rows.findIndex((tr) => tr.dataset.id === S.gLast);
  (rows[i + 1] || rows[0]).click();
}
document.addEventListener("keydown", (e) => {
  if (currentTab() !== "ground" || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (e.key === "n" || e.key === "N") groundNext();
});
async function loadGround(fit, keepView) {
  const region = $("#g-region").value, h = $("#g-filter").value, st0 = $("#g-status").value;
  const qs = new URLSearchParams({ region });
  if (h) qs.set("herkunft", h);
  if (st0) qs.set("pruefstatus", st0);
  let fc, st;
  try { [fc, st] = await Promise.all([api("/api/ground/features?" + qs), api("/api/ground/stats?region=" + encodeURIComponent(region))]); }
  catch (e) { $("#g-stats").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  groundLayer.clearLayers(); groundLayer.addData(fc);
  const by = {};
  for (const r of st.rows) { const k = r.herkunft; by[k] = by[k] || { n: 0, ha: 0, drop: 0 }; by[k].n += r.n; by[k].ha += r.ha; if (r.pruefstatus === "verworfen") by[k].drop += r.n; }
  $("#g-stats").innerHTML = Object.keys(by).length ? Object.entries(by).map(([k, v]) =>
    `<div><b>${fmt(v.n)}</b> ${esc(G_TXT[k] || k)}<br><span class="muted">${fmt(v.ha)} ha${v.drop ? ", " + fmt(v.drop) + " verworfen" : ""}</span></div>`).join("")
    : `<p class="hint">Noch keine Daten für diese Region: unten „Ausgangsbestand laden“.</p>`;
  const cand = fc.features.slice().sort((a, b) => b.properties.flaeche_zaun_m2 - a.properties.flaeche_zaun_m2);
  const sum = (k) => cand.reduce((a, f) => a + (f.properties[k] || 0), 0);
  if (cand.some((f) => f.properties.kwp_min != null))
    $("#g-stats").insertAdjacentHTML("beforeend", `<div><b>${fmt(sum("kwp_min") / 1000)}–${fmt(sum("kwp_max") / 1000)}</b> MWp<br><span class="muted">Auswahl, geschätzt</span></div>`);
  const s2 = (p) => { const z = p.zeitreihe || {}, k = Object.keys(z).sort().pop(); return k && z[k].s2 != null ? Math.round(z[k].s2 * 100) + " %" : "–"; };
  $("#g-list").innerHTML = cand.length ? `<h3>${fmt(cand.length)} Flächen in der Auswahl${cand.length > 300 ? " (300 größte gelistet)" : ""}</h3>` +
    `<table class="tbl"><tr><th>Fläche</th><th>Herkunft</th><th>Stand</th><th title="zuletzt als PV erkannt (Sentinel-2)">S2</th><th title="geschätzte DC-Leistung (Mitte der Spanne)">MWp</th><th>Name / Boden 2021</th></tr>` +
    cand.slice(0, 300).map((f) => { const p = f.properties, q = p.quellen || {}, wc = q.worldcover_2021 || {};
      return `<tr class="click${p.id === S.gLast ? " sel" : ""}" data-id="${esc(p.id)}"><td>${fmt(p.flaeche_zaun_m2 / 1e4, 1)} ha</td><td>${esc(G_TXT[p.herkunft] || "")}</td>` +
        `<td>${esc(G_STATUS[p.pruefstatus] || p.pruefstatus)}</td><td>${s2(p)}</td><td>${p.kwp_min == null ? "–" : fmt((p.kwp_min + p.kwp_max) / 2000, 1)}</td><td>${esc(q.osm_name || Object.keys(wc).join(", "))}</td></tr>`; }).join("") + "</table>"
    : `<p class="hint">Keine Flächen in dieser Auswahl.</p>`;
  $$("#g-list tr.click").forEach((tr) => tr.onclick = () => {
    const l = groundLayer.getLayers().find((x) => x.feature.properties.id === tr.dataset.id);
    S.gLast = tr.dataset.id;
    $$("#g-list tr.sel").forEach((x) => x.classList.remove("sel")); tr.classList.add("sel");
    if (l) { map.fitBounds(l.getBounds(), { maxZoom: 17, padding: [40, 40] }); l.openPopup(); }
  });
  if (fit && !keepView && groundLayer.getLayers().length) map.fitBounds(groundLayer.getBounds(), { maxZoom: 15 });
  groundOrtho();
}
function groundOrtho() {
  if (currentTab() !== "ground") return;
  const cc = (S.gRegions?.regions?.[$("#g-region").value] || {}).cc;
  const src = S.sources.filter((s) => s.country === cc).sort((a, b) => a.priority - b.priority)[0];
  if ($("#g-ortho").checked && src) { if (S.srcLayer?.srcId !== src.id) { showSource(src.id); if (S.srcLayer) S.srcLayer.srcId = src.id; } }
  else if (!$("#g-ortho").checked && S.srcLayer) { map.removeLayer(S.srcLayer); layerCtl.removeLayer(S.srcLayer); S.srcLayer = null; }
}
const Z_COL = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#dc2626", "#0891b2", "#a16207"];
async function loadZubau() {
  const region = $("#g-region").value, group = $("#g-zgroup").value, unit = $("#g-zunit").value, conf = $("#g-zconf").checked;
  const qs = new URLSearchParams({ region, group, confirmed: conf });
  $("#g-zcsv").href = "/api/ground/zubau?" + qs + "&format=csv";
  let z;
  try { z = await api("/api/ground/zubau?" + qs); } catch (e) { $("#g-zubau").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  const val = (e, k) => unit === "n" ? (k === "mid" ? e.n : e.n) : k === "min" ? e[unit + "_mw_min"] : k === "max" ? e[unit + "_mw_max"] : (e[unit + "_mw_min"] + e[unit + "_mw_max"]) / 2;
  if (!z.reihe.length) { $("#g-zubau").innerHTML = `<p class="hint">Noch keine datierten Anlagen: „Baudatum bestimmen“ unter Aufträge.</p>`; return; }
  const qOf = (q) => { const [y, n] = q.split("-Q"); return +y * 4 + (+n - 1); };
  const lab = (i) => `${Math.floor(i / 4)}-Q${i % 4 + 1}`;
  const reg = (z.register || []).filter((r) => unit === "n" || r[unit + "_mw"] != null);
  const all = z.reihe.map((e) => qOf(e.quartal)).concat(reg.map((r) => qOf(r.quartal)));
  const q0 = Math.max(Math.min(...all), qOf("2015-Q3")), q1 = Math.max(...all);
  const segs = [...new Set(z.reihe.map((e) => e.segment))];
  const tot = {};
  for (const e of z.reihe) { const t = tot[e.quartal] = tot[e.quartal] || { mid: 0, min: 0, max: 0 }; t.mid += val(e, "mid"); t.min += val(e, "min"); t.max += val(e, "max"); }
  const regV = Object.fromEntries(reg.map((r) => [r.quartal, unit === "n" ? r.n : r[unit + "_mw"]]));
  const ymax = Math.max(1, ...Object.values(tot).map((t) => t.max), ...Object.values(regV));
  const W = 420, H = 190, L = 34, B = 26, n = q1 - q0 + 1, bw = (W - L - 4) / n;
  const y = (v) => H - B - v / ymax * (H - B - 8);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="font:10px sans-serif">`;
  for (let k = 0; k <= 4; k++) { const v = ymax * k / 4; svg += `<line x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}" stroke="#e5e7eb"/><text x="${L - 3}" y="${y(v) + 3}" text-anchor="end" fill="#64748b">${fmt(v, unit === "n" || v >= 10 ? 0 : 1)}</text>`; }
  for (let i = q0; i <= q1; i++) {
    const x = L + (i - q0) * bw, q = lab(i); let acc = 0;
    segs.forEach((sg, si) => { const e = z.reihe.find((r) => r.quartal === q && r.segment === sg); if (!e) return; const v = val(e, "mid");
      svg += `<rect x="${x + 1}" y="${y(acc + v)}" width="${Math.max(bw - 2, 1)}" height="${y(acc) - y(acc + v)}" fill="${Z_COL[si % Z_COL.length]}"><title>${esc(q)} · ${esc(sg)}: ${fmt(v, 1)} (${e.n} Anlagen)</title></rect>`; acc += v; });
    const t = tot[q];
    if (t && unit !== "n") svg += `<line x1="${x + bw / 2}" x2="${x + bw / 2}" y1="${y(t.min)}" y2="${y(t.max)}" stroke="#111" stroke-width="1"/>`;
    if (regV[q] != null) svg += `<circle cx="${x + bw / 2}" cy="${y(regV[q])}" r="2.5" fill="#fff" stroke="#111"><title>Register ${esc(q)}: ${fmt(regV[q], 1)}</title></circle>`;
    if (i % 4 === 0) svg += `<text x="${x}" y="${H - 8}" fill="#64748b">${Math.floor(i / 4)}</text>`;
  }
  svg += "</svg>";
  const legend = segs.length > 1 ? segs.map((sg, si) => `<span style="white-space:nowrap"><i style="display:inline-block;width:9px;height:9px;background:${Z_COL[si % Z_COL.length]}"></i> ${esc(sg)}</span>`).join(" ") : "";
  const r = z.ohne_quartal;
  $("#g-zubau").innerHTML = svg + `<div class="hint">${legend}${legend ? "<br>" : ""}ohne Quartal: ${fmt(r.vor_2015)} vor 2015/16, ${fmt(r.nur_jahr)} nur Baujahr, ${fmt(r.ohne_datum)} ohne Datum</div>`;
}
async function initGround() {
  if (!S.gRegions) {
    S.gRegions = await api("/api/ground/regions");
    $("#g-region").innerHTML = Object.entries(S.gRegions.regions).map(([k, r]) => `<option value="${esc(k)}"${k === S.gRegions.pilot ? " selected" : ""}>${esc(r.name)} (${esc(r.cc)})</option>`).join("");
    const d = new Date(); d.setMonth(d.getMonth() - 1); $("#g-month").value = d.toISOString().slice(0, 7);
    ["#g-region", "#g-filter", "#g-status"].forEach((q) => $(q).onchange = () => loadGround(q === "#g-region"));
    $("#g-next").onclick = groundNext;
    $("#g-ortho").onchange = groundOrtho;
    $("#g-reload").onclick = () => loadGround(false);
    const start = async (kind, params) => {
      try { const j = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, params }) });
        started(j); $("#g-msg").textContent = "Gestartet – Fortschritt unter Ergebnisse."; }
      catch (e) { $("#g-msg").textContent = e.message; }
    };
    $("#g-inv").onclick = () => start("ground_inventory", { region: $("#g-region").value });
    $("#g-cap").onclick = () => start("ground_capacity", { region: $("#g-region").value });
    $("#g-reg").onclick = () => start("ground_register", { region: $("#g-region").value });
    $("#g-hist").onclick = () => start("ground_history", { region: $("#g-region").value });
    ["#g-zgroup", "#g-zunit", "#g-zconf"].forEach((q) => $(q).onchange = loadZubau);
    $("#g-zubau-box").ontoggle = () => { if ($("#g-zubau-box").open) loadZubau(); };
    $("#g-scan").onclick = () => start("ground_scan", { region: $("#g-region").value, month: $("#g-month").value, dry_run: $("#g-dry").checked });
    await loadGround(true);
  } else loadGround(false);
}

// OSM buildings picked for the roof potential ('Fläche auswerten' → Dachpotenzial)
S.osm = new Map(); S.picked = new Set();
const osmLayer = L.geoJSON(null, {
  style: (f) => S.picked.has(f.properties.id) ? { color: "#1d4ed8", weight: 2, fillColor: "#3b82f6", fillOpacity: .35 }
    : { color: "#64748b", weight: 1, fillOpacity: .03, dashArray: "3 3" },
  onEachFeature: (f, l) => l.on("click", (e) => {
    if (!pickActive() || S.drawing) return;
    L.DomEvent.stopPropagation(e); togglePick(f.properties.id);
  }),
}).addTo(map);
layerCtl.addOverlay(osmLayer, "OSM-Gebäude (Auswahl Dachpotenzial)");
function addOsm(fc) {
  let n = 0;
  for (const f of fc.features || []) if (!S.osm.has(f.properties.id)) { S.osm.set(f.properties.id, f); osmLayer.addData(f); n++; }
  return n;
}
function pickInfo() {
  const n = S.picked.size;
  $("#a-pick-info").classList.toggle("hidden", !n && !S.osm.size);
  $("#a-pick-info").innerHTML = `${fmt(n)} Gebäude ausgewählt${S.osm.size ? ` · ${fmt(S.osm.size)} Umrisse geladen` : ""} <span class="muted">· © OpenStreetMap-Mitwirkende</span>`;
  summary();
}
function restyleOsm() { osmLayer.setStyle(osmLayer.options.style); }
function togglePick(id) { S.picked.has(id) ? S.picked.delete(id) : S.picked.add(id); restyleOsm(); pickInfo(); }
function inRing(pt, ring) {
  let c = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function containsPt(g, pt) {
  const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
  return polys.some((p) => inRing(pt, p[0]) && !p.slice(1).some((h) => inRing(pt, h)));
}
async function pickAt(ll) {
  const pt = [ll.lng, ll.lat];
  let hit = [...S.osm.values()].find((f) => containsPt(f.geometry, pt));
  if (!hit) {
    const d = 0.0004;
    try { addOsm(await api("/api/osm/buildings", { json: { bbox: [ll.lng - d, ll.lat - d * 0.65, ll.lng + d, ll.lat + d * 0.65] } })); }
    catch (e) { info("#a-pick-info", e.message, true); $("#a-pick-info").classList.remove("hidden"); return; }
    hit = [...S.osm.values()].find((f) => containsPt(f.geometry, pt));
  }
  if (hit) togglePick(hit.properties.id); else pickInfo();
}
$("#a-pick-poly").onclick = async () => {
  if (!S.drawn) { info("#a-pick-info", "Erst einen Ort suchen oder ein Rechteck/Polygon zeichnen (Werkzeuge oben links).", true); $("#a-pick-info").classList.remove("hidden"); return; }
  $("#a-pick-info").classList.remove("hidden"); $("#a-pick-info").textContent = "Lade Gebäude aus OpenStreetMap …";
  try {
    const fc = await api("/api/osm/buildings", { json: { geometry: S.drawn.geometry, inside: true } });
    addOsm(fc); fc.features.forEach((f) => S.picked.add(f.properties.id)); restyleOsm(); pickInfo();
  } catch (e) { info("#a-pick-info", e.message, true); }
};
$("#a-pick-view").onclick = async () => {
  if (map.getZoom() < 16) { info("#a-pick-info", "Bitte näher heranzoomen (ab Zoomstufe 16).", true); $("#a-pick-info").classList.remove("hidden"); return; }
  const b = map.getBounds();
  try { addOsm(await api("/api/osm/buildings", { json: { bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] } })); pickInfo(); }
  catch (e) { info("#a-pick-info", e.message, true); $("#a-pick-info").classList.remove("hidden"); }
};
$("#a-pick-clear").onclick = () => { S.picked.clear(); restyleOsm(); pickInfo(); };
const drawn = new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({
  draw: { polyline: false, circle: false, marker: false, circlemarker: false, polygon: { showArea: true }, rectangle: true },
  edit: { featureGroup: drawn },
}));
// place / region search on the map (OSM Nominatim): zooms there and takes the boundary as the selected area
const SearchCtl = L.Control.extend({
  options: { position: "topleft" },
  onAdd() {
    const box = L.DomUtil.create("div", "leaflet-bar geo-search");
    box.innerHTML = `<input id="geo-q" type="search" placeholder="Ort, Kreis, Adresse suchen …" autocomplete="off"><div id="geo-res" class="hidden"></div>`;
    L.DomEvent.disableClickPropagation(box); L.DomEvent.disableScrollPropagation(box);
    return box;
  },
});
const searchCtl = new SearchCtl();
map.addControl(searchCtl);
{ const corner = searchCtl.getContainer().parentNode; corner.insertBefore(searchCtl.getContainer(), corner.firstChild); }   // search on top
let geoT = null;
function geoSearch(q) {
  clearTimeout(geoT);
  if (q.trim().length < 2) { $("#geo-res").classList.add("hidden"); return; }
  geoT = setTimeout(async () => {
    try {
      const rs = await api(`/api/geocode?q=${encodeURIComponent(q.trim())}`);
      S.geo = rs;
      $("#geo-res").innerHTML = rs.length ? rs.map((r, i) => `<div class="geo-item" data-i="${i}"><b>${esc(r.name)}</b> <span class="muted">${esc(r.kind || "")}</span><br><span class="muted small">${esc(r.label || "")}</span></div>`).join("")
        : `<div class="geo-item muted">nichts gefunden</div>`;
      $("#geo-res").classList.remove("hidden");
      $$("#geo-res .geo-item[data-i]").forEach((el) => el.onclick = () => geoPick(S.geo[+el.dataset.i]));
    } catch (e) { $("#geo-res").innerHTML = `<div class="geo-item err">${esc(e.message)}</div>`; $("#geo-res").classList.remove("hidden"); }
  }, 400);
}
function geoPick(r) {
  $("#geo-res").classList.add("hidden"); $("#geo-q").value = r.name;
  const [w, s, e, n] = r.bbox;
  map.fitBounds([[s, w], [n, e]], { maxZoom: 18 });
  if (r.geometry) {                                         // the boundary becomes the selected area (like a drawn polygon)
    drawn.clearLayers();
    const lyr = L.geoJSON(r.geometry, { style: { color: "#2563eb", weight: 2, fillOpacity: .05 } });
    lyr.eachLayer((l) => drawn.addLayer(l));
    S.drawn = { type: "Feature", geometry: r.geometry, properties: { name: r.name } };
    $("#a-bbox").value = "";
    if (typeof onAreaChosen === "function") onAreaChosen(r.name);
  }
}
document.addEventListener("input", (e) => { if (e.target.id === "geo-q") geoSearch(e.target.value); });
document.addEventListener("keydown", (e) => {
  if (e.target.id === "geo-q" && e.key === "Enter" && S.geo && S.geo[0]) { e.preventDefault(); geoPick(S.geo[0]); }
  if (e.target.id === "geo-q" && e.key === "Escape") $("#geo-res").classList.add("hidden");
});

// map tool "Gebäude wählen": click picks OSM buildings for the roof potential; a polygon drawn while it is on takes all buildings in it
function pickActive() { return currentTab() === "check" && S.mode === "pick"; }
map.on(L.Draw.Event.CREATED, (e) => {
  drawn.clearLayers(); drawn.addLayer(e.layer); S.drawn = e.layer.toGeoJSON(); $("#a-bbox").value = "";
  onAreaChosen();
});
// an area was drawn or found by the search: what it means depends on where the user is
function onAreaChosen() {
  if (currentTab() !== "check") return;
  if (S.mode === "pick") $("#a-pick-poly").click();
  else if (S.mode === "area") { estimateArea(); summary(); }
}
map.on(L.Draw.Event.DRAWSTART, () => { S.drawing = true; });
map.on(L.Draw.Event.DRAWSTOP, () => { setTimeout(() => { S.drawing = false; }, 50); });
map.on("click", (e) => { if (S.drawing || currentTab() !== "check") return; if (S.mode === "pick") pickAt(e.latlng); else if (S.mode === "point") probe(e.latlng); });

function sourceLayer(src) {
  if (!src) return null;
  if (src.type === "xyz") return L.tileLayer(src.url.replace("{z}", "{z}").replace("{y}", "{y}").replace("{x}", "{x}"), { maxZoom: 21, maxNativeZoom: src.max_zoom, attribution: src.attribution });
  return L.tileLayer.wms(src.url, { layers: src.layers, format: "image/jpeg", transparent: false, version: "1.3.0", maxZoom: 22, attribution: src.attribution });
}
function showSource(id) {
  const src = S.sources.find((s) => s.id === id);
  if (S.srcLayer) { map.removeLayer(S.srcLayer); layerCtl.removeLayer(S.srcLayer); }
  S.srcLayer = sourceLayer(src);
  if (!S.srcLayer) return;
  S.srcLayer.addTo(map); layerCtl.addOverlay(S.srcLayer, "Luftbild: " + src.name);
  dataLayer.bringToFront();
  if (src.bbox && !map.getBounds().intersects(L.latLngBounds([src.bbox[1], src.bbox[0]], [src.bbox[3], src.bbox[2]])))
    map.fitBounds([[src.bbox[1], src.bbox[0]], [src.bbox[3], src.bbox[2]]]);
}

// ------------------------------------------------------------------ tabs
function currentTab() { return $("#tabs button.active").dataset.tab; }
const MODE_HINT = {
  area: "Gebiet: Ort suchen (oben links) oder Rechteck/Polygon zeichnen",
  pick: "Gebäude anklicken · Rechteck/Polygon zeichnen oder Ort suchen übernimmt alle Gebäude darin",
  list: "", point: "In die Karte klicken: dieser Ort wird sofort ausgewertet",
};
// the map does only what the current place in the app needs: drawing and picking while choosing what to check,
// result layers while looking at results
function mapMode() {
  const t = currentTab(), checking = t === "check", drawOK = (checking && ["area", "pick"].includes(S.mode)) || (t === "train" && S.sub === "roof");
  document.querySelectorAll(".leaflet-draw").forEach((el) => { el.style.display = drawOK ? "" : "none"; });
  const hint = checking ? MODE_HINT[S.mode] : (t === "train" && S.sub === "roof" ? "Trainingsgebiet: Ort suchen oder Rechteck/Polygon zeichnen (NRW)" : "");
  $("#maphint").textContent = hint || ""; $("#maphint").classList.toggle("hidden", !hint);
  map.getContainer().classList.toggle("picking", checking && S.mode === "pick");
  map.getContainer().style.cursor = checking && S.mode === "point" ? "crosshair" : "";
  const want = (lyr, on) => { if (on && !map.hasLayer(lyr)) lyr.addTo(map); if (!on && map.hasLayer(lyr)) map.removeLayer(lyr); };
  want(osmLayer, checking && S.mode === "pick");
  want(drawn, drawOK);
  want(roofLayer, t === "results");
  want(instLayer, t === "results" || (checking && S.mode === "point"));
  want(dataLayer, t === "results" || (checking && ["list", "point"].includes(S.mode)));
  want(groundLayer, t === "ground");
  showInstLegend(map.hasLayer(instLayer) && instLayer.getLayers().length > 0);
}
function tab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== "tab-" + name));
  if (name === "results") { loadJobs(); if (S.job) loadResults(); }
  if (name === "check" && S.mode === "list" && S.input) { dataLayer.clearLayers(); dataLayer.addData(S.input.preview); }
  if (name === "train" && S.sub === "data") { loadDatasets(M.ds); if (M.item) showEditor(true); } else showEditor(false);
  if (name === "train" && S.sub === "models") loadModels();
  if (name === "ground") initGround();
  if (name === "train" && S.sub === "clean") loadCleanup();
  mapMode();
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => tab(b.dataset.tab)));
S.mode = "area"; S.sub = "data";
function setMode(m) {
  S.mode = m;
  $$("#c-modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  $$(".mode").forEach((d) => d.classList.toggle("hidden", d.id !== "mode-" + m));
  $("#c-what").classList.toggle("hidden", m === "point");
  const roofOK = m !== "area";
  $("#c-roof").disabled = !roofOK; $("#c-roof-l").classList.toggle("dim", !roofOK); $("#c-roof-note").classList.toggle("hidden", roofOK);
  if (m !== "point") { dataLayer.clearLayers(); instLayer.clearLayers(); }
  if (m === "list" && S.input) dataLayer.addData(S.input.preview);
  if (m === "pick") pickInfo();
  mapMode(); summary();
}
$$("#c-modes button").forEach((b) => b.onclick = () => setMode(b.dataset.mode));
function setSub(x) {
  S.sub = x;
  $$("#t-subs button").forEach((b) => b.classList.toggle("on", b.dataset.sub === x));
  $$(".sub").forEach((d) => d.classList.toggle("hidden", d.id !== "sub-" + x));
  if (x === "data") { loadDatasets(M.ds); if (M.item) showEditor(true); } else showEditor(false);
  if (x === "models") loadModels();
  if (x === "clean") loadCleanup();
  mapMode();
}
$$("#t-subs button").forEach((b) => b.onclick = () => setSub(b.dataset.sub));

// ------------------------------------------------------------------ models page (Trainieren → Modelle)
async function loadModels() {
  const c = S.info.config;
  const own = S.info.profiles.filter((p) => p.id.startsWith("custom:"));
  const base = S.info.profiles.filter((p) => !p.id.startsWith("custom:"));
  const row = async (p) => {
    let cal = {}; try { cal = await api(`/api/calibration/${encodeURIComponent(p.id)}`); } catch (e) { /* none */ }
    const ff = Object.entries(cal.fill_factor || {}).map(([k, v]) => `${esc(k)} ${fmt(v, 2)}`).join(", ");
    return `<tr><td>${p.id === c.profile ? "★ " : ""}${esc(p.label)}</td><td>${ff ? "Leistung kalibriert (" + ff + ") <a href=\"#\" data-uncal=\"" + esc(p.id) + "\">zurücknehmen</a>" : "–"}</td>` +
      `<td>${p.id === c.profile ? "Standard" : `<button class="secondary small" data-def="${esc(p.id)}">als Standard</button>`}</td></tr>`;
  };
  const rows = await Promise.all([...own, ...base].map(row));
  let roofs = []; try { roofs = await api("/api/roofshape/models"); } catch (e) { /* none */ }
  $("#t-models").innerHTML = `<h3>Erkennungsmodelle</h3><div class="tablewrap"><table><tr><th>Modell</th><th>Leistungskalibrierung</th><th></th></tr>${rows.join("")}</table></div>` +
    `<h3>Dachformen-Modelle</h3>` + (roofs.length ? `<div class="tablewrap"><table><tr><th>Name</th><th>richtig</th><th>Bilder</th><th></th></tr>` +
      roofs.map((m) => `<tr><td>${m.name === c.roof_model ? "★ " : ""}${esc(m.name)}</td><td class="num">${m.accuracy != null ? fmt(m.accuracy * 100, 0) + " %" : "–"}</td><td class="num">${fmt(m.n_train)}</td>` +
        `<td>${m.name === c.roof_model ? "aktiv" : `<button class="secondary small" data-roof="${esc(m.name)}">verwenden</button>`}</td></tr>`).join("") + `</table></div>`
      : `<p class="hint">Noch keins – unter „Dachformen“ trainieren.</p>`);
  const set = async (body) => { await api("/api/config", { json: body }); S.info = await api("/api/info"); renderSettings(); $$(".profile-select").forEach((el) => { el.value = S.info.config.profile; }); loadModels(); };
  $$("#t-models [data-def]").forEach((b) => b.onclick = () => set({ profile: b.dataset.def }));
  $$("#t-models [data-roof]").forEach((b) => b.onclick = () => set({ roof_model: b.dataset.roof }));
  $$("#t-models [data-uncal]").forEach((a) => a.onclick = async (e) => { e.preventDefault(); await api(`/api/calibration/${encodeURIComponent(a.dataset.uncal)}`, { method: "DELETE" }); loadModels(); });
}

// ------------------------------------------------------------------ step 3: summary and start
function summary() {
  const m = S.mode, el = $("#c-summary"), btn = $("#c-start");
  let txt = "", ok = false;
  if (m === "area") { ok = !!(S.drawn || $("#a-bbox").value.trim()); txt = ok ? ($("#a-est").textContent || "Gebiet gewählt") : "Noch kein Gebiet: Ort suchen oder in der Karte zeichnen."; }
  else if (m === "pick") { ok = S.picked.size > 0; txt = ok ? `${fmt(S.picked.size)} Gebäude ausgewählt` : "Noch keine Gebäude ausgewählt."; }
  else if (m === "list") { ok = !!(S.input && S.input.count); txt = ok ? `${fmt(S.input.count)} Gebäude in der Liste` : "Noch keine Gebäudeliste geladen."; }
  el.textContent = txt; btn.disabled = !ok;
  btn.textContent = m === "area" ? "Gebiet auswerten" : ok ? `${fmt(m === "pick" ? S.picked.size : S.input.count)} Gebäude prüfen` : "Prüfung starten";
}
function checkFlags() {
  return { typology: $("#c-typ").checked, age_check: $("#c-age").checked, cross_check: $("#c-cross").checked,
           roof_potential: S.mode !== "area" && $("#c-roof").checked, sources: $("#c-source").value, profile: $("#c-profile").value,
           save_chips: $("#c-chips").value };
}
$("#c-start").onclick = async () => {
  const f = checkFlags();
  try {
    let j;
    if (S.mode === "area") j = await api("/api/jobs", { json: { kind: "area", params: { ...areaParams(), ...f } } });
    else if (S.mode === "pick") {
      const feats = [...S.picked].map((id) => S.osm.get(id)).filter(Boolean);
      j = await api("/api/roofs/start", { json: { features: feats, ...f } });
    } else {
      j = await api("/api/jobs", { json: { kind: "buildings", params: { input: S.input.input, id_field: $("#b-id").value,
        query: ($("#b-sql").value || "").trim() || null, ...f } } });
    }
    started(j);
  } catch (e) { info("#c-msg", e.message, true); }
};
// a job was started: stay here, say so, offer the way to its results
function started(j) {
  toast(`${esc(KIND[j.kind] || j.kind)}-Auftrag gestartet. <a href="#" data-job="${esc(j.id)}">Ergebnisse ansehen →</a>`);
  S.job = j.id; loadJobs(); jobIndicator();
}
function toast(html, ms = 9000) {
  const t = $("#toast"); t.innerHTML = html + ` <button class="tclose" title="schließen">×</button>`; t.classList.remove("hidden");
  const a = t.querySelector("a[data-job]"); if (a) a.onclick = (e) => { e.preventDefault(); t.classList.add("hidden"); tab("results"); openJob(a.dataset.job); };
  t.querySelector(".tclose").onclick = () => t.classList.add("hidden");
  clearTimeout(S.toastT); S.toastT = setTimeout(() => t.classList.add("hidden"), ms);
}
// running job in the header, on every page
async function jobIndicator() {
  try {
    const js = await api("/api/jobs");
    const run = js.find((j) => j.status === "running") || js.find((j) => j.status === "queued");
    const el = $("#jobind");
    if (!run) { el.classList.add("hidden"); return; }
    const pct = run.total ? Math.round(100 * run.done / run.total) : 0;
    const more = js.filter((j) => ["running", "queued"].includes(j.status)).length - 1;
    el.innerHTML = `<span class="dot"></span>${esc(KIND[run.kind] || run.kind)} · ${pct} %${more > 0 ? ` · +${more}` : ""}<i style="width:${pct}%"></i>`;
    el.classList.remove("hidden"); el.onclick = () => { tab("results"); openJob(run.id); };
  } catch (e) { /* server busy */ }
}
setInterval(jobIndicator, 5000);


// ------------------------------------------------------------------ init
async function init() {
  S.info = await api("/api/info");
  S.sources = await api("/api/sources");
  const d = S.info.device;
  $("#device").textContent = d.gpu ? `${d.gpu} · ${d.backend.toUpperCase()}${d.vram_gb ? " · " + d.vram_gb + " GB" : ""}` : "CPU (keine GPU erkannt)";
  const countries = [...new Set(S.sources.map((s) => s.country))];
  const srcOpts = `<option value="auto">automatisch nach Lage</option>` +
    countries.map((c) => `<option value="${c}">nur ${c} (alle Quellen)</option>`).join("") +
    S.sources.map((s) => `<option value="${s.id}">${esc(s.name)}</option>`).join("");
  $$(".source-select").forEach((el) => { el.innerHTML = srcOpts; el.onchange = () => { if (!["auto", ...countries].includes(el.value)) showSource(el.value); }; });
  const profOpts = S.info.profiles.map((p) => `<option value="${p.id}">${esc(p.label)}${p.downloaded ? "" : " (wird geladen)"}</option>`).join("");
  $$(".profile-select").forEach((el) => { el.innerHTML = profOpts; el.value = S.info.config.profile; });
  $("#s-sync-profile").innerHTML = `<option value="">Standard-Modell</option>` + profOpts;
  const c = S.info.config;                                   // defaults of "Was prüfen?" from the settings
  $("#c-typ").checked = c.typology !== false; $("#c-age").checked = c.age_check !== false;
  $("#c-cross").checked = c.cross_check !== false; $("#c-roof").checked = !!c.roof_potential;
  renderSources(); renderSettings(); loadJobs(); setMode(S.mode); jobIndicator();
}

// ------------------------------------------------------------------ buildings tab
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

async function uploadFile(f) {
  info("#b-info", `Lade ${esc(f.name)} hoch …`);
  const fd = new FormData(); fd.append("file", f);
  try { setInput(await api("/api/upload", { method: "POST", body: fd })); }
  catch (e) { info("#b-info", e.message, true); }
}
$("#b-inspect").onclick = async () => {
  info("#b-info", "Lese Eingabe …");
  try { setInput(await api("/api/inspect", { json: { input: $("#b-path").value.trim(), query: $("#b-sql").value.trim() } })); }
  catch (e) { info("#b-info", e.message, true); }
};
function setInput(d) {
  S.input = d;
  info("#b-info", `<b>${fmt(d.count)}</b> Gebäude gelesen.` + (d.count > 300 ? " Karte zeigt die ersten 300." : ""));
  $("#b-id").innerHTML = `<option value="">automatisch</option>` + d.fields.map((f) => `<option>${esc(f)}</option>`).join("");
  dataLayer.clearLayers(); dataLayer.addData(d.preview); instLayer.clearLayers(); showInstLegend(false);
  if (d.bbox) map.fitBounds([[d.bbox[1], d.bbox[0]], [d.bbox[3], d.bbox[2]]], { maxZoom: 18 });
  summary();
}

function regionParams(v) {
  const p = {}; const parts = v.split(",").map((x) => x.trim()).filter(Boolean);
  const lau = parts.filter((x) => x.includes("_")), n3 = parts.filter((x) => !x.includes("_"));
  if (lau.length) p.lau = lau.join(","); if (n3.length) p.nuts3 = n3.join(","); return p;
}
$("#pm-load").onclick = async () => {
  const q = new URLSearchParams({ cc: $("#pm-cc").value, ...regionParams($("#pm-region").value), sub: $("#pm-sub").value, pv: $("#pm-pv").value });
  [...q.keys()].forEach((k) => { if (!q.get(k)) q.delete(k); });
  info("#b-info", "Frage PV-Map ab …");
  try {
    const d = await api("/api/inspect", { json: { input: "pvmap:" + q.toString() } });
    setInput(d);
    if (d.breakdown) info("#b-info", `<b>${fmt(d.count)}</b> Gebäude in PV-Map.<br>` + d.breakdown.map((r) => `${esc(r.sub)} · ${esc(r.pv)}: ${fmt(r.n)}`).join("<br>"));
  } catch (e) { info("#b-info", e.message, true); }
};

// ------------------------------------------------------------------ training tab
function trainParams() {
  const bins = $("#t-bins").value.split(",").map(Number).filter((x) => x > 0);
  return { cc: $("#t-cc").value, ...regionParams($("#t-region").value), sub: $("#t-sub").value || null, bins,
    per_stratum: +$("#t-per").value, negatives: +$("#t-neg").value, profile: $("#t-profile").value, epochs: +$("#t-epochs").value,
    name: $("#t-name").value.trim() || null, sources: $("#t-source").value, use_weak_negatives: $("#t-weak").checked,
    labels_file: $("#t-file").value.trim() || null, use_medium_positives: !$("#t-secure").checked,
    min_kwp: +$("#t-minkwp").value || null };
}
const LBL = { det_pv: "Erkennung: PV", det_none: "Erkennung: keine PV", pos_high: "PV sicher", pos_medium: "PV wahrscheinlich", neg_high: "keine PV (gemeldet)", neg_weak: "keine PV bekannt" };
$("#t-strata").onclick = async () => {
  $("#t-out").innerHTML = `<div class="info">Frage PV-Map ab …</div>`;
  try {
    const d = await api("/api/pvmap/strata", { json: trainParams() });
    $("#t-out").innerHTML = `<div class="tablewrap"><table><tr><th>Label</th><th>Art</th><th>Größe</th><th>verfügbar</th><th>kWp</th></tr>` +
      d.strata.map((r) => `<tr><td>${LBL[r.label_kind] || r.label_kind}</td><td>${esc(r.sub)}</td><td>${esc(r.size_bin)}</td><td class="num">${fmt(r.n)}</td><td class="num">${fmt(r.kwp)}</td></tr>`).join("") + `</table></div>`;
  } catch (e) { $("#t-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#t-start").onclick = async () => {
  try { const j = await api("/api/jobs", { json: { kind: "train", params: trainParams() } }); started(j); }
  catch (e) { $("#t-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
const ROOF_DE = { flat: "Flachdach", mono: "Pultdach", gabled: "Satteldach", hipped: "Walmdach", complex: "komplex" };
function roofReport(id, r) {
  const C = r.classes, M = r.confusion;
  let h = `<h3>Dachformen-Modell „${esc(r.name)}“: ${fmt(r.accuracy * 100, 0)} % richtig</h3><div class="hint">${fmt(r.n_train)} Bilder Training, ${fmt(r.n_val)} Prüfung · flach gegen geneigt ${fmt(r.flat_vs_pitched * 100, 0)} % richtig · ${fmt(r.minutes, 1)} min</div>`;
  h += `<div class="tablewrap"><table><tr><th>LoD2 ↓ / Modell →</th>${C.map((c) => `<th>${esc(ROOF_DE[c] || c)}</th>`).join("")}<th>Trefferquote</th></tr>` +
    C.map((c, i) => `<tr><td>${esc(ROOF_DE[c] || c)}</td>${M[i].map((v, j) => `<td class="num">${i === j ? "<b>" + v + "</b>" : v}</td>`).join("")}<td class="num">${fmt(r.per_class[c].recall * 100, 0)} %</td></tr>`).join("") + `</table></div>`;
  if ((r.wrong || []).length) h += `<h3>Abweichungen (LoD2 → Modell)</h3><div class="hint">Oft liegt LoD2 daneben (generalisierte Dachformen) – ansehen lohnt.</div><div class="roofgrid">` +
    r.wrong.map((w) => `<figure><img loading="lazy" src="/api/jobs/${id}/roofimg/${encodeURIComponent(w.id)}"><figcaption>${esc(ROOF_DE[w.label] || w.label)} → <b>${esc(ROOF_DE[w.pred] || w.pred)}</b> (${fmt(w.p * 100, 0)} %)</figcaption></figure>`).join("") + `</div>`;
  h += `<div class="hint">In den Einstellungen unter „Dachformen-Modell“ wählen, damit es für Gebäude ohne LoD2 genutzt wird.</div>`;
  $("#jd-report").innerHTML = h;
}
async function showReport(id) {
  let r; try { r = await api(`/api/jobs/${id}/report`); } catch { $("#jd-report").innerHTML = ""; return; }
  if (r.kind === "roofshape") return roofReport(id, r);
  const row = (name, a, b) => `<tr><td>${esc(name)}</td><td class="num">${fmt(a?.n)}</td><td class="num">${fmt(a?.recall, 2)} → <b>${fmt(b?.recall, 2)}</b></td><td class="num">${fmt(a?.precision, 2)} → <b>${fmt(b?.precision, 2)}</b></td><td class="num">${fmt(a?.f1, 2)} → <b>${fmt(b?.f1, 2)}</b></td></tr>`;
  const V = r.vorher, N = r.nachher;
  let h = `<h3>Bewertung: vorher → nachher</h3><div class="hint">${r.n_train} Training, ${r.n_val} Validierung · Schwelle ${r.cls_threshold} · Segmentierung ${r.segmentation_trained ? "nachtrainiert" : "unverändert"}</div>
    <div class="tablewrap"><table><tr><th>Gruppe</th><th>n</th><th>Trefferquote</th><th>Präzision</th><th>F1</th></tr>` +
    row("alle sicheren Labels", V.nur_sichere_labels, N.nur_sichere_labels);
  for (const [k, title] of [["nach_art", "Art"], ["nach_groesse", "Größe"], ["nach_label", "Label"]])
    for (const g of Object.keys(N[k])) h += row(`${title}: ${LBL[g] || g}`, V[k][g], N[k][g]);
  h += `</table></div>`;
  const cal = Object.entries(r.kwp_per_m2_by_size || {});
  if (cal.length) h += `<div class="hint">kWp je m² erkannter Fläche (Median): ${cal.map(([k, v]) => `${esc(k)}: ${fmt(v, 3)}`).join(" · ")}</div>`;
  const ab = r.abweichungen || [];
  if (ab.length) {
    const yn = (x) => x ? "PV" : "–";
    h += `<h3>Zum Ansehen: ${ab.length} Gebäude, bei denen die Modelle sich unterscheiden oder das Label verfehlen</h3>
      <div class="hint">Bild · vorher (rot) · nachher (blau). Bei „keine PV bekannt“ entscheidet dein Blick: ist dort PV, liegt das neue Modell richtig.</div>` +
      ab.map((a) => `<div class="valrow"><div class="hint"><b>${esc(a.id)}</b> · ${esc(LBL[a.label_kind] || a.label_kind)} · ${esc(a.sub || "")} · vorher ${yn(a.vorher)} → nachher <b>${yn(a.nachher)}</b></div>
        <img loading="lazy" style="max-width:100%;border-radius:6px" src="/api/jobs/${id}/chips/${encodeURIComponent(a.chip)}"></div>`).join("");
  }
  $("#jd-report").innerHTML = h;
}

// ------------------------------------------------------------------ area tab
function areaParams() {
  const p = { sources: $("#c-source").value, profile: $("#c-profile").value,
    buildings: $("#a-buildings").value.trim() || null, only_on_buildings: $("#a-only").checked,
    osm_buildings: $("#a-osm").checked && !$("#a-buildings").value.trim() };
  const bb = $("#a-bbox").value.trim().split(/[\s,;]+/).map(Number).filter((x) => !isNaN(x));
  if (bb.length === 4) p.bbox = bb; else if (S.drawn) p.geometry = S.drawn.geometry;
  return p;
}
async function estimateArea() {
  const p = areaParams();
  if (!p.bbox && !p.geometry) return;
  try {
    const e = await api("/api/area/estimate", { json: p });
    info("#a-est", `${fmt(e.km2, 2)} km² · ${fmt(e.cells)} Kacheln à ${fmt(e.cell_m)} m · ${fmt(e.gigapixel, 1)} Gigapixel`);
  } catch (err) { info("#a-est", err.message, true); }
  summary();
}
$("#a-bbox").addEventListener("input", () => summary());
$("#a-bbox").addEventListener("change", () => {
  const bb = $("#a-bbox").value.trim().split(/[\s,;]+/).map(Number);
  if (bb.length === 4 && bb.every((x) => !isNaN(x))) {
    drawn.clearLayers(); S.drawn = null;
    const r = L.rectangle([[bb[1], bb[0]], [bb[3], bb[2]]]); drawn.addLayer(r); map.fitBounds(r.getBounds());
  }
  estimateArea();
});
$("#c-profile").addEventListener("change", estimateArea);

// ------------------------------------------------------------------ jobs
const KIND = { buildings: "Gebäude", area: "Fläche", train: "Training", ai_masks: "KI-Masken", sync: "PV-Map-Abgleich", roofshape: "Dachformen-Modell" };
const STATUS = { queued: "wartet", running: "läuft", done: "fertig", failed: "Fehler", cancelled: "abgebrochen" };
const KEEP_JOBS = 3;             // newest jobs shown; older (not running) ones go to the archive
function jobCard(j) {
  const pct = j.total ? Math.round(100 * j.done / j.total) : 0;
  const what = j.kind === "sync" ? (j.params.filter || "") + (j.params.auto ? " · Nachtbetrieb" : "")
    : j.kind === "buildings" ? (j.params.input || "").split(/[\\/]/).pop()
    : j.kind === "train" ? ((j.params.prepare_only ? "Datensatz " : "") + (j.params.name || j.params.cc || j.params.dataset_from || "Labels"))
    : j.kind === "ai_masks" ? `${j.params.provider} · ${j.params.dataset}` : (j.params.bbox ? "Rechteck" : "Polygon");
  const busy = ["running", "queued"].includes(j.status);
  const res = ["buildings", "sync", "area"].includes(j.kind) ? "Ergebnisse" : j.kind === "train" ? (j.params.prepare_only ? "Masken" : "Bericht") : j.kind === "roofshape" ? "Bericht" : "Details";
  return `<div class="job ${S.job === j.id ? "sel" : ""}" data-id="${j.id}">
    <div class="top"><span>${KIND[j.kind]} · ${esc(what)}</span><span class="jbtns"><span class="badge ${j.status}">${STATUS[j.status]}</span>${busy ? "" : `<button class="jdel" data-id="${j.id}" data-kind="${j.kind}" title="Auftrag löschen">×</button>`}</span></div>
    <div class="progress"><div style="width:${pct}%"></div></div>
    <div class="meta"><span>${j.id} · ${fmt(j.done)} / ${fmt(j.total)} ${j.counters.pv != null ? "· " + fmt(j.counters.pv) + " mit PV" : ""}${j.counters.anlagen != null ? "· " + fmt(j.counters.anlagen) + " Anlagen" : ""}</span>
      <button class="jopen small" data-id="${j.id}" data-kind="${j.kind}" data-prep="${j.params.prepare_only ? 1 : ""}">${res} →</button></div></div>`;
}
async function loadJobs() {
  S.jobs = await api("/api/jobs");
  const sorted = [...S.jobs].sort((a, b) => (a.id < b.id ? 1 : -1));
  const recent = sorted.slice(0, KEEP_JOBS), old = sorted.slice(KEEP_JOBS);
  const keep = [...recent, ...old.filter((j) => ["running", "queued"].includes(j.status))];
  const arch = old.filter((j) => !keep.includes(j));
  if (arch.some((j) => j.id === S.job)) S.archOpen = true;
  $("#job-list").innerHTML = S.jobs.length ? keep.map(jobCard).join("") +
    (arch.length ? `<details id="job-archive" ${S.archOpen ? "open" : ""}><summary>Archiv (${arch.length})</summary>${arch.map(jobCard).join("")}</details>` : "")
    : `<p class="hint">Noch keine Aufträge.</p>`;
  const ad = $("#job-archive"); if (ad) ad.ontoggle = () => { S.archOpen = ad.open; };
  $$(".job").forEach((el) => el.onclick = () => openJob(el.dataset.id));
  $$(".jopen").forEach((b) => b.onclick = (e) => {
    e.stopPropagation();
    if (b.dataset.kind === "train" && b.dataset.prep) { M.ds = b.dataset.id; M.item = null; S.sub = "data"; tab("train"); setSub("data"); return; }   // dataset → mask editor
    openJob(b.dataset.id);
    const t = $("#jd-table") || $("#job-detail"); if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $$(".jdel").forEach((b) => b.onclick = async (e) => {
    e.stopPropagation();
    if (!b.classList.contains("confirm")) {          // first click arms, second click deletes
      b.classList.add("confirm");
      b.textContent = ["train", "ai_masks"].includes(b.dataset.kind) ? "Mit Masken löschen?" : "Löschen?";
      setTimeout(() => { if (b.isConnected) { b.classList.remove("confirm"); b.textContent = "×"; } }, 4000);
      return;
    }
    try {
      await api(`/api/jobs/${b.dataset.id}`, { method: "DELETE" });
      if (S.job === b.dataset.id) { S.job = null; clearInterval(S.poll); $("#job-detail").classList.add("hidden"); dataLayer.clearLayers(); instLayer.clearLayers(); showInstLegend(false); }
      loadJobs();
    } catch (err) { b.textContent = "Fehler"; b.title = err.message; }
  });
}
function openJob(id) {
  S.job = id; $("#job-detail").classList.remove("hidden");
  clearInterval(S.poll); refreshJob(true);
  S.poll = setInterval(() => refreshJob(false), 3000);
}
async function refreshJob(withResults) {
  if (!S.job) return;
  const j = await api(`/api/jobs/${S.job}`);
  const pct = j.total ? 100 * j.done / j.total : 0;
  $("#jd-title").textContent = `${KIND[j.kind]} · ${j.id} · ${STATUS[j.status]}`;
  $("#jd-bar").style.width = pct + "%";
  const c = j.counters, stats = [
    [fmt(j.done) + " / " + fmt(j.total), j.kind === "area" ? "Kacheln" : ["train", "roofshape"].includes(j.kind) ? "Schritte" : "Gebäude"],
    [fmt(j.rate_per_s * 3600), "pro Stunde"],
    [j.eta_s ? dur(j.eta_s) : (j.status === "done" ? "fertig" : "–"), "Restzeit"],
  ];
  if (j.kind === "buildings" || j.kind === "sync") stats.push([fmt(c.pv), "mit PV"], [fmt(c.kein_pv), "ohne PV"], [fmt(c.fehler), "ohne Bild"]);
  if (j.kind === "sync") stats.push([fmt(c.gesendet), "an PV-Map gesendet"], [fmt(c.unveraendert), "Luftbild unverändert"]);
  else if (j.kind === "ai_masks") stats.push([fmt(c.maske), "Masken"], [fmt(c.keine_pv_laut_ki), "keine PV laut KI"], [fmt(c.fehler), "Fehler"]);
  else if (j.kind === "train" && j.params.prepare_only) stats.push([fmt(c.datensatz), "Bilder im Datensatz"], [fmt(c.bild_fehlt), "ohne Bild"], ["→ Masken", "nächster Schritt"]);
  else if (j.kind === "roofshape") stats.push([fmt(c.bild), "Bilder"], [fmt(c.bild_fehlt), "ohne Bild"], [c.treffer != null ? fmt(c.treffer * 100, 0) + " %" : "–", "richtig (Prüfmenge)"]);
  else if (j.kind === "train") stats.push([fmt(c.f1_vorher, 2), "F1 vorher"], [fmt(c.f1_nachher, 2), "F1 nachher"], [c.modell ? "gespeichert" : "–", "neues Modell"]);
  else if (j.kind === "area") stats.push([fmt(c.zellen_mit_pv), "Kacheln mit PV"], [fmt(c.anlagen), "Anlagen"], [fmt(c.fehler), "Fehler"], [fmt(c.baujahr_bestimmt), "Baujahr bestimmt"]);
  if (j.kind === "buildings" || j.kind === "sync") stats.push([fmt(c.baujahr_bestimmt), "Baujahr bestimmt"]);
  if (c.dach_bewertet) stats.push([fmt(c.dach_bewertet), "Dächer bewertet"], [fmt(c.kwp_potenzial, 0) + " kWp", "Potenzial frei"]);
  if (!["train", "ai_masks", "roofshape"].includes(j.kind)) stats.push([fmt(c.kwp_sum, 0) + " kWp", "geschätzt gesamt"]);
  $("#jd-stats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  $("#jd-files").innerHTML = j.files.filter((f) => /\.(csv|geojson|json)$/.test(f) && f !== "job.json")
    .map((f) => `<a class="link" href="/api/jobs/${j.id}/files/${f}">${f}</a>`).join(" · ");
  $("#jd-log").textContent = j.log.join("\n");
  $("#jd-cancel").classList.toggle("hidden", !["running", "queued"].includes(j.status));
  $("#jd-resume").classList.toggle("hidden", !["cancelled", "failed"].includes(j.status));
  $("#jd-status").classList.toggle("hidden", !["buildings", "sync"].includes(j.kind));
  $$(".filters").forEach((x) => x.classList.toggle("hidden", ["train", "ai_masks", "roofshape"].includes(j.kind)));
  $("#jd-trainbox").classList.toggle("hidden", !["buildings", "sync", "area"].includes(j.kind));
  if (["train", "ai_masks", "roofshape"].includes(j.kind)) { $("#jd-table").innerHTML = ""; if (["train", "roofshape"].includes(j.kind) && (withResults || j.status === "done")) showReport(j.id); else $("#jd-report").innerHTML = ""; }
  else { $("#jd-report").innerHTML = ""; if (withResults || (j.status !== S.lastStatus && j.status === "done")) loadResults(); }
  S.lastStatus = j.status;
  if (!["running", "queued"].includes(j.status)) { clearInterval(S.poll); loadJobs(); }
}
function dur(s) { const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return h ? `${h} h ${m} min` : `${m} min`; }
$("#jd-cancel").onclick = async () => { await api(`/api/jobs/${S.job}/cancel`, { method: "POST" }); refreshJob(); };
$("#jd-delete").onclick = async () => {
  if (!confirm(`Auftrag ${S.job} mit allen Dateien löschen?`)) return;
  try { const r = await api(`/api/jobs/${S.job}`, { method: "DELETE" }); S.job = null; clearInterval(S.poll); $("#job-detail").classList.add("hidden"); loadJobs();
    if (r.hand_masks) alert(`Gelöscht (darunter ${r.hand_masks} Bilder mit eigenen Masken).`); }
  catch (e) { alert(e.message); }
};
async function loadCleanup() {
  const l = await api("/api/cleanup");
  $("#t-clean-list").innerHTML = l.length ? `<table><tr><th></th><th>Auftrag</th><th>Art</th><th>Bilder</th><th>eigene Masken</th></tr>` +
    l.map((j) => `<tr><td><input type="checkbox" data-id="${j.id}" ${j.hand_masks ? "" : "checked"}></td><td>${esc(j.id)} ${esc(j.name)}</td><td>${j.kind === "train" ? "Training/Datensatz" : "KI-Masken"}</td><td class="num">${fmt(j.images)}</td><td class="num">${j.hand_masks ? "<b>" + fmt(j.hand_masks) + "</b>" : "–"}</td></tr>`).join("") + "</table>"
    : `<p class="hint">Keine alten Trainingsdaten vorhanden.</p>`;
};
$("#t-clean-go").onclick = async () => {
  const ids = $$("#t-clean-list input:checked").map((x) => x.dataset.id);
  if (!ids.length || !confirm(`${ids.length} Aufträge mit allen Dateien löschen?`)) return;
  try { const r = await api("/api/cleanup", { json: { ids } }); $("#t-clean-msg").innerHTML = `<div class="info">${r.deleted.length} gelöscht.</div>`;
    loadCleanup(); loadJobs(); if (typeof loadDatasets === "function") loadDatasets(); }
  catch (e) { $("#t-clean-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#jd-resume").onclick = async () => { await api(`/api/jobs/${S.job}/resume`, { method: "POST" }); openJob(S.job); loadJobs(); };
$("#jd-reload").onclick = loadResults;
$("#jd-status").onchange = loadResults; $("#jd-class").onchange = loadResults;

async function loadResults() {
  const job = S.jobs.find((x) => x.id === S.job) || (await api(`/api/jobs/${S.job}`));
  const q = new URLSearchParams({ status: $("#jd-status").value, size_class: $("#jd-class").value, limit: 500 });
  const d = await api(`/api/jobs/${S.job}/results?` + q);
  $("#jd-class").innerHTML = `<option value="">alle Größen</option>` + (S.info.config.size_classes || [])
    .map((c) => `<option value="${c.name}" ${c.name === $("#jd-class").value ? "selected" : ""}>${c.name} (${esc(c.label || "")})</option>`).join("");
  dataLayer.clearLayers(); dataLayer.addData(d.geojson);
  instLayer.clearLayers(); if (d.installations) instLayer.addData(d.installations);
  roofLayer.clearLayers(); if (d.roofs) roofLayer.addData(d.roofs);
  instLayer.bringToFront(); showInstLegend(instLayer.getLayers().length > 0);
  if (d.geojson.features.length) map.fitBounds(dataLayer.getBounds(), { maxZoom: 18 });
  const t = $("#jd-table");
  if (job.kind === "area") {
    t.innerHTML = `<tr><th><input type="checkbox" id="jd-selall" title="alle angezeigten"></th><th>#</th><th>Fläche m²</th><th>Teilfl.</th><th>Bauform</th><th>Baujahr ≈</th><th>kWp</th><th>Spanne</th><th>Klasse</th><th>Gebäude</th></tr>` +
      d.rows.map((r) => `<tr class="click" data-lat="${r.lat}" data-lon="${r.lon}"><td><input type="checkbox" class="jd-sel" data-id="${r.installation_id}" ${S.sel && S.sel.has(String(r.installation_id)) ? "checked" : ""}></td><td>${r.installation_id}</td><td class="num">${fmt(r.pv_area_m2, 1)}</td><td class="num">${r.fields || 1}</td><td>${esc(LAYOUT_TXT[r.layout] || "")}</td><td class="num">${r.install_year || ""}</td><td class="num">${fmt(r.kwp_est, 1)}</td><td class="num">${r.kwp_low != null ? fmt(r.kwp_low, 0) + "–" + fmt(r.kwp_high, 0) : ""}</td><td>${r.size_class}</td><td>${esc(r.building_ids)}</td></tr>`).join("");
    $$("#jd-table tr.click").forEach((tr) => tr.onclick = (e) => { if (!e.target.classList.contains("jd-sel")) map.setView([+tr.dataset.lat, +tr.dataset.lon], 19); });
    S.sel = S.selJob === S.job ? (S.sel || new Set()) : new Set(); S.selJob = S.job;
    $$("#jd-table .jd-sel").forEach((cb) => cb.onchange = () => { cb.checked ? S.sel.add(cb.dataset.id) : S.sel.delete(cb.dataset.id); selInfo(); });
    $("#jd-selall").onchange = (e) => { $$("#jd-table .jd-sel").forEach((cb) => { cb.checked = e.target.checked; cb.onchange(); }); };
    fillTargets(); selInfo();
    return;
  }
  S.rows = Object.fromEntries(d.rows.map((r) => [r.id, r]));
  S.sel = S.selJob === S.job ? (S.sel || new Set()) : new Set(); S.selJob = S.job;
  const nach = $("#jd-status").value === "nacharbeit";
  const NA = { verpasst: "verpasst (Register: PV)", "unvollständig": "unvollständig", verdacht: "Verdacht (Score hoch)" };
  const pot = d.rows.some((r) => r.usable_area_m2 != null);
  t.innerHTML = `<tr><th><input type="checkbox" id="jd-selall" title="alle angezeigten"></th>${nach ? "<th>Nacharbeit</th><th>Register kWp</th>" : ""}<th>ID</th><th>Status</th><th>PV m²</th><th>kWp</th>${pot ? "<th>frei m²</th><th>Potenzial kWp</th>" : ""}<th>Klasse</th><th>Score</th><th>Quelle</th></tr>` +
    d.rows.map((r) => `<tr class="click" data-id="${esc(r.id)}"><td><input type="checkbox" class="jd-sel" data-id="${esc(r.id)}" ${S.sel.has(r.id) ? "checked" : ""}></td>${nach ? `<td>${esc(NA[r.nacharbeit] || "")}</td><td class="num">${fmt(r.pv_kwp_register, 1)}</td>` : ""}<td>${esc(r.id)}</td><td>${r.status}</td><td class="num">${fmt(r.pv_area_m2, 1)}</td><td class="num">${fmt(r.kwp_est, 1)}</td>${pot ? `<td class="num">${fmt(r.usable_area_m2, 0)}</td><td class="num">${fmt(r.kwp_potential, 0)}</td>` : ""}<td>${r.size_class || ""}</td><td class="num">${fmt(r.max_prob, 2)}</td><td>${r.source || esc((r.error || "").slice(0, 40))}</td></tr>`).join("") +
    (d.total > d.rows.length ? `<tr><td colspan="8" class="hint">${fmt(d.rows.length)} von ${fmt(d.total)} angezeigt – vollständige Liste in buildings.csv</td></tr>` : "");
  $$("#jd-table tr.click").forEach((tr) => tr.onclick = (e) => { if (!e.target.classList.contains("jd-sel")) showBuilding(tr.dataset.id); });
  $$("#jd-table .jd-sel").forEach((cb) => cb.onchange = () => { cb.checked ? S.sel.add(cb.dataset.id) : S.sel.delete(cb.dataset.id); selInfo(); });
  $("#jd-selall").onchange = (e) => { $$("#jd-table .jd-sel").forEach((cb) => { cb.checked = e.target.checked; cb.onchange(); }); };
  selInfo();
  fillTargets();
}
function fillTargets(sel = "#jd-target") {
  api("/api/datasets").then((ds) => { const cur = $(sel).value;
    $(sel).innerHTML = `<option value="">neuer Datensatz</option>` + ds.map((x) => `<option value="${x.id}">an „${esc(x.name)}“ anhängen (${fmt(x.stats.total)} Bilder)</option>`).join("");
    $(sel).value = cur; }).catch(() => {});
}
function selInfo() { $("#jd-selinfo").textContent = S.sel && S.sel.size ? `${S.sel.size} ausgewählt` : ""; }
$("#jd-selrand").onclick = () => {
  const cbs = $$("#jd-table .jd-sel"), n = Math.min(+$("#jd-seln").value || 40, cbs.length);
  cbs.forEach((cb) => { cb.checked = false; }); S.sel = new Set();
  for (const cb of [...cbs].sort(() => Math.random() - 0.5).slice(0, n)) { cb.checked = true; S.sel.add(cb.dataset.id); }
  selInfo();
};
$("#jd-todataset").onclick = async () => {
  if (!S.sel || !S.sel.size) { alert("Erst Gebäude oder Anlagen auswählen (Häkchen oder „Zufällig wählen“)."); return; }
  const target = $("#jd-target").value;
  const name = target ? null : prompt("Name des Datensatzes", `${(S.jobs.find((x) => x.id === S.job) || {}).kind === "area" ? "Fläche" : "Prüfung"} ${S.job} – ${S.sel.size} Auswahl`); if (!target && name === null) return;
  try { const j = await api("/api/jobs", { json: { kind: "train", params: { from_job: S.job, ids: [...S.sel], name, append_to: target || null } } });
    S.sel = new Set(); openJob(j.id); loadJobs(); }
  catch (e) { alert(e.message); }
};
const ROOF_TXT = { flat: "Flachdach", pitched: "Steildach", mixed: "Flach- und Steildachteile", complex: "komplexes Dach" };
function roofText(r) {
  if (r.usable_area_m2 == null) return "";
  const R = r.roof || {};
  const fs = (R.faces || []).filter((f) => f.kind === "pitched").sort((a, b) => (b.plan_area_m2 || 0) - (a.plan_area_m2 || 0));
  const faces = fs.slice(0, 6).map((f) => `${f.azimuth_deg != null ? Math.round(f.azimuth_deg) + "°" : "?"}/${fmt(f.tilt_deg, 0)}° Neigung: ${f.excluded_north ? "Nordseite, nicht genutzt" : fmt(f.usable_area_m2, 0) + " m² → " + fmt(f.kwp_potential, 0) + " kWp"}`).join(" · ") + (fs.length > 6 ? ` · +${fs.length - 6} weitere` : "");
  const assume = R.roof_type_basis === "LoD2" ? "Dachflächen aus LoD2" : `Annahme ${esc(LAYOUT_TXT[(R.layout_assumed || "").replace(/_(flat|pitched)$/, "")] || R.layout_assumed || "")}${R.tilt_assumed_deg != null ? ", " + fmt(R.tilt_assumed_deg, 0) + "°" : ""}`;
  return `<br><b>Dach</b>: ${esc(ROOF_TXT[r.roof_type] || r.roof_type)} (${esc(R.roof_type_basis || "")}) · ${fmt(r.roof_area_m2, 0)} m² · Aufbauten ${fmt(R.obstacles_area_m2, 0)} m²` +
    `${R.shade_area_m2 ? " · verschattet " + fmt(R.shade_area_m2, 0) + " m²" : ""}${R.pv_area_m2 ? " · PV " + fmt(R.pv_area_m2, 0) + " m²" : ""}` +
    ` → <b>frei ${fmt(r.usable_area_m2, 0)} m² ≈ ${fmt(r.kwp_potential, 0)} kWp</b> (${fmt(r.kwp_pot_low, 0)}–${fmt(r.kwp_pot_high, 0)}; ${assume})${faces ? "<br>Dachflächen: " + faces : ""}`;
}
function yearText(i) {
  if (i.install_year == null) return i.install_year_high != null ? ` · Baujahr vor ${i.install_year_high} (ältere Flüge unklar)` : "";
  const lo = i.install_year_low, hi = i.install_year_high;
  const span = lo == null ? `vor ${hi}` : (lo === hi ? `${hi}` : `${lo}–${hi}`);
  return ` · Baujahr ≈ ${i.install_year} (${span}${i.module_efficiency ? ", Wirkungsgrad " + Math.round(i.module_efficiency * 1000) / 10 + " %" : ""})`;
}
function instText(r) {
  if (!r.installations || !r.installations.length) return "";
  const az = (a) => (a == null ? "" : ` ${Math.round(a)}°`);
  const part = (i) => `${esc(LAYOUT_TXT[i.layout] || i.layout)}${az(i.azimuth_deg)}${i.azimuth2_deg != null ? "/" + Math.round(i.azimuth2_deg) + "°" : ""}` +
    `${i.tilt_class ? " · Neigung " + esc(i.tilt_class) + "°" : ""} · ${fmt(i.outline_area_m2, 0)} m² × Belegung ${fmt(i.fill_factor, 2)}${i.fill_factor_source === "derived_pitch" ? " (gemessen)" : ""} → ${fmt(i.kwp_est, 1)} kWp (${fmt(i.kwp_low, 0)}–${fmt(i.kwp_high, 0)})`;
  // parts that appeared on the same flights form one installation ("system")
  const sys = {}; r.installations.forEach((i, k) => { (sys[i.system || k + 1] = sys[i.system || k + 1] || []).push(i); });
  return "<br>" + Object.entries(sys).map(([n, parts]) => {
    if (parts.length === 1) return `Anlage ${n}: ${part(parts[0])}${yearText(parts[0])}`;
    const sum = (k) => parts.reduce((a, i) => a + (i[k] || 0), 0);
    const big = parts.reduce((a, i) => ((i.outline_area_m2 || 0) > (a.outline_area_m2 || 0) ? i : a), parts[0]);
    return `Anlage ${n} (${parts.length} Teilflächen): ${fmt(sum("kwp_est"), 1)} kWp (${fmt(sum("kwp_low"), 0)}–${fmt(sum("kwp_high"), 0)})${yearText(big)}` +
      parts.map((i, k) => `<br>&nbsp;&nbsp;– Teil ${k + 1}: ${part(i)}`).join("");
  }).join("<br>");
}
function showBuilding(id) {
  const r = S.rows && S.rows[id];
  if (!r) return;
  map.setView([r.lat, r.lon], 19);
  if (r.chip) {
    $("#chipimg").src = `/api/jobs/${S.job}/chips/${encodeURIComponent(r.chip)}`;
    $("#chipcap").innerHTML = `<b>${esc(r.id)}</b> · ${r.status} · ${fmt(r.pv_area_m2, 1)} m² · ≈ ${fmt(r.kwp_est, 1)} kWp${r.kwp_low != null ? ` (${fmt(r.kwp_low, 0)}–${fmt(r.kwp_high, 0)})` : ""} · Klasse ${r.size_class || "–"} · Score ${fmt(r.max_prob, 2)} · ${r.source}` +
      instText(r) + roofText(r) + (r.check && r.check !== "ohne" ? `<br>Gegenprobe ${esc(r.source2 || "")}: ${esc(CHECK_TXT[r.check] || r.check)}` : "") +
      (r.removed && r.removed.length ? `<br><span class="muted">Verworfen: ${r.removed.map((x) => `${esc(x[0])} ${fmt(x[1], 1)} m²`).join(", ")}</span>` : "");
    $("#chipview").classList.remove("hidden");
  }
}
$("#chipclose").onclick = () => $("#chipview").classList.add("hidden");

// ------------------------------------------------------------------ probe
$("#p-todataset").onclick = async () => {
  const P = S.probe; if (!P) return;
  const target = $("#p-target").value;
  const name = target ? null : prompt("Name des Datensatzes", `Probe ${P.lat.toFixed(4)} ${P.lon.toFixed(4)}`); if (!target && name === null) return;
  const inst = (P.d.installations && P.d.installations.features) || [];
  const big = inst.reduce((a, f) => (!a || (f.properties.outline_area_m2 || 0) > (a.outline_area_m2 || 0) ? f.properties : a), null);
  try {
    const j = await api("/api/probe/dataset", { json: { lat: P.lat, lon: P.lon, radius_m: P.radius_m, source: P.source, profile: P.profile,
      polygons: P.d.polygons, kwp_est: P.d.kwp_est, layout: P.d.layout, install_year: big ? big.install_year : null, name, append_to: target || null } });
    $("#p-out").insertAdjacentHTML("beforeend", `<div class="info">Übernommen – Auftrag ${esc(j.id)}; danach im Reiter „Masken“ prüfen.</div>`);
    loadJobs();
  } catch (e) { $("#p-out").insertAdjacentHTML("beforeend", `<div class="info error">${esc(e.message)}</div>`); }
};
async function probe(ll) {
  $("#p-out").innerHTML = `<div class="info">Lade Luftbild und analysiere ${ll.lat.toFixed(5)}, ${ll.lng.toFixed(5)} …</div>`;
  try {
    const d = await api("/api/probe", { json: { lat: ll.lat, lon: ll.lng, source: $("#c-source").value, profile: $("#c-profile").value, radius_m: +$("#p-radius").value } });
    S.probe = { lat: ll.lat, lon: ll.lng, radius_m: +$("#p-radius").value, source: $("#c-source").value, profile: $("#c-profile").value, d };
    $("#p-ds").classList.remove("hidden"); fillTargets("#p-target");
    dataLayer.clearLayers(); instLayer.clearLayers(); roofLayer.clearLayers(); showInstLegend(false);
    if (d.polygons) dataLayer.addData({ type: "FeatureCollection", features: d.polygons.features.map((f) => ({ ...f, properties: { kind: "pv" } })) });
    if (d.installations) { instLayer.addData(d.installations); instLayer.bringToFront(); showInstLegend(instLayer.getLayers().length > 0); }
    $("#p-out").innerHTML = `<div class="info"><b>${esc(d.source_name)}</b><br>
      Score max ${fmt(d.max_prob, 2)} · ${d.n_positive}/${d.n_patches} Ausschnitte positiv · PV ${fmt(d.pv_area_m2, 1)} m² ≈ ${fmt(d.kwp_est, 1)} kWp${d.kwp_low != null ? ` (${fmt(d.kwp_low, 0)}–${fmt(d.kwp_high, 0)})` : ""}${d.layout ? " · " + esc(LAYOUT_TXT[d.layout] || d.layout) : ""}${instText({ installations: (d.installations ? d.installations.features : []).map((f) => f.properties) })}<br>
      Bild ${d.px[0]}×${d.px[1]} px in ${d.fetch_s}s · Analyse ${d.detect_s}s</div>
      <img src="data:image/jpeg;base64,${d.image}"><div class="hint">${esc(d.attribution)}</div>`;
  } catch (e) { $("#p-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
}

// ------------------------------------------------------------------ sources tab
function renderSources() {
  $("#src-table").innerHTML = `<tr><th>Quelle</th><th>Auflösung</th><th>Lizenz</th></tr>` + S.sources.map((s) =>
    `<tr class="click" data-id="${s.id}"><td>${esc(s.name)}${s.notes ? `<div class="hint">${esc(s.notes)}</div>` : ""}</td><td class="num">${s.native_resolution_m * 100} cm</td><td>${esc(s.license)}</td></tr>`).join("");
  $$("#src-table tr.click").forEach((tr) => tr.onclick = () => showSource(tr.dataset.id));
}

// ------------------------------------------------------------------ settings
const SET = { "#s-batch": "batch_size", "#s-fetch": "fetch_workers", "#s-overlap": "patch_overlap", "#s-cls": "cls_threshold", "#s-seg": "seg_threshold",
  "#s-kwp": "kwp_per_m2", "#s-minarea": "min_pv_area_m2", "#s-buffer": "buffer_m", "#s-radius": "point_radius_m",
  "#s-kwpmin": "kwp_min", "#s-kwpmax": "kwp_max", "#s-modeldir": "model_dir", "#s-jobsdir": "jobs_dir",
  "#s-extra": "sources_extra", "#s-pmurl": "pvmap_url", "#s-pmtoken": "pvmap_token",
  "#s-anthkey": "anthropic_api_key", "#s-anthmodel": "anthropic_model", "#s-oaikey": "openai_api_key", "#s-oaimodel": "openai_model", "#s-aihint": "ai_prompt_hint", "#s-cell": "area_cell_px", "#s-profile": "profile", "#s-device-sel": "device" };
const SET_BOOL = { "#s-pp": "postproc", "#s-cross": "cross_check", "#s-typ": "typology", "#s-age": "age_check", "#s-tta": "seg_tta", "#s-sync-on": "sync_enabled", "#s-sync-reviews": "sync_push_reviews", "#s-sync-typ": "typology_push", "#s-lod2": "lod2" };
Object.assign(SET, { "#s-sync-filter": "sync_filter", "#s-sync-start": "sync_start", "#s-sync-end": "sync_end", "#s-sync-profile": "sync_profile",
  "#s-sync-recheck": "sync_recheck_days", "#s-sync-score": "sync_confirm_score" });
const PRIO = { 1: "nie geprüft", 2: "nie geprüft, PV im Register", 3: "„möglich“ mit älterem Modell", 4: "Luftbild erneut ansehen" };
$("#s-sync-status").onclick = async () => {
  $("#s-sync-out").innerHTML = `<div class="info">Frage PV-Map …</div>`;
  try { const d = await api("/api/sync/status");
    const q = d.queue ? Object.entries(d.queue.open).map(([k, v]) => `${PRIO[k] || k}: <b>${fmt(v)}</b>`).join(" · ") + ` · bisher geprüft ${fmt(d.queue.checked)} von ${fmt(d.queue.total)}` : esc(d.queue_error || "");
    const jobs = (d.jobs || []).map((j) => `${esc(j.id)} · ${STATUS[j.status]} · ${fmt(j.done)} Gebäude · ${fmt((j.counters || {}).gesendet)} gesendet`).join("<br>");
    $("#s-sync-out").innerHTML = `<div class="info">Arbeitsliste (${esc(d.filter || "")}): ${q}<br>Nachtbetrieb ${d.enabled ? `an, ${esc(d.start)}–${esc(d.end)}${d.in_window ? " (läuft gerade im Zeitfenster)" : ""}` : "aus"}${jobs ? "<br>" + jobs : ""}</div>`; }
  catch (e) { $("#s-sync-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#s-sync-now").onclick = async () => {
  try { const j = await api("/api/sync/start", { json: { hours: 2 } }); started(j); }
  catch (e) { $("#s-sync-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
function renderSettings() {
  const c = S.info.config, d = S.info.device;
  $("#s-device").innerHTML = `PyTorch ${esc(d.torch)} · Gerät <b>${esc(d.device)}</b>` +
    (d.gpu ? ` · ${esc(d.gpu)} (${d.backend.toUpperCase()}, ${d.vram_gb} GB)` : " · keine GPU erkannt – läuft auf der CPU (langsam)") +
    `<br><span class="hint">Konfiguration: ${esc(S.info.config_path)}</span>`;
  for (const [sel, k] of Object.entries(SET)) $(sel).value = c[k] ?? "";
  for (const [sel, k] of Object.entries(SET_BOOL)) $(sel).value = c[k] === false ? "false" : "true";
  api("/api/roofshape/models").then((ms) => {
    $("#s-roofmodel").innerHTML = `<option value="">keins – OSM-Dachform bzw. Bildanalyse</option>` + ms.map((m) => `<option value="${esc(m.name)}">${esc(m.name)} · ${m.accuracy != null ? Math.round(m.accuracy * 100) + " % richtig" : ""} · ${fmt(m.n_train)} Bilder</option>`).join("");
    $("#s-roofmodel").value = c.roof_model || "";
  }).catch(() => {});
  renderClasses(c.size_classes || []);
}
function renderClasses(list) {
  $("#s-classes").innerHTML = `<tr><th>Name</th><th>Beschreibung</th><th>bis kWp</th><th></th></tr>` + list.map((c, i) =>
    `<tr><td><input value="${esc(c.name)}" data-k="name"></td><td><input value="${esc(c.label || "")}" data-k="label"></td><td><input type="number" value="${c.max ?? ""}" data-k="max"></td><td><button class="secondary" data-del="${i}">×</button></td></tr>`).join("");
  $$("#s-classes [data-del]").forEach((b) => b.onclick = () => { const l = readClasses(); l.splice(+b.dataset.del, 1); renderClasses(l); });
}
function readClasses() {
  return $$("#s-classes tr").slice(1).map((tr) => {
    const g = (k) => tr.querySelector(`[data-k=${k}]`).value.trim();
    return { name: g("name"), label: g("label"), max: g("max") === "" ? null : Number(g("max")) };
  }).filter((c) => c.name);
}
$("#s-addclass").onclick = () => { const l = readClasses(); l.push({ name: "", label: "", max: null }); renderClasses(l); };
async function saveSettings(msg) {
  const body = {};
  for (const [sel, k] of Object.entries(SET)) {
    const v = $(sel).value.trim(), num = $(sel).type === "number";
    body[k] = v === "" ? null : num ? Number(v) : v;
  }
  for (const [sel, k] of Object.entries(SET_BOOL)) {
    body[k] = $(sel).value === "true";
  }
  const cls = readClasses().sort((a, b) => (a.max ?? Infinity) - (b.max ?? Infinity));
  body.size_classes = cls;
  body.roof_model = $("#s-roofmodel").value || null;
  try { await api("/api/config", { json: body }); S.info = await api("/api/info"); renderSettings(); $(msg).innerHTML = `<div class="info">Gespeichert.</div>`; }
  catch (e) { $(msg).innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
}
$("#s-save").onclick = () => saveSettings("#s-msg");
$("#s-save-pm").onclick = () => saveSettings("#s-msg-pm");
$("#s-download").onclick = async () => {
  $("#s-msg").innerHTML = `<div class="info">Lade Gewichte für „${$("#s-profile").value}“ (bis ~650 MB) …</div>`;
  try { await api("/api/models/download", { json: { profile: $("#s-profile").value } }); $("#s-msg").innerHTML = `<div class="info">Modell bereit.</div>`; init(); }
  catch (e) { $("#s-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};

function info(sel, html, err = false) { const el = $(sel); el.innerHTML = html; el.classList.remove("hidden"); el.classList.toggle("err", err); }

init().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<div class="info err">${esc(e.message)}</div>`); });

// ================================================================== datasets & mask editor
// M.polys is always the layer being edited; M.other the other one (layer "pv" or "neg" = confusers such as dormers)
const M = { ds: null, items: [], list: [], idx: -1, item: null, img: null, polys: [], other: [], layer: "pv", hist: [], cur: [], tool: "poly",
  sel: -1, view: { s: 1, x: 0, y: 0 }, drag: null, rect: null, label: 1, dirty: false };
const SRC = { user: "von Hand", ai: "KI", pvmap: "PV-Map", model: "Erkennung ?" };

async function loadDatasets(keep) {
  const ds = await api("/api/datasets"); M.dsList = ds;
  const sel = $("#m-ds");
  sel.innerHTML = ds.length ? ds.map((d) => `<option value="${d.id}">${esc(d.name)} · ${d.id} · ${fmt(d.stats.total)} Bilder</option>`).join("")
    : `<option value="">noch kein Datensatz – im Reiter „Training“ anlegen</option>`;
  if (keep && ds.some((d) => d.id === keep)) sel.value = keep;
  if (sel.value) openDataset(sel.value);
}
$("#m-ds").onchange = () => openDataset($("#m-ds").value);
["#m-filter", "#m-sub", "#m-size"].forEach((s) => $(s).onchange = () => renderList());

async function openDataset(id) {
  M.ds = id; M.dsProfile = ((M.dsList || []).find((d) => d.id === id) || {}).profile;
  // iterative training: start from the model that produced the detections, next version as the name
  if (M.dsProfile && [...$("#m-profile").options].some((o) => o.value === M.dsProfile)) {
    $("#m-profile").value = M.dsProfile;
    const base = M.dsProfile.startsWith("custom:") ? M.dsProfile.slice(7) : "de-" + M.dsProfile, m = base.match(/^(.*?)(?:-v(\d+))?$/);
    if (!$("#m-name").value) $("#m-name").placeholder = `${m[1]}-v${(+m[2] || 1) + 1}`;
  }
  const d = await api(`/api/datasets/${id}/items`);
  M.items = d.items;
  const subs = [...new Set(d.items.map((x) => x.sub))].sort(), sizes = [...new Set(d.items.map((x) => x.size_bin))].sort();
  $("#m-sub").innerHTML = `<option value="">alle Arten</option>` + subs.map((x) => `<option>${esc(x)}</option>`).join("");
  $("#m-size").innerHTML = `<option value="">alle Größen</option>` + sizes.map((x) => `<option>${esc(x)}</option>`).join("");
  if (d.items.some((x) => String(x.label_kind || "").startsWith("det_") && x.status === "open")) $("#m-filter").value = "det";
  renderStats(d.stats); renderList();
  if (M.list.length && (!M.item || !M.items.some((x) => x.name === M.item.name))) openItem(0);   // straight into the editor
}
function renderStats(s) {
  $("#m-stats").innerHTML = [[s.total, "Bilder"], [s.positive, "mit PV (Label)"], [s.positive_without_mask, "PV ohne Maske"],
    [s.masks_user, "Masken von Hand"], [s.masks_ai_unreviewed, "KI ungeprüft"], [s.neg_marked || 0, "mit „Kein PV“-Markierung"], [s.done, "fertig"]]
    .map(([b, t]) => `<div class="stat"><b>${fmt(b)}</b><span>${t}</span></div>`).join("");
}
function matches(m) {
  const f = $("#m-filter").value, sub = $("#m-sub").value, size = $("#m-size").value;
  if (sub && m.sub !== sub) return false;
  if (size && m.size_bin !== size) return false;
  const pos = m.effective_label === 1;
  const isDet = String(m.label_kind || "").startsWith("det_");
  return f === "all" || (f === "todo" && pos && !m.has_mask && m.status !== "skip") || (f === "det" && isDet && m.status === "open")
    || (f === "ai" && m.mask_source === "ai" && !(m.ai && m.ai.reviewed)) || (f === "mask" && m.has_mask)
    || (f === "neg" && !pos) || (["open", "done", "skip"].includes(f) && m.status === f);
}
function renderList() {
  M.list = M.items.filter(matches);
  $("#m-list").innerHTML = `<tr><th>Gebäude</th><th>Label</th><th>Art</th><th>Größe</th><th>Maske</th></tr>` +
    M.list.map((m, i) => `<tr class="click ${M.item && M.item.name === m.name ? "cur" : ""}" data-i="${i}"><td>${esc(m.id)}${m.val ? ' <span class="tag">Val</span>' : ""}</td>
      <td>${m.effective_label ? "PV" : "–"}${m.effective_label !== m.label ? ' <span class="tag">korr.</span>' : ""}</td><td>${esc(m.sub)}</td><td>${esc(m.size_bin)}</td>
      <td>${m.has_mask ? `<span class="tag ${m.mask_source}">${SRC[m.mask_source] || m.mask_source}${m.mask_source === "ai" && !(m.ai && m.ai.reviewed) ? " ?" : ""}</span>` : ""}${m.status === "skip" ? '<span class="tag skip">übersprungen</span>' : m.status === "done" ? " ✓" : ""}</td></tr>`).join("") +
    (M.list.length ? "" : `<tr><td colspan="5" class="hint">Nichts in diesem Filter.</td></tr>`);
  $$("#m-list tr.click").forEach((tr) => tr.onclick = () => openItem(+tr.dataset.i));
}

// ------------------------------------------------------------------ editor
// Tools: poly / rect / rrect (draw), select (pointer: move vertices and areas), pan (hand), zoom (magnifier).
// Hold Shift = hand, Cmd/Ctrl = pointer, Alt = magnifier (click: in, right-click: out, drag: zoom to frame).
// Snap (S): corners found in the image, vertices of all areas and of the footprint, then edges.
const cv = $("#e-canvas"), cx = cv.getContext("2d");
const store = { get(k, d) { try { const v = localStorage.getItem("pvm." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("pvm." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } } };
M.snap = store.get("snap", true); M.full = store.get("full", true); M.corners = [];
function showEditor(on) {
  $("#editor").classList.toggle("hidden", !on);
  document.body.classList.toggle("e-full", on && M.full);
  if (on) fitCanvas();
}
function setFull(f) { M.full = f; store.set("full", f); document.body.classList.toggle("e-full", f && !$("#editor").classList.contains("hidden"));
  $("#e-full").classList.toggle("on", !f); if (M.img) { fitView(); draw(); } }
async function openItem(i) {
  if (i < 0 || i >= M.list.length) { $("#e-info").innerHTML = "Ende der Liste."; return; }
  M.idx = i; const m = M.list[i];
  const base = `/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}`;
  const d = await api(base);
  M.item = { ...m, ...d.meta, width: d.width, height: d.height, footprint: d.footprint };
  M.polys = d.polygons; M.other = d.negatives || []; M.layer = "pv"; setLayerButtons();
  M.hist = []; M.cur = []; M.sel = -1; M.vsel = -1; M.label = d.effective_label; M.dirty = false; M.corners = [];
  M.typo = null; M.typoKey = null;
  $("#e-label").checked = !!M.label; $("#e-usereg").checked = !!M.item.use_register;
  M.img = new Image();
  M.img.onload = () => { showEditor(true); fitView(); draw(); };
  M.img.src = base + "/image";
  api(base + "/corners").then((c) => { if (M.item && M.item.name === m.name) M.corners = c.points; }).catch(() => {});
  info2(); renderList();
  const tr = $(`#m-list tr[data-i="${i}"]`); if (tr) tr.scrollIntoView({ block: "nearest" });
}
const TOOL_TXT = { poly: "Polygon", rect: "Rechteck", rrect: "gedrehtes Rechteck", select: "Zeiger", pan: "Hand", zoom: "Lupe" };
function info2(extra = "") {
  const m = M.item; if (!m) return;
  const ai = m.ai ? ` · KI (${esc(m.ai.provider)}/${esc(m.ai.mode)}): ${m.ai.pv_visible ? "PV" : "keine PV"}${m.ai.confidence != null ? ", Sicherheit " + fmt(m.ai.confidence, 2) : ""}${m.ai.reviewed ? ", geprüft" : ", <b>ungeprüft</b>"}${m.ai.note ? " – " + esc(m.ai.note) : ""}` : "";
  const ty = M.typo && M.typo.installations.length ? " · " + M.typo.installations.map((i) => `${esc(LAYOUT_TXT[i.layout] || i.layout)}${i.azimuth_deg != null ? " " + Math.round(i.azimuth_deg) + "°" + (i.azimuth2_deg != null ? "/" + Math.round(i.azimuth2_deg) + "°" : "") : ""} ${fmt(i.kwp_est, 1)} kWp`).join(", ") +
    ` → <b>${fmt(M.typo.summary.kwp_est, 1)} kWp</b> (${fmt(M.typo.summary.kwp_low, 0)}–${fmt(M.typo.summary.kwp_high, 0)})` : "";
  $("#e-info").innerHTML = `<b>${esc(m.id)}</b> · ${M.idx + 1}/${M.list.length} · ${esc(LBL[m.label_kind] || m.label_kind)} · ${esc(m.sub)} · ${esc(m.size_bin)}${m.pv_kwp ? " · Register " + fmt(m.pv_kwp, 1) + " kWp" : ""} · PV-Fläche ${fmt(areaM2(), 1)} m²${ty}${m.kwp_detected != null ? " · erkannt " + fmt(m.kwp_detected, 1) + " kWp" : ""}${m.install_year ? " · Baujahr ≈ " + m.install_year : ""}${regText(m)} · ${negPolys().length} Aufbauten${ai}${extra ? " · " + extra : ""}
    <span class="hint" style="color:#8f9bb3"> · Ebene <b>${M.layer === "neg" ? "Aufbauten (kein PV)" : "PV"}</b> · Werkzeug <b>${TOOL_TXT[M.tool]}</b>${M.snap ? " · Fang an" : ""} · <a href="#" id="e-helplink">Tasten (?)</a></span>`;
  const hl = $("#e-helplink"); if (hl) hl.onclick = (e) => { e.preventDefault(); toggleHelp(); };
}
// installation typology of the drawn mask (layout, orientation, kWp), refreshed shortly after every change
function typoRefresh() {
  const m = M.item; if (!m) return;
  const polys = pvPolys(), key = m.name + JSON.stringify(polys);
  if (key === M.typoKey) return;
  M.typoKey = key; clearTimeout(M.typoT);
  if (!polys.length) { M.typo = null; return; }
  M.typoT = setTimeout(async () => {
    try {
      const t = await api(`/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}/typology`, { json: { polygons: polys } });
      if (M.typoKey !== key) return;
      M.typo = t; draw(); info2();
    } catch (e) { /* typology is a hint only */ }
  }, 700);
}
function regText(m) {
  const R = m.register; if (!R) return "";
  const hit = (R.buildings || []).filter((b) => b.register);
  if (!hit.length) return ` · Register: <span class="muted">nichts (${(R.buildings || []).length ? "Gebäude ohne Register-PV" : "kein PV-Map-Gebäude in der Nähe"})</span>`;
  const est = M.typo && M.typo.summary && M.typo.summary.kwp_est;
  const cmp = est && hit.length === 1 ? ` · Register / Schätzung ${fmt(hit[0].register.kwp / est, 2)}${m.use_register ? " (für Kalibrierung)" : ""}` : "";
  return " · Register: " + hit.map((b) => { const r = b.register, yrs = Object.entries(r.by_year || {}).sort().map(([y, k]) => `${y}: ${fmt(k, 1)}`).join(", ");
    return `<b>${fmt(r.kwp, 1)} kWp</b> (${r.n} Einheit${r.n === 1 ? "" : "en"}${yrs ? "; " + yrs : ""}; Zuordnung ${esc(r.grade || "?")}${b.distance_m ? "; " + fmt(b.distance_m, 0) + " m entfernt" : ""}${b.name ? "; " + esc(b.name) : ""})`; }).join(" + ") + cmp;
}
async function regLookup(names) {
  $("#m-reg-msg").innerHTML = `<div class="info">frage PV-Map …</div>`; if (names && names.length === 1) info2("Register: frage PV-Map …");
  try {
    const r = await api(`/api/datasets/${M.ds}/register`, { json: { names, radius_m: +$("#m-reg-r").value || 30 } });
    $("#m-reg-msg").innerHTML = `<div class="info">${fmt(r.images)} Bilder abgefragt, ${fmt(r.with_register)} mit Registerdaten${r.errors ? `, ${r.errors} Fehler` : ""}.</div>`;
    const d = await api(`/api/datasets/${M.ds}/items`); M.items = d.items;
    if (M.item) { const it = await api(`/api/datasets/${M.ds}/items/${encodeURIComponent(M.item.name)}`); M.item.register = it.meta.register; info2(); }
  } catch (e) { $("#m-reg-msg").innerHTML = `<div class="info error">${esc(e.message)}</div>`; info2(`<span style="color:#ff8a8a">Register: ${esc(e.message)}</span>`); }
}
$("#m-reg-all").onclick = () => regLookup(null);
$("#m-reg-one").onclick = () => { if (M.item) regLookup([M.item.name]); };
$("#e-reg").onclick = () => { if (M.item) regLookup([M.item.name]); };
function toggleHelp(on) { const h = $("#e-help"), show = on === undefined ? h.classList.contains("hidden") : on; h.classList.toggle("hidden", !show); }
function areaM2() {
  const res = (S.info.profiles.find((p) => p.id === (M.dsProfile || S.info.config.profile)) || { resolution_m: 0.1 }).resolution_m;
  let a = 0; for (const p of pvPolys()) { let s = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; s += x1 * y2 - x2 * y1; } a += Math.abs(s) / 2; }
  return a * res * res;
}
function fitCanvas() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; }
function fitView() { fitCanvas(); const s = Math.min(cv.width / M.img.width, cv.height / M.img.height) * 0.97; M.view = { s, x: (cv.width - M.img.width * s) / 2, y: (cv.height - M.img.height * s) / 2 }; }
window.addEventListener("resize", () => { if (!$("#editor").classList.contains("hidden") && M.img) { fitView(); draw(); } });
const toScreen = ([x, y]) => [M.view.x + x * M.view.s, M.view.y + y * M.view.s];
function evPx(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * devicePixelRatio, (e.clientY - r.top) * devicePixelRatio]; }
function toImgRaw(e) { const [px, py] = evPx(e); return [(px - M.view.x) / M.view.s, (py - M.view.y) / M.view.s]; }
function clampImg([x, y]) { return [Math.max(0, Math.min(M.img.width, x)), Math.max(0, Math.min(M.img.height, y))]; }
function toImg(e) { return clampImg(toImgRaw(e)); }
const r1 = (v) => Math.round(v * 10) / 10;
function path(p, close = true) { cx.beginPath(); p.forEach((q, i) => { const [x, y] = toScreen(q); i ? cx.lineTo(x, y) : cx.moveTo(x, y); }); if (close) cx.closePath(); }
function zoomAt(px, py, f) { M.view.x = px - (px - M.view.x) * f; M.view.y = py - (py - M.view.y) * f; M.view.s *= f; draw(); }

// ---------------------------------------------------------------- snapping
function snapRadius() { return 10 * devicePixelRatio / M.view.s; }        // 10 screen px in image units
function nearestOnSeg(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return [a[0] + t * dx, a[1] + t * dy];
}
function snapPoint(p, skip) {
  // skip: {poly, v} vertex being dragged (not a target for itself)
  M.snapHit = null;
  if (!M.snap || !M.img) return p;
  const R = snapRadius(); let best = null, bd = R;
  const rings = [...M.polys.map((q, i) => ({ q, i, own: true })), ...M.other.map((q) => ({ q })), ...M.item.footprint.map((q) => ({ q }))];
  const consider = (c, kind) => { const d = Math.hypot(c[0] - p[0], c[1] - p[1]); if (d < bd) { bd = d; best = { c, kind }; } };
  for (const { q, i, own } of rings) q.forEach((v, j) => { if (!(own && skip && skip.poly === i && skip.v === j)) consider(v, "Ecke"); });
  M.cur.forEach((v) => consider(v, "Ecke"));
  if (!best) for (const c of M.corners) consider(c, "Bildecke");
  if (!best) for (const { q, i, own } of rings) for (let j = 0; j < q.length; j++) {
    if (own && skip && skip.poly === i && (skip.v === j || skip.v === (j + 1) % q.length)) continue;
    consider(nearestOnSeg(p, q[j], q[(j + 1) % q.length]), "Kante");
  }
  if (!best) return p;
  M.snapHit = best; return [r1(best.c[0]), r1(best.c[1])];
}

// ---------------------------------------------------------------- drawing
function draw() {
  if (!M.img) return;
  cx.setTransform(1, 0, 0, 1, 0, 0); cx.fillStyle = "#1b2232"; cx.fillRect(0, 0, cv.width, cv.height);
  cx.imageSmoothingEnabled = M.view.s < 2;
  cx.drawImage(M.img, M.view.x, M.view.y, M.img.width * M.view.s, M.img.height * M.view.s);
  const lw = 2 * devicePixelRatio;
  cx.lineWidth = lw; cx.strokeStyle = "#ffdc00"; cx.setLineDash([6, 4]); M.item.footprint.forEach((p) => { path(p); cx.stroke(); }); cx.setLineDash([]);
  if (M.typo && M.showTypo !== false) M.typo.installations.forEach((i) => { if (!i.outline_px || i.outline_px.length < 3) return;
    path(i.outline_px); cx.strokeStyle = LAYOUT_COL[i.layout] || "#94a3b8"; cx.lineWidth = lw * 1.2; cx.setLineDash([3, 3]); cx.stroke(); cx.setLineDash([]); });
  typoRefresh();
  const ai = M.item.mask_source === "ai" && !(M.item.ai && M.item.ai.reviewed);
  const drawNeg = (p, act, sel) => { path(p); cx.fillStyle = "rgba(0,200,255,.22)"; cx.fill(); cx.setLineDash([5, 3]);
    cx.strokeStyle = sel ? "#fff" : "#38d6ff"; cx.lineWidth = act ? lw * 1.3 : lw * 0.8; cx.stroke(); cx.setLineDash([]); };
  const drawPv = (p, act, sel) => { path(p); cx.fillStyle = ai ? "rgba(140,90,255,.33)" : "rgba(230,30,60,.33)"; if (act) cx.fill();
    cx.strokeStyle = sel ? "#fff" : (ai ? "#a78bfa" : "#ff3b5c"); cx.lineWidth = sel ? lw * 1.8 : act ? lw : lw * 0.8; cx.stroke(); };
  const act = M.layer;
  (act === "neg" ? M.other : M.polys).forEach((p, i) => drawPv(p, act === "pv", act === "pv" && i === M.sel));
  (act === "neg" ? M.polys : M.other).forEach((p, i) => drawNeg(p, act === "neg", act === "neg" && i === M.sel));
  if (M.sel >= 0 && M.polys[M.sel] && (eff() === "select")) M.polys[M.sel].forEach((v, j) => {
    const [x, y] = toScreen(v); cx.fillStyle = j === M.vsel ? "#ffdc00" : "#fff"; cx.strokeStyle = "#000"; cx.lineWidth = 1;
    cx.fillRect(x - 4 * devicePixelRatio, y - 4 * devicePixelRatio, 8 * devicePixelRatio, 8 * devicePixelRatio);
    cx.strokeRect(x - 4 * devicePixelRatio, y - 4 * devicePixelRatio, 8 * devicePixelRatio, 8 * devicePixelRatio); });
  const curCol = M.layer === "neg" ? "#38d6ff" : "#00e5ff";
  if (M.cur.length) {
    path(M.tool === "poly" && M.hover ? [...M.cur, M.hover] : M.cur, false); cx.strokeStyle = curCol; cx.lineWidth = lw; cx.stroke();
    M.cur.forEach((q) => { const [x, y] = toScreen(q); cx.fillStyle = curCol; cx.fillRect(x - 3, y - 3, 6, 6); });
  }
  if (M.rect) { const [a, b] = M.rect; path([a, [b[0], a[1]], b, [a[0], b[1]]]); cx.strokeStyle = curCol; cx.lineWidth = lw; cx.stroke(); }
  if (M.zrect) { const [a, b] = M.zrect; path([a, [b[0], a[1]], b, [a[0], b[1]]]); cx.strokeStyle = "#fff"; cx.setLineDash([4, 4]); cx.lineWidth = 1.5 * devicePixelRatio; cx.stroke(); cx.setLineDash([]); }
  if (M.tool === "rrect" && M.cur.length && M.hover) {
    const pv = M.cur.length === 1 ? [M.cur[0], M.hover] : rrect(M.cur[0], M.cur[1], M.hover);
    path(pv, M.cur.length === 2); cx.strokeStyle = curCol; cx.lineWidth = lw; cx.stroke();
  }
  if (M.snapHit) { const [x, y] = toScreen(M.snapHit.c); cx.strokeStyle = "#ff3df5"; cx.lineWidth = 2 * devicePixelRatio;
    cx.beginPath(); cx.arc(x, y, 7 * devicePixelRatio, 0, 7); cx.stroke(); }
}
function setLayerButtons() { $("#e-layer-pv").classList.toggle("on", M.layer === "pv"); $("#e-layer-neg").classList.toggle("on", M.layer === "neg"); }
function setLayer(l) {
  if (l === M.layer) return;
  if (M.cur.length >= 3) closePoly();
  [M.polys, M.other] = [M.other, M.polys]; M.layer = l; M.hist = []; M.cur = []; M.sel = -1; M.vsel = -1; setLayerButtons(); draw(); info2();
}
$("#e-layer-pv").onclick = () => setLayer("pv");
$("#e-layer-neg").onclick = () => setLayer("neg");
const pvPolys = () => M.layer === "pv" ? M.polys : M.other, negPolys = () => M.layer === "neg" ? M.polys : M.other;
function push() { M.hist.push(JSON.stringify(M.polys)); if (M.hist.length > 80) M.hist.shift(); M.dirty = true; }
function undo() { if (M.cur.length) { M.cur.pop(); } else if (M.hist.length) { M.polys = JSON.parse(M.hist.pop()); M.sel = -1; M.vsel = -1; } draw(); info2(); }
function closePoly() { if (M.cur.length >= 3) { push(); M.polys.push(M.cur.map(([x, y]) => [r1(x), r1(y)])); } M.cur = []; draw(); info2(); }
function inside(pt, poly) { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j];
  if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c; } return c; }
function rrect(a, b, c) {
  // rectangle with edge a–b, extended perpendicular to the edge up to the projection of c
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
  const t = (c[0] - a[0]) * nx + (c[1] - a[1]) * ny;
  return [a, b, [b[0] + nx * t, b[1] + ny * t], [a[0] + nx * t, a[1] + ny * t]].map(([x, y]) => [r1(x), r1(y)]);
}

// ---------------------------------------------------------------- tools and modifiers
let space = false;
const mods = { shift: false, alt: false, meta: false };
function eff() {
  // temporary tools while a modifier is held (not in the middle of drawing a polygon)
  if (space || mods.shift) return "pan";
  if (mods.alt) return "zoom";
  if (mods.meta) return "select";
  return M.tool;
}
function setCursor() {
  const t = eff();
  cv.style.cursor = M.drag ? "grabbing" : t === "pan" ? "grab" : t === "zoom" ? "zoom-in" : t === "select" ? (M.hoverVertex ? "move" : "default") : "crosshair";
}
function hitVertex(p) {
  const R = 8 * devicePixelRatio / M.view.s;
  const order = M.sel >= 0 ? [M.sel, ...M.polys.keys()] : [...M.polys.keys()];
  for (const i of order) { const q = M.polys[i]; if (!q) continue; const j = q.findIndex((v) => Math.hypot(v[0] - p[0], v[1] - p[1]) < R); if (j >= 0) return { i, j }; }
  return null;
}
function hitEdge(p) {
  const R = 6 * devicePixelRatio / M.view.s;
  for (let i = 0; i < M.polys.length; i++) { const q = M.polys[i];
    for (let j = 0; j < q.length; j++) { const c = nearestOnSeg(p, q[j], q[(j + 1) % q.length]); if (Math.hypot(c[0] - p[0], c[1] - p[1]) < R) return { i, j, c }; } }
  return null;
}
cv.addEventListener("mousedown", (e) => {
  if (!M.img) return;
  mods.shift = e.shiftKey; mods.alt = e.altKey; mods.meta = e.metaKey || e.ctrlKey;
  const t = eff();
  if (t === "pan" || e.button === 1) { M.drag = { x: e.clientX, y: e.clientY, vx: M.view.x, vy: M.view.y }; setCursor(); e.preventDefault(); return; }
  if (t === "zoom") {
    const [px, py] = evPx(e);
    if (e.button === 2) { zoomAt(px, py, 1 / 2); return; }
    M.zrect = [toImgRaw(e), toImgRaw(e)]; M.zstart = [px, py]; return;
  }
  if (e.button === 2) { M.drag = { x: e.clientX, y: e.clientY, vx: M.view.x, vy: M.view.y }; setCursor(); return; }
  const raw = toImg(e);
  if (t === "select") {
    const v = hitVertex(raw);
    if (v) { push(); M.sel = v.i; M.vsel = v.j; M.vdrag = v; draw(); return; }
    let hit = M.polys.findIndex((q) => inside(raw, q));
    if (hit < 0) {                                              // area of the other layer → switch layer
      const ho = M.other.findIndex((q) => inside(raw, q));
      if (ho >= 0) { setLayer(M.layer === "pv" ? "neg" : "pv"); hit = ho; }
    }
    M.sel = hit; M.vsel = -1;
    if (hit >= 0) { push(); M.pdrag = { i: hit, start: raw, orig: JSON.parse(JSON.stringify(M.polys[hit])), moved: false }; }
    draw(); info2(); return;
  }
  const p = snapPoint(raw);
  if (M.tool === "rect") { M.rect = [p, p]; return; }
  if (M.tool === "rrect") {
    M.cur.push(p);
    if (M.cur.length === 3) { const q = rrect(M.cur[0], M.cur[1], M.cur[2]); M.cur = []; push(); M.polys.push(q); info2(); }
    draw(); return;
  }
  if (M.cur.length >= 3) { const [fx, fy] = toScreen(M.cur[0]), [sx, sy] = toScreen(p); if (Math.hypot(fx - sx, fy - sy) < 10 * devicePixelRatio) { closePoly(); return; } }
  M.cur.push(p); draw();
});
cv.addEventListener("mousemove", (e) => {
  if (!M.img) return;
  mods.shift = e.shiftKey; mods.alt = e.altKey; mods.meta = e.metaKey || e.ctrlKey;
  if (M.drag) { M.view.x = M.drag.vx + (e.clientX - M.drag.x) * devicePixelRatio; M.view.y = M.drag.vy + (e.clientY - M.drag.y) * devicePixelRatio; draw(); return; }
  if (M.zrect) { M.zrect[1] = toImgRaw(e); draw(); return; }
  const raw = toImg(e);
  if (M.vdrag) { M.polys[M.vdrag.i][M.vdrag.j] = snapPoint(raw, { poly: M.vdrag.i, v: M.vdrag.j }).map(r1); draw(); info2(); return; }
  if (M.pdrag) { const dx = raw[0] - M.pdrag.start[0], dy = raw[1] - M.pdrag.start[1]; M.pdrag.moved = true;
    M.polys[M.pdrag.i] = M.pdrag.orig.map(([x, y]) => [r1(x + dx), r1(y + dy)]); draw(); return; }
  const t = eff();
  if (t === "select") { M.hoverVertex = !!hitVertex(raw); M.snapHit = null; setCursor(); draw(); return; }
  if (t === "pan" || t === "zoom") { M.snapHit = null; setCursor(); draw(); return; }
  const p = snapPoint(raw);
  if (M.rect) M.rect[1] = p;
  M.hover = p; setCursor(); draw();
});
window.addEventListener("mouseup", (e) => {
  if (M.drag) { M.drag = null; setCursor(); return; }
  if (M.zrect) {
    const [a, b] = M.zrect, [sx, sy] = M.zstart, [ex, ey] = evPx(e); M.zrect = null;
    if (Math.hypot(ex - sx, ey - sy) < 6 * devicePixelRatio) zoomAt(sx, sy, 2);            // click: zoom in
    else {                                                                                   // frame: fit it
      const w = Math.abs(b[0] - a[0]) || 1, h = Math.abs(b[1] - a[1]) || 1, s = Math.min(cv.width / w, cv.height / h) * 0.95;
      M.view = { s, x: cv.width / 2 - (a[0] + b[0]) / 2 * s, y: cv.height / 2 - (a[1] + b[1]) / 2 * s }; draw();
    }
    return;
  }
  if (M.vdrag) { M.vdrag = null; M.snapHit = null; draw(); return; }
  if (M.pdrag) { if (!M.pdrag.moved) M.hist.pop(); M.pdrag = null; draw(); return; }
  if (M.rect) { const [[x1, y1], [x2, y2]] = M.rect; M.rect = null;
    if (Math.abs(x2 - x1) > 2 && Math.abs(y2 - y1) > 2) { push(); M.polys.push([[x1, y1], [x2, y1], [x2, y2], [x1, y2]].map(([x, y]) => [r1(x), r1(y)])); }
    draw(); info2(); }
});
cv.addEventListener("dblclick", (e) => {
  e.preventDefault();
  if (eff() === "select") {                                    // double-click on an edge: insert a vertex
    const h = hitEdge(toImg(e)); if (h) { push(); M.polys[h.i].splice(h.j + 1, 0, h.c.map(r1)); M.sel = h.i; M.vsel = h.j + 1; draw(); }
    return;
  }
  if (M.tool === "poly") { if (M.cur.length > 3) M.cur.pop(); closePoly(); }
});
cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("mouseleave", () => { M.hover = null; M.snapHit = null; if (M.img) draw(); });
cv.addEventListener("wheel", (e) => { e.preventDefault(); const [px, py] = evPx(e);
  if (e.ctrlKey && !mods.meta) { zoomAt(px, py, Math.exp(-e.deltaY * 0.01)); return; }   // pinch on trackpads
  zoomAt(px, py, e.deltaY < 0 ? 1.2 : 1 / 1.2); }, { passive: false });

function setTool(t) { M.tool = t; M.cur = []; M.rect = null; if (t !== "select") M.vsel = -1;
  $$(".etool [data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t)); setCursor(); draw(); info2(); }
function setSnap(on) { M.snap = on; store.set("snap", on); $("#e-snap").classList.toggle("on", on); M.snapHit = null; if (M.img) { draw(); info2(); } }
$$(".etool [data-tool]").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
$("#e-snap").onclick = () => setSnap(!M.snap);
$("#e-full").onclick = () => setFull(!M.full);
$("#e-fit").onclick = () => { fitView(); draw(); };
$("#e-undo").onclick = undo;
$("#e-clear").onclick = () => { if (M.polys.length) { push(); M.polys = []; M.sel = -1; draw(); info2(); } };
$("#e-label").onchange = () => { M.label = $("#e-label").checked ? 1 : 0; M.dirty = true; };
$("#e-prev").onclick = () => openItem(M.idx - 1);
$("#e-next").onclick = () => openItem(M.idx + 1);
$("#e-save").onclick = () => saveItem("done");
$("#e-skip").onclick = () => saveItem("skip");
$("#e-model").onclick = () => suggest(null);
$("#e-ai").onclick = () => suggest($("#m-provider").value);
$("#e-close").onclick = () => { showEditor(false); };
setSnap(M.snap); $("#e-full").classList.toggle("on", !M.full);

async function saveItem(status) {
  const m = M.item; if (!m) return;
  if (M.cur.length >= 3) closePoly();
  try {
    const meta = await api(`/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}`, { json: { polygons: pvPolys(), negatives: negPolys(), label: M.label, status } });
    const it = M.items.find((x) => x.name === m.name); Object.assign(it, meta);
    const keepList = M.list; renderStats(computeStats()); M.list = keepList;
    openItem(M.idx + 1);
  } catch (e) { info2(`<span style="color:#ff8a8a">${esc(e.message)}</span>`); }
}
function computeStats() {
  const s = { total: M.items.length, neg_marked: M.items.filter((m) => m.has_neg).length, positive: 0, positive_without_mask: 0, masks_user: 0, masks_ai_unreviewed: 0, done: 0 };
  for (const m of M.items) { const pos = m.effective_label === 1; s.positive += pos; if (m.status === "done") s.done++;
    if (m.has_mask && m.mask_source === "user") s.masks_user++; if (m.has_mask && m.mask_source === "ai" && !(m.ai && m.ai.reviewed)) s.masks_ai_unreviewed++;
    if (pos && !m.has_mask && m.status !== "skip") s.positive_without_mask++; }
  return s;
}
async function suggest(provider) {
  const m = M.item; if (!m) return;
  info2(provider ? "frage KI …" : "Modell rechnet …");
  try {
    const d = await api(`/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}/suggest`, { json: provider ? { provider, mode: $("#m-mode").value } : { cls_threshold: 0.3 } });
    setLayer("pv"); push(); M.polys = d.polygons;
    draw();
    info2(provider ? `KI: ${d.pv_visible ? "PV sichtbar" : "keine PV"} (${fmt(d.confidence, 2)}) ${esc(d.note || "")} – prüfen und speichern` : `Modellvorschlag (Score ${fmt(d.max_prob, 2)}) – prüfen und speichern`);
  } catch (e) { info2(`<span style="color:#ff8a8a">${esc(e.message)}</span>`); }
}
function editorActive() { return currentTab() === "train" && S.sub === "data" && !$("#editor").classList.contains("hidden"); }
document.addEventListener("keydown", (e) => {
  if (!editorActive()) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement.type !== "checkbox") return;
  const k = e.key, kl = k.toLowerCase();
  if (k === "Shift" || k === "Alt" || k === "Meta" || k === "Control") {
    mods.shift = e.shiftKey; mods.alt = e.altKey; mods.meta = e.metaKey || e.ctrlKey; setCursor(); draw();
    if (k === "Alt") e.preventDefault(); return;
  }
  if (k === " ") { space = true; setCursor(); e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && kl === "z") { undo(); e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey) return;                            // leave browser shortcuts alone
  if (k === "Enter") { e.preventDefault(); if (M.cur.length) closePoly(); else saveItem("done"); }
  else if (k === "Escape") { if (!$("#e-help").classList.contains("hidden")) toggleHelp(false); M.cur = []; M.rect = null; M.zrect = null; M.sel = -1; M.vsel = -1; draw(); }
  else if (k === "Delete" || (k === "Backspace" && M.sel >= 0)) {
    if (M.sel >= 0) { push();
      if (M.vsel >= 0 && M.polys[M.sel].length > 3) { M.polys[M.sel].splice(M.vsel, 1); M.vsel = -1; }
      else { M.polys.splice(M.sel, 1); M.sel = -1; M.vsel = -1; }
      draw(); info2(); }
    e.preventDefault(); }
  else if (k === "Backspace") { undo(); e.preventDefault(); }
  else if (k === "ArrowRight") openItem(M.idx + 1);
  else if (k === "ArrowLeft") openItem(M.idx - 1);
  else if (kl === "n" || kl === "g") setLayer(M.layer === "pv" ? "neg" : "pv");
  else if (k === "1") setTool("poly");
  else if (k === "2") setTool("rect");
  else if (k === "3") setTool("rrect");
  else if (kl === "v") setTool("select");
  else if (kl === "h") setTool("pan");
  else if (kl === "z") setTool("zoom");
  else if (kl === "s") setSnap(!M.snap);
  else if (kl === "b") setFull(!M.full);
  else if (k === "?") toggleHelp();
  else if (k === "+" || k === "=") zoomAt(cv.width / 2, cv.height / 2, 1.5);
  else if (k === "-") zoomAt(cv.width / 2, cv.height / 2, 1 / 1.5);
  else if (kl === "m") suggest(null);
  else if (kl === "k") suggest($("#m-provider").value);
  else if (kl === "r" && M.item) regLookup([M.item.name]);
  else if (kl === "x") saveItem("skip");
  else if (kl === "l") { $("#e-label").checked = !$("#e-label").checked; $("#e-label").onchange(); }
  else if (kl === "f" || k === "0") { fitView(); draw(); }
});
document.addEventListener("keyup", (e) => {
  if (e.key === " ") space = false;
  mods.shift = e.shiftKey; mods.alt = e.altKey; mods.meta = e.metaKey || e.ctrlKey;
  if (editorActive()) { setCursor(); draw(); }
});
window.addEventListener("blur", () => { space = false; mods.shift = mods.alt = mods.meta = false; });

$("#m-ai-batch").onclick = async () => {
  if (!M.ds) return;
  try { const j = await api("/api/jobs", { json: { kind: "ai_masks", params: { dataset: M.ds, provider: $("#m-provider").value, mode: $("#m-mode").value, limit: +$("#m-limit").value } } });
    $("#m-msg").innerHTML = `<div class="info">KI-Auftrag ${j.id} gestartet – Fortschritt unter „Aufträge“. Danach Filter „KI ungeprüft“.</div>`; }
  catch (e) { $("#m-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#m-train").onclick = async () => {
  if (!M.ds) return;
  try { const j = await api("/api/jobs", { json: { kind: "train", params: { dataset_from: M.ds, profile: $("#m-profile").value, epochs: +$("#m-epochs").value,
      name: $("#m-name").value.trim() || $("#m-name").placeholder || null, ai_masks: $("#m-aiuse").value, min_seg_samples: 20,
      calibrate_kwp: $("#m-cal").checked ? $("#m-caluse").value : null } } });
    started(j); }
  catch (e) { $("#m-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
function calText(c) {
  const L = Object.entries(c.layouts || {}).map(([k, v]) => `${esc(LAYOUT_TXT[k.replace(/_(flat|pitched)$/, "")] || k)}${k.endsWith("_flat") ? " (Flachdach)" : k.endsWith("_pitched") ? " (Steildach)" : ""}: ${v.n} Bilder, Belegung ${fmt(v.f_median, 2)} (bisher ${fmt(v.default, 2)})${c.fill_factor[k] != null ? " ✓" : " – zu wenige"}`).join("<br>");
  const sk = Object.entries(c.skipped || {}).map(([k, v]) => `${esc(k)}: ${v}`).join(", ");
  return `${fmt(c.n_images)} Bilder mit Registerleistung${c.ratio_median ? `, Register / Schätzung im Median ${fmt(c.ratio_median, 2)}` : ""}${L ? "<br>" + L : ""}${sk ? `<br><span class="muted">nicht verwendet: ${sk}</span>` : ""}`;
}
$("#m-calonly").onclick = async () => {
  if (!M.ds) return;
  $("#m-calmsg").innerHTML = `<div class="info">rechne …</div>`;
  try {
    const c = await api(`/api/datasets/${M.ds}/calibrate`, { json: { use: $("#m-caluse").value, profile: $("#m-profile").value, save: true } });
    $("#m-calmsg").innerHTML = `<div class="info">${calText(c)}<br>${c.path ? `Gespeichert für <b>${esc(c.profile)}</b> – gilt ab sofort für dieses Modell. <a href="#" id="m-calundo">Zurücknehmen</a>` : "Nichts gespeichert (je Bauform mindestens " + c.min_n + " Bilder nötig)."}</div>`;
    const u = $("#m-calundo"); if (u) u.onclick = async (e) => { e.preventDefault(); await api(`/api/calibration/${encodeURIComponent(c.profile)}`, { method: "DELETE" }); $("#m-calmsg").innerHTML = `<div class="info">Kalibrierung für ${esc(c.profile)} entfernt.</div>`; };
  } catch (e) { $("#m-calmsg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#e-usereg").onchange = async () => {
  const m = M.item; if (!m) return; const on = $("#e-usereg").checked;
  try { await api(`/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}/flags`, { json: { use_register: on } });
    m.use_register = on; const it = M.items.find((x) => x.name === m.name); if (it) it.use_register = on;
    if (on && !m.register) regLookup([m.name]); else info2(); }
  catch (e) { info2(`<span style="color:#ff8a8a">${esc(e.message)}</span>`); $("#e-usereg").checked = !on; }
};
$("#r-start").onclick = async () => {
  const bb = $("#r-bbox").value.trim().split(/[\s,;]+/).map(Number).filter((x) => !isNaN(x));
  const params = { per_class: +$("#r-per").value || 300, epochs: +$("#r-epochs").value || 8, name: $("#r-name").value.trim() || null };
  if (bb.length === 4) params.bbox = bb; else if (S.drawn) params.geometry = S.drawn.geometry;
  try { const j = await api("/api/jobs", { json: { kind: "roofshape", params } }); started(j); }
  catch (e) { $("#r-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#t-prepare").onclick = async () => {
  const tp = trainParams(); if ($("#t-posonly").checked) tp.negatives = 0;
  try { const j = await api("/api/jobs", { json: { kind: "train", params: { ...tp, prepare_only: true } } }); started(j); }
  catch (e) { $("#t-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
