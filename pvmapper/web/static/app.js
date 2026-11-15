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
const drawn = new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({
  draw: { polyline: false, circle: false, marker: false, circlemarker: false, polygon: { showArea: true }, rectangle: true },
  edit: { featureGroup: drawn },
}));
map.on(L.Draw.Event.CREATED, (e) => { drawn.clearLayers(); drawn.addLayer(e.layer); S.drawn = e.layer.toGeoJSON(); $("#a-bbox").value = ""; tab("area"); estimateArea(); });
map.on("click", (e) => { if (currentTab() === "probe") probe(e.latlng); });

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
function tab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== "tab-" + name));
  map.getContainer().style.cursor = name === "probe" ? "crosshair" : "";
  if (name === "jobs") loadJobs();
  if (name === "masks") { loadDatasets(M.ds); if (M.item) showEditor(true); } else showEditor(false);
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => tab(b.dataset.tab)));

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
  renderSources(); renderSettings(); loadJobs();
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
  dataLayer.clearLayers(); dataLayer.addData(d.preview);
  if (d.bbox) map.fitBounds([[d.bbox[1], d.bbox[0]], [d.bbox[3], d.bbox[2]]], { maxZoom: 18 });
  $("#b-start").disabled = !d.count;
}
$("#b-start").onclick = async () => {
  const params = { input: S.input.input, id_field: $("#b-id").value, query: $("#b-sql").value.trim() || null,
    sources: $("#b-source").value, profile: $("#b-profile").value, save_chips: $("#b-chips").value };
  try { const j = await api("/api/jobs", { json: { kind: "buildings", params } }); tab("jobs"); openJob(j.id); }
  catch (e) { info("#b-info", e.message, true); }
};

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
const LBL = { pos_high: "PV sicher", pos_medium: "PV wahrscheinlich", neg_high: "keine PV (gemeldet)", neg_weak: "keine PV bekannt" };
$("#t-strata").onclick = async () => {
  $("#t-out").innerHTML = `<div class="info">Frage PV-Map ab …</div>`;
  try {
    const d = await api("/api/pvmap/strata", { json: trainParams() });
    $("#t-out").innerHTML = `<div class="tablewrap"><table><tr><th>Label</th><th>Art</th><th>Größe</th><th>verfügbar</th><th>kWp</th></tr>` +
      d.strata.map((r) => `<tr><td>${LBL[r.label_kind] || r.label_kind}</td><td>${esc(r.sub)}</td><td>${esc(r.size_bin)}</td><td class="num">${fmt(r.n)}</td><td class="num">${fmt(r.kwp)}</td></tr>`).join("") + `</table></div>`;
  } catch (e) { $("#t-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#t-start").onclick = async () => {
  try { const j = await api("/api/jobs", { json: { kind: "train", params: trainParams() } }); tab("jobs"); openJob(j.id); }
  catch (e) { $("#t-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
async function showReport(id) {
  let r; try { r = await api(`/api/jobs/${id}/report`); } catch { $("#jd-report").innerHTML = ""; return; }
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
  $("#jd-report").innerHTML = h;
}

// ------------------------------------------------------------------ area tab
function areaParams() {
  const p = { sources: $("#a-source").value, profile: $("#a-profile").value,
    buildings: $("#a-buildings").value.trim() || null, only_on_buildings: $("#a-only").checked };
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
}
$("#a-bbox").addEventListener("change", () => {
  const bb = $("#a-bbox").value.trim().split(/[\s,;]+/).map(Number);
  if (bb.length === 4 && bb.every((x) => !isNaN(x))) {
    drawn.clearLayers(); S.drawn = null;
    const r = L.rectangle([[bb[1], bb[0]], [bb[3], bb[2]]]); drawn.addLayer(r); map.fitBounds(r.getBounds());
  }
  estimateArea();
});
$("#a-profile").addEventListener("change", estimateArea);
$("#a-start").onclick = async () => {
  try { const j = await api("/api/jobs", { json: { kind: "area", params: areaParams() } }); tab("jobs"); openJob(j.id); }
  catch (e) { info("#a-est", e.message, true); }
};

// ------------------------------------------------------------------ jobs
const KIND = { buildings: "Gebäude", area: "Fläche", train: "Training", ai_masks: "KI-Masken" };
const STATUS = { queued: "wartet", running: "läuft", done: "fertig", failed: "Fehler", cancelled: "abgebrochen" };
async function loadJobs() {
  S.jobs = await api("/api/jobs");
  $("#job-list").innerHTML = S.jobs.length ? S.jobs.map((j) => {
    const pct = j.total ? Math.round(100 * j.done / j.total) : 0;
    const what = j.kind === "buildings" ? (j.params.input || "").split(/[\\/]/).pop()
      : j.kind === "train" ? ((j.params.prepare_only ? "Datensatz " : "") + (j.params.name || j.params.cc || j.params.dataset_from || "Labels"))
      : j.kind === "ai_masks" ? `${j.params.provider} · ${j.params.dataset}` : (j.params.bbox ? "Rechteck" : "Polygon");
    return `<div class="job ${S.job === j.id ? "sel" : ""}" data-id="${j.id}">
      <div class="top"><span>${KIND[j.kind]} · ${esc(what)}</span><span class="badge ${j.status}">${STATUS[j.status]}</span></div>
      <div class="progress"><div style="width:${pct}%"></div></div>
      <div class="meta">${j.id} · ${fmt(j.done)} / ${fmt(j.total)} ${j.counters.pv != null ? "· " + fmt(j.counters.pv) + " mit PV" : ""}</div></div>`;
  }).join("") : `<p class="hint">Noch keine Aufträge.</p>`;
  $$(".job").forEach((el) => el.onclick = () => openJob(el.dataset.id));
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
    [fmt(j.done) + " / " + fmt(j.total), j.kind === "area" ? "Kacheln" : j.kind === "train" ? "Schritte" : "Gebäude"],
    [fmt(j.rate_per_s * 3600), "pro Stunde"],
    [j.eta_s ? dur(j.eta_s) : (j.status === "done" ? "fertig" : "–"), "Restzeit"],
  ];
  if (j.kind === "buildings") stats.push([fmt(c.pv), "mit PV"], [fmt(c.kein_pv), "ohne PV"], [fmt(c.fehler), "ohne Bild"]);
  else if (j.kind === "ai_masks") stats.push([fmt(c.maske), "Masken"], [fmt(c.keine_pv_laut_ki), "keine PV laut KI"], [fmt(c.fehler), "Fehler"]);
  else if (j.kind === "train" && j.params.prepare_only) stats.push([fmt(c.datensatz), "Bilder im Datensatz"], [fmt(c.bild_fehlt), "ohne Bild"], ["→ Masken", "nächster Schritt"]);
  else if (j.kind === "train") stats.push([fmt(c.f1_vorher, 2), "F1 vorher"], [fmt(c.f1_nachher, 2), "F1 nachher"], [c.modell ? "gespeichert" : "–", "neues Modell"]);
  else stats.push([fmt(c.zellen_mit_pv), "Kacheln mit PV"], [fmt(c.anlagen), "Anlagen"], [fmt(c.fehler), "Fehler"]);
  if (!["train", "ai_masks"].includes(j.kind)) stats.push([fmt(c.kwp_sum, 0) + " kWp", "geschätzt gesamt"]);
  $("#jd-stats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  $("#jd-files").innerHTML = j.files.filter((f) => /\.(csv|geojson|json)$/.test(f) && f !== "job.json")
    .map((f) => `<a class="link" href="/api/jobs/${j.id}/files/${f}">${f}</a>`).join(" · ");
  $("#jd-log").textContent = j.log.join("\n");
  $("#jd-cancel").classList.toggle("hidden", !["running", "queued"].includes(j.status));
  $("#jd-resume").classList.toggle("hidden", !["cancelled", "failed"].includes(j.status));
  $("#jd-status").classList.toggle("hidden", j.kind !== "buildings");
  $$(".filters").forEach((x) => x.classList.toggle("hidden", j.kind === "train" || j.kind === "ai_masks"));
  if (j.kind === "train" || j.kind === "ai_masks") { $("#jd-table").innerHTML = ""; if (j.kind === "train" && (withResults || j.status === "done")) showReport(j.id); else $("#jd-report").innerHTML = ""; }
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
$("#t-clean").ontoggle = async () => {
  if (!$("#t-clean").open) return;
  const l = await api("/api/cleanup");
  $("#t-clean-list").innerHTML = l.length ? `<table><tr><th></th><th>Auftrag</th><th>Art</th><th>Bilder</th><th>eigene Masken</th></tr>` +
    l.map((j) => `<tr><td><input type="checkbox" data-id="${j.id}" ${j.hand_masks ? "" : "checked"}></td><td>${esc(j.id)} ${esc(j.name)}</td><td>${j.kind === "train" ? "Training/Datensatz" : "KI-Masken"}</td><td class="num">${fmt(j.images)}</td><td class="num">${j.hand_masks ? "<b>" + fmt(j.hand_masks) + "</b>" : "–"}</td></tr>`).join("") + "</table>"
    : `<p class="hint">Keine alten Trainingsdaten vorhanden.</p>`;
};
$("#t-clean-go").onclick = async () => {
  const ids = $$("#t-clean-list input:checked").map((x) => x.dataset.id);
  if (!ids.length || !confirm(`${ids.length} Aufträge mit allen Dateien löschen?`)) return;
  try { const r = await api("/api/cleanup", { json: { ids } }); $("#t-clean-msg").innerHTML = `<div class="info">${r.deleted.length} gelöscht.</div>`;
    $("#t-clean").ontoggle(); loadJobs(); if (typeof loadDatasets === "function") loadDatasets(); }
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
  if (d.geojson.features.length) map.fitBounds(dataLayer.getBounds(), { maxZoom: 18 });
  const t = $("#jd-table");
  if (job.kind === "area") {
    t.innerHTML = `<tr><th>#</th><th>Fläche m²</th><th>kWp</th><th>Klasse</th><th>Gebäude</th></tr>` +
      d.rows.map((r) => `<tr class="click" data-lat="${r.lat}" data-lon="${r.lon}"><td>${r.installation_id}</td><td class="num">${fmt(r.pv_area_m2, 1)}</td><td class="num">${fmt(r.kwp_est, 1)}</td><td>${r.size_class}</td><td>${esc(r.building_ids)}</td></tr>`).join("");
    $$("#jd-table tr.click").forEach((tr) => tr.onclick = () => map.setView([+tr.dataset.lat, +tr.dataset.lon], 19));
    return;
  }
  S.rows = Object.fromEntries(d.rows.map((r) => [r.id, r]));
  t.innerHTML = `<tr><th>ID</th><th>Status</th><th>PV m²</th><th>kWp</th><th>Klasse</th><th>Score</th><th>Quelle</th></tr>` +
    d.rows.map((r) => `<tr class="click" data-id="${esc(r.id)}"><td>${esc(r.id)}</td><td>${r.status}</td><td class="num">${fmt(r.pv_area_m2, 1)}</td><td class="num">${fmt(r.kwp_est, 1)}</td><td>${r.size_class || ""}</td><td class="num">${fmt(r.max_prob, 2)}</td><td>${r.source || esc((r.error || "").slice(0, 40))}</td></tr>`).join("") +
    (d.total > d.rows.length ? `<tr><td colspan="7" class="hint">${fmt(d.rows.length)} von ${fmt(d.total)} angezeigt – vollständige Liste in buildings.csv</td></tr>` : "");
  $$("#jd-table tr.click").forEach((tr) => tr.onclick = () => showBuilding(tr.dataset.id));
}
function showBuilding(id) {
  const r = S.rows && S.rows[id];
  if (!r) return;
  map.setView([r.lat, r.lon], 19);
  if (r.chip) {
    $("#chipimg").src = `/api/jobs/${S.job}/chips/${encodeURIComponent(r.chip)}`;
    $("#chipcap").innerHTML = `<b>${esc(r.id)}</b> · ${r.status} · ${fmt(r.pv_area_m2, 1)} m² · ≈ ${fmt(r.kwp_est, 1)} kWp · Klasse ${r.size_class || "–"} · Score ${fmt(r.max_prob, 2)} · ${r.source}` +
      (r.check && r.check !== "ohne" ? `<br>Gegenprobe ${esc(r.source2 || "")}: ${esc(CHECK_TXT[r.check] || r.check)}` : "") +
      (r.removed && r.removed.length ? `<br><span class="muted">Verworfen: ${r.removed.map((x) => `${esc(x[0])} ${fmt(x[1], 1)} m²`).join(", ")}</span>` : "");
    $("#chipview").classList.remove("hidden");
  }
}
$("#chipclose").onclick = () => $("#chipview").classList.add("hidden");

// ------------------------------------------------------------------ probe
async function probe(ll) {
  $("#p-out").innerHTML = `<div class="info">Lade Luftbild und analysiere ${ll.lat.toFixed(5)}, ${ll.lng.toFixed(5)} …</div>`;
  try {
    const d = await api("/api/probe", { json: { lat: ll.lat, lon: ll.lng, source: $("#p-source").value, profile: $("#p-profile").value, radius_m: +$("#p-radius").value } });
    dataLayer.clearLayers();
    if (d.polygons) dataLayer.addData({ type: "FeatureCollection", features: d.polygons.features.map((f) => ({ ...f, properties: { kind: "pv" } })) });
    $("#p-out").innerHTML = `<div class="info"><b>${esc(d.source_name)}</b><br>
      Score max ${fmt(d.max_prob, 2)} · ${d.n_positive}/${d.n_patches} Ausschnitte positiv · PV ${fmt(d.pv_area_m2, 1)} m² ≈ ${fmt(d.kwp_est, 1)} kWp<br>
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
const SET_BOOL = { "#s-pp": "postproc", "#s-cross": "cross_check", "#s-tta": "seg_tta" };
function renderSettings() {
  const c = S.info.config, d = S.info.device;
  $("#s-device").innerHTML = `PyTorch ${esc(d.torch)} · Gerät <b>${esc(d.device)}</b>` +
    (d.gpu ? ` · ${esc(d.gpu)} (${d.backend.toUpperCase()}, ${d.vram_gb} GB)` : " · keine GPU erkannt – läuft auf der CPU (langsam)") +
    `<br><span class="hint">Konfiguration: ${esc(S.info.config_path)}</span>`;
  for (const [sel, k] of Object.entries(SET)) $(sel).value = c[k] ?? "";
  for (const [sel, k] of Object.entries(SET_BOOL)) $(sel).value = c[k] === false ? "false" : "true";
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
$("#s-save").onclick = async () => {
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
  try { await api("/api/config", { json: body }); S.info = await api("/api/info"); renderSettings(); $("#s-msg").innerHTML = `<div class="info">Gespeichert.</div>`; }
  catch (e) { $("#s-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
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
const SRC = { user: "von Hand", ai: "KI", pvmap: "PV-Map", model: "Modell" };

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
  const d = await api(`/api/datasets/${id}/items`);
  M.items = d.items;
  const subs = [...new Set(d.items.map((x) => x.sub))].sort(), sizes = [...new Set(d.items.map((x) => x.size_bin))].sort();
  $("#m-sub").innerHTML = `<option value="">alle Arten</option>` + subs.map((x) => `<option>${esc(x)}</option>`).join("");
  $("#m-size").innerHTML = `<option value="">alle Größen</option>` + sizes.map((x) => `<option>${esc(x)}</option>`).join("");
  renderStats(d.stats); renderList();
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
  return f === "all" || (f === "todo" && pos && !m.has_mask && m.status !== "skip")
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
const cv = $("#e-canvas"), cx = cv.getContext("2d");
function showEditor(on) { $("#editor").classList.toggle("hidden", !on); if (on) fitCanvas(); }
async function openItem(i) {
  if (i < 0 || i >= M.list.length) { $("#e-info").innerHTML = "Ende der Liste."; return; }
  M.idx = i; const m = M.list[i];
  const d = await api(`/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}`);
  M.item = { ...m, ...d.meta, width: d.width, height: d.height, footprint: d.footprint };
  M.polys = d.polygons; M.other = d.negatives || []; M.layer = "pv"; setLayerButtons();
  M.hist = []; M.cur = []; M.sel = -1; M.label = d.effective_label; M.dirty = false;
  $("#e-label").checked = !!M.label;
  M.img = new Image();
  M.img.onload = () => { showEditor(true); fitView(); draw(); };
  M.img.src = `/api/datasets/${M.ds}/items/${encodeURIComponent(m.name)}/image`;
  info2(); renderList();
  const tr = $(`#m-list tr[data-i="${i}"]`); if (tr) tr.scrollIntoView({ block: "nearest" });
}
function info2(extra = "") {
  const m = M.item; if (!m) return;
  const ai = m.ai ? ` · KI (${esc(m.ai.provider)}/${esc(m.ai.mode)}): ${m.ai.pv_visible ? "PV" : "keine PV"}${m.ai.confidence != null ? ", Sicherheit " + fmt(m.ai.confidence, 2) : ""}${m.ai.reviewed ? ", geprüft" : ", <b>ungeprüft</b>"}${m.ai.note ? " – " + esc(m.ai.note) : ""}` : "";
  $("#e-info").innerHTML = `<b>${esc(m.id)}</b> · ${M.idx + 1}/${M.list.length} · ${esc(m.label_kind)} · ${esc(m.sub)} · ${esc(m.size_bin)}${m.pv_kwp ? " · Register " + fmt(m.pv_kwp, 1) + " kWp" : ""} · Fläche ${fmt(areaM2(), 1)} m²${ai}${extra ? " · " + extra : ""}
    <span class="hint" style="color:#8f9bb3"> · ${M.layer === "neg" ? "<b>Ebene „Kein PV“</b> (Gauben, Fenster …) · " : ""}N wechselt PV/Kein PV · 1 Polygon · 2 Rechteck · 3 gedreht (Kante + Breite) · Doppelklick/Enter schließt · Entf löscht Auswahl · Mausrad zoomt, Leertaste+Ziehen verschiebt</span>`;
}
function areaM2() {
  const res = (S.info.profiles.find((p) => p.id === (M.dsProfile || S.info.config.profile)) || { resolution_m: 0.1 }).resolution_m;
  let a = 0; for (const p of pvPolys()) { let s = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; s += x1 * y2 - x2 * y1; } a += Math.abs(s) / 2; }
  return a * res * res;
}
function fitCanvas() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; }
function fitView() { fitCanvas(); const s = Math.min(cv.width / M.img.width, cv.height / M.img.height) * 0.96; M.view = { s, x: (cv.width - M.img.width * s) / 2, y: (cv.height - M.img.height * s) / 2 }; }
window.addEventListener("resize", () => { if (!$("#editor").classList.contains("hidden") && M.img) { fitView(); draw(); } });
const toScreen = ([x, y]) => [M.view.x + x * M.view.s, M.view.y + y * M.view.s];
function toImg(e) { const r = cv.getBoundingClientRect(); const px = (e.clientX - r.left) * devicePixelRatio, py = (e.clientY - r.top) * devicePixelRatio;
  return [Math.max(0, Math.min(M.img.width, (px - M.view.x) / M.view.s)), Math.max(0, Math.min(M.img.height, (py - M.view.y) / M.view.s))]; }
function path(p, close = true) { cx.beginPath(); p.forEach((q, i) => { const [x, y] = toScreen(q); i ? cx.lineTo(x, y) : cx.moveTo(x, y); }); if (close) cx.closePath(); }
function draw() {
  if (!M.img) return;
  cx.setTransform(1, 0, 0, 1, 0, 0); cx.fillStyle = "#1b2232"; cx.fillRect(0, 0, cv.width, cv.height);
  cx.imageSmoothingEnabled = M.view.s < 2;
  cx.drawImage(M.img, M.view.x, M.view.y, M.img.width * M.view.s, M.img.height * M.view.s);
  const lw = 2 * devicePixelRatio;
  cx.lineWidth = lw; cx.strokeStyle = "#ffdc00"; cx.setLineDash([6, 4]); M.item.footprint.forEach((p) => { path(p); cx.stroke(); }); cx.setLineDash([]);
  const ai = M.item.mask_source === "ai" && !(M.item.ai && M.item.ai.reviewed);
  const negs = M.layer === "neg" ? M.polys : M.other, pvs = M.layer === "pv" ? M.polys : M.other;
  negs.forEach((p, i) => { path(p); cx.fillStyle = "rgba(0,200,255,.18)"; cx.fill(); cx.setLineDash([4, 3]);
    cx.strokeStyle = M.layer === "neg" && i === M.sel ? "#fff" : "#38d6ff"; cx.lineWidth = M.layer === "neg" ? lw * 1.4 : lw * 0.8; cx.stroke(); cx.setLineDash([]); });
  if (M.layer === "neg") { cx.globalAlpha = 0.45; pvs.forEach((p) => { path(p); cx.strokeStyle = "#ff3b5c"; cx.lineWidth = lw; cx.stroke(); }); cx.globalAlpha = 1; }
  else M.polys.forEach((p, i) => { path(p); cx.fillStyle = ai ? "rgba(140,90,255,.35)" : "rgba(230,30,60,.35)"; cx.fill(); cx.strokeStyle = i === M.sel ? "#fff" : (ai ? "#a78bfa" : "#ff3b5c"); cx.lineWidth = i === M.sel ? lw * 1.8 : lw; cx.stroke(); });
  if (M.cur.length) { path(M.cur, false); cx.strokeStyle = "#00e5ff"; cx.lineWidth = lw; cx.stroke(); M.cur.forEach((q) => { const [x, y] = toScreen(q); cx.fillStyle = "#00e5ff"; cx.fillRect(x - 3, y - 3, 6, 6); }); }
  if (M.rect) { const [a, b] = M.rect; path([a, [b[0], a[1]], b, [a[0], b[1]]]); cx.strokeStyle = "#00e5ff"; cx.stroke(); }
  if (M.tool === "rrect" && M.cur.length && M.hover) {
    const pv = M.cur.length === 1 ? [M.cur[0], M.hover] : rrect(M.cur[0], M.cur[1], M.hover);
    path(pv, M.cur.length === 2); cx.strokeStyle = "#00e5ff"; cx.lineWidth = 2 * devicePixelRatio; cx.stroke();
  }
}
function setLayerButtons() { $("#e-layer-pv").classList.toggle("on", M.layer === "pv"); $("#e-layer-neg").classList.toggle("on", M.layer === "neg"); }
function setLayer(l) {
  if (l === M.layer) return;
  if (M.cur.length >= 3) closePoly();
  [M.polys, M.other] = [M.other, M.polys]; M.layer = l; M.hist = []; M.cur = []; M.sel = -1; setLayerButtons(); draw(); info2();
}
$("#e-layer-pv").onclick = () => setLayer("pv");
$("#e-layer-neg").onclick = () => setLayer("neg");
const pvPolys = () => M.layer === "pv" ? M.polys : M.other, negPolys = () => M.layer === "neg" ? M.polys : M.other;
function push() { M.hist.push(JSON.stringify(M.polys)); if (M.hist.length > 50) M.hist.shift(); M.dirty = true; }
function undo() { if (M.cur.length) { M.cur.pop(); } else if (M.hist.length) { M.polys = JSON.parse(M.hist.pop()); } draw(); info2(); }
function closePoly() { if (M.cur.length >= 3) { push(); M.polys.push(M.cur.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10])); } M.cur = []; draw(); info2(); }
function inside(pt, poly) { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j];
  if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c; } return c; }

let space = false;
cv.addEventListener("mousedown", (e) => {
  if (!M.img) return;
  if (space || e.button === 1 || e.button === 2) { M.drag = { x: e.clientX, y: e.clientY, vx: M.view.x, vy: M.view.y }; e.preventDefault(); return; }
  const p = toImg(e);
  if (M.tool === "rect") { M.rect = [p, p]; return; }
  if (M.tool === "rrect") {
    M.cur.push(p);
    if (M.cur.length === 3) { const q = rrect(M.cur[0], M.cur[1], M.cur[2]); M.cur = []; push(); M.polys.push(q); info2(); }
    draw(); return;
  }
  if (!M.cur.length) { const hit = M.polys.findIndex((q) => inside(p, q)); if (hit >= 0 && !e.shiftKey) { M.sel = hit; draw(); return; } M.sel = -1; }
  if (M.cur.length >= 3) { const [fx, fy] = toScreen(M.cur[0]), [sx, sy] = toScreen(p); if (Math.hypot(fx - sx, fy - sy) < 10 * devicePixelRatio) { closePoly(); return; } }
  M.cur.push(p); draw();
});
cv.addEventListener("mousemove", (e) => {
  if (M.drag) { M.view.x = M.drag.vx + (e.clientX - M.drag.x) * devicePixelRatio; M.view.y = M.drag.vy + (e.clientY - M.drag.y) * devicePixelRatio; draw(); return; }
  if (M.rect) { M.rect[1] = toImg(e); draw(); }
  else if (M.tool === "rrect" && M.cur.length) { M.hover = toImg(e); draw(); }
});
function rrect(a, b, c) {
  // rectangle with edge a–b, extended perpendicular to the edge up to the projection of c
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
  const t = (c[0] - a[0]) * nx + (c[1] - a[1]) * ny;
  return [a, b, [b[0] + nx * t, b[1] + ny * t], [a[0] + nx * t, a[1] + ny * t]].map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
}
window.addEventListener("mouseup", () => {
  if (M.drag) { M.drag = null; return; }
  if (M.rect) { const [[x1, y1], [x2, y2]] = M.rect; M.rect = null;
    if (Math.abs(x2 - x1) > 2 && Math.abs(y2 - y1) > 2) { push(); M.polys.push([[x1, y1], [x2, y1], [x2, y2], [x1, y2]].map(([x, y]) => [Math.round(x), Math.round(y)])); }
    draw(); info2(); }
});
cv.addEventListener("dblclick", (e) => { e.preventDefault(); if (M.cur.length > 3) M.cur.pop(); closePoly(); });
cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("wheel", (e) => { e.preventDefault(); const r = cv.getBoundingClientRect(); const px = (e.clientX - r.left) * devicePixelRatio, py = (e.clientY - r.top) * devicePixelRatio;
  const f = e.deltaY < 0 ? 1.2 : 1 / 1.2; M.view.x = px - (px - M.view.x) * f; M.view.y = py - (py - M.view.y) * f; M.view.s *= f; draw(); }, { passive: false });

function setTool(t) { M.tool = t; M.cur = []; $$(".etool [data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t)); draw(); }
$$(".etool [data-tool]").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
$("#e-undo").onclick = undo;
$("#e-clear").onclick = () => { if (M.polys.length) { push(); M.polys = []; draw(); info2(); } };
$("#e-label").onchange = () => { M.label = $("#e-label").checked ? 1 : 0; M.dirty = true; };
$("#e-prev").onclick = () => openItem(M.idx - 1);
$("#e-next").onclick = () => openItem(M.idx + 1);
$("#e-save").onclick = () => saveItem("done");
$("#e-skip").onclick = () => saveItem("skip");
$("#e-model").onclick = () => suggest(null);
$("#e-ai").onclick = () => suggest($("#m-provider").value);

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
document.addEventListener("keydown", (e) => {
  if (currentTab() !== "masks" || $("#editor").classList.contains("hidden")) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement.type !== "checkbox") return;
  const k = e.key;
  if (k === " ") { space = true; e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { undo(); e.preventDefault(); return; }
  if (k === "Enter") { e.preventDefault(); if (M.cur.length) closePoly(); else saveItem("done"); }
  else if (k === "Escape") { M.cur = []; M.rect = null; M.sel = -1; draw(); }
  else if (k === "Delete" || (k === "Backspace" && M.sel >= 0)) { if (M.sel >= 0) { push(); M.polys.splice(M.sel, 1); M.sel = -1; draw(); info2(); } e.preventDefault(); }
  else if (k === "Backspace") { undo(); e.preventDefault(); }
  else if (k === "ArrowRight") openItem(M.idx + 1);
  else if (k === "ArrowLeft") openItem(M.idx - 1);
  else if (k.toLowerCase() === "n") setLayer(M.layer === "pv" ? "neg" : "pv");
  else if (k === "1") setTool("poly");
  else if (k === "2") setTool("rect");
  else if (k === "3") setTool("rrect");
  else if (k.toLowerCase() === "m") suggest(null);
  else if (k.toLowerCase() === "k") suggest($("#m-provider").value);
  else if (k.toLowerCase() === "x") saveItem("skip");
  else if (k.toLowerCase() === "l") { $("#e-label").checked = !$("#e-label").checked; $("#e-label").onchange(); }
  else if (k.toLowerCase() === "f") { fitView(); draw(); }
});
document.addEventListener("keyup", (e) => { if (e.key === " ") space = false; });

$("#m-ai-batch").onclick = async () => {
  if (!M.ds) return;
  try { const j = await api("/api/jobs", { json: { kind: "ai_masks", params: { dataset: M.ds, provider: $("#m-provider").value, mode: $("#m-mode").value, limit: +$("#m-limit").value } } });
    $("#m-msg").innerHTML = `<div class="info">KI-Auftrag ${j.id} gestartet – Fortschritt unter „Aufträge“. Danach Filter „KI ungeprüft“.</div>`; }
  catch (e) { $("#m-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#m-train").onclick = async () => {
  if (!M.ds) return;
  try { const j = await api("/api/jobs", { json: { kind: "train", params: { dataset_from: M.ds, profile: $("#m-profile").value, epochs: +$("#m-epochs").value,
      name: $("#m-name").value.trim() || null, ai_masks: $("#m-aiuse").value, min_seg_samples: 20 } } });
    tab("jobs"); openJob(j.id); }
  catch (e) { $("#m-msg").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
$("#t-prepare").onclick = async () => {
  const tp = trainParams(); if ($("#t-posonly").checked) tp.negatives = 0;
  try { const j = await api("/api/jobs", { json: { kind: "train", params: { ...tp, prepare_only: true } } }); tab("jobs"); openJob(j.id); }
  catch (e) { $("#t-out").innerHTML = `<div class="info err">${esc(e.message)}</div>`; }
};
