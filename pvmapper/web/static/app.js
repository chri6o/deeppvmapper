// PV-Mapper web UI — plain JS, Leaflet for the map.
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
const KIND = { buildings: "Gebäude", area: "Fläche" };
const STATUS = { queued: "wartet", running: "läuft", done: "fertig", failed: "Fehler", cancelled: "abgebrochen" };
async function loadJobs() {
  S.jobs = await api("/api/jobs");
  $("#job-list").innerHTML = S.jobs.length ? S.jobs.map((j) => {
    const pct = j.total ? Math.round(100 * j.done / j.total) : 0;
    const what = j.kind === "buildings" ? (j.params.input || "").split(/[\\/]/).pop() : (j.params.bbox ? "Rechteck" : "Polygon");
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
    [fmt(j.done) + " / " + fmt(j.total), j.kind === "area" ? "Kacheln" : "Gebäude"],
    [fmt(j.rate_per_s * 3600), "pro Stunde"],
    [j.eta_s ? dur(j.eta_s) : (j.status === "done" ? "fertig" : "–"), "Restzeit"],
  ];
  if (j.kind === "buildings") stats.push([fmt(c.pv), "mit PV"], [fmt(c.kein_pv), "ohne PV"], [fmt(c.fehler), "ohne Bild"]);
  else stats.push([fmt(c.zellen_mit_pv), "Kacheln mit PV"], [fmt(c.anlagen), "Anlagen"], [fmt(c.fehler), "Fehler"]);
  stats.push([fmt(c.kwp_sum, 0) + " kWp", "geschätzt gesamt"]);
  $("#jd-stats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  $("#jd-files").innerHTML = j.files.filter((f) => /\.(csv|geojson|json)$/.test(f) && f !== "job.json")
    .map((f) => `<a class="link" href="/api/jobs/${j.id}/files/${f}">${f}</a>`).join(" · ");
  $("#jd-log").textContent = j.log.join("\n");
  $("#jd-cancel").classList.toggle("hidden", !["running", "queued"].includes(j.status));
  $("#jd-resume").classList.toggle("hidden", !["cancelled", "failed"].includes(j.status));
  $("#jd-status").classList.toggle("hidden", j.kind === "area");
  if (withResults || (j.status !== S.lastStatus && j.status === "done")) loadResults();
  S.lastStatus = j.status;
  if (!["running", "queued"].includes(j.status)) { clearInterval(S.poll); loadJobs(); }
}
function dur(s) { const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return h ? `${h} h ${m} min` : `${m} min`; }
$("#jd-cancel").onclick = async () => { await api(`/api/jobs/${S.job}/cancel`, { method: "POST" }); refreshJob(); };
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
    $("#chipcap").innerHTML = `<b>${esc(r.id)}</b> · ${r.status} · ${fmt(r.pv_area_m2, 1)} m² · ≈ ${fmt(r.kwp_est, 1)} kWp · Klasse ${r.size_class || "–"} · Score ${fmt(r.max_prob, 2)} · ${r.source}`;
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
  "#s-extra": "sources_extra", "#s-cell": "area_cell_px", "#s-profile": "profile", "#s-device-sel": "device" };
function renderSettings() {
  const c = S.info.config, d = S.info.device;
  $("#s-device").innerHTML = `PyTorch ${esc(d.torch)} · Gerät <b>${esc(d.device)}</b>` +
    (d.gpu ? ` · ${esc(d.gpu)} (${d.backend.toUpperCase()}, ${d.vram_gb} GB)` : " · keine GPU erkannt – läuft auf der CPU (langsam)") +
    `<br><span class="hint">Konfiguration: ${esc(S.info.config_path)}</span>`;
  for (const [sel, k] of Object.entries(SET)) $(sel).value = c[k] ?? "";
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
