"""Build date of ground-mounted PV from the Sentinel-2 history, and the addition per quarter (docs/freiflaechen.md).

openEO computes, for every known fence and a ring around it (reference for the season and the weather), the monthly median of
Sentinel-2 L2A bands since July 2015 (cloud, shadow and snow masked with SCL) — one value per polygon and month, so only the
fences are read, not whole tiles. Locally:

    dNDVI(t)  = NDVI(ring) − NDVI(fence)          ≈ 0 while both are fields, clearly > 0 once modules cover the fence
    dBRIGHT(t)= brightness(fence) − brightness(ring)   bare soil (construction) bright, modules dark

Completion month = first month from which dNDVI stays above the midpoint between the level before and after (3 of the next 4
valid months), construction start = earliest month in the 12 months before with bare-soil signature. Fences already covered
in the first year of data get "vor <first month>". Accuracy is checked against MaStR commissioning dates (Brandenburg).

Kept in table ground_history (re-imports of the baseline do not touch it); `apply` copies the dates to freiflaeche.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import statistics
import time

from shapely import wkb
from shapely.geometry import mapping
from shapely.ops import unary_union

from . import store

S2_START = "2015-07-01"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def schema(con):
    con.execute("""
      CREATE TABLE IF NOT EXISTS ground_history (
        fid           text PRIMARY KEY,
        bau_monat     date,              -- completion: modules visible from this month on
        beginn_monat  date,              -- construction start (bare soil), if seen
        vor           date,              -- already built in the first month with data (build date before it)
        unsicher_mon  smallint,          -- months between the last month before and the completion month (data gaps)
        konfidenz     real,
        sprung        real,              -- dNDVI after − before
        serie         jsonb,             -- {month: [dNDVI, dBRIGHT]}
        berechnet     timestamptz NOT NULL DEFAULT now())""")


# -- openEO ------------------------------------------------------------------------------------------------------------

def rings(fences: list[tuple[str, object]], inner_m: float = 30, outer_m: float = 150) -> list:
    """ring 30–150 m around each fence, without any fence (reference area)"""
    deg = 1 / 111320
    allf = unary_union([g for _, g in fences])
    out = []
    for _, g in fences:
        lat = g.centroid.y
        k = 1 / max(math.cos(math.radians(lat)), 0.2)
        r = g.buffer(outer_m * deg * k).difference(g.buffer(inner_m * deg * k)).difference(allf)
        out.append(r if not r.is_empty else g.buffer(outer_m * deg * k).difference(g))
    return out


def geometries(fences: list[tuple[str, object]]) -> dict:
    """FeatureCollection: fence i at index 2i, its ring at 2i+1"""
    feats = []
    for (fid, g), r in zip(fences, rings(fences)):
        feats.append({"type": "Feature", "properties": {"fid": fid, "teil": "flaeche"}, "geometry": mapping(g.simplify(1e-5))})
        feats.append({"type": "Feature", "properties": {"fid": fid, "teil": "ring"}, "geometry": mapping(r.simplify(1e-5))})
    return {"type": "FeatureCollection", "features": feats}


def build(conn, fc: dict, start: str, end: str):
    bbox = _bbox(fc)
    s2 = conn.load_collection("SENTINEL2_L2A", spatial_extent=bbox, temporal_extent=[start, end], bands=BANDS + ["SCL"],
                              max_cloud_cover=80)
    scl = s2.band("SCL")
    bad = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11) | (scl == 0) | (scl == 1)
    s2 = s2.filter_bands(BANDS).mask(bad)
    m = s2.aggregate_temporal_period("month", reducer="median")
    return m.aggregate_spatial(geometries=fc, reducer="mean")


def _bbox(fc: dict) -> dict:
    from shapely.geometry import shape
    b = unary_union([shape(f["geometry"]) for f in fc["features"]]).bounds
    return {"west": b[0], "south": b[1], "east": b[2], "north": b[3]}


def parse_result(text: str, n_geoms: int) -> dict[int, dict[str, list]]:
    """openEO aggregate_spatial result (JSON {date: [[bands] per geometry]} or CSV date,feature_index,bands…) →
    {geometry index: {YYYY-MM: [B02, B03, B04, B08, B11, B12]}}"""
    out: dict[int, dict[str, list]] = {i: {} for i in range(n_geoms)}
    t = text.lstrip()
    if t.startswith("{"):
        d = json.loads(t)
        for date, per in d.items():
            for i, vals in enumerate(per or []):
                if vals and i < n_geoms and all(v is not None for v in vals[:len(BANDS)]):
                    out[i][date[:7]] = [float(v) for v in vals[:len(BANDS)]]
        return out
    rd = csv.DictReader(io.StringIO(text))
    for row in rd:
        date = row.get("date") or row.get("t") or ""
        i = int(float(row.get("feature_index") or row.get("index") or 0))
        cols = [k for k in row if k not in ("date", "t", "feature_index", "index", "")]
        try:
            vals = [float(row[k]) for k in cols[:len(BANDS)]]
        except (TypeError, ValueError):
            continue
        if len(vals) == len(BANDS) and i < n_geoms:
            out[i][date[:7]] = vals
    return out


# -- change point ------------------------------------------------------------------------------------------------------

def features(fence: dict[str, list], ring: dict[str, list]) -> dict[str, tuple[float, float]]:
    """{month: (dNDVI, dBRIGHT)} where both have data"""
    def ndvi(v):
        return (v[3] - v[2]) / (v[3] + v[2]) if (v[3] + v[2]) > 0 else 0.0

    def bright(v):
        return (v[0] + v[1] + v[2]) / 3

    out = {}
    for m in sorted(set(fence) & set(ring)):
        f, r = fence[m], ring[m]
        out[m] = (round(ndvi(r) - ndvi(f), 4), round((bright(f) - bright(r)) / max(bright(r), 1e-6), 4))
    return out


def _mon(m: str) -> int:
    y, mm = m.split("-")
    return int(y) * 12 + int(mm) - 1


def change_point(series: dict[str, tuple[float, float]], min_jump: float = 0.1) -> dict:
    """completion and construction month from the monthly (dNDVI, dBRIGHT) series"""
    ms = sorted(series)
    if len(ms) < 8:
        return {"status": "zu_wenig_daten"}
    s = [series[m][0] for m in ms]
    head, tail = s[:min(12, len(s) // 3)], s[-min(12, len(s) // 3):]
    before, after = statistics.median(head), statistics.median(tail)
    jump = after - before
    if jump < min_jump:
        if after >= min_jump and before >= min_jump:
            return {"status": "vor", "vor": ms[0], "sprung": round(jump, 3)}
        return {"status": "keine_aenderung", "sprung": round(jump, 3)}
    thr = before + jump / 2
    b = [series[m][1] for m in ms]
    covered = [s[i] > thr and b[i] < 0.1 for i in range(len(ms))]       # vegetation gone and not bright bare soil: modules
    done = None
    for i in range(len(ms)):
        nxt = covered[i:i + 4]
        if covered[i] and sum(nxt) >= min(3, len(nxt)):
            done = i
            break
    if done is None:
        return {"status": "unklar", "sprung": round(jump, 3)}
    if done < len(head) // 2:
        return {"status": "vor", "vor": ms[0], "sprung": round(jump, 3)}
    gap = _mon(ms[done]) - _mon(ms[done - 1]) - 1 if done else 0
    start = None
    for j in range(done - 1, -1, -1):                       # construction: bare soil (bright) with vegetation gone
        if _mon(ms[done]) - _mon(ms[j]) > 12:
            break
        if s[j] > thr * 0.6 and series[ms[j]][1] > 0.15:
            start = ms[j]
    after_vals = s[done:]
    spread = statistics.pstdev(after_vals) if len(after_vals) > 1 else 0.1
    conf = max(0.0, min(1.0, (jump / (jump + spread)) * (1 - min(gap, 6) / 12)))
    return {"status": "gebaut", "bau": ms[done], "beginn": start, "unsicher": gap, "sprung": round(jump, 3),
            "konfidenz": round(conf, 2)}


# -- store, apply, accuracy --------------------------------------------------------------------------------------------

def save(con, fid: str, cp: dict, series: dict):
    d = lambda m: (m + "-01") if m else None                     # noqa: E731
    con.execute("""INSERT INTO ground_history (fid, bau_monat, beginn_monat, vor, unsicher_mon, konfidenz, sprung, serie)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (fid) DO UPDATE SET bau_monat = EXCLUDED.bau_monat,
                   beginn_monat = EXCLUDED.beginn_monat, vor = EXCLUDED.vor, unsicher_mon = EXCLUDED.unsicher_mon,
                   konfidenz = EXCLUDED.konfidenz, sprung = EXCLUDED.sprung, serie = EXCLUDED.serie, berechnet = now()""",
                [fid, d(cp.get("bau")), d(cp.get("beginn")), d(cp.get("vor")), cp.get("unsicher"), cp.get("konfidenz"),
                 cp.get("sprung"), json.dumps({m: list(v) for m, v in series.items()})])


def apply(con, region: str, min_conf: float = 0.3) -> int:
    """copy the dates to freiflaeche (build year and first_seen; a later monthly scan keeps its own first_seen for new finds)"""
    schema(con)
    cur = con.execute("""
      UPDATE freiflaeche f SET baujahr_schaetzung = extract(year FROM h.bau_monat)::smallint,
             first_seen = h.bau_monat, updated_at = now()
      FROM ground_history h WHERE h.fid = f.id AND f.region = %s AND h.bau_monat IS NOT NULL AND coalesce(h.konfidenz, 0) >= %s""",
                      [region, min_conf])
    return cur.rowcount


def accuracy(con, region: str) -> dict:
    """detected completion month vs MaStR commissioning (earliest unit of the fence): error in months"""
    rows = con.execute("""
      SELECT h.fid, h.bau_monat, min(r.start_date), count(*) FROM ground_history h
      JOIN ground_register_match m ON m.fid = h.fid AND m.source = 'mastr'
      JOIN ground_register r ON r.source = m.source AND r.unit_id = m.unit_id
      JOIN freiflaeche f ON f.id = h.fid WHERE f.region = %s AND h.bau_monat IS NOT NULL AND r.start_date IS NOT NULL
      GROUP BY h.fid, h.bau_monat""", [region]).fetchall()
    errs = [(b.year * 12 + b.month) - (s.year * 12 + s.month) for _, b, s, _ in rows]
    if not errs:
        return {"n": 0}
    return {"n": len(errs), "median_monate": statistics.median(errs), "median_abs_monate": statistics.median(abs(e) for e in errs),
            "anteil_le_3": round(sum(abs(e) <= 3 for e in errs) / len(errs), 2),
            "anteil_selbes_quartal": round(sum(abs(e) <= 1 for e in errs) / len(errs), 2)}


# -- addition per quarter ----------------------------------------------------------------------------------------------

SIZE_CLASSES_MW = [0.5, 1, 5, 20]          # upper bounds; above the last: "> 20 MW"


def size_class(mw: float, bounds=SIZE_CLASSES_MW) -> str:
    lo = 0
    for b in bounds:
        if mw < b:
            return f"{lo:g}–{b:g} MW"
        lo = b
    return f"> {bounds[-1]:g} MW"


def zubau(con, region: str | None = None, cc: str | None = None, group: str = "none", only_confirmed: bool = False,
          bounds=SIZE_CLASSES_MW) -> dict:
    """addition per quarter (DC/AC min–max, count) by completion month (history, else first_seen); by size class or type"""
    schema(con)
    w, a = ["f.pruefstatus <> 'verworfen'"], []
    if region:
        w.append("f.region = %s"); a.append(region)
    if cc:
        w.append("f.land = %s"); a.append(cc)
    if only_confirmed:
        w.append("f.pruefstatus IN ('manuell_geprueft','gegenprobe_bestaetigt')")
    rows = con.execute(f"""
      SELECT coalesce(h.bau_monat, f.first_seen), h.vor, f.baujahr_schaetzung, f.typ, f.kwp_min, f.kwp_max, f.ac_kw_min, f.ac_kw_max
      FROM freiflaeche f LEFT JOIN ground_history h ON h.fid = f.id WHERE {' AND '.join(w)}""", a).fetchall()
    series: dict[tuple, dict] = {}
    rest = {"vor_2015": 0, "nur_jahr": 0, "ohne_datum": 0}
    for when, vor, bj, typ, k0, k1, a0, a1 in rows:
        if when is None:
            rest["vor_2015" if vor else ("nur_jahr" if bj else "ohne_datum")] += 1
            continue
        q = f"{when.year}-Q{(when.month - 1) // 3 + 1}"
        mid = ((k0 or 0) + (k1 or 0)) / 2000
        g = size_class(mid, bounds) if group == "groesse" else (typ or "unbekannt") if group == "typ" else "alle"
        e = series.setdefault((q, g), {"quartal": q, "segment": g, "n": 0, "dc_mw_min": 0, "dc_mw_max": 0, "ac_mw_min": 0, "ac_mw_max": 0})
        e["n"] += 1
        e["dc_mw_min"] += (k0 or 0) / 1000; e["dc_mw_max"] += (k1 or 0) / 1000
        e["ac_mw_min"] += (a0 or 0) / 1000; e["ac_mw_max"] += (a1 or 0) / 1000
    out = sorted(series.values(), key=lambda e: (e["quartal"], e["segment"]))
    for e in out:
        for k in ("dc_mw_min", "dc_mw_max", "ac_mw_min", "ac_mw_max"):
            e[k] = round(e[k], 2)
    return {"reihe": out, "ohne_quartal": rest, "gruppe": group}


def register_quarters(con, region: str) -> list[dict]:
    """additions per quarter from the register of the region (MaStR: DC gross; MIOZE: count and AC, units ≥ 0.5 MW)"""
    rows = con.execute("""
      SELECT source, to_char(start_date, 'YYYY') || '-Q' || extract(quarter FROM start_date)::int, count(*),
             sum(brutto_kw) / 1000, sum(netto_kw) / 1000
      FROM ground_register WHERE region = %s AND start_date IS NOT NULL AND coalesce(brutto_kw, netto_kw, 0) >= 500
      GROUP BY 1, 2 ORDER BY 2""", [region]).fetchall()
    return [{"quelle": s, "quartal": q, "n": n, "dc_mw": round(b, 2) if b else None, "ac_mw": round(nn, 2) if nn else None}
            for s, q, n, b, nn in rows]


# -- job ---------------------------------------------------------------------------------------------------------------

def run(job, cfg: dict):
    """job kind 'ground_history': params region, chunk (fences per openEO job), max_fences (test), start"""
    from . import capacity, coarse
    key = job.params.get("region") or cfg.get("ground_pilot")
    con = store.connect(cfg)
    store.migrate(con)
    schema(con)
    rows = con.execute("SELECT id, ST_AsBinary(geom_zaun) FROM freiflaeche WHERE region = %s AND pruefstatus <> 'verworfen' "
                       "ORDER BY id", [key]).fetchall()
    fences = [(fid, wkb.loads(bytes(g))) for fid, g in rows]
    if job.params.get("max_fences"):
        fences = fences[: int(job.params["max_fences"])]
    chunk = int(job.params.get("chunk") or cfg.get("ground_history_chunk", 150))
    start = job.params.get("start") or S2_START
    end = dt.date.today().replace(day=1).isoformat()
    groups = [fences[i:i + chunk] for i in range(0, len(fences), chunk)]
    from . import inventory
    d = os.path.join(inventory.workdir(cfg, key), "history")
    os.makedirs(d, exist_ok=True)
    job.total, job.done = len(groups), 0
    job.log(f"Baudatum aus Sentinel-2 ({start} bis {end}): {len(fences)} Flächen in {len(groups)} openEO-Aufträgen")
    conn = coarse.connect(cfg, job.log)
    pending, results, t0 = {}, {}, {}
    par = max(1, min(int(cfg.get("ground_openeo_parallel", 2)), 2))
    queue = list(range(len(groups)))
    while queue or pending:
        job.check_cancel()
        while queue and len(pending) < par:
            i = queue.pop(0)
            out = os.path.join(d, f"teil{i}.json")
            if os.path.exists(out):
                results[i] = open(out).read()
                job.step()
                continue
            fc = geometries(groups[i])
            bj = coarse.retry(lambda: build(conn, fc, start, end).create_job(title=f"pvmapper Baudatum {key} {i}", out_format="JSON"),
                              job, "Auftrag anlegen")
            coarse.retry(bj.start, job, "Auftrag starten")
            pending[i], t0[i] = bj, time.time()
            time.sleep(6)
        time.sleep(float(cfg.get("ground_openeo_poll_s", 30)))
        for i, bj in list(pending.items()):
            st = coarse.retry(bj.status, job, f"Status Teil {i + 1}")
            if st == "finished":
                assets = coarse.retry(lambda: bj.get_results().get_assets(), job, f"Ergebnis Teil {i + 1}")
                a = next((x for x in assets if x.name.endswith((".json", ".csv"))), assets[0])
                p = os.path.join(d, f"teil{i}" + os.path.splitext(a.name)[1])
                coarse.retry(lambda: a.download(p), job, f"Download Teil {i + 1}")
                if not p.endswith(".json"):
                    os.replace(p, os.path.join(d, f"teil{i}.json"))
                results[i] = open(os.path.join(d, f"teil{i}.json")).read()
                try:
                    md = bj.describe()
                    job.log(f"Teil {i + 1}/{len(groups)} fertig nach {(time.time() - t0[i]) / 60:.1f} min, Credits {md.get('costs')}")
                except Exception:                                    # noqa: BLE001
                    pass
                pending.pop(i)
                job.step()
            elif st in ("error", "canceled"):
                job.log(f"Teil {i + 1}: openEO-Auftrag {st}")
                pending.pop(i)
                job.step(fehler=1)
    n = {"gebaut": 0, "vor": 0, "keine_aenderung": 0, "unklar": 0, "zu_wenig_daten": 0}
    for i, text in results.items():
        g = groups[i]
        data = parse_result(text, 2 * len(g))
        for k, (fid, _) in enumerate(g):
            ser = features(data[2 * k], data[2 * k + 1])
            cp = change_point(ser)
            n[cp["status"]] = n.get(cp["status"], 0) + 1
            save(con, fid, cp, ser)
    k = apply(con, key)
    job.log(f"Ergebnis: {n['gebaut']} mit Baudatum, {n['vor']} schon vor Mitte 2015/16 vorhanden, {n['keine_aenderung']} ohne "
            f"erkennbaren Umbruch, {n['unklar']} unklar, {n['zu_wenig_daten']} mit zu wenig Daten; {k} Daten übernommen")
    acc = accuracy(con, key)
    if acc.get("n"):
        job.log(f"Genauigkeit gegen MaStR-Inbetriebnahme ({acc['n']} Flächen): Median {acc['median_monate']:+.0f} Monate, "
                f"Betrag {acc['median_abs_monate']:.0f} Monate, ±3 Monate: {acc['anteil_le_3']:.0%}, ±1 Monat: {acc['anteil_selbes_quartal']:.0%}")
    capacity.apply(con, key, cfg, job.log)
    job.counters.update(n)
