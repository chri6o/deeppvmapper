"""
Building mode: check a list of buildings for rooftop PV.

For every building the matching open-data orthophoto is fetched (footprint +
buffer, at the model's ground resolution), classified and segmented patch by
patch; detections are clipped to the footprint (+1.5 m) and summarised as
PV area, estimated kWp and size class. Results go to results.sqlite so an
interrupted job resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from shapely import wkt as shp_wkt
from shapely.geometry import mapping
from shapely.ops import unary_union

from . import buildings as bld
from . import config as cfgmod
from . import geo, inference, postproc, runtime, sources

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY, status TEXT, has_pv INTEGER, pv_area_m2 REAL, kwp_est REAL,
  size_class TEXT, in_range INTEGER, max_prob REAL, n_installations INTEGER,
  footprint_m2 REAL, source TEXT, lon REAL, lat REAL, error TEXT, chip TEXT,
  props TEXT, footprint_wkt TEXT, pv_wkt TEXT
)"""

COLUMNS = ["id", "status", "has_pv", "pv_area_m2", "kwp_est", "size_class", "in_range",
           "max_prob", "n_installations", "footprint_m2", "source", "lon", "lat", "error", "chip",
           "check", "source2", "removed"]
EXTRA = [("check", "TEXT"), ("source2", "TEXT"), ("removed", "TEXT")]     # added after the first release


def ensure_columns(db):
    have = {r[1] for r in db.execute("PRAGMA table_info(results)")}
    for col, typ in EXTRA:
        if have and col not in have:
            db.execute(f'ALTER TABLE results ADD COLUMN "{col}" {typ}')


def _footprint(b: bld.Building, cfg) -> tuple:
    """Returns (search geometry WGS84, footprint or None)."""
    g = b.geometry
    if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty:
        return g.buffer(0), g.buffer(0)
    c = g.centroid
    utm = f"EPSG:{geo.utm_epsg(c.x, c.y)}"
    circ = geo.reproject(c, "EPSG:4326", utm).buffer(float(cfg["point_radius_m"]))
    return geo.reproject(circ, utm, "EPSG:4326"), None


def _summarise(b, polys, footprint, search, win, src, prob, cfg):
    c = search.centroid
    utm = f"EPSG:{geo.utm_epsg(c.x, c.y)}"
    clip = geo.reproject(footprint if footprint is not None else search, "EPSG:4326", utm)
    if footprint is not None:
        clip = clip.buffer(1.5)
    kept = []
    for p in polys:
        q = geo.reproject(p, "EPSG:4326", utm).intersection(clip)
        if not q.is_empty and q.area >= 0.5:
            kept.append(q)
    merged = unary_union(kept) if kept else None
    area = merged.area if merged is not None else 0.0
    parts = [] if merged is None else list(getattr(merged, "geoms", [merged]))
    # unary_union returns a single Polygon when the buffered parts touch
    n_inst = len(getattr(u := unary_union([p.buffer(1.0) for p in parts]), "geoms", [u])) if len(parts) > 1 else len(parts)
    has_pv = area >= float(cfg["min_pv_area_m2"])
    kwp = round(area * float(cfg["kwp_per_m2"]), 1) if has_pv else 0.0
    return {
        "id": b.id, "status": "pv" if has_pv else "kein_pv", "has_pv": int(has_pv),
        "pv_area_m2": round(area, 1), "kwp_est": kwp,
        "size_class": cfgmod.size_class(kwp, cfg["size_classes"]) if has_pv else "",
        "in_range": int(has_pv and cfgmod.in_kwp_range(kwp, cfg)),
        "max_prob": round(prob, 4), "n_installations": n_inst if has_pv else 0,
        "footprint_m2": round(geo.reproject(footprint, "EPSG:4326", utm).area, 1) if footprint is not None else None,
        "source": src.id, "lon": round(c.x, 7), "lat": round(c.y, 7), "error": None, "chip": None,
        "props": json.dumps(b.props, ensure_ascii=False),
        "footprint_wkt": footprint.simplify(1e-6).wkt if footprint is not None else None,
        "pv_wkt": geo.reproject(merged, utm, "EPSG:4326").wkt if has_pv else None,
    }


def run(job, cfg: dict):
    p = job.params
    src_filter = runtime.source_filter(cfg, p.get("sources"))
    cat = runtime.catalog(cfg)
    det = runtime.detector(cfg, p.get("profile"))
    prof = det.p
    chips_mode = p.get("save_chips", cfg.get("save_chips", "positive"))
    chip_dir = os.path.join(job.dir, "chips")
    os.makedirs(chip_dir, exist_ok=True)

    from .jobs import open_db
    db = open_db(os.path.join(job.dir, "results.sqlite"))
    db.execute(SCHEMA)
    ensure_columns(db)
    done = {r[0] for r in db.execute("SELECT id FROM results WHERE status != 'fehler'")}
    db.execute("DELETE FROM results WHERE status = 'fehler'")

    read_kw = dict(id_field=p.get("id_field") or None, query=p.get("query") or None,
                   layer=p.get("layer") or None, cfg=cfg)
    job.log(f"Lese Gebäude aus {p['input']} …")
    job.total = bld.count_buildings(p["input"], **read_kw)
    job.done = len(done)
    job.log(f"{job.total} Gebäude, davon {len(done)} bereits erledigt. Profil {prof.id}, "
            f"Gerät {det.device}, Auflösung {prof.resolution_m} m/px.")
    job.save()

    fetched: queue.Queue = queue.Queue(maxsize=int(cfg["fetch_workers"]) * 4)
    SENTINEL = object()
    buffer_m = float(cfg["buffer_m"])

    cross = bool(p.get("cross_check", cfg.get("cross_check", True)))
    pp_cfg = dict(cfg, **{k: p[k] for k in postproc.DEFAULTS if k in p})
    job.log("Nachbearbeitung: " + ("an" if pp_cfg.get("postproc", True) else "aus")
            + ", Gegenprobe mit zweitem Luftbild: " + ("an (wo vorhanden)" if cross else "aus"))

    def fetch(b):
        search, fp = _footprint(b, cfg)
        try:
            img, win, src = sources.fetch_for_geometry(
                cat, search, prof.resolution_m, buffer_m if fp is not None else 0.0,
                src_filter, min_px=prof.patch_px)
            img2 = src2 = None
            if cross and getattr(src, "alt", None):
                img2, src2 = sources.fetch_second(cat, src, win, img)
            return b, search, fp, img, win, (src, img2, src2), None
        except Exception as e:                                   # noqa: BLE001
            return b, search, fp, None, None, None, str(e)[:500]

    def producer():
        try:
            with ThreadPoolExecutor(int(cfg["fetch_workers"])) as pool:
                pending = []
                for b in bld.read_buildings(p["input"], **read_kw):
                    if job._cancel.is_set():
                        break
                    if b.id in done:
                        continue
                    pending.append(pool.submit(fetch, b))
                    if len(pending) >= int(cfg["fetch_workers"]) * 3:
                        fetched.put(pending.pop(0).result())
                for f in pending:
                    if job._cancel.is_set():
                        break
                    fetched.put(f.result())
        finally:
            fetched.put(SENTINEL)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    keys = COLUMNS + ["props", "footprint_wkt", "pv_wkt"]
    insert = ("INSERT OR REPLACE INTO results (" + ",".join(f'"{k}"' for k in keys) + ") VALUES ("
              + ",".join("?" * len(keys)) + ")")
    group, group_patches = [], 0
    target_patches = max(det.batch_size * 2, 16)

    def flush():
        nonlocal group, group_patches
        if not group:
            return
        items = [(i, item[3]) for i, item in enumerate(group)]
        items += [((i, 2), item[5][1]) for i, item in enumerate(group) if item[5][1] is not None]
        res = det.run(items)
        rows = []
        for i, (b, search, fp, img, win, (src, img2, src2), _) in enumerate(group):
            r = res[i]
            clip = runtime.clip_mask(fp, win, img.shape) if fp is not None else None
            mask, info = postproc.clean(img, r.mask, prof.resolution_m, clip, pp_cfg)
            check = "ohne"
            if img2 is not None:
                mask2, _ = postproc.clean(img2, res[(i, 2)].mask, prof.resolution_m, clip, pp_cfg)
                mask, check = postproc.fuse(mask, mask2, prof.resolution_m)
            polys = inference.mask_to_polygons(mask, win)
            row = _summarise(b, polys, fp, search, win, src, r.max_prob, cfg)
            row.update(check=check, source2=src2.id if src2 is not None else None,
                       removed=json.dumps(info["removed"][:20], ensure_ascii=False) if info["removed"] else None)
            job.counters["pruefung_" + check] += 1
            if chips_mode == "all" or (chips_mode == "positive" and row["has_pv"]):
                name = runtime.safe_name(b.id) + ".jpg"
                runtime.render_chip(img, mask, win, fp if fp is not None else search,
                                    second=img2).save(os.path.join(chip_dir, name), quality=82)
                row["chip"] = name
            rows.append(tuple(row[k] for k in keys))
            job.counters[row["status"]] += 1
            if row["has_pv"]:
                job.counters["kwp_sum"] = round(job.counters.get("kwp_sum", 0) + row["kwp_est"], 1)
        db.executemany(insert, rows)
        job.step(len(rows))
        group, group_patches = [], 0

    while True:
        item = fetched.get()
        if item is SENTINEL:
            break
        job.check_cancel()
        b, search, fp, img, win, src, err = item
        if err:
            c = search.centroid
            db.execute(insert, (b.id, "fehler", 0, None, None, "", 0, None, None, None, None,
                                round(c.x, 7), round(c.y, 7), err, None, None, None, None,
                                json.dumps(b.props, ensure_ascii=False),
                                fp.wkt if fp is not None else None, None))
            job.counters["fehler"] += 1
            job.step(1)
            if job.counters["fehler"] <= 20:
                job.log(f"Bild für {b.id} nicht verfügbar: {err[:160]}")
            continue
        group.append(item)
        n = len(inference.patch_grid(max(img.shape[1], prof.patch_px),
                                     max(img.shape[0], prof.patch_px), prof.patch_px, det.overlap))
        group_patches += n
        if group_patches >= target_patches:
            flush()
    flush()
    job.check_cancel()
    export(job, db)
    job.log(f"Fertig: {job.counters.get('pv', 0)} mit PV, {job.counters.get('kein_pv', 0)} ohne, "
            f"{job.counters.get('fehler', 0)} Fehler.")


def export(job, db=None):
    """Writes buildings.csv, buildings.geojson and pv_polygons.geojson."""
    import csv
    from .jobs import open_db
    db = db or open_db(os.path.join(job.dir, "results.sqlite"))
    names = [d[0] for d in db.execute("SELECT * FROM results LIMIT 0").description]

    def rows():
        yield from db.execute("SELECT * FROM results ORDER BY rowid")

    prop_keys: list = []
    for r in db.execute("SELECT * FROM results ORDER BY rowid LIMIT 2000"):
        for k in json.loads(r[names.index("props")] or "{}"):
            if k not in prop_keys and k not in COLUMNS:
                prop_keys.append(k)

    with open(os.path.join(job.dir, "buildings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS + prop_keys)
        for r in rows():
            d = dict(zip(names, r))
            props = json.loads(d["props"] or "{}")
            w.writerow([d[k] for k in COLUMNS] + [props.get(k) for k in prop_keys])

    def write_fc(path, features):
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"type":"FeatureCollection","features":[\n')
            first = True
            for ft in features:
                if not first:
                    f.write(",\n")
                f.write(json.dumps(ft, ensure_ascii=False))
                first = False
            f.write("\n]}\n")

    def bfeats():
        for r in rows():
            d = dict(zip(names, r))
            g = shp_wkt.loads(d["footprint_wkt"]) if d["footprint_wkt"] else None
            geom = mapping(g) if g is not None else {"type": "Point", "coordinates": [d["lon"], d["lat"]]}
            props = {k: d[k] for k in COLUMNS}
            props.update({k: v for k, v in json.loads(d["props"] or "{}").items() if k not in props})
            yield {"type": "Feature", "geometry": geom, "properties": props}

    def pvfeats():
        for r in rows():
            d = dict(zip(names, r))
            if d["pv_wkt"]:
                yield {"type": "Feature", "geometry": mapping(shp_wkt.loads(d["pv_wkt"])),
                       "properties": {"building_id": d["id"], "pv_area_m2": d["pv_area_m2"],
                                      "kwp_est": d["kwp_est"], "size_class": d["size_class"],
                                      "source": d["source"]}}

    write_fc(os.path.join(job.dir, "buildings.geojson"), bfeats())
    compare_with_known(job, db)
    write_fc(os.path.join(job.dir, "pv_polygons.geojson"), pvfeats())
    job.log("Exporte geschrieben: buildings.csv, buildings.geojson, pv_polygons.geojson")


def compare_with_known(job, db):
    """If the buildings came with a known PV state (PV-Map: pv_conf from registers/layers/reports), compare it with the
    detection: agreement by source confidence and by building type → comparison.json (and the job log)."""
    import collections
    stats = collections.defaultdict(lambda: collections.Counter())
    any_known = False
    for status, props, kwp in db.execute("SELECT status, props, kwp_est FROM results WHERE status != 'fehler'"):
        p = json.loads(props or "{}")
        if "pv_conf" not in p:
            continue
        any_known = True
        known = p.get("pv_conf") or ("no_pv_reported" if p.get("reported_no_pv") else "unknown")
        det = "pv" if status == "pv" else "kein_pv"
        for key in ("alle", f"art:{p.get('sub') or '?'}"):
            stats[(key, known)][det] += 1
    if not any_known:
        return
    rows = []
    for (group, known), c in sorted(stats.items()):
        n = c["pv"] + c["kein_pv"]
        rows.append({"gruppe": group, "bekannt": known, "n": n, "erkannt_pv": c["pv"],
                     "anteil_erkannt": round(c["pv"] / n, 3) if n else None})
    with open(os.path.join(job.dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    for r in rows:
        if r["gruppe"] == "alle":
            job.log(f"Abgleich PV-Map: bekannt={r['bekannt']}: {r['erkannt_pv']}/{r['n']} als PV erkannt")
