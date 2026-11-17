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
import re
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from shapely import wkt as shp_wkt
from shapely.geometry import mapping
from shapely.ops import unary_union

from . import buildings as bld
from . import config as cfgmod
from . import geo, inference, postproc, runtime, sources, typology

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY, status TEXT, has_pv INTEGER, pv_area_m2 REAL, kwp_est REAL,
  size_class TEXT, in_range INTEGER, max_prob REAL, n_installations INTEGER,
  footprint_m2 REAL, source TEXT, lon REAL, lat REAL, error TEXT, chip TEXT,
  props TEXT, footprint_wkt TEXT, pv_wkt TEXT
)"""


def _year(src) -> int | None:
    """year of the imagery, from the source (`year`) or its id (de-nw-2024, nl-pdok-hr-2025)"""
    y = getattr(src, "year", None)
    if y:
        return int(y)
    m = re.search(r"(?:^|-)(20\d\d)(?:-|$)", getattr(src, "id", "") or "")
    return int(m.group(1)) if m else None


def apply_typology(row: dict, img, mask, win, clip, src, props: dict, res_m: float, cfg: dict) -> list:
    """installations of one building (typology.py) → row gets kwp_est/low/high, layout, kwp_method, installations"""
    row.update(kwp_low=None, kwp_high=None, layout=None, kwp_method="area_factor", installations=None)
    if not row["has_pv"] or not cfg.get("typology", True):
        return []
    try:
        insts = typology.analyse(img, mask, res_m, clip, props, cfg, year=_year(src))
    except Exception as e:                          # never lose the detection over the classification
        row["error"] = f"Typologie: {type(e).__name__}: {e}"[:300]
        return []
    feats = typology.to_wgs84(insts, win, row["id"], {"source": src.id, "year": _year(src)})
    if not feats:
        return []
    P = [f["properties"] for f in feats]
    kwp = round(sum(p["kwp_est"] for p in P), 1)
    row.update(kwp_est=kwp, kwp_low=round(sum(p["kwp_low"] for p in P), 1),
               kwp_high=round(sum(p["kwp_high"] for p in P), 1), kwp_method="typology",
               installations=json.dumps(feats, ensure_ascii=False, separators=(",", ":")))
    by = {}
    for p in P:
        by[p["layout"]] = by.get(p["layout"], 0) + p["outline_area_m2"]
    row["layout"] = max(by, key=by.get) if len(by) == 1 or max(by.values()) >= 0.8 * sum(by.values()) else "mixed"
    row["size_class"] = cfgmod.size_class(kwp, cfg["size_classes"])
    row["in_range"] = int(cfgmod.in_kwp_range(kwp, cfg))
    return feats

COLUMNS = ["id", "status", "has_pv", "pv_area_m2", "kwp_est", "size_class", "in_range",
           "max_prob", "n_installations", "footprint_m2", "source", "lon", "lat", "error", "chip",
           "check", "source2", "removed", "kwp_low", "kwp_high", "layout", "kwp_method"]
EXTRA = [("check", "TEXT"), ("source2", "TEXT"), ("removed", "TEXT"),      # added after the first release
         ("kwp_low", "REAL"), ("kwp_high", "REAL"), ("layout", "TEXT"), ("kwp_method", "TEXT"),
         ("installations", "TEXT")]


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
        "props": json.dumps({k: v for k, v in b.props.items() if not k.startswith("_")}, ensure_ascii=False),
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
    if p.get("max"):
        job.total = min(job.total, len(done) + int(p["max"]))
    job.done = len(done)
    job.log(f"{job.total} Gebäude, davon {len(done)} bereits erledigt. Profil {prof.id}, "
            f"Gerät {det.device}, Auflösung {prof.resolution_m} m/px.")
    job.save()

    # results back to PV-Map (sync job with the work list, PV-Map docs/pvmapper-ergebnisse.md)
    push = bool(p.get("push"))
    client = pending = None
    if push:
        from . import pvmap
        client = pvmap.Client.from_config(cfg)
        pending = []
    until = None
    if p.get("until"):
        import datetime as _dt
        until = _dt.datetime.fromisoformat(p["until"])
    max_n = int(p.get("max") or 0)
    confirm = float(cfg.get("sync_confirm_score", 0.9))
    push_typology = bool(cfg.get("typology_push", False))

    def send(force=False):
        if push and pending and (force or len(pending) >= 200):
            try:
                out = client.send_results(list(pending), run=job.id, model=prof.id)
                job.counters["gesendet"] += len(pending)
                if out.get("unknown"):
                    job.counters["unbekannt_in_pvmap"] += out["unknown"]
            except Exception as e:                               # noqa: BLE001
                job.counters["senden_fehler"] += len(pending)
                if job.counters["senden_fehler"] <= len(pending):
                    job.log(f"Senden an PV-Map fehlgeschlagen: {str(e)[:200]}")
                return
            pending.clear()

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
            if push:
                b.props["_img_hash"] = h = sources.image_hash(img)
                sc = b.props.get("_scan") or {}
                # work list: same aerial image and same model as at the last check → nothing new to compute
                if sc.get("img_hash") and sources.hash_distance(sc["img_hash"], h) <= 12 and sc.get("model") == prof.id:
                    b.props["_unchanged"] = True
                    return b, search, fp, None, win, src, None
            return b, search, fp, img, win, src, None
        except Exception as e:                                   # noqa: BLE001
            return b, search, fp, None, None, None, str(e)[:500]

    def producer():
        try:
            with ThreadPoolExecutor(int(cfg["fetch_workers"])) as pool:
                pending = []
                n_read = 0
                for b in bld.read_buildings(p["input"], **read_kw):
                    if job._cancel.is_set():
                        break
                    if until is not None and _dt.datetime.now(until.tzinfo) >= until:
                        job.log("Zeitfenster zu Ende – Rest beim nächsten Lauf.")
                        break
                    n_read += 1
                    if max_n and n_read > max_n:
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

    keys = COLUMNS + ["props", "footprint_wkt", "pv_wkt", "installations"]
    insert = ("INSERT OR REPLACE INTO results (" + ",".join(f'"{k}"' for k in keys) + ") VALUES ("
              + ",".join("?" * len(keys)) + ")")
    group, group_patches = [], 0
    target_patches = max(det.batch_size * 2, 16)
    import time as _t
    t_start, n_start, last_report = _t.time(), job.done, job.done
    det.stats.update(patches=0, segmented=0, gpu_s=0.0)
    job.log(f"Grafikkarte: {'halbe Genauigkeit (fp16)' if det.fp16 else 'volle Genauigkeit (fp32)'}, "
            f"Batch {det.batch_size}, Spiegel-Mittelung {'an' if det.tta else 'aus'}.")

    def flush():
        nonlocal group, group_patches
        if not group:
            return
        res = det.run([(i, item[3]) for i, item in enumerate(group)])
        masks, infos, clips = {}, {}, {}
        for i, (b, search, fp, img, win, src, _) in enumerate(group):
            clips[i] = runtime.clip_mask(fp, win, img.shape) if fp is not None else None
            masks[i], infos[i] = postproc.clean(img, res[i].mask, prof.resolution_m, clips[i], pp_cfg, prob=res[i].prob)
        # cross-check only where the first image shows PV: a second year of imagery for the hits
        seconds = {}
        hits = [i for i, item in enumerate(group) if cross and masks[i].any() and getattr(item[5], "alt", None)]
        if hits:
            with ThreadPoolExecutor(int(cfg["fetch_workers"])) as pool:
                got = pool.map(lambda i: (i, *sources.fetch_second(cat, group[i][5], group[i][4], group[i][3])), hits)
                seconds = {i: (im2, s2) for i, im2, s2 in got if im2 is not None}
            if seconds:
                res2 = det.run([(i, im2) for i, (im2, _) in seconds.items()])
        rows = []
        for i, (b, search, fp, img, win, src, _) in enumerate(group):
            r = res[i]
            mask, info = masks[i], infos[i]
            img2, src2 = seconds.get(i, (None, None))
            check = "ohne"
            if img2 is not None:
                mask2, _ = postproc.clean(img2, res2[i].mask, prof.resolution_m, clips[i], pp_cfg, prob=res2[i].prob)
                mask, check = postproc.fuse(mask, mask2, prof.resolution_m)
            polys = inference.mask_to_polygons(mask, win)
            row = _summarise(b, polys, fp, search, win, src, r.max_prob, cfg)
            feats = apply_typology(row, img, mask, win, clips[i], src, b.props, prof.resolution_m, cfg)
            row.update(check=check, source2=src2.id if src2 is not None else None,
                       removed=json.dumps(info["removed"][:20], ensure_ascii=False) if info["removed"] else None)
            job.counters["pruefung_" + check] += 1
            if chips_mode == "all" or (chips_mode == "positive" and row["has_pv"]):
                name = runtime.safe_name(b.id) + ".jpg"
                runtime.render_chip(img, mask, win, fp if fp is not None else search,
                                    second=img2).save(os.path.join(chip_dir, name), quality=82)
                row["chip"] = name
            rows.append(tuple(row[k] for k in keys))
            if push:
                tier = 1 if (check == "bestaetigt" and r.max_prob >= confirm) else 2
                pending.append({"id": b.id, "result": "pv" if row["has_pv"] else "none",
                                "tier": tier if row["has_pv"] else None, "kwp": row["kwp_est"], "area_m2": row["pv_area_m2"],
                                "max_prob": row["max_prob"], "check": check, "source": src.id,
                                "source2": src2.id if src2 is not None else None, "img_hash": b.props.get("_img_hash"),
                                "polygons": mapping(shp_wkt.loads(row["pv_wkt"])) if row["pv_wkt"] else None,
                                "n_installations": row["n_installations"], "size_class": row["size_class"]})
                if push_typology and feats:
                    pending[-1].update(kwp_low=row["kwp_low"], kwp_high=row["kwp_high"], layout=row["layout"],
                                       installations=[{**f["properties"], "geometry": f["geometry"]} for f in feats])
            job.counters[row["status"]] += 1
            if row["has_pv"]:
                job.counters["kwp_sum"] = round(job.counters.get("kwp_sum", 0) + row["kwp_est"], 1)
        db.executemany(insert, rows)
        job.step(len(rows))
        send()
        group, group_patches = [], 0
        nonlocal last_report
        if job.done - last_report >= 2000:
            last_report = job.done
            el = max(_t.time() - t_start, 1e-6)
            st = det.stats
            job.log(f"{job.done}/{job.total}: {3600 * (job.done - n_start) / el:,.0f} Gebäude/h · Grafikkarte "
                    f"{100 * st['gpu_s'] / el:.0f} % der Zeit beschäftigt · {st['segmented']} von {st['patches']} "
                    f"Ausschnitten segmentiert".replace(",", "."))

    while True:
        item = fetched.get()
        if item is SENTINEL:
            break
        job.check_cancel()
        b, search, fp, img, win, src, err = item
        if err:
            c = search.centroid
            erow = dict.fromkeys(keys)
            erow.update(id=b.id, status="fehler", has_pv=0, size_class="", in_range=0, lon=round(c.x, 7),
                        lat=round(c.y, 7), error=err, props=json.dumps(b.props, ensure_ascii=False),
                        footprint_wkt=fp.wkt if fp is not None else None)
            db.execute(insert, tuple(erow[k] for k in keys))
            job.counters["fehler"] += 1
            job.step(1)
            if job.counters["fehler"] <= 20:
                job.log(f"Bild für {b.id} nicht verfügbar: {err[:160]}")
            continue
        if b.props.get("_unchanged"):
            pending.append({"id": b.id, "result": "unchanged"})
            job.counters["unveraendert"] += 1
            job.step(1)
            send()
            continue
        group.append(item)
        n = len(inference.patch_grid(max(img.shape[1], prof.patch_px),
                                     max(img.shape[0], prof.patch_px), prof.patch_px, det.overlap))
        group_patches += n
        if group_patches >= target_patches:
            flush()
    flush()
    if push:
        send(force=True)
        try:
            client.finish()
        except Exception as e:                                   # noqa: BLE001
            job.log(f"PV-Map: Abschluss nicht gemeldet ({str(e)[:120]})")
        job.log(f"An PV-Map gesendet: {job.counters.get('gesendet', 0)} Ergebnisse, davon "
                f"{job.counters.get('unveraendert', 0)} mit unverändertem Luftbild.")
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
                                      "kwp_est": d["kwp_est"], "kwp_low": d.get("kwp_low"),
                                      "kwp_high": d.get("kwp_high"), "layout": d.get("layout"),
                                      "size_class": d["size_class"],
                                      "source": d["source"]}}

    def instfeats():
        if "installations" not in names:
            return
        for r in rows():
            d = dict(zip(names, r))
            for ft in json.loads(d["installations"] or "[]"):
                pr = ft["properties"]
                pr["osm"] = typology.osm_tags(pr)
                pr["mastr"] = typology.mastr_fields(pr)
                yield ft

    write_fc(os.path.join(job.dir, "buildings.geojson"), bfeats())
    compare_with_known(job, db)
    write_fc(os.path.join(job.dir, "pv_polygons.geojson"), pvfeats())
    write_fc(os.path.join(job.dir, "installations.geojson"), instfeats())
    job.log("Exporte geschrieben: buildings.csv, buildings.geojson, pv_polygons.geojson, installations.geojson")


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
