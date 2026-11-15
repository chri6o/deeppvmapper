"""
Area mode: wall-to-wall detection for a region (bbox or polygon) from WMS/WMTS.

The region is cut into square cells (area_cell_px × resolution, e.g. 2560 px
at 0.1 m = 256 m). Each cell is fetched from the best covering open-data
source, analysed, and its PV polygons stored. At the end polygons are merged
across cell borders, grouped into installations (parts closer than 1 m) and
characterised (area, estimated kWp, size class). Optionally detections are
restricted to / matched with a building layer.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from shapely import wkt as shp_wkt
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from . import buildings as bld
from . import config as cfgmod
from . import geo, inference, postproc, runtime, sources

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS cells (id TEXT PRIMARY KEY, status TEXT, source TEXT, "
    "n_polys INTEGER, max_prob REAL, error TEXT)",
    "CREATE TABLE IF NOT EXISTS polys (cell TEXT, wkt TEXT)",
]

MAX_CELLS = 200_000


def region_geometry(p: dict):
    if p.get("bbox"):
        a, b, c, d = [float(x) for x in p["bbox"]]
        return box(min(a, c), min(b, d), max(a, c), max(b, d))
    g = p.get("geometry")
    if isinstance(g, str):
        g = json.loads(g)
    if g and g.get("type") == "FeatureCollection":
        return unary_union([shape(f["geometry"]) for f in g["features"]])
    if g and g.get("type") == "Feature":
        return shape(g["geometry"])
    if g:
        return shape(g)
    raise ValueError("area job needs 'bbox' or 'geometry'")


def make_cells(region_wgs84, cell_m: float):
    c = region_wgs84.centroid
    utm = f"EPSG:{geo.utm_epsg(c.x, c.y)}"
    r = geo.reproject(region_wgs84, "EPSG:4326", utm)
    minx, miny, maxx, maxy = r.bounds
    nx, ny = math.ceil((maxx - minx) / cell_m), math.ceil((maxy - miny) / cell_m)
    if nx * ny > MAX_CELLS:
        raise ValueError(f"Region zu groß ({nx * ny} Zellen). Bitte kleinere Fläche wählen.")
    cells = []
    for j in range(ny):
        for i in range(nx):
            cb = box(minx + i * cell_m, miny + j * cell_m, minx + (i + 1) * cell_m, miny + (j + 1) * cell_m)
            if cb.intersects(r):
                cells.append((f"{i}_{j}", geo.reproject(cb, utm, "EPSG:4326")))
    return cells, utm


def estimate(p: dict, cfg: dict) -> dict:
    """Quick size estimate for the UI: cells, km², rough imagery volume."""
    from .models import get_profile
    prof = get_profile(p.get("profile") or cfg["profile"], cfg["model_dir"])
    cell_m = int(cfg["area_cell_px"]) * prof.resolution_m
    region = region_geometry(p)
    cells, _ = make_cells(region, cell_m)
    return {"cells": len(cells), "km2": round(geo.ground_area_m2(region) / 1e6, 2),
            "cell_m": cell_m, "gigapixel": round(len(cells) * int(cfg["area_cell_px"]) ** 2 / 1e9, 2)}


def run(job, cfg: dict):
    from .jobs import open_db
    p = job.params
    cat = runtime.catalog(cfg)
    det = runtime.detector(cfg, p.get("profile"))
    prof = det.p
    src_filter = runtime.source_filter(cfg, p.get("sources"))
    cell_px = int(cfg["area_cell_px"])
    cell_px -= cell_px % prof.patch_px or 0
    cell_px = max(cell_px, prof.patch_px)
    cell_m = cell_px * prof.resolution_m

    db = open_db(os.path.join(job.dir, "results.sqlite"))
    for s in SCHEMA:
        db.execute(s)
    region = region_geometry(p)
    cells, utm = make_cells(region, cell_m)
    done = {r[0] for r in db.execute("SELECT id FROM cells WHERE status='ok'")}
    db.execute("DELETE FROM cells WHERE status != 'ok'")
    job.total, job.done = len(cells), len(done)
    job.log(f"Region {geo.ground_area_m2(region) / 1e6:.2f} km² → {len(cells)} Zellen à "
            f"{cell_m:.0f} m, {len(done)} bereits erledigt. Profil {prof.id}, Gerät {det.device}.")
    job.save()

    chip_dir = os.path.join(job.dir, "chips")
    os.makedirs(chip_dir, exist_ok=True)
    chips_mode = p.get("save_chips", cfg.get("save_chips", "positive"))
    q: queue.Queue = queue.Queue(maxsize=int(cfg["fetch_workers"]) * 2)
    SENTINEL = object()

    def fetch(cell):
        cid, g = cell
        try:
            img, win, src = sources.fetch_for_geometry(cat, g, prof.resolution_m, 0.0, src_filter,
                                                       min_px=prof.patch_px, max_blank=0.95)
            return cid, img, win, src, None
        except Exception as e:                                   # noqa: BLE001
            return cid, None, None, None, str(e)[:400]

    def producer():
        try:
            with ThreadPoolExecutor(int(cfg["fetch_workers"])) as pool:
                pending = []
                for cell in cells:
                    if job._cancel.is_set():
                        break
                    if cell[0] in done:
                        continue
                    pending.append(pool.submit(fetch, cell))
                    if len(pending) >= int(cfg["fetch_workers"]) * 2:
                        q.put(pending.pop(0).result())
                for f in pending:
                    if job._cancel.is_set():
                        break
                    q.put(f.result())
        finally:
            q.put(SENTINEL)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        job.check_cancel()
        cid, img, win, src, err = item
        if err:
            db.execute("INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?)", (cid, "fehler", None, 0, None, err))
            job.step(1, fehler=1)
            if job.counters["fehler"] <= 20:
                job.log(f"Zelle {cid}: {err[:160]}")
            continue
        r = det.run([(cid, img)])[cid]
        # no footprints here: only the filters (contrast, dormer rows, rectangles), no colour completion
        mask, _ = postproc.clean(img, r.mask, det.p.resolution_m, None, dict(cfg, postproc_complete=False))
        r.mask = mask
        polys = inference.mask_to_polygons(r.mask, win)
        db.execute("BEGIN")
        db.execute("DELETE FROM polys WHERE cell=?", (cid,))
        db.executemany("INSERT INTO polys VALUES (?,?)", [(cid, pg.wkt) for pg in polys])
        db.execute("INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?)",
                   (cid, "ok", src.id, len(polys), r.max_prob, None))
        db.execute("COMMIT")
        if polys and chips_mode in ("positive", "all") or chips_mode == "all":
            runtime.render_chip(img, r.mask, win).save(os.path.join(chip_dir, f"cell_{cid}.jpg"), quality=80)
        job.step(1, zellen_mit_pv=int(bool(polys)), polygone=len(polys))

    job.check_cancel()
    export(job, cfg, db, utm, region)


def export(job, cfg, db, utm, region):
    from .jobs import open_db  # noqa: F401
    p = job.params
    raw = [geo.reproject(shp_wkt.loads(w), "EPSG:4326", utm) for (w,) in db.execute("SELECT wkt FROM polys")]
    clip = geo.reproject(region, "EPSG:4326", utm)
    merged = unary_union([g.buffer(0) for g in raw]).intersection(clip) if raw else None
    parts = [] if merged is None or merged.is_empty else list(getattr(merged, "geoms", [merged]))
    parts = [g for g in parts if g.geom_type == "Polygon"]
    # installations: parts closer than 1 m belong together
    groups = []
    if parts:
        blobs = unary_union([g.buffer(0.5) for g in parts])
        blobs = list(getattr(blobs, "geoms", [blobs]))
        tree = STRtree(parts)
        for bb in blobs:
            members = [parts[i] for i in tree.query(bb) if parts[i].intersects(bb)]
            if members:
                groups.append(unary_union(members))

    bindex, bitems = None, []
    if p.get("buildings"):
        job.log("Gleiche Erkennungen mit Gebäudelayer ab …")
        bitems = [(b.id, geo.reproject(b.geometry, "EPSG:4326", utm).buffer(1.5))
                  for b in bld.read_buildings(p["buildings"], id_field=p.get("id_field") or None, cfg=cfg)
                  if b.geometry.geom_type in ("Polygon", "MultiPolygon")]
        bindex = STRtree([g for _, g in bitems]) if bitems else None

    feats, rows = [], []
    min_area = float(cfg["min_pv_area_m2"])
    for k, g in enumerate(groups):
        bids = []
        if bindex is not None:
            bids = [bitems[i][0] for i in bindex.query(g) if bitems[i][1].intersects(g)]
            if p.get("only_on_buildings", True) and not bids:
                continue
        area = g.area
        if area < min_area:
            continue
        kwp = round(area * float(cfg["kwp_per_m2"]), 1)
        cw = geo.reproject(g.centroid, utm, "EPSG:4326")
        props = {"installation_id": k, "pv_area_m2": round(area, 1), "kwp_est": kwp,
                 "size_class": cfgmod.size_class(kwp, cfg["size_classes"]),
                 "in_range": int(cfgmod.in_kwp_range(kwp, cfg)),
                 "lon": round(cw.x, 7), "lat": round(cw.y, 7), "building_ids": ";".join(map(str, bids))}
        rows.append(props)
        feats.append({"type": "Feature", "geometry": mapping(geo.reproject(g, utm, "EPSG:4326")),
                      "properties": props})

    with open(os.path.join(job.dir, "installations.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    with open(os.path.join(job.dir, "installations.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["installation_id", "pv_area_m2", "kwp_est", "size_class", "in_range", "lon", "lat",
                "building_ids"]
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    by_class: dict = {}
    for r in rows:
        c = by_class.setdefault(r["size_class"], {"anzahl": 0, "kwp": 0.0})
        c["anzahl"] += 1
        c["kwp"] = round(c["kwp"] + r["kwp_est"], 1)
    job.counters["anlagen"] = len(rows)
    job.counters["kwp_sum"] = round(sum(r["kwp_est"] for r in rows), 1)
    with open(os.path.join(job.dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"anlagen": len(rows), "kwp_sum": job.counters["kwp_sum"], "nach_klasse": by_class},
                  f, ensure_ascii=False, indent=1)
    job.log(f"Fertig: {len(rows)} Anlagen, ≈ {job.counters['kwp_sum']} kWp. "
            "Exporte: installations.geojson, installations.csv, summary.json")
