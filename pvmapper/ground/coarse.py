"""Coarse search for new ground-mounted PV (docs/freiflaechen.md, step 3).

openEO on the Copernicus Data Space runs the Eurac process `eurac_pv_farm_detection` (ESA APEx algorithm catalogue, CC BY 4.0):
Sentinel-2 L2A, cloud mask, weekly means, gap filling, temporal median, pixel classifier (ONNX), 3×3 clean-up → a 0/1 map at
10 m. It was validated up to 20 × 20 km, so the region is cut into tiles of `ground_tile_km`; one batch job per tile and
window, at most two at a time (CDSE limit). Only the GeoTIFF of each tile comes to the PC (a few hundred kB); the imagery
stays in the cloud. Optionally a second band with ESA WorldCover 2021 (class per pixel) for the confidence — not as a hard
mask: forest cleared for a new park or a lake with floating PV are exactly what we look for.

Locally: polygons from the map (OpenCV), at least `ground_min_ha`; > `ground_max_building_share` OSM buildings → dropped;
overlap with a known row → time series of that row; otherwise a new candidate (herkunft 'suche', first_seen = last month of
the window: the panels were there for at least half of it, so the true date is at most ~3 months earlier).

Credentials only from the environment: OPENEO_AUTH_CLIENT_ID / OPENEO_AUTH_CLIENT_SECRET (/ OPENEO_AUTH_PROVIDER_ID), set in
/etc/pvmapper/openeo.env (read by systemd, root only). Without them a stored refresh token of the service user is used.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import time

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .. import geo
from . import inventory, store

BACKEND = "https://openeo.dataspace.copernicus.eu"
UDP_ID = "eurac_pv_farm_detection"
UDP_URL = ("https://raw.githubusercontent.com/ESA-APEx/apex_algorithms/refs/heads/main/algorithm_catalog/eurac/"
           "eurac_pv_farm_detection/openeo_udp/eurac_pv_farm_detection.json")
CREDITS_PER_KM2 = 0.02            # measured 18.11.2026: 8 credits for a 395 km² tile with WorldCover (catalogue: 0.01)
WC_CLASSES = {10: "baum", 20: "strauch", 30: "gras", 40: "acker", 50: "bebaut", 60: "kahl", 70: "schnee", 80: "wasser",
              90: "feucht", 95: "mangrove", 100: "moos"}


# -- tiles and windows -------------------------------------------------------------------------------------------------

def tiles(region_wgs84, km: float = 20) -> list[dict]:
    """grid in EPSG:3035 (aligned to multiples of `km`), tiles that touch the region; bbox in WGS84 with a 100 m margin"""
    r = geo.reproject(region_wgs84, "EPSG:4326", "EPSG:3035")
    s = km * 1000
    x0, y0, x1, y1 = r.bounds
    out = []
    for ix in range(math.floor(x0 / s), math.ceil(x1 / s)):
        for iy in range(math.floor(y0 / s), math.ceil(y1 / s)):
            t = box(ix * s, iy * s, (ix + 1) * s, (iy + 1) * s)
            if not t.intersects(r):
                continue
            w = geo.reproject(t.buffer(100), "EPSG:3035", "EPSG:4326")
            a, b, c, d = w.bounds
            out.append({"id": f"E{ix * int(km)}N{iy * int(km)}", "bbox": [a, b, c, d],
                        "km2": round(t.intersection(r).area / 1e6, 1)})
    return out


def window(month: str, months: int = 3) -> tuple[str, str]:
    """'2026-10' → ['2026-08-01', '2026-11-01'): the last `months` months up to and including `month`"""
    y, m = (int(v) for v in month.split("-"))
    end = dt.date(y + (m == 12), m % 12 + 1, 1)
    sy, sm = y, m - months + 1
    while sm < 1:
        sm += 12
        sy -= 1
    return dt.date(sy, sm, 1).isoformat(), end.isoformat()


def last_month(today: dt.date | None = None) -> str:
    t = today or dt.date.today()
    y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    return f"{y}-{m:02d}"


def skipped(month: str, cfg: dict) -> bool:
    """winter windows (snow, low sun, clouds) give unreliable maps; skipped by default for Dec–Feb"""
    return int(month.split("-")[1]) in set(cfg.get("ground_scan_skip_months", [12, 1, 2]) or [])


# -- GeoTIFF without GDAL ----------------------------------------------------------------------------------------------

def read_geotiff(path: str) -> tuple[np.ndarray, tuple, str]:
    """bands (b, h, w), affine (x0, dx, y0, dy) and CRS ('EPSG:n') of a north-up GeoTIFF as written by openEO (tifffile, no GDAL)"""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        pg = tf.pages[0]
        arr = pg.asarray()
        tags = {t.code: t.value for t in pg.tags.values()}
        if arr.ndim == 3 and pg.planarconfig == 1 and arr.shape[-1] == pg.samplesperpixel:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim == 2 and len(tf.pages) > 1:
            arr = np.stack([q.asarray() for q in tf.pages])
    scale, tie, keys = tags.get(33550), tags.get(33922), tags.get(34735)
    epsg = None
    if keys:
        for i in range(4, len(keys), 4):
            if keys[i] == 3072 and keys[i + 1] == 0:
                epsg = keys[i + 3]
                break
            if keys[i] == 2048 and keys[i + 1] == 0:
                epsg = keys[i + 3]
    x0 = tie[3] - tie[0] * scale[0]
    y0 = tie[4] + tie[1] * scale[1]
    return arr, (x0, scale[0], y0, -scale[1]), f"EPSG:{epsg or 4326}"


def polygons(mask: np.ndarray, aff: tuple, crs: str, min_px: int = 20) -> list:
    """0/1 map → polygons (WGS84), holes kept; tiny blobs dropped before reprojection"""
    import cv2
    m = (mask > 0).astype(np.uint8)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    x0, dx, y0, dy = aff

    def ring(c):
        pts = c.reshape(-1, 2).astype(float)
        return [(x0 + (u + 0.5) * dx, y0 + (v + 0.5) * dy) for u, v in pts]
    out = []
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1 or len(c) < 3:
            continue                                            # holes are handled with their parent
        if cv2.contourArea(c) + len(c) / 2 < min_px:
            continue
        holes = [ring(cs[j]) for j in range(len(cs)) if hier[0][j][3] == i and len(cs[j]) >= 3]
        p = Polygon(ring(c), holes).buffer(abs(dx) / 2, join_style=2)     # pixel centres → pixel edges
        p = p if p.is_valid else p.buffer(0)
        if not p.is_empty:
            out.append(geo.reproject(p, crs, "EPSG:4326"))
    return out


def raster_mask(g_wgs84, shape, aff, crs) -> np.ndarray:
    """polygon (WGS84) → 0/1 mask on the tile grid"""
    import cv2
    m = np.zeros(shape, np.uint8)
    pg = geo.reproject(g_wgs84, "EPSG:4326", crs)
    for part in getattr(pg, "geoms", [pg]):
        if part.is_empty:
            continue
        ext = np.array([((x - aff[0]) / aff[1], (y - aff[2]) / aff[3]) for x, y in part.exterior.coords], np.int32)
        cv2.fillPoly(m, [ext], 1)
        for h in part.interiors:
            cv2.fillPoly(m, [np.array([((x - aff[0]) / aff[1], (y - aff[2]) / aff[3]) for x, y in h.coords], np.int32)], 0)
    return m


def valid_area(pv: np.ndarray, wc, aff, crs, bbox) -> np.ndarray:
    """pixels inside the tile (bbox) and, with the WorldCover band, where it has data: outside, the process marks phantom areas"""
    v = raster_mask(box(*bbox), pv.shape, aff, crs).astype(bool)
    if wc is not None:
        v &= np.isfinite(wc)
    return v


def coverage(con, pv: np.ndarray, valid: np.ndarray, aff, crs, bbox, month: str) -> tuple[list, int]:
    """share of PV pixels inside every known fence in the tile, written to its time series (also 0: seen, not found);
    fences mostly outside the valid area are left out. Returns the masks (for excluding known areas) and the count."""
    from shapely import wkb
    rows = con.execute("SELECT id, ST_AsBinary(geom_zaun), flaeche_zaun_m2 FROM freiflaeche "
                       "WHERE geom_zaun && ST_MakeEnvelope(%s,%s,%s,%s,4326) AND pruefstatus <> 'verworfen'", list(bbox)).fetchall()
    masks, n = [], 0
    px_m2 = abs(aff[1] * aff[3])
    for fid, gb, area in rows:
        g = wkb.loads(bytes(gb))
        m = raster_mask(g, pv.shape, aff, crs).astype(bool)
        masks.append(m)
        inside = m & valid
        if area and inside.sum() * px_m2 < 0.7 * area:
            continue                                            # mostly in a neighbouring tile: counted there
        if not inside.any():
            continue
        cover = round(float((pv[inside] > 0).mean()), 3)
        con.execute("UPDATE freiflaeche SET zeitreihe = coalesce(zeitreihe,'{}'::jsonb) || jsonb_build_object(%s::text, %s::jsonb), "
                    "updated_at = now() WHERE id = %s", [month, json.dumps({"s2": cover}), fid])
        n += 1
    return masks, n


def worldcover_shares(classes: np.ndarray, mask_poly_px: np.ndarray) -> dict:
    v = classes[mask_poly_px > 0]
    v = v[np.isfinite(v)] if v.dtype.kind == "f" else v             # no-data at the tile edge
    if v.size == 0:
        return {}
    u, n = np.unique(v, return_counts=True)
    return {WC_CLASSES.get(int(k), str(int(k))): round(float(c) / v.size, 3) for k, c in zip(u, n) if c / v.size >= 0.05}


# -- openEO ------------------------------------------------------------------------------------------------------------

def connect(cfg: dict, log=print):
    import openeo
    conn = openeo.connect(cfg.get("openeo_backend") or BACKEND)
    if os.environ.get("OPENEO_AUTH_CLIENT_ID") and os.environ.get("OPENEO_AUTH_CLIENT_SECRET"):
        conn.authenticate_oidc_client_credentials(provider_id=os.environ.get("OPENEO_AUTH_PROVIDER_ID") or None)
        log("openEO: angemeldet (Client-Zugang aus der Umgebung)")
    else:
        conn.authenticate_oidc_refresh_token()
        log("openEO: angemeldet (gespeichertes Refresh-Token)")
    return conn


def build(conn, bbox, start: str, end: str, worldcover: bool):
    ext = dict(zip(("west", "south", "east", "north"), bbox))
    pv = conn.datacube_from_process(UDP_ID, namespace=UDP_URL, spatial_extent=ext, temporal_extent=[start, end])
    if not worldcover:
        return pv
    wc = conn.load_collection("ESA_WORLDCOVER_10M_2021_V2", spatial_extent=ext, temporal_extent=["2021-01-01", "2022-01-01"],
                              bands=["MAP"]).reduce_dimension("t", reducer="max")
    pv = pv.add_dimension("bands", "pv", "bands")
    return pv.merge_cubes(wc.resample_cube_spatial(pv, method="near"))


def run_tiles(conn, todo: list[dict], start: str, end: str, d: str, cfg: dict, job, state: dict, save_state):
    """batch jobs, at most `ground_openeo_parallel` at a time; restartable: openEO job ids are kept in state"""
    par = max(1, min(int(cfg.get("ground_openeo_parallel", 2)), 2))
    wc = bool(cfg.get("ground_scan_worldcover", True))
    running: dict[str, object] = {}
    queue = [t for t in todo if not os.path.exists(os.path.join(d, t["id"] + ".tif"))]
    for tid, jid in list(state.get("running", {}).items()):          # resume after a restart
        running[tid] = conn.job(jid)
        queue = [t for t in queue if t["id"] != tid]
    while queue or running:
        job.check_cancel()
        while queue and len(running) < par:
            t = queue.pop(0)
            cube = build(conn, t["bbox"], start, end, wc)
            bj = cube.create_job(title=f"pvmapper {t['id']} {start}..{end}", out_format="GTiff")
            bj.start()
            running[t["id"]] = bj
            state.setdefault("running", {})[t["id"]] = bj.job_id
            save_state()
            time.sleep(6)                                            # 12 requests per minute
        time.sleep(float(cfg.get("ground_openeo_poll_s", 30)))
        for tid, bj in list(running.items()):
            st = bj.status()
            if st == "finished":
                assets = bj.get_results().get_assets()
                tif = next((a for a in assets if a.name.endswith((".tif", ".tiff"))), assets[0])
                tif.download(os.path.join(d, tid + ".tif"))
                try:
                    md = bj.describe()
                    state.setdefault("credits", {})[tid] = (md.get("costs") or md.get("usage", {}).get("credits"))
                except Exception:                                    # noqa: BLE001
                    pass
                running.pop(tid)
                state["running"].pop(tid, None)
                state.setdefault("done", []).append(tid)
                job.step(kacheln=1)
                save_state()
            elif st in ("error", "canceled"):
                try:
                    msg = "; ".join(str(x.get("message"))[:200] for x in bj.logs(level="error")[:2])
                except Exception:                                    # noqa: BLE001
                    msg = ""
                job.log(f"Kachel {tid}: openEO-Auftrag {st} {msg}")
                running.pop(tid)
                state["running"].pop(tid, None)
                state.setdefault("failed", []).append(tid)
                job.step(fehler=1)
                save_state()


# -- matching ----------------------------------------------------------------------------------------------------------

def match(con, polys: list, month: str, cc: str, region: str, cfg: dict, buildings: list, wc_info: list | None = None) -> dict:
    """scan polygons → known rows get a time series point, the rest become candidates"""
    min_m2 = float(cfg.get("ground_min_ha", 0.5)) * 1e4
    max_share = float(cfg.get("ground_max_building_share", 0.2))
    from shapely.strtree import STRtree
    btree = STRtree(buildings) if buildings else None
    n = {"flaechen": 0, "zu_klein": 0, "gebaeude": 0, "bekannt": 0, "neu": 0}
    rows, seen = [], set()
    for k, g in enumerate(polys):
        a = geo.ground_area_m2(g)
        if a < min_m2:
            n["zu_klein"] += 1
            continue
        n["flaechen"] += 1
        if inventory.building_share(g, buildings, btree) > max_share:
            n["gebaeude"] += 1
            continue
        hits = con.execute(
            "SELECT id, herkunft, ST_Area(ST_Intersection(geom_zaun, g)::geography), ST_Area(geom_zaun::geography) "
            "FROM freiflaeche, ST_SetSRID(ST_GeomFromWKB(decode(%s,'hex')),4326) g WHERE geom_zaun && g AND ST_Intersects(geom_zaun, g)",
            [g.wkb_hex]).fetchall()
        known = [h for h in hits if h[2] > 0.3 * min(h[3], a)]
        if known:
            for fid, herk, inter, area in known:
                if herk == "suche" and fid not in seen:          # a candidate seen again
                    seen.add(fid)
                    con.execute("UPDATE freiflaeche SET zeitreihe = coalesce(zeitreihe,'{}'::jsonb) || jsonb_build_object(%s::text, "
                                "%s::jsonb), updated_at = now() WHERE id = %s",
                                [month, json.dumps({"s2": round(min(1.0, inter / area), 3) if area else 1.0}), fid])
            n["bekannt"] += 1
            continue
        c = g.centroid
        rid = f"ff-{cc.lower()}-s2-{month.replace('-', '')}-{abs(round(c.x * 1e4))}-{abs(round(c.y * 1e4))}"
        q = {"s2": month, "s2_flaeche_m2": round(a)}
        if wc_info and wc_info[k]:
            q["worldcover_2021"] = wc_info[k]
        rows.append({"id": rid, "land": cc, "region": region, "geom_zaun": g, "flaeche_zaun_m2": round(a),
                     "first_seen": month + "-01", "status": "in_betrieb", "typ": "floating" if (wc_info and wc_info[k] and
                     wc_info[k].get("wasser", 0) > 0.5) else "unbekannt", "methode": "s2_polygon",
                     "konfidenz": 0.5, "pruefstatus": "kandidat", "herkunft": "suche", "split": inventory.split_of(g),
                     "quellen": q, "zeitreihe": {month: {"s2": 1.0}}})
        n["neu"] += 1
    if rows:
        store.upsert(con, rows)
    return n


def confidence(con, region: str):
    """share of scanned windows since first_seen in which a candidate was seen again"""
    con.execute("""
      UPDATE freiflaeche f SET konfidenz = round((x.hit::numeric / greatest(x.tot, 1)), 2)
      FROM (SELECT id, count(*) FILTER (WHERE (v->>'s2')::numeric > 0.3) hit, count(*) tot
            FROM freiflaeche, jsonb_each(coalesce(zeitreihe,'{}'::jsonb)) e(k, v) WHERE region = %s AND herkunft = 'suche'
            GROUP BY id) x WHERE f.id = x.id""", [region])


# -- job ---------------------------------------------------------------------------------------------------------------

def run(job, cfg: dict):
    """job kind 'ground_scan': params region, month ('YYYY-MM', default last month), max_tiles (test runs), dry_run"""
    p = job.params
    key = p.get("region") or cfg.get("ground_pilot") or "pl-wielkopolskie"
    reg = inventory.regions(cfg)[key]
    cc, base = reg["cc"], inventory.workdir(cfg, key)
    month = p.get("month") or last_month()
    if skipped(month, cfg) and not p.get("force"):
        job.log(f"{month}: Winterfenster, übersprungen (ground_scan_skip_months).")
        return
    start, end = window(month, int(cfg.get("ground_scan_months", 3)))
    poly_path = os.path.join(base, "region.poly")
    if not os.path.exists(poly_path):
        raise RuntimeError("Erst den Ausgangsbestand dieser Region laden (Auftrag ground_inventory).")
    region_geom = inventory.read_poly(open(poly_path).read())
    ts = tiles(region_geom, float(cfg.get("ground_tile_km", 20)))
    if p.get("tiles"):
        ts = [t for t in ts if t["id"] in set(p["tiles"])]
    if p.get("max_tiles"):
        ts = ts[: int(p["max_tiles"])]
    km2 = sum(t["km2"] for t in ts)
    de = lambda x: f"{x:,.0f}".replace(",", ".")                     # noqa: E731
    job.log(f"Grobsuche {reg['name']} {month} (Fenster {start} – {end}): {len(ts)} Kacheln, {de(km2)} km², "
            f"geschätzt {de(km2 * CREDITS_PER_KM2)} openEO-Credits")
    if p.get("dry_run"):
        return
    d = os.path.join(base, "scan", month)
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}

    def save_state():
        json.dump(state, open(sp, "w"))
    job.total = len(ts) * 2
    job.done = len(state.get("done", []))
    conn = connect(cfg, job.log)
    run_tiles(conn, ts, start, end, d, cfg, job, state, save_state)
    con = store.connect(cfg)
    store.migrate(con)
    pbf = next((os.path.join(base, f) for f in sorted(os.listdir(base)) if f.endswith(".osm.pbf")
                and f not in ("solar.osm.pbf", "near.osm.pbf", "buildings.osm.pbf")), None)
    total = {}
    for t in ts:                     # a repeated run of a month replaces that month's untouched candidates in these tiles
        con.execute("DELETE FROM freiflaeche WHERE region = %s AND herkunft = 'suche' AND pruefstatus = 'kandidat' "
                    "AND first_seen = %s AND (coalesce(zeitreihe,'{}'::jsonb) - %s) = '{}'::jsonb "
                    "AND ST_Intersects(geom_zaun, ST_Expand(ST_MakeEnvelope(%s,%s,%s,%s,4326), 0.03))",   # incl. the raster margin
                    [key, month + "-01", month] + list(t["bbox"]))
    for t in ts:
        f = os.path.join(d, t["id"] + ".tif")
        if not os.path.exists(f):
            job.step()
            continue
        arr, aff, crs = read_geotiff(f)
        pv = arr[0] if arr.ndim == 3 else arr
        pv = np.nan_to_num(pv, nan=0.0)
        wc = arr[1] if arr.ndim == 3 and arr.shape[0] > 1 else None
        valid = valid_area(pv, wc, aff, crs, t["bbox"])
        pv = np.where(valid, pv, 0)
        known_masks, n_cov = coverage(con, pv, valid, aff, crs, t["bbox"], month)
        total["anlagen_geprueft"] = total.get("anlagen_geprueft", 0) + n_cov
        rest = pv.copy()
        for m in known_masks:
            rest[m] = 0                                          # known plants are not candidates
        polys = polygons(rest, aff, crs)
        wc_info = None
        if wc is not None and polys:
            wc_info = []
            for g in polys:
                m = raster_mask(g, pv.shape, aff, crs)
                wc_info.append(worldcover_shares(wc, m))
        bld = inventory.extract_buildings(pbf, os.path.join(d, t["id"]), polys) if (pbf and polys) else []
        n = match(con, polys, month, cc, key, cfg, bld, wc_info)
        for k2, v in n.items():
            total[k2] = total.get(k2, 0) + v
        job.step()
    confidence(con, key)
    job.counters.update(total)
    cr = [v for v in (state.get("credits") or {}).values() if isinstance(v, (int, float))]
    job.log(f"Fertig: {total.get('anlagen_geprueft', 0)} bekannte Anlagen geprüft; außerhalb davon {total.get('flaechen', 0)} Flächen "
            f"ab {cfg.get('ground_min_ha', 0.5)} ha, davon {total.get('bekannt', 0)} schon Kandidaten, "
            f"{total.get('neu', 0)} neue Kandidaten, {total.get('gebaeude', 0)} verworfen (Gebäude)"
            + (f"; verbraucht {sum(cr):.0f} Credits" if cr else "") + (f"; {len(state.get('failed', []))} Kacheln fehlgeschlagen"
                                                                     if state.get("failed") else ""))
    recall(con, key, month, job.log)


def recall(con, region: str, month: str, log=print) -> dict:
    """detection rate against the baseline (test split, from 1 ha, in operation before the window)"""
    r = con.execute("""
      SELECT count(*), count(*) FILTER (WHERE (zeitreihe->%s->>'s2')::numeric > 0.3)
      FROM freiflaeche WHERE region = %s AND herkunft <> 'suche' AND pruefstatus <> 'verworfen' AND split = 'test'
        AND flaeche_zaun_m2 >= 10000 AND zeitreihe ? %s""", [month, region, month]).fetchone()
    tot, hit = r
    if tot:
        log(f"Trefferquote Bestand (Testmenge, ab 1 ha, in den gesuchten Kacheln): {hit}/{tot} = {hit / tot:.0%} (Ziel ≥ 90 %; "
            f"erkannt = mehr als 30 % der Fläche als PV markiert)")
    return {"n": tot, "hits": hit}

