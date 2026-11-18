"""
Official 3D building models (LoD2) as roof source – PVMAPPER.md "Dachformen".

NRW: Geobasis NRW, 1 km tiles in CityGML (EPSG:25832), dl-de/zero-2.0, ~10 million buildings with a generalised roof type
(bldg:roofType, AdV code list) and their roof surfaces in 3D. From the surfaces every roof face gets its tilt, azimuth, true
area and plan outline. Tiles are downloaded on demand and kept in a cache (raw GML and a compact parsed JSON).

Used for
  - the roof potential (roofs.py): roof type and the real faces (tilt, azimuth) instead of the gable-roof assumption,
  - the typology: tilt and azimuth of flush installations from the face below them,
  - training the roof-shape model (roofshape.py) with the LoD2 roof types as labels.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import threading
import xml.etree.ElementTree as ET

import requests
from shapely.geometry import Polygon, box, mapping, shape
from shapely.strtree import STRtree

NRW_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/LoD2_32_{e}_{n}_1_NW.gml"
NRW_BBOX = (5.86, 50.32, 9.46, 52.53)
CRS = "EPSG:25832"
UA = "PV-Mapper (https://github.com/chri6o/deeppvmapper)"

# AdV roof types → (OSM roof:shape, roof-shape class of roofshape.py)
ROOF_TYPES = {
    "1000": ("flat", "flat"), "2100": ("skillion", "mono"), "2200": ("skillion", "mono"),
    "3100": ("gabled", "gabled"), "3200": ("hipped", "hipped"), "3300": ("half-hipped", "hipped"),
    "3400": ("mansard", "complex"), "3500": ("pyramidal", "hipped"), "3600": ("cone", "complex"),
    "3700": ("dome", "complex"), "3800": ("sawtooth", "complex"), "3900": ("round", "complex"),
    "4000": ("spire", "complex"), "5000": ("complex", "complex"), "9999": (None, None),
}
NS = {"bldg": "http://www.opengis.net/citygml/building/1.0", "gml": "http://www.opengis.net/gml"}
_MEM: dict = {}
_LOCK = threading.Lock()


def covers(lon: float, lat: float) -> bool:
    a, b, c, d = NRW_BBOX
    return a <= lon <= c and b <= lat <= d


def cache_dir(cfg: dict) -> str:
    d = cfg.get("lod2_dir") or os.path.join(cfg.get("jobs_dir") or ".", "_lod2")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------------
# geometry of one roof face

def _ring(text: str):
    v = [float(x) for x in text.split()]
    return [(v[i], v[i + 1], v[i + 2]) for i in range(0, len(v) - 2, 3)]


def face(points3d):
    """3D ring → tilt (°), azimuth (° from north, downhill direction), true area (m²), plan polygon (2D)"""
    # Newell's method: normal and area of a planar polygon
    nx = ny = nz = 0.0
    n = len(points3d)
    for i in range(n):
        x1, y1, z1 = points3d[i]
        x2, y2, z2 = points3d[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0:
        return None
    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz                    # upward normal
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, nz / norm))))
    az = (math.degrees(math.atan2(nx, ny)) + 360) % 360 if tilt > 1 else None
    plan = Polygon([(x, y) for x, y, _ in points3d]).buffer(0)
    return {"tilt_deg": round(tilt, 1), "azimuth_deg": None if az is None else round(az), "area_m2": round(norm / 2, 2),
            "plan": plan}


# ---------------------------------------------------------------------------------------------------------------------
# tiles

def parse_gml(path: str) -> list[dict]:
    """CityGML LoD2 tile → [{id, roof_type, footprint (2D, EPSG:25832), faces: [{tilt, azimuth, area, plan}]}]"""
    out = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != f"{{{NS['bldg']}}}Building":
            continue
        bid = el.get(f"{{{NS['gml']}}}id")
        rt = el.findtext(".//bldg:roofType", default="", namespaces=NS)
        faces, ground = [], []
        for rs in el.iterfind(".//bldg:RoofSurface", NS):
            for pl in rs.iterfind(".//gml:posList", NS):
                f = face(_ring(pl.text or ""))
                if f and f["area_m2"] >= 0.5:
                    faces.append(f)
        for gs in el.iterfind(".//bldg:GroundSurface", NS):
            for pl in gs.iterfind(".//gml:posList", NS):
                p = Polygon([(x, y) for x, y, _ in _ring(pl.text or "")]).buffer(0)
                if not p.is_empty:
                    ground.append(p)
        if faces or ground:
            from shapely.ops import unary_union
            fp = unary_union(ground) if ground else unary_union([f["plan"] for f in faces])
            out.append({"id": bid, "roof_type": rt, "footprint": mapping(fp),
                        "faces": [{**{k: f[k] for k in ("tilt_deg", "azimuth_deg", "area_m2")}, "plan": mapping(f["plan"])}
                                  for f in faces]})
        el.clear()
    return out


def tile_names(geom_25832) -> list[tuple]:
    minx, miny, maxx, maxy = geom_25832.bounds
    return [(e, n) for e in range(int(minx // 1000), int(maxx // 1000) + 1)
            for n in range(int(miny // 1000), int(maxy // 1000) + 1)]


def load_tile(cfg: dict, e: int, n: int, log=None) -> list[dict]:
    """parsed buildings of one 1 km tile (cache: parsed JSON; else download + parse)"""
    key = (e, n)
    with _LOCK:
        if key in _MEM:
            return _MEM[key]
    d = cache_dir(cfg)
    js = os.path.join(d, f"LoD2_32_{e}_{n}.json.gz")
    if os.path.isfile(js):
        with gzip.open(js, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        gml = os.path.join(d, f"LoD2_32_{e}_{n}_1_NW.gml")
        if not os.path.isfile(gml):
            if log:
                log(f"Lade LoD2-Kachel {e}/{n} (Geobasis NRW) …")
            r = requests.get(NRW_URL.format(e=e, n=n), timeout=180, headers={"User-Agent": UA}, stream=True)
            if r.status_code == 404:
                data = []
                with gzip.open(js, "wt", encoding="utf-8") as f:
                    json.dump(data, f)
                with _LOCK:
                    _MEM[key] = data
                return data
            r.raise_for_status()
            tmp = gml + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, gml)
        data = parse_gml(gml)
        with gzip.open(js, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        if not cfg.get("lod2_keep_gml"):
            os.remove(gml)                                    # the parsed JSON is enough and ~10× smaller
    with _LOCK:
        _MEM[key] = data
        if len(_MEM) > 64:                                    # keep memory bounded on long runs
            _MEM.pop(next(iter(_MEM)))
    return data


def match(geom_wgs84, cfg: dict, log=None, min_share: float = 0.3) -> dict | None:
    """LoD2 buildings that make up a footprint (each overlapping it by ≥ min_share of its own area, or covering ≥ min_share of
    it) → {roof_type, roof_shape, class, ids, faces (plan as shapely in EPSG:25832), share} or None outside NRW / no match"""
    from . import geo
    c = geom_wgs84.centroid
    if not covers(c.x, c.y):
        return None
    g = geo.reproject(geom_wgs84, "EPSG:4326", CRS)
    blds = []
    for e, n in tile_names(g.buffer(5)):
        blds += load_tile(cfg, e, n, log)
    if not blds:
        return None
    fps = [shape(b["footprint"]) for b in blds]
    tree = STRtree(fps)
    hits = []
    for i in tree.query(g.buffer(1)):
        fp = fps[i]
        inter = fp.intersection(g).area
        if fp.area > 0 and (inter / fp.area >= min_share or inter / max(g.area, 1e-6) >= min_share):
            hits.append((inter, blds[i]))
    if not hits:
        return None
    hits.sort(key=lambda h: -h[0])
    main = hits[0][1]
    rt = main["roof_type"]
    faces = []
    for _, b in hits:
        for f in b["faces"]:
            faces.append({**{k: f[k] for k in ("tilt_deg", "azimuth_deg", "area_m2")}, "plan": shape(f["plan"])})
    types = {b["roof_type"] for _, b in hits}
    shape_tag, cls = ROOF_TYPES.get(rt, (None, None))
    if len(types) > 1:
        shape_tag, cls = "complex", "complex"
    return {"roof_type": rt, "roof_shape": shape_tag, "class": cls, "ids": [b["id"] for _, b in hits], "faces": faces,
            "share": round(sum(h[0] for h in hits) / max(g.area, 1e-6), 2), "crs": CRS}


def face_masks(m: dict, win, shape_hw) -> list[tuple]:
    """faces of a match rasterised into an image window: [(face dict, bool mask)]"""
    import numpy as np
    from PIL import Image, ImageDraw
    from . import geo
    out = []
    H, W = shape_hw[:2]
    ulx, xres, _, uly, _, yres = win.geotransform
    for f in m["faces"]:
        p = geo.reproject(f["plan"], m["crs"], win.crs) if win.crs != m["crs"] else f["plan"]
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        for q in getattr(p, "geoms", [p]):
            if q.geom_type != "Polygon" or q.is_empty:
                continue
            d.polygon([((x - ulx) / xres, (y - uly) / yres) for x, y in q.exterior.coords], fill=1)
            for h in q.interiors:
                d.polygon([((x - ulx) / xres, (y - uly) / yres) for x, y in h.coords], fill=0)
        mk = np.asarray(img, bool)
        if mk.any():
            out.append((f, mk))
    return out


def buildings_in(region_wgs84, cfg: dict, log=None) -> list[dict]:
    """all LoD2 buildings in a region (for training samples): [{id, roof_type, class, footprint (WGS84 GeoJSON)}]"""
    from . import geo
    g = geo.reproject(region_wgs84, "EPSG:4326", CRS)
    out = []
    for e, n in tile_names(g):
        for b in load_tile(cfg, e, n, log):
            fp = shape(b["footprint"])
            if not fp.intersects(g):
                continue
            cls = ROOF_TYPES.get(b["roof_type"], (None, None))[1]
            out.append({"id": b["id"], "roof_type": b["roof_type"], "class": cls, "area_m2": round(fp.area, 1),
                        "footprint": mapping(geo.reproject(fp, CRS, "EPSG:4326"))})
    return out


def refine_installations(feats: list, m: dict, conf: dict | None = None) -> int:
    """tilt and azimuth of flush installations from the LoD2 face below them (typology features, WGS84). The kWp scale with
    the true module area: × cos(old tilt) / cos(new tilt). Returns the number of installations changed."""
    from . import geo, typology
    n = 0
    if not m or not m.get("faces"):
        return 0
    for ft in feats:
        p = ft["properties"]
        if p.get("layout") != "flush":
            continue
        o = geo.reproject(shape(ft["geometry"]), "EPSG:4326", m["crs"])
        best, ov = None, 0.0
        for f in m["faces"]:
            a = f["plan"].intersection(o).area
            if a > ov:
                best, ov = f, a
        if best is None or ov < 0.5 * o.area:
            continue
        old = float(p.get("tilt_deg") or 0.0)
        new = float(best["tilt_deg"])
        k = math.cos(math.radians(min(old, 85))) / max(math.cos(math.radians(min(new, 85))), 0.087)
        for key in ("kwp_est", "kwp_low", "kwp_high"):
            if p.get(key) is not None:
                p[key] = round(p[key] * k, 1)
        p.update(tilt_deg=round(new, 1), tilt_class=typology.tilt_class(new),
                 azimuth_deg=best["azimuth_deg"] if new >= 10 else p.get("azimuth_deg"),
                 roof_shape="flat" if new < 10 else (p.get("roof_shape") if p.get("roof_shape") not in (None, "unknown", "flat") else "pitched"),
                 tilt_basis="LoD2")
        p.setdefault("confidence", {})["tilt_deg"] = 0.9
        n += 1
    return n
