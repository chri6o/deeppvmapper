"""
Building footprints from OpenStreetMap (Overpass API) for 'Fläche auswerten' (PVMAPPER.md).

With `osm_buildings` the area job fetches every way/relation tagged building=* in the region once, keeps it in the job
folder (buildings_osm.geojson) and then
  - skips image cells without a building (less imagery, less GPU time),
  - clips the PV mask of every cell to the footprints (+1.5 m), so ground-mounted arrays, carports and fields drop out,
  - assigns every installation to its building (parts on one building that appeared on the same flights = one installation).
Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import json
import math
import time

import requests
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union

ENDPOINTS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
TILE_DEG = 0.04            # ≈ 3–4 km tiles: Overpass answers quickly and stays below its size limits
UA = "PV-Mapper (https://github.com/chri6o/deeppvmapper)"


def query(s: float, w: float, n: float, e: float) -> str:
    return (f"[out:json][timeout:120];(way[\"building\"]({s},{w},{n},{e});"
            f"relation[\"building\"][\"type\"=\"multipolygon\"]({s},{w},{n},{e}););out geom tags;")


def _fetch(q: str, timeout: int = 180) -> dict:
    err = None
    for url in ENDPOINTS:
        for attempt in range(2):
            try:
                r = requests.post(url, data={"data": q}, timeout=timeout, headers={"User-Agent": UA})
                if r.status_code == 200:
                    return r.json()
                err = f"{url}: HTTP {r.status_code}"
                if r.status_code in (429, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                break
            except Exception as ex:                                   # noqa: BLE001
                err = f"{url}: {type(ex).__name__}: {ex}"
    raise RuntimeError(f"OpenStreetMap (Overpass) nicht erreichbar – {err}")


def parse(data: dict) -> list[dict]:
    """Overpass JSON (out geom) → GeoJSON features with id osm:way/123 and the tags that matter for roofs"""
    out = []
    keep = ("building", "name", "roof:shape", "roof:levels", "building:levels", "height", "roof:direction", "roof:orientation")
    for el in data.get("elements") or []:
        g = None
        if el.get("type") == "way" and el.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(pts) >= 4:
                g = Polygon(pts)
        elif el.get("type") == "relation":
            outer = [Polygon([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in el.get("members") or []
                     if m.get("role") == "outer" and m.get("geometry") and len(m["geometry"]) >= 4]
            inner = [Polygon([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in el.get("members") or []
                     if m.get("role") == "inner" and m.get("geometry") and len(m["geometry"]) >= 4]
            if outer:
                g = unary_union(outer)
                if inner:
                    g = g.difference(unary_union(inner))
        if g is None:
            continue
        g = g.buffer(0)
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        tags = el.get("tags") or {}
        out.append({"type": "Feature", "id": f"osm:{el['type']}/{el['id']}", "geometry": mapping(g),
                    "properties": {"id": f"osm:{el['type']}/{el['id']}", **{k: tags[k] for k in keep if k in tags}}})
    return out


def buildings(region_wgs84, log=None) -> list[dict]:
    """all OSM buildings intersecting a region (WGS84), fetched in tiles; duplicates across tiles removed"""
    minx, miny, maxx, maxy = region_wgs84.bounds
    nx, ny = max(1, math.ceil((maxx - minx) / TILE_DEG)), max(1, math.ceil((maxy - miny) / TILE_DEG))
    seen, out = set(), []
    for j in range(ny):
        for i in range(nx):
            t = box(minx + i * TILE_DEG, miny + j * TILE_DEG, min(maxx, minx + (i + 1) * TILE_DEG),
                    min(maxy, miny + (j + 1) * TILE_DEG))
            if not t.intersects(region_wgs84):
                continue
            a, b, c, d = t.bounds
            for f in parse(_fetch(query(b, a, d, c))):
                if f["id"] in seen or not shape(f["geometry"]).intersects(region_wgs84):
                    continue
                seen.add(f["id"])
                out.append(f)
            if log and nx * ny > 1:
                log(f"OSM-Gebäude: Kachel {j * nx + i + 1}/{nx * ny}, bisher {len(out)}")
    return out


def save(feats: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats,
                   "attribution": "© OpenStreetMap contributors, ODbL"}, f, ensure_ascii=False)
