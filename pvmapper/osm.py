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
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely.ops import polygonize, unary_union

ENDPOINTS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter")
OSM_API = "https://api.openstreetmap.org/api/0.6/map"
TILE_DEG = 0.04            # ≈ 3–4 km tiles: Overpass answers quickly and stays below its size limits
API_TILE_DEG = 0.01        # fallback: the OSM editing API answers small boxes (≈ 0.7 × 1.1 km) with everything in them
API_MAX_TILES = 400
UA = "PV-Mapper (https://github.com/chri6o/deeppvmapper)"


def query(s: float, w: float, n: float, e: float) -> str:
    return (f"[out:json][timeout:120];(way[\"building\"]({s},{w},{n},{e});"
            f"relation[\"building\"][\"type\"=\"multipolygon\"]({s},{w},{n},{e}););out geom tags;")


_DOWN = {"until": 0.0}      # Overpass failed recently: interactive requests go straight to the OSM API for a while


def _fetch(q: str, timeout: int = 25, endpoints=ENDPOINTS) -> dict:
    err = []
    for url in endpoints:
        try:
            r = requests.post(url, data={"data": q}, timeout=timeout, headers={"User-Agent": UA})
            if r.status_code == 200:
                return r.json()
            err.append(f"{url.split('/')[2]}: HTTP {r.status_code}")
        except Exception as ex:                                       # noqa: BLE001
            err.append(f"{url.split('/')[2]}: {type(ex).__name__}")
    raise RuntimeError("Overpass nicht erreichbar (" + "; ".join(err) + ")")


def _fetch_api(s: float, w: float, n: float, e: float, timeout: int = 60) -> dict:
    """fallback: OSM API 0.6 /map for a small box → the same JSON shape as Overpass 'out geom' (buildings only)"""
    import xml.etree.ElementTree as ET
    r = requests.get(OSM_API, params={"bbox": f"{w},{s},{e},{n}"}, timeout=timeout, headers={"User-Agent": UA})
    if r.status_code != 200:
        raise RuntimeError(f"OSM-API: HTTP {r.status_code} {r.text[:120]}")
    root = ET.fromstring(r.content)
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    ways, out = {}, []
    for w_ in root.iter("way"):
        refs = [nd.get("ref") for nd in w_.iter("nd")]
        if not all(x in nodes for x in refs):
            continue
        geom = [{"lon": nodes[x][0], "lat": nodes[x][1]} for x in refs]
        tags = {t.get("k"): t.get("v") for t in w_.iter("tag")}
        ways[w_.get("id")] = geom
        if "building" in tags:
            out.append({"type": "way", "id": int(w_.get("id")), "tags": tags, "geometry": geom})
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.iter("tag")}
        if "building" not in tags or tags.get("type") != "multipolygon":
            continue
        members = [{"role": m.get("role"), "geometry": ways[m.get("ref")]} for m in rel.iter("member")
                   if m.get("type") == "way" and m.get("ref") in ways]
        if members:
            out.append({"type": "relation", "id": int(rel.get("id")), "tags": tags, "members": members})
    return {"elements": out}


def parse(data: dict) -> list[dict]:
    """Overpass JSON (out geom) → GeoJSON features with id osm:way/123 and the tags that matter for roofs"""
    out = []
    keep = ("building", "name", "roof:shape", "roof:levels", "building:levels", "height", "roof:direction", "roof:orientation")
    for el in data.get("elements") or []:
        g = None
        if el.get("type") == "way" and el.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(pts) >= 4:
                g = Polygon(pts).buffer(0)
        elif el.get("type") == "relation":
            # member ways may be pieces of a ring: assemble the rings with polygonize
            def rings(role):
                lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in el.get("members") or []
                         if m.get("role") == role and m.get("geometry") and len(m["geometry"]) >= 2]
                return [q.buffer(0) for q in polygonize(lines)] if lines else []
            try:
                outer, inner = rings("outer"), rings("inner")
                if outer:
                    g = unary_union(outer)
                    if inner:
                        g = g.difference(unary_union(inner))
            except Exception:                                         # noqa: BLE001 – broken relation: skip it
                g = None
        if g is None:
            continue
        g = g.buffer(0)
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        tags = el.get("tags") or {}
        out.append({"type": "Feature", "id": f"osm:{el['type']}/{el['id']}", "geometry": mapping(g),
                    "properties": {"id": f"osm:{el['type']}/{el['id']}", **{k: tags[k] for k in keep if k in tags}}})
    return out


def _tiles(region, deg):
    minx, miny, maxx, maxy = region.bounds
    nx, ny = max(1, math.ceil((maxx - minx) / deg)), max(1, math.ceil((maxy - miny) / deg))
    out = []
    for j in range(ny):
        for i in range(nx):
            t = box(minx + i * deg, miny + j * deg, min(maxx, minx + (i + 1) * deg), min(maxy, miny + (j + 1) * deg))
            if t.intersects(region):
                out.append(t)
    return out


def buildings(region_wgs84, log=None, quick: bool = False) -> list[dict]:
    """all OSM buildings intersecting a region (WGS84): Overpass in tiles; if Overpass does not answer, the OSM API in small
    tiles (fine for building lists and areas of a few km²). Duplicates across tiles removed."""
    seen, out = set(), []

    def take(feats):
        for f in feats:
            if f["id"] in seen or not shape(f["geometry"]).intersects(region_wgs84):
                continue
            seen.add(f["id"])
            out.append(f)

    tiles = _tiles(region_wgs84, TILE_DEG)
    # interactive (picking on the map): one Overpass server with a short wait, then the OSM API
    kw = {"timeout": 8, "endpoints": ENDPOINTS[:1]} if quick else {}
    try:
        if quick and time.time() < _DOWN["until"]:
            raise RuntimeError("Overpass antwortete zuletzt nicht")
        for k, t in enumerate(tiles):
            a, b, c, d = t.bounds
            take(parse(_fetch(query(b, a, d, c), **kw)))
            if log and len(tiles) > 1:
                log(f"OSM-Gebäude: Kachel {k + 1}/{len(tiles)}, bisher {len(out)}")
        return out
    except RuntimeError as e:
        over_err = str(e)
        _DOWN["until"] = time.time() + 600
    small = _tiles(region_wgs84, API_TILE_DEG)
    if len(small) > API_MAX_TILES:
        raise RuntimeError(f"{over_err}; das Gebiet ist für die OSM-API zu groß ({len(small)} Kacheln) – bitte kleiner wählen")
    if log:
        log(f"{over_err} – lade über die OSM-API ({len(small)} Kacheln).")
    for k, t in enumerate(small):
        a, b, c, d = t.bounds
        try:
            take(parse(_fetch_api(b, a, d, c)))
        except Exception as ex:                                       # noqa: BLE001
            raise RuntimeError(f"OpenStreetMap nicht erreichbar – {over_err}; OSM-API: {type(ex).__name__}: {ex}"[:400])
        if log and len(small) > 1 and (k + 1) % 10 == 0:
            log(f"OSM-Gebäude: Kachel {k + 1}/{len(small)}, bisher {len(out)}")
        if len(small) > 1:
            time.sleep(0.5)                                          # polite to the editing API
    return out


def save(feats: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats,
                   "attribution": "© OpenStreetMap contributors, ODbL"}, f, ensure_ascii=False)
