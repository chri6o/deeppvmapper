"""Baseline of known ground-mounted PV (docs/freiflaechen.md, step 2): OSM via Geofabrik extracts + osmium, TZ-SAM via Zenodo
or a file dropped by hand. Duplicates merged; roof systems and candidates overlapping OSM buildings by more than
`ground_max_building_share` are discarded; areas below `ground_min_ha` are skipped. The baseline is training and test set
(spatial hold-out in 10 km blocks).

Overpass is not used: whole regions are too large for it, and it is often unreachable. Geofabrik extracts are updated daily;
only their solar objects and the buildings next to them are kept.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import subprocess

import requests
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .. import geo
from . import store

GEOFABRIK = "https://download.geofabrik.de"
ZENODO_CONCEPT = "11368203"          # TZ-SAM, all versions; newer quarters are only on solar.transitionzero.org (manual download)
UA = "pvmapper/1.0 (+https://github.com/chri6o/deeppvmapper)"
REGIONS = {
    "pl-wielkopolskie": {"cc": "PL", "name": "Wielkopolskie", "geofabrik": "europe/poland/wielkopolskie"},
    "cz": {"cc": "CZ", "name": "Česko", "geofabrik": "europe/czech-republic"},
}
ROOF = {"roof", "rooftop", "roofs"}
FLOAT = {"water", "floating"}


def regions(cfg: dict) -> dict:
    return {**REGIONS, **(cfg.get("ground_regions") or {})}


def workdir(cfg: dict, region: str) -> str:
    d = os.path.join(cfg.get("ground_dir") or os.path.join(cfg.get("jobs_dir") or "jobs", "_ground"), region)
    os.makedirs(d, exist_ok=True)
    return d


# -- helpers -----------------------------------------------------------------------------------------------------------

def parse_power(s) -> float | None:
    """'12 MW', '12,5 MWp', '5000 kW', '800 kWp', '1.2 GW', '750000 W' → kWp; bare numbers and 'yes' are ambiguous → None"""
    if s is None:
        return None
    m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*([kMG]?)(W)p?\s*$", str(s).split(";")[0].strip(), re.I)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v * {"": 1e-3, "K": 1, "M": 1e3, "G": 1e6}[m.group(2).upper()]


def parse_year(s) -> int | None:
    m = re.search(r"(19[5-9][0-9]|20[0-9]{2})", str(s or ""))
    return int(m.group(1)) if m else None


def split_of(geom) -> str:
    """spatial hold-out: 10 km blocks in EPSG:3035, one block in five is test"""
    c = geo.reproject(geom.centroid, "EPSG:4326", "EPSG:3035")
    key = f"{math.floor(c.x / 10000)},{math.floor(c.y / 10000)}"
    return "test" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def building_share(fence, buildings: list, tree: STRtree | None = None) -> float:
    """share of the fence area covered by building footprints (WGS84 geometries; ratio is computed in a metric CRS)"""
    if not buildings:
        return 0.0
    idx = tree.query(fence) if tree is not None else range(len(buildings))
    hit = [buildings[i] for i in idx if buildings[i].intersects(fence)]
    if not hit:
        return 0.0
    inter = fence.intersection(unary_union(hit))
    a = geo.ground_area_m2(fence)
    return geo.ground_area_m2(inter) / a if a > 0 and not inter.is_empty else 0.0


def same_site(a, b) -> bool:
    """one installation in two sources: overlap > 50 % of the smaller and > 25 % of the larger area (a large cluster that
    only touches a small plant is a different site)"""
    if not a.intersects(b):
        return False
    i = a.intersection(b).area
    return i > 0.5 * min(a.area, b.area) and i > 0.25 * max(a.area, b.area)


def overlap_small(a, b) -> float:
    """intersection / area of the smaller geometry"""
    if not a.intersects(b):
        return 0.0
    i = a.intersection(b).area
    return i / max(min(a.area, b.area), 1e-18)


# -- OSM ---------------------------------------------------------------------------------------------------------------

def _is_solar(t: dict) -> bool:
    return (t.get("power") == "plant" and t.get("plant:source") == "solar") or \
           (t.get("power") == "generator" and t.get("generator:source") == "solar")


def _is_roof(t: dict) -> bool:
    return (t.get("location") in ROOF or t.get("generator:place") in ROOF or "building" in t or t.get("rooftop") == "yes"
            or t.get("generator:location") in ROOF)


def _osm_key(p: dict) -> str:
    return f"{(p.get('@type') or 'x')[0]}{p.get('@id')}"


def osm_features(lines) -> list[dict]:
    """geojsonseq lines from `osmium export -a type,id` → solar polygons [{key, tags, geom}] (no roof systems)"""
    out = []
    for ln in lines:
        ln = ln.strip().lstrip("\x1e")
        if not ln:
            continue
        f = json.loads(ln)
        p = f.get("properties") or {}
        if f.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon") or not _is_solar(p) or _is_roof(p):
            continue
        g = shape(f["geometry"])
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        out.append({"key": _osm_key(p), "tags": {k: v for k, v in p.items() if not k.startswith("@")}, "geom": g})
    return out


def building_geoms(lines) -> list:
    out = []
    for ln in lines:
        ln = ln.strip().lstrip("\x1e")
        if not ln:
            continue
        f = json.loads(ln)
        if f.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            g = shape(f["geometry"])
            out.append(g if g.is_valid else g.buffer(0))
    return out


def osm_rows(feats: list[dict], buildings: list, cc: str, region: str, cfg: dict) -> tuple[list[dict], dict]:
    """plants (fence) with their generators (module rows); standalone generator rows clustered into one installation"""
    min_m2 = float(cfg.get("ground_min_ha", 0.5)) * 1e4
    max_share = float(cfg.get("ground_max_building_share", 0.2))
    count = {"osm_objekte": len(feats), "zu_klein": 0, "gebaeude": 0, "dubletten": 0}
    plants = sorted([f for f in feats if f["tags"].get("power") == "plant"], key=lambda f: -f["geom"].area)
    gens = [f for f in feats if f["tags"].get("power") == "generator"]
    # duplicate plants (way and relation, or overlapping mapping): keep the largest, remember the others
    kept: list[dict] = []
    for p in plants:
        dup = next((k for k in kept if overlap_small(k["geom"], p["geom"]) > 0.5), None)
        if dup:
            dup.setdefault("dups", []).append(p["key"])
            count["dubletten"] += 1
        else:
            kept.append(p)
    ptree = STRtree([p["geom"] for p in kept]) if kept else None
    loose = []
    for g in gens:
        hit = None
        if ptree is not None:
            for i in ptree.query(g["geom"].representative_point()):
                if kept[i]["geom"].contains(g["geom"].representative_point()):
                    hit = kept[i]
                    break
        (hit.setdefault("gens", []) if hit else loose).append(g)
    # standalone generator rows: cluster rows closer than 15 m into one installation
    clusters = []
    if loose:
        m15 = 15 / 111320
        blobs = unary_union([g["geom"].buffer(m15) for g in loose])
        for blob in getattr(blobs, "geoms", [blobs]):
            members = [g for g in loose if blob.contains(g["geom"].representative_point())]
            if members:
                members.sort(key=lambda g: -g["geom"].area)
                fence = unary_union([g["geom"] for g in members]).buffer(m15).buffer(-m15)
                clusters.append({"cluster": True, "key": members[0]["key"], "tags": members[0]["tags"], "geom": fence if not fence.is_empty
                                 else unary_union([g["geom"] for g in members]),
                                 "gens": members if len(members) > 1 else [], "dups": [g["key"] for g in members[1:]]})
    btree = STRtree(buildings) if buildings else None
    rows = []
    for p in kept + clusters:
        fence = p["geom"]
        a = geo.ground_area_m2(fence)
        if a < min_m2:
            count["zu_klein"] += 1
            continue
        t = p["tags"]
        q = {"osm": [p["key"]] + p.get("dups", [])}
        mod = None
        if p.get("gens"):
            m = unary_union([g["geom"] for g in p["gens"]])
            if not p.get("cluster"):
                m = m.intersection(fence)
            ma = geo.ground_area_m2(m) if not m.is_empty else 0
            if ma > 0.9 * a:
                q["osm_module_flaechig"] = True       # one generator drawn over the whole plant: no row geometry
            elif ma > 0:
                mod = m
            q["osm_generatoren"] = len(p["gens"])
        kwp = parse_power(t.get("plant:output:electricity") or t.get("generator:output:electricity"))
        if kwp:
            q["osm_leistung_kwp"] = round(kwp, 1)     # stated in OSM (operator/press), reference only — never used to calibrate
        for k in ("name", "operator", "start_date", "ref"):
            if t.get(k):
                q[f"osm_{k.replace(':', '_')}"] = t[k]
        if t.get("agrivoltaics") == "yes" or t.get("agrivoltaic") == "yes":
            q["osm_agri"] = True
        share = building_share(fence, buildings, btree)
        row = {"id": f"ff-{cc.lower()}-osm-{p['key']}", "land": cc, "region": region, "geom_zaun": fence, "geom_module": mod,
               "flaeche_zaun_m2": round(a), "flaeche_module_m2": round(geo.ground_area_m2(mod)) if mod is not None else None,
               "status": "in_betrieb", "baujahr_schaetzung": parse_year(t.get("start_date")),
               "typ": "floating" if (t.get("location") in FLOAT or t.get("floating") == "yes") else "unbekannt",
               "pruefstatus": "kandidat", "herkunft": "bestand_osm", "split": split_of(fence), "quellen": q}
        if share > max_share:
            row.update(pruefstatus="verworfen", verwurf_grund="gebaeude")
            q["gebaeude_anteil"] = round(share, 3)
            count["gebaeude"] += 1
        rows.append(row)
    count["osm_anlagen"] = len(rows) - count["gebaeude"]
    return rows, count


# -- TZ-SAM ------------------------------------------------------------------------------------------------------------

def tzsam_rows(records: list[dict], region_geom, osm: list[dict], buildings: list, cc: str, region: str, cfg: dict,
               version: str) -> tuple[list[dict], dict]:
    """TZ-SAM polygons (CC BY-NC 4.0: training/test, owner only in PV-Map): merged into an OSM row when they overlap it by
    more than half of the smaller area, otherwise a row of their own"""
    min_m2 = float(cfg.get("ground_min_ha", 0.5)) * 1e4
    max_share = float(cfg.get("ground_max_building_share", 0.2))
    count = {"tzsam_objekte": 0, "tzsam_zusammengefuehrt": 0, "tzsam_neu": 0, "zu_klein": 0, "gebaeude": 0}
    otree = STRtree([r["geom_zaun"] for r in osm]) if osm else None
    btree = STRtree(buildings) if buildings else None
    new = []
    for rec in records:
        g = rec["geom"]
        if region_geom is not None and not region_geom.intersects(g.representative_point()):
            continue
        count["tzsam_objekte"] += 1
        info = {"tzsam": [str(rec["cluster_id"])], "tzsam_version": version, "lizenz_nc": True}
        if rec.get("capacity_mw") is not None:
            info["tzsam_kwp_schaetzung"] = round(float(rec["capacity_mw"]) * 1000, 1)   # TZ-SAM's own estimate, not stated
        for k in ("constructed_before", "constructed_after"):
            if rec.get(k):
                info[f"tzsam_{k}"] = str(rec[k])[:10]
        hit, touched = None, []
        if otree is not None:
            for i in otree.query(g):
                if same_site(osm[i]["geom_zaun"], g):
                    hit = osm[i]
                    break
                if osm[i]["geom_zaun"].intersects(g):
                    touched.append(osm[i])
        if hit is not None:
            q = hit["quellen"]
            q["tzsam"] = sorted(set(q.get("tzsam", []) + info.pop("tzsam")))
            q.update(info)
            if hit.get("baujahr_schaetzung") is None and info.get("tzsam_constructed_before"):
                hit["baujahr_schaetzung"] = int(info["tzsam_constructed_before"][:4])
            count["tzsam_zusammengefuehrt"] += 1
            continue
        for t in touched:                       # partial overlap: reference only, no capacity or dates taken over
            t["quellen"]["tzsam_teilweise"] = sorted(set(t["quellen"].get("tzsam_teilweise", []) + info["tzsam"]))
            t["quellen"]["lizenz_nc"] = True
        if touched:
            g = g.difference(unary_union([t["geom_zaun"] for t in touched]))
            g = g if g.is_valid else g.buffer(0)
            info["ohne_osm_teil"] = True
            count["tzsam_teilweise"] = count.get("tzsam_teilweise", 0) + 1
        a = geo.ground_area_m2(g) if not g.is_empty else 0
        if a < min_m2:
            count["zu_klein"] += 1
            continue
        fs = info.get("tzsam_constructed_before")
        row = {"id": f"ff-{cc.lower()}-tz-{rec['cluster_id']}", "land": cc, "region": region, "geom_zaun": g, "geom_module": None,
               "flaeche_zaun_m2": round(a), "status": "in_betrieb", "typ": "unbekannt",
               "first_seen": (fs[:7] + "-01") if fs else None, "baujahr_schaetzung": int(fs[:4]) if fs else None,
               "pruefstatus": "kandidat", "herkunft": "bestand_tzsam", "split": split_of(g), "quellen": info}
        share = building_share(g, buildings, btree)
        if share > max_share:
            row.update(pruefstatus="verworfen", verwurf_grund="gebaeude")
            info["gebaeude_anteil"] = round(share, 3)
            count["gebaeude"] += 1
        else:
            count["tzsam_neu"] += 1
        new.append(row)
    return new, count


def tzsam_file(cfg: dict, region_dir: str, log=print) -> tuple[str | None, str | None]:
    """(path, version): newest file dropped by hand into <ground_dir>/tzsam/ (newer quarters from solar.transitionzero.org),
    else the newest Zenodo version (analysis polygons, GeoPackage)"""
    base = os.path.join(os.path.dirname(region_dir), "tzsam")
    os.makedirs(base, exist_ok=True)
    manual = sorted(glob.glob(os.path.join(base, "*.gpkg")), key=os.path.getmtime)
    manual = [m for m in manual if not os.path.basename(m).startswith("zenodo-")]
    if manual:
        return manual[-1], "datei:" + os.path.basename(manual[-1])
    try:
        r = requests.get(f"https://zenodo.org/api/records/{ZENODO_CONCEPT}/versions/latest",
                         headers={"User-Agent": UA, "Accept": "application/json"}, timeout=40)
        r.raise_for_status()
        d = r.json()
        f = next(x for x in d["files"] if x["key"].endswith("analysis_polygons.gpkg"))
        path = os.path.join(base, f"zenodo-{d['id']}-{f['key']}")
        if not os.path.exists(path):
            log(f"TZ-SAM: lade {f['key']} (Zenodo {d['id']}, {f['size'] / 1e6:.0f} MB) …")
            download(f["links"]["self"], path)
        return path, f"zenodo:{d['id']}"
    except Exception as e:                                            # noqa: BLE001
        log(f"TZ-SAM nicht verfügbar ({e}); weiter nur mit OSM.")
        return None, None


def read_tzsam(path: str, bbox) -> list[dict]:
    from pyogrio.raw import read
    from shapely import wkb
    meta, _, geoms, fields = read(path, bbox=tuple(bbox))
    names = list(meta["fields"])
    out = []
    for i, gb in enumerate(geoms):
        rec = {n: (fields[j][i].item() if hasattr(fields[j][i], "item") else fields[j][i]) for j, n in enumerate(names)}
        g = wkb.loads(bytes(gb))
        rec["geom"] = g if g.is_valid else g.buffer(0)
        out.append(rec)
    return out


# -- download and osmium -----------------------------------------------------------------------------------------------

def download(url: str, path: str, timeout: int = 60):
    """resumable download (Range on <path>.part)"""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    h = {"User-Agent": UA}
    if have:
        h["Range"] = f"bytes={have}-"
    with requests.get(url, headers=h, stream=True, timeout=timeout) as r:
        if r.status_code == 416:
            os.replace(part, path)
            return
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(part, path)


def geofabrik_latest(path: str) -> tuple[str, str]:
    """(dated file URL, version) — the -latest link redirects to the dated file"""
    r = requests.head(f"{GEOFABRIK}/{path}-latest.osm.pbf", headers={"User-Agent": UA}, allow_redirects=False, timeout=30)
    loc = r.headers.get("Location")
    if r.status_code in (301, 302, 307, 308) and loc:
        url = loc if loc.startswith("http") else f"{GEOFABRIK}/{os.path.dirname(path)}/{loc}"
        return url, os.path.basename(url)
    r.raise_for_status()
    return f"{GEOFABRIK}/{path}-latest.osm.pbf", r.headers.get("Last-Modified", "latest")


def read_poly(text: str):
    """Osmosis .poly → (Multi)Polygon; '!'-sections are holes"""
    from shapely.geometry import Polygon
    lines = [ln.strip() for ln in text.splitlines()]
    outer, holes, cur, name = [], [], [], None
    for ln in lines[1:]:
        if not ln:
            continue
        if ln == "END":
            if name is None:
                break
            (holes if name.startswith("!") else outer).append(cur)
            cur, name = [], None
        elif name is None:
            name = ln
        else:
            x, y = ln.split()[:2]
            cur.append((float(x), float(y)))
    polys = [Polygon(o) for o in outer if len(o) >= 3]
    g = unary_union(polys)
    for h in holes:
        if len(h) >= 3:
            g = g.difference(Polygon(h))
    return g


def osmium(*args, log=print):
    cmd = ["osmium", *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"osmium {args[0]}: {p.stderr.strip()[-500:]}")


def extract_osm(pbf: str, d: str, log=print) -> tuple[list[dict], list]:
    """solar objects and the buildings within 50 m of them, via osmium (no whole-region building load)"""
    solar_pbf, solar_js = os.path.join(d, "solar.osm.pbf"), os.path.join(d, "solar.geojsonseq")
    osmium("tags-filter", pbf, "nwr/power=plant,generator", "-o", solar_pbf, "--overwrite")
    osmium("export", solar_pbf, "-f", "geojsonseq", "-a", "type,id", "--geometry-types=polygon", "-o", solar_js, "--overwrite")
    with open(solar_js) as f:
        feats = osm_features(f)
    log(f"OSM: {len(feats)} Solarflächen (ohne Dachanlagen)")
    if not feats:
        return feats, []
    bld = extract_buildings(pbf, d, [x["geom"] for x in feats])
    log(f"OSM: {len(bld)} Gebäude im Umkreis von 50 m")
    return feats, bld


def extract_buildings(pbf: str, d: str, geoms: list) -> list:
    """OSM buildings within 50 m of the given geometries (osmium extract on the zone, then building filter)"""
    if not geoms:
        return []
    os.makedirs(d, exist_ok=True)
    zone = unary_union([g.buffer(50 / 111320) for g in geoms]).simplify(5 / 111320)
    zone_js = os.path.join(d, "zone.geojson")
    with open(zone_js, "w") as f:
        json.dump({"type": "Feature", "properties": {}, "geometry": geo.mapping(zone)}, f)
    near, b_pbf, b_js = (os.path.join(d, n) for n in ("near.osm.pbf", "buildings.osm.pbf", "buildings.geojsonseq"))
    osmium("extract", "-p", zone_js, "-s", "complete_ways", pbf, "-o", near, "--overwrite")
    osmium("tags-filter", near, "wr/building", "-o", b_pbf, "--overwrite")
    osmium("export", b_pbf, "-f", "geojsonseq", "--geometry-types=polygon", "-o", b_js, "--overwrite")
    with open(b_js) as f:
        return building_geoms(f)


# -- job ---------------------------------------------------------------------------------------------------------------

def run(job, cfg: dict):
    """job kind 'ground_inventory': params region (key of ground_regions), force (re-import unchanged inputs)"""
    p = job.params
    key = p.get("region") or cfg.get("ground_pilot") or "pl-wielkopolskie"
    reg = regions(cfg).get(key)
    if not reg:
        raise ValueError(f"Unbekannte Region {key}")
    cc, d = reg["cc"], workdir(cfg, key)
    job.total, job.done = 6, 0
    con = store.connect(cfg)
    store.migrate(con)
    job.log(f"Ausgangsbestand {reg['name']} ({cc}): Mindestgröße {cfg.get('ground_min_ha', 0.5)} ha, "
            f"Gebäudeanteil höchstens {float(cfg.get('ground_max_building_share', 0.2)):.0%}")
    poly_path = os.path.join(d, "region.poly")
    if not os.path.exists(poly_path):
        r = requests.get(f"{GEOFABRIK}/{reg['geofabrik']}.poly", headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        open(poly_path, "w").write(r.text)
    region_geom = read_poly(open(poly_path).read())
    job.step()
    url, ver = geofabrik_latest(reg["geofabrik"])
    pbf = os.path.join(d, ver if ver.endswith(".pbf") else "region.osm.pbf")
    if not os.path.exists(pbf):
        for old in glob.glob(os.path.join(d, "*.osm.pbf")):
            if os.path.basename(old) not in ("solar.osm.pbf", "near.osm.pbf", "buildings.osm.pbf"):
                os.remove(old)
        job.log(f"Lade OSM-Auszug {os.path.basename(url)} (Geofabrik) …")
        download(url, pbf)
    job.step()
    job.check_cancel()
    feats, bld = extract_osm(pbf, d, job.log)
    osm, c1 = osm_rows(feats, bld, cc, key, cfg)
    job.log(f"OSM: {c1['osm_anlagen']} Anlagen ab {cfg.get('ground_min_ha', 0.5)} ha, {c1['dubletten']} Dubletten zusammengeführt, "
            f"{c1['zu_klein']} zu klein, {c1['gebaeude']} verworfen (Gebäude)")
    job.step()
    job.check_cancel()
    tz_path, tz_ver = tzsam_file(cfg, d, job.log)
    tz, c2 = [], {}
    if tz_path:
        recs = read_tzsam(tz_path, region_geom.bounds)
        tz, c2 = tzsam_rows(recs, region_geom, osm, bld, cc, key, cfg, tz_ver)
        job.log(f"TZ-SAM ({tz_ver}): {c2['tzsam_objekte']} in der Region, {c2['tzsam_zusammengefuehrt']} mit OSM zusammengeführt, "
                f"{c2['tzsam_neu']} zusätzlich, {c2['zu_klein']} zu klein, {c2['gebaeude']} verworfen (Gebäude)")
    job.step()
    n = store.upsert(con, osm + tz)
    job.log(f"Gespeichert: {n['neu']} neu, {n['aktualisiert']} aktualisiert, {n['geschuetzt']} von Hand geprüft (nur Quellen ergänzt)")
    store.mark_imported(con, f"osm:{key}", ver, len(osm))
    if tz_ver:
        store.mark_imported(con, f"tzsam:{key}", tz_ver, len(tz))
    job.step()
    rows = osm + tz
    test = sum(1 for r in rows if r["split"] == "test")
    ref = sum(1 for r in rows if r["quellen"].get("osm_leistung_kwp"))
    job.counters.update({**{k: v for k, v in {**c1, **c2}.items() if isinstance(v, int)}, "test": test, "mit_leistung": ref})
    job.log(f"Fertig: {len(rows)} Flächen, davon {test} Testmenge (räumlich), {ref} mit Leistungsangabe in OSM (nur Gegencheck).")
    from . import capacity
    capacity.apply(con, key, cfg, job.log)
    job.step()
