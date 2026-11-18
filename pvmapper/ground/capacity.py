"""Capacity of ground-mounted PV (docs/freiflaechen.md, step 5, first stage).

DC = module area × efficiency(build year) × 1 kW/m² ÷ cos(tilt) where module rows are known (OSM generators now, the fine
analysis later), otherwise fence area × density(type, build year) from ground_density.yml. Always a range and the method.
AC = DC ÷ (1.2 … 1.4). Yield in kWh/kWp from PVGIS (JRC), fixed mounting at the optimal angle, cached per 0.1° cell.
Never calibrated against registers; the capacities stated in OSM are only compared in the log.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time

import requests
import yaml

from .. import typology

CONF_PATH = os.path.join(os.path.dirname(__file__), "ground_density.yml")
UA = "pvmapper/1.0 (+https://github.com/chri6o/deeppvmapper)"


def conf(cfg: dict | None = None) -> dict:
    with open((cfg or {}).get("ground_density_file") or CONF_PATH) as f:
        return yaml.safe_load(f)


def density(typ: str | None, year: int | None, c: dict) -> tuple[float, float]:
    """MWp per fenced hectare (min, max)"""
    tab = c["density_mwp_per_ha"].get(typ or "unbekannt") or c["density_mwp_per_ha"]["unbekannt"]
    pts = sorted((int(y), v) for y, v in tab.items())
    if year is None:
        if len(pts) == 1:                       # type with one value for all years (tracker, agri)
            return tuple(pts[0][1])
        return tuple(c.get("unknown_year") or (min(v[0] for _, v in pts), max(v[1] for _, v in pts)))
    if year <= pts[0][0]:
        return tuple(pts[0][1])
    if year >= pts[-1][0]:
        return tuple(pts[-1][1])
    for (y0, v0), (y1, v1) in zip(pts, pts[1:]):
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            return (v0[0] + t * (v1[0] - v0[0]), v0[1] + t * (v1[1] - v0[1]))
    return tuple(pts[-1][1])


def build_year(row: dict) -> int | None:
    if row.get("baujahr_schaetzung"):
        return int(row["baujahr_schaetzung"])
    fs = row.get("first_seen")
    if fs:
        return int(str(fs)[:4])
    return None


def estimate(row: dict, c: dict, tconf: dict) -> dict:
    """{kwp_min, kwp_max, ac_kw_min, ac_kw_max, methode} for one row (fields as in table freiflaeche)"""
    typ, year = row.get("typ") or "unbekannt", build_year(row)
    tilt = (c.get("tilt_deg") or {}).get(typ, (c.get("tilt_deg") or {}).get("unbekannt"))
    mod = row.get("flaeche_module_m2")
    if mod and tilt:
        eff = typology.efficiency_for_year(year, tconf)
        lo_e, hi_e = (eff, eff) if year else tuple(tconf["module"].get("efficiency_range") or (eff, eff))
        cos_lo, cos_hi = math.cos(math.radians(tilt[0])), math.cos(math.radians(tilt[1]))
        kmin, kmax = mod * lo_e / cos_lo, mod * hi_e / cos_hi
        method = "modulflaeche"
    else:
        d = density(typ, year, c)
        ha = (row.get("flaeche_zaun_m2") or 0) / 1e4
        kmin, kmax = ha * d[0] * 1000, ha * d[1] * 1000
        method = "zaunflaeche" if row.get("herkunft") != "suche" or row.get("geom_module") else "s2_polygon"
    r = c.get("dc_ac_ratio") or [1.2, 1.4]
    return {"kwp_min": round(kmin, 1), "kwp_max": round(kmax, 1), "ac_kw_min": round(kmin / r[1], 1),
            "ac_kw_max": round(kmax / r[0], 1), "methode": method}


# -- PVGIS -------------------------------------------------------------------------------------------------------------

def _cell(lat: float, lon: float, g: float) -> tuple[float, float]:
    return round(math.floor(lat / g) * g + g / 2, 4), round(math.floor(lon / g) * g + g / 2, 4)


def pvgis_yield(con, lat: float, lon: float, c: dict, session=None) -> float | None:
    """kWh/kWp per year, fixed mounting at the optimal angle, centre of the grid cell; cached in table ground_pvgis"""
    p = c.get("pvgis") or {}
    clat, clon = _cell(lat, lon, float(p.get("grid_deg", 0.1)))
    con.execute("CREATE TABLE IF NOT EXISTS ground_pvgis (lat double precision, lon double precision, kwh_kwp real, slope real, src text, "
                "fetched timestamptz DEFAULT now(), PRIMARY KEY (lat, lon))")
    hit = con.execute("SELECT kwh_kwp FROM ground_pvgis WHERE lat = %s AND lon = %s", [clat, clon]).fetchone()
    if hit:
        return hit[0]
    s = session or requests
    try:
        r = s.get(p.get("url", "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc"), timeout=60, headers={"User-Agent": UA},
                  params={"lat": clat, "lon": clon, "peakpower": 1, "loss": p.get("loss_pct", 14), "optimalangles": 1,
                          "outputformat": "json"})
        r.raise_for_status()
        d = r.json()
        ey = float(d["outputs"]["totals"]["fixed"]["E_y"])
        slope = d["inputs"]["mounting_system"]["fixed"]["slope"]["value"]
        src = d["inputs"]["meteo_data"]["radiation_db"]
    except Exception:                                                   # noqa: BLE001
        return None
    con.execute("INSERT INTO ground_pvgis (lat, lon, kwh_kwp, slope, src) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [clat, clon, ey, slope, src])
    time.sleep(0.05)                                                    # PVGIS allows 30 requests/s; stay far below
    return ey


# -- apply -------------------------------------------------------------------------------------------------------------

def apply(con, region: str, cfg: dict | None = None, log=print, session=None) -> dict:
    """recompute capacity and yield for all rows of a region (except discarded); OSM capacities only compared"""
    c, tconf = conf(cfg), typology.settings({})                         # no profile: no register-based calibration
    rows = con.execute("""
      SELECT id, typ, baujahr_schaetzung, first_seen, flaeche_zaun_m2, flaeche_module_m2, herkunft, geom_module IS NOT NULL,
             ST_Y(ST_Centroid(geom_zaun)), ST_X(ST_Centroid(geom_zaun)), quellen
      FROM freiflaeche WHERE region = %s AND pruefstatus <> 'verworfen'""", [region]).fetchall()
    n, cmp = 0, []
    for (fid, typ, bj, fs, zaun, mod, herk, has_mod, lat, lon, q) in rows:
        row = {"typ": typ, "baujahr_schaetzung": bj, "first_seen": fs, "flaeche_zaun_m2": zaun, "flaeche_module_m2": mod,
               "herkunft": herk, "geom_module": has_mod}
        e = estimate(row, c, tconf)
        ey = pvgis_yield(con, lat, lon, c, session)
        con.execute("UPDATE freiflaeche SET kwp_min=%s, kwp_max=%s, ac_kw_min=%s, ac_kw_max=%s, methode=%s, ertrag_kwh_kwp=%s, "
                    "updated_at = now() WHERE id = %s",
                    [e["kwp_min"], e["kwp_max"], e["ac_kw_min"], e["ac_kw_max"], e["methode"], ey, fid])
        n += 1
        ref = (q or {}).get("osm_leistung_kwp")
        if ref:
            cmp.append({"id": fid, "ref": ref, "min": e["kwp_min"], "max": e["kwp_max"], "methode": e["methode"]})
    out = {"anlagen": n, "vergleich": cmp}
    if cmp:
        errs = [((x["min"] + x["max"]) / 2 - x["ref"]) / x["ref"] for x in cmp]
        inside = sum(1 for x in cmp if x["min"] <= x["ref"] <= x["max"])
        out.update(median_fehler=statistics.median(errs), median_abs_fehler=statistics.median(abs(e) for e in errs), in_spanne=inside)
        log(f"Gegencheck mit Leistungsangaben aus OSM ({len(cmp)} Anlagen, nicht zum Kalibrieren): Median-Abweichung der Mitte "
            f"{out['median_fehler']:+.0%}, Median des Betrags {out['median_abs_fehler']:.0%}, {inside} von {len(cmp)} in der Spanne")
    log(f"Leistung berechnet für {n} Flächen.")
    return out


def run(job, cfg: dict):
    """job kind 'ground_capacity': params region"""
    from . import store
    key = job.params.get("region") or cfg.get("ground_pilot") or "pl-wielkopolskie"
    con = store.connect(cfg)
    store.migrate(con)
    res = apply(con, key, cfg, job.log)
    job.counters.update({"anlagen": res["anlagen"], "gegencheck": len(res["vergleich"])})
    with open(os.path.join(job.dir, "gegencheck.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
