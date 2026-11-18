"""Registers for ground-mounted PV (docs/freiflaechen.md, decision 18.11.2026).

- MaStR (DE, Gesamtdatenexport, dl-de/by-2-0): solar units on open land ("Freifläche") with coordinates, gross capacity
  (modules, DC) and net capacity (inverters, AC). Matched to the fences; used to calibrate the density table (MWp/ha) with
  safeguards against wrong entries: unambiguous matches only, plausibility limits, median and quartiles, hold-out test.
- MIOZE (PL, URE register of small installations up to 1 MW, api.ure.gov.pl): place name and AC capacity, no coordinates.
  Place → centre from the OSM extract of the region; used to show where registered capacity has no known fence nearby
  (search priority), never as a capacity of a fence.
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
import statistics
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import requests

from . import inventory

UA = "pvmapper/1.0 (+https://github.com/chri6o/deeppvmapper)"
MASTR_PAGE = "https://www.marktstammdatenregister.de/MaStR/Datendownload"
MIOZE_URL = "https://api.ure.gov.pl/api/MIOZERegistry/XML"
DENSITY_LIMITS = (0.2, 2.0)        # MWp/ha outside → register entry or fence implausible, not used for calibration
YEAR_BINS = [(None, 2011), (2012, 2018), (2019, 2021), (2022, None)]


_FOLD = str.maketrans({"ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ß": "ss"})   # no NFKD decomposition


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", (s or "").translate(_FOLD).lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _num(v):
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


# -- MIOZE (PL) --------------------------------------------------------------------------------------------------------

def parse_mioze(data: bytes, area: str) -> list[dict]:
    """PV installations (RodzajInstalacji PV*) of one voivodeship"""
    out = []
    want = norm(area)
    for _, el in ET.iterparse(io.BytesIO(data)):
        if el.tag != "MIOZERegistry":
            continue
        r = {c.tag: (c.text or "").strip() for c in el}
        el.clear()
        if not r.get("RodzajInstalacji", "").startswith("PV") or norm(r.get("WojewodztwoInstalacji")) != want:
            continue
        mw = _num(r.get("MocEEInstalacji"))
        out.append({"unit_id": f"{r.get('DKN')}-{r.get('IdInstalacji') or 1}", "name": r.get("Nazwa", "").strip(),
                    "ort": r.get("MiejscowoscInstalacji"), "netto_kw": mw * 1000 if mw is not None else None,
                    "start_date": (r.get("DataRozpoczeciaDzialalnosci") or "")[:10] or None,
                    "attrs": {"nr_wpisu": r.get("NrWpisu"), "rodzaj": r.get("RodzajInstalacji")}})
    return out


def places(pbf: str, d: str) -> dict[str, list]:
    """normalised place name → [(lon, lat, place type)] from the OSM extract (cities, towns, villages, hamlets)"""
    js = os.path.join(d, "places.geojsonseq")
    pp = os.path.join(d, "places.osm.pbf")
    inventory.osmium("tags-filter", pbf, "n/place=city,town,village,hamlet,suburb,isolated_dwelling", "-o", pp, "--overwrite")
    inventory.osmium("export", pp, "-f", "geojsonseq", "--geometry-types=point", "-o", js, "--overwrite")
    out: dict[str, list] = {}
    with open(js) as f:
        for ln in f:
            ln = ln.strip().lstrip("\x1e")
            if not ln:
                continue
            ft = json.loads(ln)
            p = ft.get("properties") or {}
            if p.get("name"):
                x, y = ft["geometry"]["coordinates"][:2]
                out.setdefault(norm(p["name"]), []).append((x, y, p.get("place")))
    return out


RANK = {"city": 0, "town": 1, "village": 2, "suburb": 3, "hamlet": 4, "isolated_dwelling": 5}


def locate(units: list[dict], pl: dict[str, list]) -> dict:
    """place centre per unit: unique name → 'ort'; several places of that name → the one of the highest rank if unique,
    else 'mehrdeutig' (no point: better no search hint than a wrong one)"""
    n = {"ort": 0, "mehrdeutig": 0, "fehlt": 0}
    for u in units:
        cand = pl.get(norm(u["ort"])) or []
        if not cand:
            u["geom_kind"], u["lonlat"] = "fehlt", None
        elif len(cand) == 1:
            u["geom_kind"], u["lonlat"] = "ort", cand[0][:2]
        else:
            best = min(RANK.get(c[2], 9) for c in cand)
            top = [c for c in cand if RANK.get(c[2], 9) == best]
            u["geom_kind"], u["lonlat"] = ("ort", top[0][:2]) if len(top) == 1 else ("mehrdeutig", None)
        n[u["geom_kind"]] += 1
    return n


# -- MaStR (DE) --------------------------------------------------------------------------------------------------------

def mastr_url(session=None) -> str:
    s = session or requests
    r = s.get(MASTR_PAGE, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    urls = re.findall(r"https?://[^\"']+/Gesamtdatenexport_\d{8}_[\d.]+\.zip", r.text)
    if not urls:
        raise RuntimeError("MaStR: Link zum Gesamtdatenexport nicht gefunden")
    return max(set(urls), key=lambda u: re.search(r"_(\d{8})_", u).group(1))      # newest date, not the quarterly 'Stichtag' file


def _xml_records(fh, tag: str):
    for _, el in ET.iterparse(fh):
        if el.tag == tag:
            yield {c.tag: (c.text or "").strip() for c in el}
            el.clear()


def catalog(z: zipfile.ZipFile) -> dict[str, str]:
    name = next(n for n in z.namelist() if n.lower().startswith("katalogwerte"))
    with z.open(name) as fh:
        return {r.get("Id"): r.get("Wert", "") for r in _xml_records(fh, "Katalogwert")}


def parse_mastr(zip_path: str, area: str, log=print) -> list[dict]:
    """solar units on open land in one federal state, in operation or temporarily shut down"""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        kat = catalog(z)
        files = sorted(n for n in z.namelist() if re.match(r"EinheitenSolar(_\d+)?\.xml$", n))
        for i, name in enumerate(files):
            with z.open(name) as fh:
                for r in _xml_records(fh, "EinheitSolar"):
                    if kat.get(r.get("Bundesland")) != area:
                        continue
                    lage = kat.get(r.get("ArtDerSolaranlage") or r.get("Lage"), "")      # 852 = Freiflächensolaranlage
                    if "Freifl" not in lage:
                        continue
                    status = kat.get(r.get("EinheitBetriebsstatus"), "")
                    if status not in ("In Betrieb", "Vorübergehend stillgelegt"):
                        continue
                    lon, lat = _num(r.get("Laengengrad")), _num(r.get("Breitengrad"))
                    out.append({"unit_id": r.get("EinheitMastrNummer"), "name": r.get("NameStromerzeugungseinheit"),
                                "ort": r.get("Gemeinde") or r.get("Ort"), "brutto_kw": _num(r.get("Bruttoleistung")),
                                "netto_kw": _num(r.get("Nettonennleistung")),
                                "start_date": (r.get("Inbetriebnahmedatum") or "")[:10] or None,
                                "lonlat": (lon, lat) if lon and lat else None, "geom_kind": "einheit" if lon and lat else "fehlt",
                                "attrs": {k: v for k, v in {"art": lage, "status": status, "module": _num(r.get("AnzahlModule")),
                                                            "wr_kw": _num(r.get("ZugeordneteWirkleistungWechselrichter")),
                                                            "landkreis": r.get("Landkreis"),
                                                            "ausrichtung": kat.get(r.get("Hauptausrichtung")),
                                                            "neigung": kat.get(r.get("HauptausrichtungNeigungswinkel"))}.items() if v}})
            log(f"MaStR: {name} gelesen ({i + 1}/{len(files)}), bisher {len(out)} Freiflächen-Einheiten in {area}")
    return out


# -- store and match ---------------------------------------------------------------------------------------------------

def save(con, source: str, region: str, units: list[dict]) -> int:
    with con.transaction():
        con.execute("DELETE FROM ground_register WHERE source = %s AND region = %s", [source, region])
        for u in units:
            ll = u.get("lonlat")
            con.execute(
                "INSERT INTO ground_register (source, unit_id, region, name, ort, geom, geom_kind, brutto_kw, netto_kw, start_date, attrs) "
                "VALUES (%s,%s,%s,%s,%s, CASE WHEN %s::float8 IS NULL THEN NULL ELSE ST_SetSRID(ST_MakePoint(%s,%s),4326) END, "
                "%s,%s,%s,%s,%s::jsonb) ON CONFLICT (source, unit_id) DO UPDATE SET region = EXCLUDED.region, geom = EXCLUDED.geom, "
                "geom_kind = EXCLUDED.geom_kind, brutto_kw = EXCLUDED.brutto_kw, netto_kw = EXCLUDED.netto_kw, attrs = EXCLUDED.attrs",
                [source, u["unit_id"], region, u.get("name"), u.get("ort"), ll[0] if ll else None, ll[0] if ll else None,
                 ll[1] if ll else None, u.get("geom_kind"), u.get("brutto_kw"), u.get("netto_kw"), u.get("start_date"),
                 json.dumps(u.get("attrs") or {}, ensure_ascii=False)])
    return len(units)


def match_mastr(con, region: str, max_m: float = 100) -> dict:
    """unit → fence: point inside a fence, or the only fence within `max_m`; units near several fences stay unmatched"""
    con.execute("DELETE FROM ground_register_match WHERE source = 'mastr' AND unit_id IN "
                "(SELECT unit_id FROM ground_register WHERE source = 'mastr' AND region = %s)", [region])
    con.execute("""
      INSERT INTO ground_register_match (source, unit_id, fid, dist_m)
      SELECT 'mastr', r.unit_id, (array_agg(f.id ORDER BY d))[1], min(d)
      FROM ground_register r
      JOIN LATERAL (SELECT f.id, ST_Distance(f.geom_zaun::geography, r.geom::geography) d FROM freiflaeche f
                    WHERE f.region = %s AND f.pruefstatus <> 'verworfen' AND ST_DWithin(f.geom_zaun::geography, r.geom::geography, %s)) f ON TRUE
      WHERE r.source = 'mastr' AND r.region = %s AND r.geom IS NOT NULL
      GROUP BY r.unit_id
      HAVING count(*) = 1 OR min(d) = 0 AND count(*) FILTER (WHERE d = 0) = 1""", [region, max_m, region])
    n = con.execute("SELECT count(*), count(DISTINCT fid) FROM ground_register_match m JOIN ground_register r USING (source, unit_id) "
                    "WHERE r.region = %s AND m.source = 'mastr'", [region]).fetchone()
    tot = con.execute("SELECT count(*) FROM ground_register WHERE source = 'mastr' AND region = %s AND geom IS NOT NULL", [region]).fetchone()[0]
    return {"einheiten": tot, "zugeordnet": n[0], "flaechen": n[1]}


def pairs(con, region: str) -> list[dict]:
    """fences with matched MaStR units: gross capacity (DC) summed per fence, first commissioning year"""
    rows = con.execute("""
      SELECT f.id, f.typ, f.flaeche_zaun_m2, f.split, sum(r.brutto_kw), sum(r.netto_kw), min(r.start_date), count(*)
      FROM ground_register_match m JOIN ground_register r USING (source, unit_id) JOIN freiflaeche f ON f.id = m.fid
      WHERE m.source = 'mastr' AND f.region = %s GROUP BY f.id, f.typ, f.flaeche_zaun_m2, f.split""", [region]).fetchall()
    return [{"id": a, "typ": b, "ha": (c or 0) / 1e4, "split": d, "brutto_kw": e, "netto_kw": f_, "year": g.year if g else None,
             "units": h} for a, b, c, d, e, f_, g, h in rows]


def _bin(year):
    for lo, hi in YEAR_BINS:
        if (lo is None or (year or 0) >= lo) and (hi is None or (year or 9999) <= hi):
            return f"{lo or ''}-{hi or ''}"
    return "?"


def calibrate(ps: list[dict], min_n: int = 5) -> dict:
    """density (MWp/ha) per year bin from train pairs: median and quartiles; plausibility limits; evaluated on the test pairs"""
    good = [p for p in ps if p["brutto_kw"] and p["ha"] > 0 and DENSITY_LIMITS[0] <= p["brutto_kw"] / 1000 / p["ha"] <= DENSITY_LIMITS[1]]
    train = [p for p in good if p["split"] != "test"]
    bins: dict[str, list] = {}
    for p in train:
        bins.setdefault(_bin(p["year"]), []).append(p["brutto_kw"] / 1000 / p["ha"])
    table = {}
    for b, v in bins.items():
        if len(v) >= min_n:
            q = statistics.quantiles(v, n=4) if len(v) >= 4 else [min(v), statistics.median(v), max(v)]
            table[b] = {"median": round(statistics.median(v), 3), "range": [round(q[0], 3), round(q[2], 3)], "n": len(v)}
    return {"table": table, "paare": len(ps), "plausibel": len(good), "train": len(train), "verworfen_unplausibel": len(ps) - len(good)}


def lookup(cal: dict, year) -> tuple[float, float] | None:
    hit = (cal or {}).get("table", {}).get(_bin(year))
    return tuple(hit["range"]) if hit else None


def evaluate(ps: list[dict], cal: dict, c: dict, tconf) -> dict:
    """test pairs: median relative error of the range centre and share inside the range, start values vs calibrated"""
    from . import capacity
    test = [p for p in ps if p["split"] == "test" and p["brutto_kw"] and p["ha"] > 0]
    res = {}
    for label, use in (("startwerte", None), ("kalibriert", cal)):
        errs, inside = [], 0
        for p in test:
            d = (lookup(use, p["year"]) if use else None) or capacity.density(p["typ"], p["year"], c)
            lo, hi = p["ha"] * d[0] * 1000, p["ha"] * d[1] * 1000
            errs.append(((lo + hi) / 2 - p["brutto_kw"]) / p["brutto_kw"])
            inside += lo <= p["brutto_kw"] <= hi
        if errs:
            res[label] = {"n": len(errs), "median_fehler": round(statistics.median(errs), 3),
                          "median_abs_fehler": round(statistics.median(abs(e) for e in errs), 3), "in_spanne": inside}
    return res


def gaps(con, region: str, radius_m: float = 3000, min_kw: float = 500) -> list[dict]:
    """MIOZE: registered PV capacity per place vs known/candidate fences within `radius_m` (search priority, PL)"""
    rows = con.execute("""
      SELECT r.ort, ST_X(r.geom), ST_Y(r.geom), count(*), sum(r.netto_kw),
             (SELECT count(*) FROM freiflaeche f WHERE f.region = %s AND f.pruefstatus <> 'verworfen'
                AND ST_DWithin(f.geom_zaun::geography, r.geom::geography, %s)) AS bekannt
      FROM ground_register r WHERE r.source = 'mioze' AND r.region = %s AND r.geom IS NOT NULL AND r.netto_kw >= %s
      GROUP BY r.ort, r.geom ORDER BY 5 DESC""", [region, radius_m, region, min_kw]).fetchall()
    return [{"ort": o, "lon": x, "lat": y, "anlagen": n, "kw_ac": kw, "flaechen_in_der_naehe": b, "luecke": max(0, n - b)}
            for o, x, y, n, kw, b in rows]


# -- job ---------------------------------------------------------------------------------------------------------------

def run(job, cfg: dict):
    """job kind 'ground_register': params region; MaStR → match → calibrate (DE) or MIOZE → places → gaps (PL)"""
    from . import capacity, store
    key = job.params.get("region") or cfg.get("ground_pilot")
    reg = inventory.regions(cfg)[key]
    d = inventory.workdir(cfg, key)
    con = store.connect(cfg)
    store.migrate(con)
    pbf = next((os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".osm.pbf")
                and f not in ("solar.osm.pbf", "near.osm.pbf", "buildings.osm.pbf", "places.osm.pbf")), None)
    out: dict = {"region": key}
    if reg.get("register") == "mioze":
        r = requests.get(MIOZE_URL, headers={"User-Agent": UA}, timeout=180)
        r.raise_for_status()
        units = parse_mioze(r.content, reg["register_area"])
        if not pbf:
            raise RuntimeError("Erst den Ausgangsbestand laden (OSM-Auszug fehlt).")
        loc = locate(units, places(pbf, d))
        save(con, "mioze", key, units)
        g = gaps(con, key)
        out.update(einheiten=len(units), orte=loc, luecken=g)
        job.log(f"MIOZE ({reg['register_area']}): {len(units)} PV-Anlagen, {sum(u['netto_kw'] or 0 for u in units) / 1000:,.0f} MW AC; "
                f"Ort gefunden {loc['ort']}, mehrdeutig {loc['mehrdeutig']}, unbekannt {loc['fehlt']}".replace(",", "."))
        top = [x for x in g if x["luecke"] > 0][:10]
        job.log(f"Orte mit Anlagen ab 0,5 MW, aber ohne bekannte Fläche im Umkreis von 3 km: {sum(1 for x in g if x['luecke'] > 0)}"
                + ("; größte Lücken: " + ", ".join(f"{x['ort']} ({x['luecke']})" for x in top) if top else ""))
    elif reg.get("register") == "mastr":
        base = os.path.join(os.path.dirname(d), "mastr")
        os.makedirs(base, exist_ok=True)
        url = mastr_url()
        zp = os.path.join(base, os.path.basename(url))
        if not os.path.exists(zp):
            for old in glob.glob(os.path.join(base, "Gesamtdatenexport_*.zip")):
                os.remove(old)
            job.log(f"MaStR: lade {os.path.basename(url)} (gut 3 GB) …")
            inventory.download(url, zp, timeout=120)
        units = parse_mastr(zp, reg["register_area"], job.log)
        save(con, "mastr", key, units)
        m = match_mastr(con, key)
        ps = pairs(con, key)
        cal = calibrate(ps)
        ev = evaluate(ps, cal, capacity.conf(cfg), None)
        out.update(einheiten=len(units), zuordnung=m, kalibrierung=cal, test=ev)
        with open(os.path.join(d, "calibration.json"), "w") as f:
            json.dump({**cal, "test": ev, "quelle": os.path.basename(zp)}, f, ensure_ascii=False, indent=1)
        job.log(f"MaStR: {len(units)} Freiflächen-Einheiten in {reg['register_area']}, {m['zugeordnet']} eindeutig {m['flaechen']} Flächen "
                f"zugeordnet; {cal['plausibel']} Paare plausibel ({cal['verworfen_unplausibel']} außerhalb "
                f"{DENSITY_LIMITS[0]}–{DENSITY_LIMITS[1]} MWp/ha verworfen)")
        for b, v in sorted(cal["table"].items()):
            job.log(f"  Baujahre {b}: Median {v['median']:.2f} MWp/ha, Spanne {v['range'][0]:.2f}–{v['range'][1]:.2f} (n={v['n']})")
        for k, v in ev.items():
            job.log(f"  Testmenge {k}: Median-Abweichung {v['median_fehler']:+.0%}, Betrag {v['median_abs_fehler']:.0%}, "
                    f"in Spanne {v['in_spanne']}/{v['n']}")
    else:
        raise ValueError(f"Für {key} ist kein Register hinterlegt.")
    with open(os.path.join(job.dir, "register.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
