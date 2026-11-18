"""PostGIS store for ground-mounted PV (table freiflaeche, schema.sql)."""

from __future__ import annotations

import json
import os

from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon, mapping

FIELDS = ("land", "region", "geom_zaun", "geom_module", "flaeche_zaun_m2", "flaeche_module_m2", "first_seen", "status", "typ",
          "baujahr_schaetzung", "kwp_min", "kwp_max", "ac_kw_min", "ac_kw_max", "ertrag_kwh_kwp", "methode", "orthofoto_quelle",
          "orthofoto_datum", "orthofoto_lizenz", "orthofoto_nennung", "wartet_auf", "konfidenz", "pruefstatus", "verwurf_grund",
          "herkunft", "split", "quellen", "zeitreihe")
GEOMS = ("geom_zaun", "geom_module")
JSONS = ("quellen", "zeitreihe")
# set by hand in the review list: an import never overwrites these rows' review fields
MANUAL = ("manuell_geprueft", "gegenprobe_bestaetigt")


def dsn(cfg: dict) -> str:
    return os.environ.get("PVMAPPER_GROUND_DSN") or cfg.get("ground_db") or "postgresql:///pvmapper_ground"


def connect(cfg: dict):
    import psycopg
    con = psycopg.connect(dsn(cfg), autocommit=True)
    return con


def migrate(con):
    with open(os.path.join(os.path.dirname(__file__), "schema.sql")) as f:
        con.execute(f.read())


def multi(g):
    if g is None or g.is_empty:
        return None
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, MultiPolygon):
        return g
    if hasattr(g, "geoms"):
        ps = [p for p in g.geoms if isinstance(p, Polygon)] + [q for p in g.geoms if isinstance(p, MultiPolygon) for q in p.geoms]
        return MultiPolygon(ps) if ps else None
    return None


def _val(k, v):
    if v is None:
        return None
    if k in GEOMS:
        m = multi(v)
        return m.wkb_hex if m is not None else None
    if k in JSONS:
        return json.dumps(v, ensure_ascii=False)
    return v


def upsert(con, rows: list[dict]) -> dict:
    """insert or update; rows reviewed by hand keep their review status, type and geometries (only `quellen` is refreshed);
    a row discarded by hand (verwurf_grund 'manuell:<reason>') stays discarded"""
    n = {"neu": 0, "aktualisiert": 0, "geschuetzt": 0}
    cols = ("id",) + FIELDS
    ph = ", ".join(f"ST_SetSRID(ST_GeomFromWKB(decode(%s,'hex')),4326)" if c in GEOMS else
                   ("%s::jsonb" if c in JSONS else "%s") for c in cols)
    upd = ", ".join(("typ = CASE WHEN freiflaeche.historie @> '[{\"aktion\": \"type\"}]'::jsonb THEN freiflaeche.typ ELSE EXCLUDED.typ END"
                     if c == "typ" else f"{c} = EXCLUDED.{c}") for c in FIELDS)   # a type set by hand survives re-imports
    sql = (f"INSERT INTO freiflaeche ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT (id) DO UPDATE SET {upd}, updated_at = now() "
           f"WHERE freiflaeche.pruefstatus NOT IN ('manuell_geprueft','gegenprobe_bestaetigt') "
           f"AND coalesce(freiflaeche.verwurf_grund,'') NOT LIKE 'manuell%%' RETURNING (xmax = 0)")
    prot = ("UPDATE freiflaeche SET quellen = freiflaeche.quellen || %s::jsonb, updated_at = now() WHERE id = %s")
    with con.transaction():
        for r in rows:
            cur = con.execute(sql, [r["id"]] + [_val(c, r.get(c)) for c in FIELDS])
            res = cur.fetchone()
            if res is None:
                con.execute(prot, [json.dumps(r.get("quellen") or {}, ensure_ascii=False), r["id"]])
                n["geschuetzt"] += 1
            elif res[0]:
                n["neu"] += 1
            else:
                n["aktualisiert"] += 1
    return n


def imported(con, source: str, version: str) -> bool:
    return con.execute("SELECT 1 FROM ground_import WHERE source=%s AND version=%s", [source, version]).fetchone() is not None


def mark_imported(con, source: str, version: str, rows: int):
    con.execute("INSERT INTO ground_import (source, version, rows) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING", [source, version, rows])


def features(con, where: str = "TRUE", args=(), limit: int = 20000) -> dict:
    """GeoJSON FeatureCollection (fence polygons, all fields as properties)"""
    rows = con.execute(
        f"SELECT id, ST_AsBinary(geom_zaun), ST_AsBinary(geom_module), {', '.join(c for c in FIELDS if c not in GEOMS)}, historie "
        f"FROM freiflaeche WHERE {where} ORDER BY id LIMIT %s", list(args) + [limit]).fetchall()
    names = [c for c in FIELDS if c not in GEOMS] + ["historie"]
    out = []
    for r in rows:
        p = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in zip(names, r[3:])}
        p["id"] = r[0]
        p["has_module"] = r[2] is not None
        out.append({"type": "Feature", "id": r[0], "geometry": mapping(wkb.loads(bytes(r[1]))), "properties": p})
    return {"type": "FeatureCollection", "features": out}


def stats(con, region: str | None = None) -> dict:
    w, a = ("region = %s", [region]) if region else ("TRUE", [])
    rows = con.execute(f"SELECT herkunft, pruefstatus, count(*), round(sum(flaeche_zaun_m2)/1e4) FROM freiflaeche WHERE {w} "
                       f"GROUP BY 1, 2 ORDER BY 1, 2", a).fetchall()
    return {"rows": [{"herkunft": h, "pruefstatus": p, "n": n, "ha": float(ha or 0)} for h, p, n, ha in rows]}


TYPES = ("fest_sued", "fest_ost_west", "tracker", "agri_vertikal", "floating", "unbekannt")
REASONS = ("keine_pv", "gewaechshaus", "baustelle", "dach", "doppelt", "sonstiges")


def review(con, fid: str, action: str, reason: str | None = None, typ: str | None = None, note: str | None = None) -> dict:
    """decision by hand: confirm | discard (reason) | type (typ) | reset → back to 'kandidat'. Logged in `historie`."""
    row = con.execute("SELECT pruefstatus, typ, verwurf_grund FROM freiflaeche WHERE id = %s", [fid]).fetchone()
    if row is None:
        raise KeyError(fid)
    old = {"pruefstatus": row[0], "typ": row[1], "verwurf_grund": row[2]}
    if action == "confirm":
        new = {"pruefstatus": "manuell_geprueft", "verwurf_grund": None}
    elif action == "discard":
        if reason not in REASONS:
            raise ValueError("Grund fehlt")
        new = {"pruefstatus": "verworfen", "verwurf_grund": f"manuell:{reason}"}
    elif action == "type":
        if typ not in TYPES:
            raise ValueError("unbekannter Typ")
        new = {"typ": typ}
    elif action == "reset":
        new = {"pruefstatus": "kandidat", "verwurf_grund": None}
    else:
        raise ValueError("unbekannte Aktion")
    entry = {"zeit": __import__("datetime").datetime.now().isoformat(timespec="seconds"), "aktion": action, "von": "manuell",
             "alt": {k: old[k] for k in new}, "neu": new}
    if note:
        entry["notiz"] = note[:500]
    sets = ", ".join(f"{k} = %s" for k in new)
    con.execute(f"UPDATE freiflaeche SET {sets}, historie = historie || %s::jsonb, updated_at = now() WHERE id = %s",
                list(new.values()) + [json.dumps([entry], ensure_ascii=False), fid])
    return {**old, **new}
