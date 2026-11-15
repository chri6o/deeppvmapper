"""
Building input readers.

Supported inputs (all yield `Building` records in WGS84):
  *.geojson / *.json   FeatureCollection with Polygon/MultiPolygon/Point features
  *.csv                columns `lat`,`lon` (or `latitude`,`longitude`, `y`,`x`)
                       or a `wkt`/`geometry` column; separator auto-detected
  *.gpkg / *.shp / …   any OGR format, via pyogrio or GDAL/osgeo if installed
  pvmap:cc=DE&sub=…    PV-Map connector (see pvmap.py): footprints, building type and the PV
                       state known from registers/layers come along as properties
  postgresql://…       PostGIS; pass an SQL query returning id + geometry
                       (needs `psycopg`), e.g.
                       SELECT osm_id AS id, ST_AsText(ST_Transform(way,4326)) AS wkt,
                              name FROM planet_osm_polygon WHERE amenity='school'

The id column is taken from `id_field` (default: first of id, osm_id, fid,
gml_id, objectid) or the running index.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

from shapely import wkt as shp_wkt, wkb as shp_wkb
from shapely.geometry import shape, Point

ID_CANDIDATES = ("id", "osm_id", "fid", "gml_id", "objectid", "building_id", "uuid")


@dataclass
class Building:
    id: str
    geometry: object                 # shapely geometry, WGS84
    props: dict = field(default_factory=dict)

    @property
    def is_point(self) -> bool:
        return self.geometry.geom_type == "Point"


def _pick_id(props: dict, id_field: Optional[str], idx: int) -> str:
    if id_field and props.get(id_field) not in (None, ""):
        return str(props[id_field])
    for k in ID_CANDIDATES:
        for kk in (k, k.upper()):
            if props.get(kk) not in (None, ""):
                return str(props[kk])
    return str(idx)


def _clean(props: dict) -> dict:
    out = {}
    for k, v in props.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _read_geojson(path, id_field):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    for i, ft in enumerate(feats):
        if not ft.get("geometry"):
            continue
        props = ft.get("properties") or {}
        if "id" in ft and "id" not in props:
            props = {"id": ft["id"], **props}
        yield Building(_pick_id(props, id_field, i), shape(ft["geometry"]), _clean(props))


def _read_csv(path, id_field):
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        cols = {c.lower(): c for c in reader.fieldnames or []}

        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        clat, clon = col("lat", "latitude", "y"), col("lon", "lng", "longitude", "x")
        cwkt = col("wkt", "geometry", "geom", "the_geom")
        if not (clat and clon) and not cwkt:
            raise ValueError("CSV needs lat/lon columns or a wkt/geometry column")
        for i, row in enumerate(reader):
            try:
                if cwkt and row.get(cwkt):
                    g = shp_wkt.loads(row[cwkt])
                else:
                    g = Point(float(str(row[clon]).replace(",", ".")),
                              float(str(row[clat]).replace(",", ".")))
            except Exception:
                continue
            props = {k: v for k, v in row.items() if k not in (cwkt,)}
            yield Building(_pick_id(props, id_field, i), g, _clean(props))


def _read_ogr(path, id_field, layer=None):
    try:
        import pyogrio
        gdf = pyogrio.read_dataframe(path, layer=layer)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        for i, row in enumerate(gdf.itertuples(index=False)):
            d = row._asdict()
            g = d.pop("geometry")
            if g is None:
                continue
            yield Building(_pick_id(d, id_field, i), g, _clean(d))
        return
    except ImportError:
        pass
    from osgeo import ogr, osr
    ds = ogr.Open(path)
    if ds is None:
        raise ValueError(f"cannot open {path}")
    lyr = ds.GetLayerByName(layer) if layer else ds.GetLayer(0)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = lyr.GetSpatialRef()
    ct = osr.CoordinateTransformation(src, target) if src and not src.IsSame(target) else None
    for i, ft in enumerate(lyr):
        g = ft.GetGeometryRef()
        if g is None:
            continue
        g = g.Clone()
        if ct:
            g.Transform(ct)
        props = ft.items()
        props["fid"] = ft.GetFID()
        yield Building(_pick_id(props, id_field, i), shp_wkb.loads(bytes(g.ExportToWkb())),
                       _clean(props))


def _read_postgis(dsn, query, id_field):
    import psycopg
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(query)
        names = [d.name for d in cur.description]
        gcol = next((n for n in names if n.lower() in ("wkt", "geom", "geometry", "way")), None)
        if not gcol:
            raise ValueError("query must return a geometry column named wkt/geom/geometry")
        for i, row in enumerate(cur):
            d = dict(zip(names, row))
            g = d.pop(gcol)
            if g is None:
                continue
            if isinstance(g, (bytes, memoryview)):
                geom = shp_wkb.loads(bytes(g))
            elif isinstance(g, str) and all(c in "0123456789abcdefABCDEF" for c in g[:16]):
                geom = shp_wkb.loads(g, hex=True)
            else:
                geom = shp_wkt.loads(g)
            yield Building(_pick_id(d, id_field, i), geom, _clean(d))


def _read_pvmap(spec, cfg):
    from . import pvmap
    client = pvmap.Client.from_config(cfg)
    for ft in client.buildings(**pvmap.parse_input(spec)):
        props = dict(ft.get("properties") or {})
        if not ft.get("geometry"):
            continue
        yield Building(str(ft.get("id") or props.get("id")), shape(ft["geometry"]), _clean(props))


def read_buildings(path_or_dsn: str, id_field: Optional[str] = None, query: Optional[str] = None,
                   layer: Optional[str] = None, cfg: Optional[dict] = None) -> Iterator[Building]:
    s = str(path_or_dsn)
    if s.startswith("pvmap:"):
        yield from _read_pvmap(s, cfg)
        return
    if s.startswith(("postgresql://", "postgres://")):
        if not query:
            raise ValueError("PostGIS input needs a SQL query")
        yield from _read_postgis(s, query, id_field)
        return
    ext = os.path.splitext(s)[1].lower()
    if ext in (".geojson", ".json"):
        yield from _read_geojson(s, id_field)
    elif ext in (".csv", ".txt", ".tsv"):
        yield from _read_csv(s, id_field)
    else:
        yield from _read_ogr(s, id_field, layer)


def count_buildings(path_or_dsn, **kw) -> int:
    s = str(path_or_dsn)
    if s.startswith("pvmap:"):
        from . import pvmap
        return int(pvmap.Client.from_config(kw.get("cfg")).count(**pvmap.parse_input(s))["total"])
    return sum(1 for _ in read_buildings(path_or_dsn, **kw))
