"""Geometry helpers: metric CRS selection, transforms, image windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon, box, shape, mapping
import shapely


def utm_epsg(lon: float, lat: float) -> int:
    """ETRS89/WGS84 UTM zone EPSG code for a WGS84 point (metric, ~true scale)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def reproject(geom, src: str, dst: str):
    if src == dst:
        return geom
    t = transformer(src, dst)
    return shapely.transform(geom, lambda x, y: t.transform(x, y), interleaved=False)


def scale_factor(crs: str, lon: float, lat: float) -> float:
    """CRS units per ground metre at the given location (1.0 for UTM-like CRS)."""
    if crs.upper() in ("EPSG:3857", "EPSG:900913"):
        return 1.0 / math.cos(math.radians(lat))
    return 1.0


@dataclass
class Window:
    """A rectangular image window in a source CRS at a fixed ground resolution."""
    crs: str
    minx: float
    miny: float
    maxx: float
    maxy: float
    width: int
    height: int

    @property
    def geotransform(self):
        xres = (self.maxx - self.minx) / self.width
        yres = (self.maxy - self.miny) / self.height
        return (self.minx, xres, 0.0, self.maxy, 0.0, -yres)

    def pixel_to_crs(self, px: float, py: float):
        ulx, xres, _, uly, _, yres = self.geotransform
        return ulx + px * xres, uly + py * yres


def window_around(geom_wgs84, crs: str, resolution_m: float, buffer_m: float = 0.0,
                  min_px: int = 0, snap_px: int = 1) -> Window:
    """
    Builds a window in `crs` covering a WGS84 geometry plus a buffer, at
    `resolution_m` ground metres per pixel. Width/height are rounded up to a
    multiple of `snap_px` and at least `min_px`, centred on the geometry.
    """
    c = geom_wgs84.centroid
    k = scale_factor(crs, c.x, c.y)
    g = reproject(geom_wgs84, "EPSG:4326", crs)
    minx, miny, maxx, maxy = g.bounds
    pad = buffer_m * k
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    res = resolution_m * k

    def size(extent):
        n = max(min_px, int(math.ceil(extent / res)))
        return int(math.ceil(n / snap_px) * snap_px)

    w, h = size(maxx - minx), size(maxy - miny)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return Window(crs, cx - w * res / 2, cy - h * res / 2, cx + w * res / 2, cy + h * res / 2, w, h)


def ground_area_m2(geom_wgs84) -> float:
    c = geom_wgs84.centroid
    epsg = f"EPSG:{utm_epsg(c.x, c.y)}"
    return reproject(geom_wgs84, "EPSG:4326", epsg).area


def to_polygon(geom):
    if isinstance(geom, dict):
        geom = shape(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    return geom  # points etc. handled by caller


__all__ = ["utm_epsg", "reproject", "scale_factor", "Window", "window_around",
           "ground_area_m2", "box", "shape", "mapping", "to_polygon"]
