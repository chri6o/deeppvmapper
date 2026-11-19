"""
Imagery sources: open-data WMS and WMTS/XYZ orthophoto services.

Every source returns an RGB uint8 array (H, W, 3) for a `geo.Window`, i.e. a
rectangle in the source CRS at a fixed ground resolution. Large windows are
split into several requests and mosaicked. Sources are declared in
`sources.yml` (see that file for the field reference).
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
import yaml
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import geo

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "sources.yml")
DEFAULT_UA = "pvmapper/0.1 (+https://github.com/chri6o/deeppvmapper)"


class SourceError(RuntimeError):
    pass


def _session(user_agent: str) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    s.headers["User-Agent"] = user_agent
    return s


def blank_fraction(img: np.ndarray) -> float:
    """Share of pixels that look like 'no data' (pure white, pure black)."""
    if img.size == 0:
        return 1.0
    white = np.all(img >= 250, axis=-1)
    black = np.all(img <= 5, axis=-1)
    return float(np.mean(white | black))


@dataclass
class Source:
    id: str
    name: str
    type: str                      # 'wms' | 'xyz'
    url: str
    country: str = ""
    region: str = ""
    layers: str = ""
    styles: str = ""
    crs: str = "EPSG:3857"
    format: str = "image/jpeg"
    version: str = "1.3.0"
    native_resolution_m: float = 0.2
    bbox: Optional[list] = None    # WGS84 [minlon, minlat, maxlon, maxlat]
    license: str = ""
    attribution: str = ""
    max_px: int = 2048             # max width/height per request
    max_zoom: int = 19             # xyz only
    concurrency: int = 4
    min_interval_s: float = 0.0    # politeness delay between requests
    priority: int = 50             # lower = preferred in 'auto' routing
    enabled: bool = True
    extra_params: dict = field(default_factory=dict)
    notes: str = ""
    alt: list = field(default_factory=list)   # other flights/years of the same area, for the cross-check
    history: dict = field(default_factory=dict)  # older flights for the installation age (pvmapper/age.py)
    nir_layer: str = ""            # layer with the near infrared of the same flight (training data, 19.11.2026)

    def __post_init__(self):
        self._sem = threading.Semaphore(max(1, int(self.concurrency)))
        self._lock = threading.Lock()
        self._last = 0.0
        self._sessions = threading.local()
        self.user_agent = DEFAULT_UA

    # -- plumbing ---------------------------------------------------------

    def _http(self) -> requests.Session:
        s = getattr(self._sessions, "s", None)
        if s is None:
            s = self._sessions.s = _session(self.user_agent)
        return s

    def _get(self, url, params=None, timeout=60) -> bytes:
        with self._sem:
            if self.min_interval_s:
                with self._lock:
                    wait = self._last + self.min_interval_s - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.time()
            r = self._http().get(url, params=params, timeout=timeout)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200 or not ctype.startswith("image"):
            msg = r.text[:300] if "xml" in ctype or "text" in ctype else ctype
            raise SourceError(f"{self.id}: HTTP {r.status_code} {msg}")
        return r.content

    def covers(self, lon: float, lat: float) -> bool:
        if not self.bbox:
            return True
        a, b, c, d = self.bbox
        return a <= lon <= c and b <= lat <= d

    # -- public -------------------------------------------------------------

    def fetch_nir(self, win: geo.Window):
        """near infrared of the window as a 2-D array (None without an infrared layer)"""
        if not self.nir_layer:
            return None
        from dataclasses import replace
        s = replace(self, id=self.id + "-nir", layers=self.nir_layer, nir_layer="", alt=[], history={})
        s.user_agent = self.user_agent
        img = s.fetch(win)
        return img[..., 0] if img.ndim == 3 else img

    def fetch(self, win: geo.Window) -> np.ndarray:
        if self.type == "wms":
            return self._fetch_wms(win)
        if self.type == "xyz":
            return self._fetch_xyz(win)
        raise SourceError(f"unknown source type {self.type}")

    # -- WMS -------------------------------------------------------------------

    def _wms_request(self, minx, miny, maxx, maxy, w, h) -> np.ndarray:
        params = {
            "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": self.version,
            "LAYERS": self.layers, "STYLES": self.styles, "FORMAT": self.format,
            "WIDTH": w, "HEIGHT": h,
        }
        if self.version.startswith("1.3"):
            params["CRS"] = self.crs
            if self.crs.upper() == "EPSG:4326":      # lat/lon axis order in 1.3.0
                params["BBOX"] = f"{miny},{minx},{maxy},{maxx}"
            else:
                params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"
        else:
            params["SRS"] = self.crs
            params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"
        params.update(self.extra_params or {})
        data = self._get(self.url, params)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        return np.asarray(img)

    def _fetch_wms(self, win: geo.Window) -> np.ndarray:
        if win.crs != self.crs:
            raise SourceError(f"window CRS {win.crs} != source CRS {self.crs}")
        out = np.zeros((win.height, win.width, 3), dtype=np.uint8)
        xres = (win.maxx - win.minx) / win.width
        yres = (win.maxy - win.miny) / win.height
        step = self.max_px
        for y0 in range(0, win.height, step):
            for x0 in range(0, win.width, step):
                w = min(step, win.width - x0)
                h = min(step, win.height - y0)
                minx = win.minx + x0 * xres
                maxx = minx + w * xres
                maxy = win.maxy - y0 * yres
                miny = maxy - h * yres
                out[y0:y0 + h, x0:x0 + w] = self._wms_request(minx, miny, maxx, maxy, w, h)
        return out

    # -- XYZ / WMTS (Web Mercator tile pyramid) ---------------------------------

    def _fetch_xyz(self, win: geo.Window) -> np.ndarray:
        if win.crs.upper() != "EPSG:3857":
            raise SourceError("xyz sources need windows in EPSG:3857")
        R = 6378137.0
        origin = math.pi * R
        target_res = (win.maxx - win.minx) / win.width          # 3857 units / px
        z = self.max_zoom
        for zz in range(0, self.max_zoom + 1):
            if 2 * origin / (256 * 2 ** zz) <= target_res:
                z = zz
                break
        tres = 2 * origin / (256 * 2 ** z)
        tx0 = int((win.minx + origin) // (256 * tres))
        tx1 = int((win.maxx + origin) // (256 * tres))
        ty0 = int((origin - win.maxy) // (256 * tres))
        ty1 = int((origin - win.miny) // (256 * tres))
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 400:
            raise SourceError("window too large for xyz fetch; split it")
        mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                url = self.url.format(z=z, x=tx, y=ty)
                try:
                    tile = Image.open(io.BytesIO(self._get(url))).convert("RGB")
                except SourceError:
                    continue                                    # outside coverage
                mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
        mx0 = tx0 * 256 * tres - origin
        my0 = origin - ty0 * 256 * tres
        left = (win.minx - mx0) / tres
        top = (my0 - win.maxy) / tres
        right = (win.maxx - mx0) / tres
        bottom = (my0 - win.miny) / tres
        crop = mosaic.transform((win.width, win.height), Image.EXTENT,
                                (left, top, right, bottom), Image.BILINEAR)
        return np.asarray(crop)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class Catalog:
    def __init__(self, path: str = CATALOG_PATH, extra_paths=(), user_agent: str = DEFAULT_UA):
        self.sources: dict[str, Source] = {}
        for p in (path, *extra_paths):
            if p and os.path.isfile(p):
                self._load(p, user_agent)

    def _load(self, path, ua):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for sid, spec in (data.get("sources") or {}).items():
            src = Source(id=sid, **spec)
            src.user_agent = ua
            self.sources[sid] = src

    def get(self, sid: str) -> Source:
        if sid not in self.sources:
            raise KeyError(f"unknown imagery source '{sid}'")
        return self.sources[sid]

    def enabled(self):
        return [s for s in self.sources.values() if s.enabled]

    def candidates(self, lon: float, lat: float, allowed=None):
        """Sources covering a point, best first. `allowed`: ids or country codes."""
        cands = [s for s in self.enabled() if s.covers(lon, lat)]
        if allowed:
            allowed = set(allowed)
            cands = [s for s in cands if s.id in allowed or s.country in allowed]
        return sorted(cands, key=lambda s: (s.priority, -1 / max(s.native_resolution_m, 1e-3)))

    def as_dicts(self):
        keys = ("id", "name", "country", "region", "type", "native_resolution_m", "license",
                "attribution", "bbox", "priority", "enabled", "notes", "alt", "history", "nir_layer")
        return [{k: getattr(s, k) for k in keys} for s in self.sources.values()]


def fetch_for_geometry(catalog: Catalog, geom_wgs84, resolution_m: float, buffer_m: float,
                       source_ids=None, min_px: int = 0, snap_px: int = 1,
                       max_blank: float = 0.5):
    """
    Fetches imagery for a geometry, trying covering sources in priority order
    until one returns real (non-blank) pixels. `source_ids` may contain source
    ids, country codes, or 'auto'/None for all sources.

    Returns (rgb_array, window, source) or raises SourceError.
    """
    c = geom_wgs84.centroid
    allowed = None if not source_ids or "auto" in source_ids else source_ids
    cands = catalog.candidates(c.x, c.y, allowed)
    if not cands:
        raise SourceError(f"no imagery source covers {c.y:.5f},{c.x:.5f}")
    errors = []
    for src in cands:
        win = geo.window_around(geom_wgs84, src.crs, resolution_m, buffer_m, min_px, snap_px)
        try:
            img = src.fetch(win)
        except Exception as e:                       # network, server, decode
            errors.append(f"{src.id}: {e}")
            continue
        if blank_fraction(img) <= max_blank:
            return img, win, src
        errors.append(f"{src.id}: no coverage (blank image)")
    raise SourceError("; ".join(errors))


def fetch_second(catalog: Catalog, src: Source, win: geo.Window, primary: np.ndarray,
                 max_blank: float = 0.5, min_diff: float = 6.0):
    """Second image of the same window from another flight/year (`alt` of the source), for the cross-check.
    Skips blank answers and images identical to the primary one (the 'current' layer often equals the latest
    year). Returns (rgb, source) or (None, None)."""
    for sid in src.alt or []:
        alt = catalog.sources.get(sid)
        if alt is None or not alt.enabled or alt.crs != win.crs:
            continue
        try:
            img = alt.fetch(win)
        except Exception:                                        # noqa: BLE001
            continue
        if img.shape != primary.shape or blank_fraction(img) > max_blank:
            continue
        if float(np.abs(img.astype(np.int16) - primary.astype(np.int16)).mean()) < min_diff:
            continue
        return img, alt
    return None, None


def image_hash(img: np.ndarray) -> str:
    """fingerprint of an aerial image (16×16 brightness pattern, 64 hex digits): the same flight gives (almost) the same
    fingerprint even if the server encodes the JPEG a little differently; a new flight changes many bits"""
    from PIL import Image
    g = np.asarray(Image.fromarray(img).convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return "".join(f"{int(''.join('1' if b else '0' for b in bits[i:i + 4]), 2):x}" for i in range(0, 256, 4))


def hash_distance(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 256
    return bin(int(a, 16) ^ int(b, 16)).count("1")
