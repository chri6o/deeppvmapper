"""Shared runtime: model loading (cached), imagery catalog, chip rendering."""

from __future__ import annotations

import os
import re
import threading

import cv2
import numpy as np
from PIL import Image

from . import geo, inference, models, sources

_lock = threading.Lock()
_cache: dict = {}


def detector(cfg: dict, profile_id: str | None = None) -> inference.Detector:
    pid = profile_id or cfg["profile"]
    key = (pid, os.path.abspath(cfg["model_dir"]), cfg["device"])
    with _lock:
        if key not in _cache:
            _cache.clear()                       # keep only one model pair in VRAM
            if "cuda" in str(models.pick_device(cfg["device"])):
                import torch
                torch.cuda.empty_cache()
            _cache[key] = models.load(pid, cfg["model_dir"], cfg["device"])
        profile, cls, seg, dev = _cache[key]
    return inference.Detector(profile, cls, seg, dev,
                              cls_threshold=cfg.get("cls_threshold"),
                              seg_threshold=cfg.get("seg_threshold"),
                              batch_size=int(cfg.get("batch_size") or 32),
                              overlap=float(cfg.get("patch_overlap", 0.25) or 0))


def catalog(cfg: dict) -> sources.Catalog:
    extra = [cfg["sources_extra"]] if cfg.get("sources_extra") else []
    return sources.Catalog(extra_paths=extra, user_agent=cfg.get("user_agent") or sources.DEFAULT_UA)


def source_filter(cfg: dict, override=None):
    s = override if override is not None else cfg.get("sources", "auto")
    if not s or s == "auto":
        return None
    if isinstance(s, str):
        s = [x.strip() for x in s.split(",") if x.strip()]
    return s


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120]


def geom_to_px(geom_wgs84, win: geo.Window):
    """Exterior rings of a WGS84 (multi)polygon as int pixel arrays for drawing."""
    g = geo.reproject(geom_wgs84, "EPSG:4326", win.crs)
    ulx, xres, _, uly, _, yres = win.geotransform
    polys = getattr(g, "geoms", [g])
    out = []
    for p in polys:
        if p.geom_type != "Polygon":
            continue
        pts = np.array(p.exterior.coords)
        px = np.stack([(pts[:, 0] - ulx) / xres, (pts[:, 1] - uly) / yres], 1)
        out.append(px.round().astype(np.int32))
    return out


def render_chip(img: np.ndarray, mask: np.ndarray | None, win: geo.Window,
                outline=None, max_w: int = 900) -> Image.Image:
    """Side-by-side: original | PV mask overlay (red) with footprint outline (yellow)."""
    ov = img.copy()
    if mask is not None and mask.any():
        m = mask > 0
        ov[m] = (0.45 * ov[m] + np.array([140, 0, 0])).clip(0, 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 40, 40), max(1, img.shape[1] // 400))
    if outline is not None:
        for ring in geom_to_px(outline, win):
            cv2.polylines(ov, [ring], True, (255, 220, 0), max(1, img.shape[1] // 300))
    both = np.concatenate([img, np.full((img.shape[0], 6, 3), 255, np.uint8), ov], 1)
    im = Image.fromarray(both)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    return im
