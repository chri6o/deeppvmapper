"""
Training dataset and masks (annotation in the web UI, AI suggestions).

A dataset is the `dataset/` folder of a training job (or of a "prepare only" job). Per building:
  <name>.jpg        aerial image at the model resolution (footprint + buffer)
  <name>.fp.png     footprint mask
  <name>.mask.png   PV mask (optional) — from a PV-Map layer, drawn by a user, or suggested by AI
  <name>.neg.png    "no PV" mask (optional) — confusers marked by a user (dormers, skylights, solar thermal …);
                    training weights these pixels higher and uses them as hard negative patches
  <name>.cls.json   polygons per roof element class (optional, docs/maskieren.md): {"solarthermie": [...], "gaube": [...], …}
  <name>.cls.png    label map of all classes (0 background, index + 1 of CLASS_KEYS; PV first, later classes on top)
  <name>.nir.png    near infrared of the same window (optional, where the source has an infrared layer)
  <name>.json       meta: id, label, label_kind, sub, size_bin, pv_kwp, has_mask, mask_source,
                    status (open | done | skip), label_override, ai {provider, model, reviewed, …}

Polygons exchanged with the UI are lists of [x, y] pixel coordinates of the image.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

MASK_SOURCES = ("pvmap", "user", "ai", "model")
# roof element classes of the mask editor (docs/maskieren.md); "alt" = 'no PV' areas from before the classes (19.11.2026)
CLASS_KEYS = ("pv", "solarthermie", "gaube", "fenster", "technik", "dunkel", "schatten", "gruen", "alt")
# cause of a false hit in the verdict bar (review.CAUSES) → class of its areas
CAUSE_CLASS = {"gaube": "gaube", "schatten": "schatten", "dachfenster": "fenster", "solarthermie": "solarthermie",
               "dach_dunkel": "dunkel", "sonstiges": "alt"}


def class_polygons(d, name) -> dict:
    p = os.path.join(d, name + ".cls.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if k in CLASS_KEYS and k != "pv" and v}


def _write_classes(d, name, classes: dict, pv_polys=None) -> dict:
    """polygons per class → <name>.cls.json and the label map <name>.cls.png; returns {class: number of polygons}"""
    classes = {k: v for k, v in (classes or {}).items() if k in CLASS_KEYS and k != "pv" and v}
    jp, mp = os.path.join(d, name + ".cls.json"), os.path.join(d, name + ".cls.png")
    if not classes:
        for q in (jp, mp):
            if os.path.isfile(q):
                os.remove(q)
        return {}
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(classes, f)
    w, h = image_size(d, name)
    lab = np.zeros((h, w), np.uint8)
    if pv_polys is None:
        pv_polys = current_polygons(d, name)
    for i, k in enumerate(CLASS_KEYS):
        polys = pv_polys if k == "pv" else classes.get(k)
        if polys:
            lab[polygons_to_mask(polys, w, h) > 0] = i + 1
    Image.fromarray(lab).save(mp)
    return {k: len(v) for k, v in classes.items()}


def add_to_class(d, name, cls: str, polys) -> dict:
    """adds areas to a class (false hit with a cause → its class); returns the class counts"""
    c = class_polygons(d, name)
    if polys:
        c[cls] = (c.get(cls) or []) + list(polys)
    return _write_classes(d, name, c)


def ds_dir(job_dir: str) -> str:
    return os.path.join(job_dir, "dataset")


def _meta_path(d, name):
    return os.path.join(d, name + ".json")


def read_meta(d, name) -> dict:
    with open(_meta_path(d, name), encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("status", "open")
    m.setdefault("mask_source", "pvmap" if m.get("has_mask") else None)
    return m


def write_meta(d, meta):
    tmp = _meta_path(d, meta["name"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp, _meta_path(d, meta["name"]))


def list_items(d) -> list[dict]:
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and not f.endswith(".tmp"):
            try:
                m = read_meta(d, f[:-5])
            except Exception:                                    # noqa: BLE001
                continue
            m["effective_label"] = effective_label(m)
            out.append(m)
    return out


def effective_label(m) -> int:
    return int(m["label_override"]) if m.get("label_override") is not None else int(m.get("label", 0))


def stats(items) -> dict:
    s = {"total": len(items), "neg_marked": 0, "open": 0, "done": 0, "skip": 0, "masks": 0, "masks_user": 0, "masks_ai": 0,
         "masks_ai_unreviewed": 0, "positive": 0, "positive_without_mask": 0}
    for m in items:
        s[m.get("status", "open")] = s.get(m.get("status", "open"), 0) + 1
        pos = effective_label(m) == 1
        s["positive"] += pos
        s["neg_marked"] += bool(m.get("has_neg"))
        if m.get("has_mask"):
            s["masks"] += 1
            src = m.get("mask_source")
            if src == "user":
                s["masks_user"] += 1
            if src == "ai":
                s["masks_ai"] += 1
                if not (m.get("ai") or {}).get("reviewed"):
                    s["masks_ai_unreviewed"] += 1
        elif pos and m.get("status") != "skip":
            s["positive_without_mask"] += 1
    return s


# ---------------------------------------------------------------------------
# masks ↔ polygons
# ---------------------------------------------------------------------------

def mask_to_px_polygons(mask: np.ndarray, min_area: int = 12, eps: float = 1.2) -> list:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            out.append(c.reshape(-1, 2).astype(int).tolist())
    return out


def polygons_to_mask(polys, width: int, height: int) -> np.ndarray:
    m = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(m)
    for p in polys or []:
        pts = [(float(x), float(y)) for x, y in p]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.asarray(m, dtype=np.uint8)


def image_size(d, name):
    with Image.open(os.path.join(d, name + ".jpg")) as im:
        return im.size


def footprint_polygons(d, name) -> list:
    fp = np.asarray(Image.open(os.path.join(d, name + ".fp.png"))) > 0
    return mask_to_px_polygons(fp, min_area=4, eps=1.0)


def current_polygons(d, name, kind: str = "mask") -> list:
    p = os.path.join(d, f"{name}.{kind}.png")
    if not os.path.isfile(p):
        return []
    return mask_to_px_polygons(np.asarray(Image.open(p)) > 0)


def save_annotation(d, name, polygons=None, label=None, status="done", source="user", ai=None,
                    negatives=None, classes=None) -> dict:
    """Stores a mask (polygons; empty list removes it), optional 'no PV' regions, optional polygons per roof element class,
    an optional label correction and the status."""
    meta = read_meta(d, name)
    if negatives is not None:
        w, h = image_size(d, name)
        n = polygons_to_mask(negatives, w, h)
        path = os.path.join(d, name + ".neg.png")
        if n.any():
            Image.fromarray(n * 255).save(path)
        elif os.path.isfile(path):
            os.remove(path)
        meta["has_neg"] = bool(n.any())
    if polygons is not None:
        w, h = image_size(d, name)
        m = polygons_to_mask(polygons, w, h)
        path = os.path.join(d, name + ".mask.png")
        if m.any():
            Image.fromarray(m * 255).save(path)
            meta["has_mask"], meta["mask_source"] = True, source
        else:
            if os.path.isfile(path):
                os.remove(path)
            meta["has_mask"], meta["mask_source"] = False, None
    if classes is not None:
        meta["classes"] = _write_classes(d, name, classes, polygons)
    if label is not None:
        meta["label_override"] = None if int(label) == int(meta.get("label", 0)) else int(label)
    if ai is not None:
        meta["ai"] = ai
    elif source == "user" and meta.get("ai"):
        meta["ai"]["reviewed"] = True                            # a user saving over an AI mask has reviewed it
    meta["status"] = status
    meta["annotated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_meta(d, meta)
    meta["effective_label"] = effective_label(meta)
    return meta


def usable_mask(meta, ai_masks: str = "reviewed") -> bool:
    """Whether a mask may be used for training: PV-Map layers and users always, AI depending on the setting."""
    if not meta.get("has_mask") or meta.get("status") == "skip":
        return False
    src = meta.get("mask_source") or "pvmap"
    if src == "ai":
        return ai_masks == "all" or (ai_masks == "reviewed" and (meta.get("ai") or {}).get("reviewed"))
    return src in ("pvmap", "user")


# ---------------------------------------------------------------------------
# model suggestion
# ---------------------------------------------------------------------------

def suggest_from_model(det, d, name, cls_threshold: float | None = None) -> dict:
    """Runs the detector on the stored image; returns PV polygons (pixel space) and the max score."""
    img = np.asarray(Image.open(os.path.join(d, name + ".jpg")).convert("RGB"))
    old = det.cls_threshold
    if cls_threshold is not None:
        det.cls_threshold = cls_threshold
    try:
        r = det.run([("x", img)])["x"]
    finally:
        det.cls_threshold = old
    return {"polygons": mask_to_px_polygons(r.mask), "max_prob": round(r.max_prob, 3)}


def neg_mask(d, name):
    p = os.path.join(d, name + ".neg.png")
    return (np.asarray(Image.open(p)) > 0) if os.path.isfile(p) else None


def corner_points(d, name, max_corners: int = 2500) -> list:
    """corner-like image points (roof edges, module field corners) for snapping in the mask editor"""
    img = np.asarray(Image.open(os.path.join(d, name + ".jpg")).convert("L"))
    g = cv2.GaussianBlur(img, (3, 3), 0)
    pts = cv2.goodFeaturesToTrack(g, maxCorners=max_corners, qualityLevel=0.015, minDistance=3, blockSize=5)
    if pts is None:
        return []
    pts = cv2.cornerSubPix(g, pts.astype(np.float32), (3, 3), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05))
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts.reshape(-1, 2)]


def _window(meta):
    from shapely.geometry import Point
    from . import geo
    win = meta.get("window")
    if not win:
        return None, None
    w = geo.Window(win[0], *win[1:5], int(win[5]), int(win[6]))
    c = geo.reproject(Point((w.minx + w.maxx) / 2, (w.miny + w.maxy) / 2), w.crs, "EPSG:4326")
    res = (w.maxx - w.minx) / w.width / geo.scale_factor(w.crs, c.x, c.y)
    return w, res


def typology_of(d, name, meta, polygons=None, cfg=None) -> dict:
    """installation typology of an image in the mask editor, for the drawn mask (polygons in px; None = saved mask):
    {summary, installations: [attributes + outline_px]} – pixel outlines for the editor, nothing stored"""
    from . import typology
    win, res = _window(meta)
    if win is None:
        return {"summary": {}, "installations": []}
    w, h = image_size(d, name)
    m = polygons_to_mask(polygons, w, h) if polygons is not None else (
        np.asarray(Image.open(os.path.join(d, name + ".mask.png"))) > 0 if os.path.isfile(os.path.join(d, name + ".mask.png"))
        else np.zeros((h, w), bool))
    if not m.any():
        return {"summary": {}, "installations": []}
    img = np.asarray(Image.open(os.path.join(d, name + ".jpg")).convert("RGB"))
    fpp = os.path.join(d, name + ".fp.png")
    clip = None
    if os.path.isfile(fpp):
        fp = (np.asarray(Image.open(fpp)) > 0).astype(np.uint8)
        k = max(3, int(round(3.0 / res)))                       # footprint + 1.5 m, as in the building check
        clip = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) if fp.any() else None

    class _Src:
        id = meta.get("source") or ""
    insts = typology.analyse(img, m, res, clip, {}, cfg, year=typology.imagery_year(_Src),
                             review_status="confirmed_manual")
    keep = ("layout", "azimuth_deg", "azimuth2_deg", "tilt_class", "fill_factor", "fill_factor_source", "outline_area_m2",
            "kwp_est", "kwp_low", "kwp_high", "outline_px")
    sm = typology.summarise(insts)
    if sm:
        sm["year"] = typology.imagery_year(_Src)
    return {"summary": sm, "installations": [{k: a.get(k) for k in keep} for a in insts],
            "_insts": insts, "_win": win}


def review_result(d, name, meta, kwp_per_m2: float = 0.2, cfg: dict | None = None) -> dict | None:
    """result for PV-Map from an image checked by hand: PV polygons in WGS84 (from the stored window) or 'none'"""
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union
    from . import geo
    win = meta.get("window")
    if not win or meta.get("status") != "done":
        return None
    w = geo.Window(win[0], *win[1:5], int(win[5]), int(win[6]))
    label = effective_label(meta)
    polys = current_polygons(d, name) if label == 1 else []
    if label == 1 and not polys:
        return None                                            # PV, but no area drawn: nothing to say about where
    if not polys:
        return {"id": meta["id"], "result": "none", "reviewed": True}
    shapes = [Polygon([w.pixel_to_crs(x, y) for x, y in p]).buffer(0) for p in polys]
    g = unary_union([x for x in shapes if not x.is_empty])
    c = geo.reproject(g.centroid, w.crs, "EPSG:4326")
    k = geo.scale_factor(w.crs, c.x, c.y)
    area = g.area / (k * k)
    out = {"id": meta["id"], "result": "pv", "tier": 1, "reviewed": True, "area_m2": round(area, 1),
           "kwp": round(area * kwp_per_m2, 1), "polygons": mapping(geo.reproject(g, w.crs, "EPSG:4326"))}
    if cfg is not None and cfg.get("typology", True):            # the hand-drawn mask is the best basis for the typology
        try:
            from . import typology
            t = typology_of(d, name, meta, cfg=cfg)
            if t["summary"]:
                out.update(kwp=t["summary"]["kwp_est"])
                if cfg.get("typology_push"):
                    feats = typology.to_wgs84(t["_insts"], t["_win"], meta["id"],
                                              {"source": meta.get("source"), "year": t["summary"].get("year")})
                    out.update(kwp_low=t["summary"]["kwp_low"], kwp_high=t["summary"]["kwp_high"],
                               layout=t["summary"]["layout"],
                               installations=[{**f["properties"], "geometry": f["geometry"]} for f in feats])
        except Exception:                                        # noqa: BLE001 – the review itself must go through
            pass
    return out


# ---------------------------------------------------------------------------------------------------------------------
# register data (MaStR via PV-Map) for the images of a dataset

def _synthetic(bid: str) -> bool:
    """ids PV-Mapper made up itself (area installations 'AREA-…', probe cut-outs 'PROBE-…') are no PV-Map building ids"""
    return bid.startswith(("AREA-", "PROBE-"))


def register_lookup(d: str, client, names: list | None = None, radius_m: float = 30.0, log=None) -> dict:
    """asks PV-Map what the register says about the building(s) of each image and stores it in the image's meta
    (`register`: {buildings: [{id, name, distance_m, register: {kwp, n, by_year, first_year, grade, …} | None}], checked})
    – by building id where the image came from PV-Map, else around the image centre. Returns counts."""
    import datetime as _dt
    from shapely.geometry import Point
    from . import geo
    items = [m for m in list_items(d) if not names or m["name"] in names]
    today = _dt.date.today().isoformat()
    out = {"images": 0, "with_register": 0, "errors": 0}
    by_id = [m for m in items if not _synthetic(str(m.get("id") or ""))]
    found = {}
    for i in range(0, len(by_id), 200):
        chunk = by_id[i:i + 200]
        try:
            for b in client.register(ids=[m["id"] for m in chunk]):
                found[str(b["id"])] = b
        except Exception as e:                                       # noqa: BLE001
            out["errors"] += 1
            if log:
                log(f"PV-Map: {e}"[:200])
    for m in items:
        meta = read_meta(d, m["name"])
        hit = found.get(str(meta.get("id")))
        if hit is not None:
            res = [hit]
        else:
            lon, lat = meta.get("lon"), meta.get("lat")
            if lon is None and meta.get("window"):
                w, _ = _window(meta)
                c = geo.reproject(Point((w.minx + w.maxx) / 2, (w.miny + w.maxy) / 2), w.crs, "EPSG:4326")
                lon, lat = c.x, c.y
            if lon is None:
                continue
            try:
                res = client.register(lat=lat, lon=lon, radius_m=radius_m)
            except Exception as e:                                   # noqa: BLE001
                out["errors"] += 1
                if log and out["errors"] <= 3:
                    log(f"PV-Map: {e}"[:200])
                continue
        meta["register"] = {"buildings": res, "checked": today}
        with open(os.path.join(d, m["name"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        out["images"] += 1
        out["with_register"] += any(b.get("register") for b in res)
    return out


# ---------------------------------------------------------------------------
# "find outline" tool of the mask editor (20.11.2026): a click into a module field, a shadow, a dormer …
# ---------------------------------------------------------------------------

def _flood(smooth, sx, sy, t):
    h, w = smooth.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(smooth, mask, (sx, sy), (0, 0, 0), (t, t, t), (t, t, t), flags)
    return mask[1:-1, 1:-1] > 0


def _smooth(img):
    return cv2.bilateralFilter(cv2.medianBlur(img, 5), 9, 30, 7)


TOL_STEPS = (5, 7, 9, 12, 15, 18, 22, 27, 33, 40, 48, 58)


def auto_tolerance(smooth, sx, sy, max_share: float = 0.35, min_px: int = 30):
    """tolerance at which the filled area stops growing (a plateau = the object's boundary), before it runs out into the roof:
    the area for each tolerance step, then the step after which it grows least; [(tol, area)] for the editor"""
    h, w = smooth.shape[:2]
    curve = []
    for t in TOL_STEPS:
        a = int(_flood(smooth, sx, sy, t).sum())
        curve.append((t, a))
        if a > max_share * h * w:
            break
    ok = [(t, a) for t, a in curve if min_px <= a <= max_share * h * w]
    if not ok:
        return (curve[0][0] if curve else 18), curve
    best, best_g = ok[-1][0], None
    for i, (t, a) in enumerate(curve):
        if (t, a) not in ok or i + 1 >= len(curve):
            continue
        g = curve[i + 1][1] / max(a, 1)                 # growth to the next step
        if best_g is None or g < best_g:
            best, best_g = t, g
    return best, curve


def outline_at(d, name, x: float, y: float, tol: float | None = 18.0, max_share: float = 0.35) -> dict:
    """outer boundary of the uniform area around a clicked point: flood fill on a smoothed image (the module grid and roof tile
    texture disappear), closed and with holes filled; a nearly rectangular result becomes a (rotated) rectangle. If the fill
    runs out into the roof (more than max_share of the image), the tolerance is reduced step by step."""
    img = np.asarray(Image.open(os.path.join(d, name + ".jpg")).convert("RGB"))
    h, w = img.shape[:2]
    sx, sy = int(round(min(max(x, 0), w - 1))), int(round(min(max(y, 0), h - 1)))
    smooth = _smooth(img)
    auto = tol is None or str(tol).lower() == "auto"
    curve = None
    if auto:
        tol, curve = auto_tolerance(smooth, sx, sy, max_share)
    t = float(tol)
    region = None
    for _ in range(6):
        r = _flood(smooth, sx, sy, t)
        if r.sum() <= max_share * h * w:
            region = r
            break
        t *= 0.65
    if region is None or region.sum() < 6:
        return {"polygon": None, "tol": round(t, 1), "auto": auto, "hint": "keine einheitliche Fläche gefunden"}
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"polygon": None, "tol": round(t, 1), "hint": "keine einheitliche Fläche gefunden"}
    inside = [c for c in cnts if cv2.pointPolygonTest(c, (sx, sy), False) >= 0]
    c = max(inside or cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    rect = rw * rh > 0 and area / (rw * rh) >= 0.85
    if rect:
        poly = cv2.boxPoints(((cx, cy), (rw, rh), ang))
    else:
        poly = cv2.approxPolyDP(c, max(1.0, 0.008 * cv2.arcLength(c, True)), True).reshape(-1, 2)
    return {"polygon": [[round(float(px), 1), round(float(py), 1)] for px, py in poly], "rect": bool(rect),
            "area_px": int(area), "tol": round(t, 1), "auto": auto, **({"curve": curve} if curve else {})}


def suggest_tolerance(d, name) -> int:
    """a starting tolerance for an image (wand): from the texture of the roof – the median brightness step between neighbouring
    pixels of the smoothed image inside the footprint, times a factor; flat, uniform images get a small one"""
    img = np.asarray(Image.open(os.path.join(d, name + ".jpg")).convert("RGB"))
    sm = _smooth(img).astype(np.int16)
    g = np.abs(np.diff(sm, axis=1)).max(axis=2)
    fpp = os.path.join(d, name + ".fp.png")
    if os.path.isfile(fpp):
        fp = np.asarray(Image.open(fpp)) > 0
        vals = g[fp[:, 1:]]
        if vals.size < 100:
            vals = g.ravel()
    else:
        vals = g.ravel()
    noise = float(np.percentile(vals, 75))
    return int(min(40, max(6, round(4 + 3 * noise))))
