"""
Training dataset and masks (annotation in the web UI, AI suggestions).

A dataset is the `dataset/` folder of a training job (or of a "prepare only" job). Per building:
  <name>.jpg        aerial image at the model resolution (footprint + buffer)
  <name>.fp.png     footprint mask
  <name>.mask.png   PV mask (optional) — from a PV-Map layer, drawn by a user, or suggested by AI
  <name>.neg.png    "no PV" mask (optional) — confusers marked by a user (dormers, skylights, solar thermal …);
                    training weights these pixels higher and uses them as hard negative patches
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
                    negatives=None) -> dict:
    """Stores a mask (polygons; empty list removes it), optional 'no PV' regions, an optional label correction
    and the status."""
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
