"""
Post-processing of PV masks in image space (detector output and AI masks).

The segmentation models were trained on other regions and often mark roof structures (dormers, skylights,
bright roof elements) or only parts of a module field (glare, shading). These rules use what is known about
PV on roofs:

  clip        only the building footprint (+ margin) counts
  contrast    modules are darker than the surrounding roof or clearly bluer; regions that are as bright as the
              roof and not bluer are rejected (bright roof elements, white roofs, flashing)
  repetition  several small, similar, aligned blobs without any larger field are a row of dormers/skylights
  completion  a detected field is grown into adjacent pixels of the same colour (rows cut by glare or by the
              patch grid), and other regions on the same roof with the same colour and a rectangular shape are
              added (fields the model missed); only when modules differ clearly from the roof
  rectify     nearly rectangular fields become their rotated rectangle; holes (glare) are filled

All thresholds are in metres / 8-bit brightness and can be changed in the settings (postproc_*).
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    "postproc": True,
    "postproc_min_m2": 1.5,          # smaller blobs are noise
    "postproc_contrast": 8,          # module field must be this much darker than the roof around it …
    "postproc_blue": 6,              # … or this much bluer (B−R)
    "postproc_max_v": 205,           # brighter on average: never modules
    "postproc_fill": 0.7,            # share of its rotated bounding rectangle a field must fill …
    "postproc_strong": 25,           # … unless it is this much darker than the roof (L-shaped arrays)
    "postproc_repeat_m2": 4.0,       # "small" for the dormer-row rule
    "postproc_repeat_n": 3,
    "postproc_complete": True,
    "postproc_complete_contrast": 14,  # completion only when modules stand out this much from the roof
    "postproc_lookalike_m2": 4.0,    # minimum size of an added look-alike field (≥ 2 modules; skylights are smaller)
    "postproc_rectify": 0.8,         # fill ratio above which a field becomes its rotated rectangle (0 = off)
}


def settings(cfg: dict | None) -> dict:
    s = dict(DEFAULTS)
    for k in DEFAULTS:
        if cfg and cfg.get(k) is not None:
            s[k] = cfg[k]
    return s


def _features(V, B, sl, comp, others, clip, r_px):
    """mean brightness / blueness of a component and of the roof ring around it (all arrays cropped to sl)"""
    Vc, Bc = V[sl], B[sl]
    k = 2 * r_px + 1
    near = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    ring = near & ~comp & ~others[sl]
    if clip is not None:
        ring &= clip[sl]
    if ring.sum() < 10:
        ring = near & ~comp
    v, b = Vc[comp], Bc[comp]
    rv = Vc[ring] if ring.any() else v
    rb = Bc[ring] if ring.any() else b
    return {"v": float(v.mean()), "sv": float(v.std()), "b": float(b.mean()), "sb": float(b.std()),
            "rv": float(np.median(rv)), "rb": float(np.median(rb))}


def _rect(comp):
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    r = cv2.minAreaRect(c)
    return r, comp.sum() / max(r[1][0] * r[1][1], 1.0)


def _fill_holes(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out.astype(bool)


def _components(m, pad: int = 0):
    """connected components as (slice of the padded bounding box, mask cropped to it, area, centroid).
    Working on crops keeps memory and time independent of the image size."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    H, W = m.shape[:2]
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sl = (slice(max(0, y - pad), min(H, y + h + pad)), slice(max(0, x - pad), min(W, x + w + pad)))
        yield sl, lab[sl] == i, int(a), cen[i]


def _union_slice(a, b):
    return (slice(min(a[0].start, b[0].start), max(a[0].stop, b[0].stop)),
            slice(min(a[1].start, b[1].start), max(a[1].stop, b[1].stop)))


def _paste(sl_dst, sl_src, comp):
    """comp (cropped to sl_src) placed into a crop of sl_dst"""
    out = np.zeros((sl_dst[0].stop - sl_dst[0].start, sl_dst[1].stop - sl_dst[1].start), bool)
    y0, x0 = sl_src[0].start - sl_dst[0].start, sl_src[1].start - sl_dst[1].start
    out[y0:y0 + comp.shape[0], x0:x0 + comp.shape[1]] = comp
    return out


def clean(img: np.ndarray, mask: np.ndarray, res_m: float, clip: np.ndarray | None = None,
          cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """img (H,W,3) uint8 RGB, mask (H,W) {0,1}, clip (H,W) bool or None.
    Returns (cleaned mask uint8, info {removed: [(reason, m²)], added_m2})."""
    s = settings(cfg)
    info = {"removed": [], "added_m2": 0.0}
    m = mask.astype(bool)
    if clip is not None:
        m &= clip
    if not s["postproc"] or not m.any():
        return m.astype(np.uint8), info
    px_m2 = res_m * res_m
    f = img.astype(np.float32)
    V = f.max(-1)
    B = f[..., 2] - f[..., 0]
    del f
    r_px = max(3, int(round(1.5 / res_m)))
    pad = r_px + 2
    min_px = s["postproc_min_m2"] / px_m2
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    if clip is not None:
        m &= clip

    def filter_pass(m):
        keep = np.zeros_like(m)
        tiny, stats = [], []
        for sl, comp, a, cen in _components(m, pad):
            if a < min_px:
                if a >= 0.3 / px_m2:
                    tiny.append((a, cen))
                info["removed"].append(("klein", round(a * px_m2, 1)))
                continue
            ft = _features(V, B, sl, comp, m, clip, r_px)
            darker = ft["rv"] - ft["v"]
            bluer = ft["b"] - ft["rb"]
            if ft["v"] > s["postproc_max_v"] and bluer < 3 * s["postproc_blue"]:
                info["removed"].append(("sehr hell", round(a * px_m2, 1)))
                continue
            fill = _rect(comp)[1]
            # modules are darker than the roof and rectangular; only clearly bluer, compact, uniform regions
            # may be as bright as the roof (dark roofs). Bluish slate and lit roof faces fail both.
            dark_ok = darker >= s["postproc_contrast"] and (fill >= s["postproc_fill"] or darker >= s["postproc_strong"])
            blue_ok = bluer >= 10 + s["postproc_blue"] / 2 and fill >= 0.8 and ft["sv"] <= 12
            if not (dark_ok or blue_ok):
                reason = "kein Kontrast zum Dach" if darker < s["postproc_contrast"] else "unregelmäßige Form"
                info["removed"].append((reason, round(a * px_m2, 1)))
                continue
            stats.append((sl, comp, a, cen))
        small = [x for x in stats if x[2] * px_m2 < s["postproc_repeat_m2"]]
        big = [x for x in stats if x[2] * px_m2 >= s["postproc_repeat_m2"]]
        drop_small = False
        n_small = len(small) + len(tiny)
        if n_small >= s["postproc_repeat_n"] and small and not big:
            areas = np.array([x[2] for x in small] + [a for a, _ in tiny], float)
            pts = np.array([x[3] for x in small] + [c for _, c in tiny], float)
            cv_a = areas.std() / max(areas.mean(), 1)
            ev = np.linalg.eigvalsh(np.cov(pts.T)) if len(pts) > 2 else np.array([0, 1])
            aligned = ev[0] <= 0.1 * ev[1] + 1e-6
            drop_small = cv_a < 0.9 and (aligned or n_small >= 6)
        for sl, comp, a, cen in stats:
            if drop_small and a * px_m2 < s["postproc_repeat_m2"]:
                info["removed"].append(("Reihe gleichartiger Dachaufbauten", round(a * px_m2, 1)))
                continue
            keep[sl] |= comp
        return keep

    m = filter_pass(m)
    before = m.sum()

    if s["postproc_complete"] and m.any():
        m = _complete(m, V, B, clip, s, min_px, r_px, pad, px_m2)
        m = filter_pass(m)

    out = np.zeros_like(m)
    for sl, comp, a, _ in _components(m, 1):
        comp = _fill_holes(comp)
        if s["postproc_rectify"] and a * px_m2 >= 2.0:
            r, fill = _rect(comp)
            if fill >= s["postproc_rectify"]:
                box = np.zeros(comp.shape, np.uint8)
                cv2.fillPoly(box, [cv2.boxPoints(r).round().astype(np.int32)], 1)
                comp = box.astype(bool)
        out[sl] |= comp
    if clip is not None:
        out &= clip
    info["removed"].sort(key=lambda r: -r[1])
    info["added_m2"] = round(max(0, int(out.sum()) - int(before)) * px_m2, 1)
    return out.astype(np.uint8), info


def _complete(m, V, B, clip, s, min_px, r_px, pad, px_m2):
    """grow detected fields along same-coloured pixels; add same-coloured rectangular fields on the roof"""
    area_all = int(clip.sum()) if clip is not None else m.size
    add = np.zeros_like(m)
    budget = 3 * int(m.sum())
    k5 = np.ones((5, 5), np.uint8)
    for sl, comp, a, _ in list(_components(m, pad)):
        ft = _features(V, B, sl, comp, m, clip, r_px)
        if ft["rv"] - ft["v"] < s["postproc_complete_contrast"] and ft["b"] - ft["rb"] < 2 * s["postproc_blue"]:
            continue                                   # modules not distinguishable from the roof by colour
        tv, tb = max(10.0, 2 * ft["sv"]), max(8.0, 2 * ft["sb"])
        cand = (np.abs(V - ft["v"]) < tv) & (np.abs(B - ft["b"]) < tb)
        if clip is not None:
            cand &= clip
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k5).astype(bool)
        if cand.sum() > 0.6 * area_all:
            continue                                   # roof has the module colour: no completion
        near = cv2.dilate(comp.astype(np.uint8), k5).astype(bool)          # in sl coordinates
        for sc, c, ca, _ in _components(cand & ~m, 0):
            if ca < min_px:
                continue
            u = _union_slice(sl, sc)
            cu, nu = _paste(u, sc, c), _paste(u, sl, near)
            if (cu & nu).any():                        # extension of this field
                _, fill = _rect(cu | _paste(u, sl, comp))
                if fill >= 0.55:
                    add[sc] |= c
            else:                                      # look-alike field elsewhere on the roof
                if ca < s["postproc_lookalike_m2"] / px_m2:
                    continue
                _, fill = _rect(c)
                if fill < 0.7:
                    continue
                H, W = m.shape
                sp = (slice(max(0, sc[0].start - pad), min(H, sc[0].stop + pad)),
                      slice(max(0, sc[1].start - pad), min(W, sc[1].stop + pad)))
                f2 = _features(V, B, sp, _paste(sp, sc, c), m | add, clip, r_px)
                if f2["rv"] - f2["v"] >= s["postproc_complete_contrast"] or f2["b"] - f2["rb"] >= 2 * s["postproc_blue"]:
                    add[sc] |= c
    if add.sum() > budget + min_px * 4:
        # too much: keep only extensions touching existing fields
        _, lab = cv2.connectedComponents((m | add).astype(np.uint8))
        keep_ids = set(np.unique(lab[m])) - {0}
        add &= np.isin(lab, list(keep_ids))
    return m | add


def fuse(primary: np.ndarray, second: np.ndarray | None, res_m: float) -> tuple[np.ndarray, str]:
    """Cross-check with a second aerial image (other year/flight) of the same window.
    - second image shows no PV: keep primary (may be a newer installation), status 'nur_aktuell'
    - otherwise keep primary fields confirmed by the second image (overlap after 1 m tolerance) and add the
      matching parts of the second mask (fields cut by glare/shade in one image); unconfirmed fields go."""
    if second is None:
        return primary, "ohne"
    p = primary.astype(bool)
    q = second.astype(bool)
    if not p.any():
        return p.astype(np.uint8), "keine_pv" if not q.any() else "nur_zweitbild"
    if not q.any():
        return p.astype(np.uint8), "nur_aktuell"
    tol = max(1, int(round(1.0 / res_m)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    qd = cv2.dilate(q.astype(np.uint8), k).astype(bool)
    out = np.zeros_like(p)
    dropped = 0
    for sl, comp, a, _ in _components(p):
        if (comp & qd[sl]).sum() >= 0.2 * a:
            out[sl] |= comp
        else:
            dropped += 1
    od = cv2.dilate(out.astype(np.uint8), k).astype(bool)
    for sl, comp, a, _ in _components(q):
        if (comp & od[sl]).any():                      # same field, seen more completely in the second image
            out[sl] |= comp
    status = "bestaetigt" if out.any() else "nicht_bestaetigt"
    if out.any() and dropped:
        status = "teilweise"
    return out.astype(np.uint8), status
