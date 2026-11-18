"""
Free roof area and PV potential of a building (PVMAPPER.md "Freie Dachfläche").

For a given building list: footprint + aerial image + the PV mask of the check →

  roof type      flat / pitched from OSM roof:shape, else from the image (the two halves of a pitched roof along its long axis
                 are lit differently); pitched roofs are taken as gable roofs with the ridge along the long axis
  setback        edge distance (flat 1.0 m, pitched 0.5 m) and a strip along the ridge
  obstacles      everything on the roof that differs clearly from the roof surface: colour (Lab distance to the roof's own
                 colour, per roof half on pitched roofs), very dark (technical units) or very bright spots (skylights, light
                 domes), strong texture (railings, pipes); plus a clearance around each (0.5 m). Deviating areas larger than
                 300 m² are another roof section and get their own surface colour.
  shade          areas clearly darker than the roof surface but of its colour (≥ 2 m²): shaded at the time of the flight
  existing PV    the detected PV with the gaps between its rows (+0.3 m) is not free
  free area      what is left, without strips narrower than a module (1.0 m) and pieces smaller than 5 m²; on pitched roofs
                 without the face pointing north (±60°)
  potential      free area × fill factor of the assumed layout (flat: east-west, pitched: flush) / cos(tilt) × efficiency of
                 current modules, with a range from the fill factor and efficiency ranges

Everything here is image analysis without a trained network – a first estimate; obstacles marked in the mask editor
("Gaube / kein PV") are the material for a trained obstacle model later.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULTS = {
    "roof_setback_flat_m": 1.0,
    "roof_setback_pitched_m": 0.5,
    "roof_ridge_m": 0.3,
    "roof_obstacle_clearance_m": 0.5,
    "roof_min_width_m": 1.0,
    "roof_min_area_m2": 5.0,
    "roof_pitched_tilt_deg": 35.0,
    "roof_flat_layout": "east_west",          # east_west | tilted_mono
    "roof_exclude_north_deg": 60.0,           # faces whose azimuth is within ± this of north are not used
    "roof_color_dist": 26.0,                  # Lab distance to the roof colour that counts as an obstacle
    "roof_dark": 38.0,                        # darker than the roof by this (gray levels): shadow / technical unit
    "roof_bright": 48.0,                      # brighter by this: skylight, light dome, white unit
    "roof_halves_diff": 14.0,                 # gray difference of the two halves that marks a pitched roof (image test)
    "roof_shade": 30.0,                       # darker than the roof surface by this over ≥ 2 m²: shaded, not used
    "roof_max_obstacle_m2": 300.0,            # larger deviating areas are another roof section (own surface colour)
}
PITCHED = {"gabled", "hipped", "half-hipped", "pyramidal", "gambrel", "mansard", "saltbox", "skillion", "round", "dome",
           "side_hipped", "cross_gabled", "onion", "gabled_height_moved", "sawtooth"}


def _c(cfg):
    return {**DEFAULTS, **{k: v for k, v in (cfg or {}).items() if k in DEFAULTS and v is not None}}


def _comp_filter(m: np.ndarray, min_px: int, max_px: int | None = None) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    a = st[:, cv2.CC_STAT_AREA]
    keep[1:] = a[1:] >= min_px
    if max_px is not None:
        keep[1:] &= a[1:] <= max_px
    return keep[lab]


def _disk(px: int):
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def _axis(fp: np.ndarray):
    """long axis of the footprint: centre (x, y), unit vector along the long side, length/width in px"""
    cnts, _ = cv2.findContours(fp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    th = math.radians(ang)
    u = (math.cos(th), math.sin(th))
    if h > w:
        u = (-u[1], u[0])
        w, h = h, w
    return (cx, cy), u, w, h


def _az(vx, vy) -> float:
    """image vector (x right, y down) → compass azimuth (0 = north)"""
    return (math.degrees(math.atan2(vx, -vy)) + 360) % 360


def analyse(img: np.ndarray, fp: np.ndarray, pv: np.ndarray | None, res_m: float, props: dict | None = None,
            cfg: dict | None = None, conf: dict | None = None) -> dict:
    """fp: exact footprint (bool), pv: detected PV (bool). Returns areas (plan m²), potential and masks
    (free, obstacles) in image pixels."""
    from . import typology
    C = _c(cfg)
    conf = conf or typology.settings(cfg)
    fp = fp.astype(bool)
    pv = np.zeros_like(fp) if pv is None else (pv.astype(bool) & fp)
    px_m = 1.0 / res_m
    a_px = res_m * res_m
    roof_m2 = float(fp.sum()) * a_px
    out = {"roof_area_m2": round(roof_m2, 1), "usable_area_m2": 0.0, "kwp_potential": 0.0, "kwp_pot_low": 0.0,
           "kwp_pot_high": 0.0, "roof_type": None, "roof_type_basis": None, "obstacles_area_m2": 0.0, "shade_area_m2": 0.0,
           "pv_area_m2": round(float(pv.sum()) * a_px, 1), "faces": []}
    if roof_m2 < 10:
        out["note"] = "Dach zu klein"
        return {**out, "_free": np.zeros_like(fp), "_obst": np.zeros_like(fp)}

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (cx, cy), u, L, Wd = _axis(fp)
    H, W = fp.shape
    yy, xx = np.mgrid[0:H, 0:W]
    side = ((xx - cx) * (-u[1]) + (yy - cy) * u[0])            # signed distance from the long axis (px)
    half_a, half_b = fp & (side < 0), fp & (side >= 0)
    core = fp & (cv2.distanceTransform(fp.astype(np.uint8), cv2.DIST_L2, 3) > 2 * px_m) & ~pv

    # roof type
    shape_tag = str((props or {}).get("roof:shape") or (props or {}).get("roof_shape") or "").lower()
    if shape_tag == "flat":
        rtype, basis = "flat", "OSM"
    elif shape_tag in PITCHED:
        rtype, basis = "pitched", "OSM"
    else:
        # a gable roof: simple (nearly rectangular) footprint, each half uniform, the halves clearly different all along
        rect = float(fp.sum()) / max(L * Wd, 1.0)
        along = ((xx - cx) * u[0] + (yy - cy) * u[1])
        diffs, mads = [], []
        for k in range(3):
            lo, hi = -L / 2 + k * L / 3, -L / 2 + (k + 1) * L / 3
            seg = (along >= lo) & (along < hi)
            ga, gb = gray[half_a & core & seg], gray[half_b & core & seg]
            if ga.size < 30 or gb.size < 30:
                continue
            ma, mb = float(np.median(ga)), float(np.median(gb))
            diffs.append(ma - mb)
            mads += [float(np.median(np.abs(ga - ma))), float(np.median(np.abs(gb - mb)))]
        same = len(diffs) == 3 and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs))
        dmin = min(abs(d) for d in diffs) if diffs else 0.0
        pitched = same and rect >= 0.85 and Wd * res_m >= 5 and dmin >= C["roof_halves_diff"] and dmin >= 1.5 * max(mads or [99])
        rtype, basis = ("pitched" if pitched else "flat"), "Bild"
    out.update(roof_type=rtype, roof_type_basis=basis)

    # obstacles: compared with the roof's own surface (per half on pitched roofs, the halves are lit differently).
    # Large deviating areas are another roof section (other material or level): their obstacles are found against their own
    # surface colour in a second pass.
    tex = cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2
    tex = np.sqrt(np.maximum(tex, 0))
    inner_fp = fp & (cv2.distanceTransform(fp.astype(np.uint8), cv2.DIST_L2, 3) > 1)
    pv_ring = cv2.dilate(pv.astype(np.uint8), _disk(round(0.5 * px_m))).astype(bool)   # edges of the PV are no obstacles

    def deviating(reg):
        sel = reg & core
        if sel.sum() < 50:
            return np.zeros_like(fp), np.zeros_like(fp)
        ref = np.median(lab[sel], axis=0)
        rg = float(np.median(gray[sel]))
        rt = float(np.median(tex[sel]))
        d = np.sqrt(((lab - ref) ** 2).sum(axis=2))
        o = (d > C["roof_color_dist"]) | (gray < rg - C["roof_dark"]) | (gray > rg + C["roof_bright"]) | (tex > max(14.0, 2.5 * rt))
        shade = (gray < rg - C["roof_shade"]) & (d < 2.5 * C["roof_color_dist"])
        return o & reg & inner_fp & ~pv_ring, shade & reg & inner_fp & ~pv_ring

    def clean(o):
        o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_OPEN, _disk(1)).astype(bool)
        return cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_CLOSE, _disk(2)).astype(bool)

    regions = [half_a, half_b] if rtype == "pitched" else [fp]
    obst, shade = np.zeros_like(fp), np.zeros_like(fp)
    big_px = int(C["roof_max_obstacle_m2"] / a_px)
    for reg in regions:
        o, sh = deviating(reg)
        o = clean(o)
        small = _comp_filter(o, int(0.2 / a_px), big_px)
        obst |= small
        shade |= _comp_filter(sh & ~o, int(2.0 / a_px))
        n, labs, st, _ = cv2.connectedComponentsWithStats((o & ~small).astype(np.uint8), connectivity=8)
        for i in range(1, n):                                   # another roof section: its own obstacles
            if st[i, cv2.CC_STAT_AREA] < big_px:
                continue
            sec = labs == i
            o2, sh2 = deviating(sec)
            obst |= _comp_filter(clean(o2), int(0.2 / a_px), big_px)
            shade |= _comp_filter(sh2, int(2.0 / a_px))
    shade &= ~obst
    out["obstacles_area_m2"] = round(float(obst.sum()) * a_px, 1)
    out["shade_area_m2"] = round(float(shade.sum()) * a_px, 1)

    # free area
    set_m = C["roof_setback_pitched_m"] if rtype == "pitched" else C["roof_setback_flat_m"]
    inner = cv2.distanceTransform(fp.astype(np.uint8), cv2.DIST_L2, 5) > set_m * px_m
    # existing PV: the whole installation (rows with their gaps, as the typology outline) plus 0.3 m
    pv_area = cv2.morphologyEx(pv.astype(np.uint8), cv2.MORPH_CLOSE, _disk(round(1.5 * px_m))).astype(bool) | pv
    blocked = cv2.dilate(obst.astype(np.uint8), _disk(round(C["roof_obstacle_clearance_m"] * px_m))).astype(bool) \
        | cv2.dilate(pv_area.astype(np.uint8), _disk(round(0.3 * px_m))).astype(bool) | shade
    free = inner & ~blocked
    faces = []
    if rtype == "pitched":
        free &= np.abs(side) > C["roof_ridge_m"] * px_m
        n = (-u[1], u[0])                                        # normal of the long axis: points to the side>0 half
        for sgn, half in ((1, half_b), (-1, half_a)):
            az = _az(sgn * n[0], sgn * n[1])
            north = min(az, 360 - az) <= C["roof_exclude_north_deg"]
            faces.append({"azimuth_deg": round(az), "excluded_north": north, "_mask": half})
            if north:
                free &= ~half
    w_px = max(1, int(round(C["roof_min_width_m"] * px_m)))
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, np.ones((w_px, w_px), np.uint8)).astype(bool)
    free = _comp_filter(free, int(C["roof_min_area_m2"] / a_px))

    # potential
    eff_tab = conf["module"].get("efficiency_by_year") or {}
    eff = typology.efficiency_for_year(max(map(int, eff_tab)) if eff_tab else None, conf)
    er = conf["module"]["efficiency_range"]
    F = conf["fill_factor"]
    if rtype == "pitched":
        key, tilt = "flush_pitched", float(C["roof_pitched_tilt_deg"])
    else:
        key = C["roof_flat_layout"] if C["roof_flat_layout"] in F else "east_west"
        tilt = float(conf["tilt"].get(key, {}).get("deg", 12)) if isinstance(conf.get("tilt"), dict) else 12.0
    f, (flo, fhi) = float(F[key]["f"]), F[key].get("range", [F[key]["f"], F[key]["f"]])
    cosv = max(math.cos(math.radians(tilt)), 0.087)
    free_m2 = float(free.sum()) * a_px
    out.update(usable_area_m2=round(free_m2, 1), layout_assumed=key, tilt_assumed_deg=tilt, module_efficiency=eff,
               kwp_potential=round(free_m2 * f / cosv * eff, 1), kwp_pot_low=round(free_m2 * flo / cosv * er[0], 1),
               kwp_pot_high=round(free_m2 * fhi / cosv * max(er[1], eff), 1),
               usable_share=round(free_m2 / roof_m2, 3))
    out["kwp_pot_low"] = min(out["kwp_pot_low"], out["kwp_potential"])
    for fc in faces:
        m2 = float((free & fc.pop("_mask")).sum()) * a_px
        fc.update(usable_area_m2=round(m2, 1), kwp_potential=round(m2 * f / cosv * eff, 1))
    out["faces"] = faces
    return {**out, "_free": free, "_obst": obst}


def to_wgs84(mask: np.ndarray, win, min_px: int = 4, eps: float = 1.0):
    """pixel mask → WGS84 (Multi)Polygon GeoJSON or None"""
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union
    from . import geo
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or cv2.contourArea(c) < min_px:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        holes = []
        j = hier[0][i][2]
        while j != -1:
            h = cv2.approxPolyDP(cnts[j], eps, True).reshape(-1, 2)
            if len(h) >= 3:
                holes.append([win.pixel_to_crs(float(x), float(y)) for x, y in h])
            j = hier[0][j][0]
        p = Polygon([win.pixel_to_crs(float(x), float(y)) for x, y in c], holes).buffer(0)
        if not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    return mapping(geo.reproject(unary_union(polys), win.crs, "EPSG:4326"))
