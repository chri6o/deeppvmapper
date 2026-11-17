"""
Installation typology, outline, fill factor and capacity estimate (PVMAPPER.md "Anlagentypologie").

Input is what the detection already produces for one building: the aerial image, the final PV mask (module pixels after
post-processing), optionally the segmentation probability, the footprint (clip) and the building's properties. Everything
here is geometry and image analysis on that mask – no further neural network:

  installations   module pixels closer than `group_gap_m` form one installation; its outline is the closed hull of the
                  module rows (gaps between rows belong to the installation, gaps between fields do not)
  rows            strongest periodic pattern (2D FFT) of the mask inside the outline and of the gray image inside the
                  modules, pitch 1.9–8 m; shorter periods are the module grid itself and count as "no rows"
  layout          tilted_mono: rows with gaps (coverage < 0.65), or dense rows with two brightness levels whose faces
                  would point north/south; east_west: dense rows, two brightness levels per period (first harmonic of
                  the phase profile ≥ `two_level`), faces within 35° of east/west; flush: dense without rows, or a
                  regular pattern with one brightness level (module grid); vertical_bifacial: thin lines, wide
                  spacing; else unknown
  azimuth         normal of the rows; mono rows face away from their shadow strip, east-west gives both normals
  fill factor     measured in plan from the row geometry (`derived_pitch`), else the default of the layout (`type_default`)
  capacity        outline × f_real × efficiency, f_real = f_plan / cos(tilt); range from the fill factor and efficiency ranges

Attributes that cannot be seen from above (integration, mounting, securing, carrier, technology) are 'unknown'.
Every attribute has a confidence; below `min_confidence` the value is 'unknown' (or null for numbers).
"""

from __future__ import annotations

import math
import os
import copy

import cv2
import numpy as np
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "typology.yml")

LAYOUTS = ("flush", "tilted_mono", "east_west", "vertical_bifacial", "mixed", "unknown")
TILT_CLASSES = ("<5", "5-20", "21-40", "41-60", "61-89", "90", "non_uniform")
REVIEW = ("unreviewed", "confirmed_manual", "confirmed_crosscheck", "rejected")

SCHEMA = {   # attribute → allowed values (None = free number/text)
    "integration": ("BAPV", "BIPV_A", "BIPV_B", "unknown"),
    "roof_shape": None,
    "mounting_pitched": ("on_roof", "in_roof", "solar_tiles", "unknown", None),
    "securing_flat": ("laid_ballasted", "laid_aerodynamic", "anchored", "unknown", None),
    "carrier": ("rail_rack", "tray", "green_roof", "none", "unknown", None),
    "layout": LAYOUTS,
    "row_alignment": ("aligned", "rotated", "n/a"),
    "module_format": ("portrait", "landscape", "mixed", "unknown"),
    "technology": ("standard", "bifacial", "thin_film_flexible", "pvt", "unknown"),
    "tilt_class": TILT_CLASSES + (None,),
    "fill_factor_source": ("counted", "derived_pitch", "type_default"),
    "review_status": REVIEW,
}


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def settings(cfg: dict | None = None) -> dict:
    """typology.yml, optionally replaced by `typology_file` (path) in the main settings; cached per path"""
    p = (cfg or {}).get("typology_file") or CONFIG_PATH
    st = os.stat(p).st_mtime
    hit = _CACHE.get(p)
    if hit is None or hit[0] != st:
        hit = _CACHE[p] = (st, _load(p))
    return copy.deepcopy(hit[1])


_CACHE: dict = {}


# ---------------------------------------------------------------------------
# formulas (unit-tested)
# ---------------------------------------------------------------------------

def f_row(length_m: float, tilt_deg: float, pitch_m: float) -> float:
    """row factor in plan view: projected module depth / row pitch (front edge to front edge)"""
    return length_m * math.cos(math.radians(tilt_deg)) / pitch_m


def f_cross(width_m: float, gap_m: float) -> float:
    """cross factor within a row: module width / (width + gap)"""
    return width_m / (width_m + gap_m)


def fill_factor_pitch(length_m, tilt_deg, pitch_m, width_m, gap_m) -> float:
    """f = f_row × f_cross (plan view). For east-west, length_m is the depth of a back-to-back pair in plan and pitch_m the
    pair-to-pair distance including aisles; pass tilt_deg=0 when length_m is already projected."""
    return f_row(length_m, tilt_deg, pitch_m) * f_cross(width_m, gap_m)


def real_from_plan(f_plan: float, tilt_deg: float | None) -> float:
    """real module area per outline area: the plan value divided by cos(tilt)"""
    if not tilt_deg:
        return f_plan
    return f_plan / max(math.cos(math.radians(min(float(tilt_deg), 85.0))), 0.087)


def capacity_kwp(outline_m2: float, f_real: float, efficiency: float) -> float:
    """kWp = outline area × real fill factor × efficiency × 1 kW/m² (STC irradiance)"""
    return outline_m2 * f_real * efficiency


def efficiency_for_year(year: int | None, conf: dict) -> float:
    m = conf["module"]
    if not year:
        return float(m["efficiency"])
    table = sorted((int(k), float(v)) for k, v in (m.get("efficiency_by_year") or {}).items())
    if not table:
        return float(m["efficiency"])
    best = table[0][1]                          # older than the table: its first value
    for y, e in table:
        if int(year) >= y:
            best = e
    return best


def tilt_class(tilt_deg: float | None) -> str | None:
    """MaStR classes (registration help as of 05.12.2025)"""
    if tilt_deg is None:
        return None
    t = float(tilt_deg)
    if t < 5:
        return "<5"
    if t <= 20:
        return "5-20"
    if t <= 40:
        return "21-40"
    if t <= 60:
        return "41-60"
    if t < 90:
        return "61-89"
    return "90"


def tilt_class_legacy(tc: str | None) -> str | None:
    """for older MaStR entries: <20 / 20-40 / 40-60 / >60"""
    return {"<5": "<20", "5-20": "<20", "21-40": "20-40", "41-60": "40-60", "61-89": ">60", "90": ">60"}.get(tc or "")


COMPASS8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
MASTR8 = ("Nord", "Nord-Ost", "Ost", "Süd-Ost", "Süd", "Süd-West", "West", "Nord-West")


def compass8(az: float | None) -> str | None:
    if az is None:
        return None
    return COMPASS8[int(((float(az) % 360) + 22.5) // 45) % 8]


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def osm_tags(inst: dict) -> dict:
    """OSM tags for one installation (mounting type has no OSM tag and stays in PV-Map)"""
    t = {"power": "generator", "generator:source": "solar", "generator:method": "photovoltaic",
         "generator:type": "solar_photovoltaic_panel", "location": "roof"}
    o1, o2 = compass8(inst.get("azimuth_deg")), compass8(inst.get("azimuth2_deg"))
    if o1:
        t["generator:orientation"] = f"{o1};{o2}" if o2 else o1
    if inst.get("module_count"):
        t["generator:solar:modules"] = str(int(inst["module_count"]))
    return t


def mastr_fields(inst: dict) -> dict:
    """MaStR solar unit fields (Hauptausrichtung, Neigungswinkel, Nebenausrichtung, Anzahl der Module).
    East-west: east is the main orientation, west the secondary one (MaStR registers them as two orientations)."""
    def m8(az):
        return None if az is None else MASTR8[COMPASS8.index(compass8(az))]
    a1, a2 = inst.get("azimuth_deg"), inst.get("azimuth2_deg")
    if a2 is not None and a1 is not None and abs(((a1 - 90 + 180) % 360) - 180) > abs(((a2 - 90 + 180) % 360) - 180):
        a1, a2 = a2, a1                                          # east first
    tc = inst.get("tilt_class")
    return {"Hauptausrichtung": m8(a1), "Neigungswinkel der Hauptausrichtung": tc,
            "Nebenausrichtung": m8(a2), "Neigungswinkel der Nebenausrichtung": tc if a2 is not None else None,
            "Anzahl der Module": inst.get("module_count")}


def validate(inst: dict) -> list[str]:
    """schema check: allowed values, ranges, null where an attribute does not apply"""
    err = []
    for k, allowed in SCHEMA.items():
        if allowed is not None and inst.get(k) not in allowed:
            err.append(f"{k}={inst.get(k)!r}")
    for k, lo, hi in (("azimuth_deg", 0, 360), ("azimuth2_deg", 0, 360), ("fill_factor", 0, 1.0001),
                      ("tilt_deg", 0, 90)):
        v = inst.get(k)
        if v is not None and not (lo <= float(v) < hi or (k == "tilt_deg" and float(v) == 90)):
            err.append(f"{k}={v}")
    rs = inst.get("roof_shape")
    if rs == "flat" and inst.get("mounting_pitched") is not None:
        err.append("mounting_pitched on a flat roof")
    if rs not in (None, "flat", "unknown") and inst.get("securing_flat") is not None:
        err.append("securing_flat on a pitched roof")
    if inst.get("kwp_low") is not None and not (inst["kwp_low"] <= inst["kwp_est"] <= inst["kwp_high"]):
        err.append("kwp range")
    for k, v in (inst.get("confidence") or {}).items():
        if not 0 <= float(v) <= 1:
            err.append(f"confidence {k}={v}")
    return err


# ---------------------------------------------------------------------------
# image analysis
# ---------------------------------------------------------------------------

def _orientation(m: np.ndarray):
    """dominant direction across the rows (unit vector in image coords, y down) and coherence 0–1"""
    f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    if jxx + jyy <= 1e-9:
        return (1.0, 0.0), 0.0
    th = 0.5 * math.atan2(2 * jxy, jxx - jyy)
    lam = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy)
    return (math.cos(th), math.sin(th)), lam / (jxx + jyy)


def _profile(values: np.ndarray, t: np.ndarray, n: int) -> np.ndarray:
    s = np.bincount(t, weights=values, minlength=n)
    c = np.bincount(t, minlength=n)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _period(prof: np.ndarray, lo: int, hi: int):
    """first clear autocorrelation peak in [lo, hi] pixels → (period px, strength)"""
    x = np.nan_to_num(prof - np.nanmean(prof))
    if len(x) < 2 * lo or not x.any():
        return None, 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(hi, len(ac) - 2)
    best, bv = None, 0.0
    for k in range(max(lo, 1), hi + 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > bv:
            best, bv = k, float(ac[k])
            break                                                # first peak = fundamental period
    return best, bv


def _fft_peak(arr: np.ndarray, valid: np.ndarray, lo_px: float, hi_px: float):
    """strongest periodic pattern of arr inside `valid` with a period between lo_px and hi_px:
    (unit vector across the pattern, period px, strength = peak / median power in the band)"""
    a = np.where(valid, arr - arr[valid].mean(), 0).astype(np.float32)
    h, w = a.shape
    H, W = cv2.getOptimalDFTSize(max(h, 64)), cv2.getOptimalDFTSize(max(w, 64))
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32) if h > 8 and w > 8 else 1.0
    F = np.abs(np.fft.fft2(a * win, s=(H, W)))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    band = (r >= 1.0 / hi_px) & (r <= 1.0 / lo_px)
    if not band.any():
        return None, None, 0.0
    Fb = np.where(band, F, 0)
    k = np.unravel_index(int(np.argmax(Fb)), Fb.shape)
    med = float(np.median(F[band])) + 1e-9
    vx, vy = float(fx[0, k[1]]), float(fy[k[0], 0])
    rr = math.hypot(vx, vy)
    if rr == 0:
        return None, None, 0.0
    return (vx / rr, vy / rr), 1.0 / rr, float(Fb[k]) / med


def _az_of(vec) -> float:
    """compass azimuth of an image direction (x right, y down, north up)"""
    return (math.degrees(math.atan2(vec[0], -vec[1])) + 360) % 360


def _ang_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def analyse_installation(img, mods, outline, res_m, conf, footprint_angle=None, props=None):
    """one installation: mods = module pixels, outline = installation outline (both bool, same crop). Returns attributes."""
    C = conf["classify"]
    px_m = 1.0 / res_m
    area_px = int(outline.sum())
    cover = float(mods.sum()) / max(area_px, 1)
    (ux, uy), coh = _orientation(mods)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lo_px, hi_px = C["period_range_m"][0] * px_m, C["period_range_m"][1] * px_m
    # row pattern: in the mask (rows with gaps) or in the image inside the field (dense pairs, shadow lines)
    cand = []
    for arr, valid in ((mods.astype(np.float32), outline), (gray, mods)):
        if valid.sum() > 50:
            v, per, st = _fft_peak(arr, valid, lo_px, hi_px)
            if v is not None:
                cand.append((st, v, per))
    strength, p_px = 0.0, None
    if cand:
        st, v, per = max(cand, key=lambda c: c[0])
        strength, p_px = st, int(round(per))
        ux, uy = v
    ys, xs = np.nonzero(outline)
    t = np.round(xs * ux + ys * uy).astype(int)
    t -= t.min()
    n = int(t.max()) + 1
    extent_m = n / px_m
    strength_n = min(1.0, max(0.0, (strength - 4.0) / 16.0))          # FFT peak ratio 4 → 0, 20 → 1
    periodic = p_px is not None and p_px >= 2 and strength_n >= C["min_periodicity"] \
        and extent_m / (p_px / px_m) >= C["min_rows"]
    strength = strength_n
    roof = (props or {}).get("roof_shape") or (props or {}).get("roof:shape")

    att = {"confidence": {}, "two_level": None}
    cf = att["confidence"]
    normal = (ux, uy)
    if periodic:
        if cover <= C["vertical_coverage"]:
            layout, lc = "vertical_bifacial", 0.6
        else:
            # brightness of the module pixels by phase within one period: east-west pairs show two levels (a square
            # wave → strong first harmonic); module joints are thin dips that hardly contribute to the first harmonic
            sel = mods[ys, xs]
            ph = ((t[sel] % p_px) * 16 // p_px).astype(int)
            gb = _profile(gray[ys, xs][sel], ph, 16)
            ok = ~np.isnan(gb)
            step = 0.0
            if ok.sum() >= 8:
                g = np.where(ok, gb, np.nanmean(gb))
                a1 = abs(np.fft.rfft(g - g.mean())[1]) * 2 / len(g)
                step = float(a1 / (gray[ys, xs][sel].std() + 1e-6))
            att["two_level"] = round(step, 2)
            near_ew = min(_ang_diff(_az_of(normal), 90), _ang_diff(_az_of(normal), 270)) <= 35
            two = step >= C["two_level"]
            if cover >= C["east_west_coverage"] and two and near_ew:
                layout, lc = "east_west", min(0.9, 0.7 + 0.1 * (step >= 1.5 * C["two_level"]))
            elif cover >= C["east_west_coverage"] and two:
                # two brightness levels, but the faces would point north/south: far more often mono rows towards the
                # sun with gravel between them than an east-west system turned by 90°
                layout, lc = "tilted_mono", 0.6
            elif cover >= C["flush_coverage"]:
                # dense field with a regular pattern but one brightness level: the module grid of a flush array
                layout, lc = "flush", min(0.9, 0.6 + (cover - C["flush_coverage"]) * 2)
            elif cover >= C["east_west_coverage"]:
                layout, lc = "tilted_mono", 0.45                 # dense rows, no second brightness level
            else:
                layout, lc = "tilted_mono", 0.75
        if layout != "flush":
            lc *= min(1.0, 0.4 + strength)
    elif cover >= C["flush_coverage"]:
        layout, lc = "flush", min(0.9, 0.5 + (cover - C["flush_coverage"]) * 2 + 0.2 * (1 - coh))
    else:
        layout, lc = "unknown", 0.3
    att["layout"] = layout if lc >= C["min_confidence"] else "unknown"
    cf["layout"] = round(lc, 2)
    layout = att["layout"]

    # azimuth
    az1 = az2 = None
    if layout == "tilted_mono":
        # the shadow strip lies behind the modules: compare the gap right after the module strip in +t and in -t
        nonmod = outline[ys, xs] & ~mods[ys, xs]
        k = np.where(mods[ys, xs])[0]
        phase = (t % p_px).astype(int)
        cov_ph = _profile(mods[ys, xs].astype(np.float32), phase, p_px)
        gray_ph = _profile(np.where(nonmod, gray[ys, xs], np.nan), phase, p_px)
        mod_ph = np.nan_to_num(cov_ph) > 0.5
        az_a, az_b = _az_of(normal), (_az_of(normal) + 180) % 360
        dark_after = dark_before = None
        if mod_ph.any() and (~mod_ph).any() and len(k):
            idx = np.where(mod_ph)[0]
            # gaps directly after / before the module phase
            after = [(i + 1) % p_px for i in idx if not mod_ph[(i + 1) % p_px]]
            before = [(i - 1) % p_px for i in idx if not mod_ph[(i - 1) % p_px]]
            g = np.nan_to_num(gray_ph, nan=np.nanmean(gray_ph) if np.isfinite(np.nanmean(gray_ph)) else 0)
            dark_after = float(np.mean(g[after])) if after else None
            dark_before = float(np.mean(g[before])) if before else None
        if dark_after is not None and dark_before is not None and abs(dark_after - dark_before) > 8:
            # shadow in +t direction → modules face -t
            az1 = az_b if dark_after < dark_before else az_a
            ac = 0.75
        else:
            az1 = az_a if _ang_diff(az_a, 180) <= _ang_diff(az_b, 180) else az_b    # towards the equator
            ac = 0.5
        cf["azimuth_deg"] = ac
    elif layout == "east_west":
        a = _az_of(normal)
        az1 = a if _ang_diff(a, 90) <= _ang_diff(a, 270) else (a + 180) % 360
        az2 = (az1 + 180) % 360
        cf["azimuth_deg"] = round(cf["layout"], 2)
    elif layout == "vertical_bifacial":
        az1 = _az_of(normal)
        az1 = az1 if _ang_diff(az1, 90) <= _ang_diff(az1, 270) else (az1 + 180) % 360
        az2 = (az1 + 180) % 360
        cf["azimuth_deg"] = 0.5
    if az1 is not None and cf.get("azimuth_deg", 0) < C["min_confidence"]:
        az1 = az2 = None
    att["azimuth_deg"] = None if az1 is None else round(az1)
    att["azimuth2_deg"] = None if az2 is None else round(az2)

    # roof shape: OSM/LoD2 when given; row layouts imply a flat roof
    if roof:
        att["roof_shape"], cf["roof_shape"] = roof, 0.9
    elif layout in ("tilted_mono", "east_west", "vertical_bifacial"):
        att["roof_shape"], cf["roof_shape"] = "flat", 0.8
    else:
        att["roof_shape"], cf["roof_shape"] = "unknown", 0.0
    flat = att["roof_shape"] == "flat"

    # tilt
    T = conf["tilt"]
    key = "flush_flat" if (layout == "flush" and flat) else layout
    if key in T:
        att["tilt_deg"], cf["tilt_deg"] = float(T[key]["deg"]), float(T[key]["confidence"])
    else:
        att["tilt_deg"], cf["tilt_deg"] = None, 0.0
    att["tilt_class"] = tilt_class(att["tilt_deg"]) if cf["tilt_deg"] >= C["min_confidence"] - 0.15 else None
    if att["tilt_class"] is None:
        att["tilt_deg"] = None

    # row alignment against the building edges
    if layout in ("tilted_mono", "east_west", "vertical_bifacial") and footprint_angle is not None:
        if isinstance(footprint_angle, np.ndarray):              # local edges of the footprint around this field
            (ex, ey), ecoh = _orientation(footprint_angle)
            footprint_angle = (_az_of((ex, ey)) + 90) % 180 if ecoh > 0.1 else None
    if layout in ("tilted_mono", "east_west", "vertical_bifacial") and footprint_angle is not None:
        row_dir = (_az_of(normal) + 90) % 180
        d = min(_ang_diff(row_dir, footprint_angle % 180), _ang_diff(row_dir, (footprint_angle + 90) % 180))
        d = min(d, 180 - d)
        att["row_alignment"] = "aligned" if d <= C["aligned_deg"] else "rotated"
        cf["row_alignment"] = 0.8
    else:
        att["row_alignment"] = "n/a"

    # fill factor
    F = conf["fill_factor"]
    if periodic and layout in ("tilted_mono", "east_west", "vertical_bifacial"):
        fp = cover * float(F["f_cross"])
        src = "derived_pitch"
        tol = float(F["measured_tolerance"])
        rng = (max(0.02, fp - tol), min(1.0, fp + tol))
        att["row_pitch_m"] = round(p_px / px_m, 2)
        att["module_depth_m"] = round(cover * p_px / px_m, 2)            # projected depth of a row (or pair)
        if layout == "east_west" and cover < 0.78:
            key = "east_west_aisles"
    else:
        key = ("flush_flat" if flat else "flush_pitched") if layout == "flush" else ("unknown" if layout not in F else layout)
        fp, rng, src = float(F[key]["f"]), tuple(F[key]["range"]), "type_default"
    att["fill_factor"], att["fill_factor_range"], att["fill_factor_source"] = round(fp, 3), [round(rng[0], 3), round(rng[1], 3)], src
    cf["fill_factor"] = round(min(0.9, 0.4 + strength), 2) if src == "derived_pitch" else 0.4

    # attributes that cannot be seen from above at this resolution
    att["integration"] = "BAPV" if layout in ("tilted_mono", "east_west", "vertical_bifacial") else "unknown"
    cf["integration"] = 0.9 if att["integration"] == "BAPV" else 0.0
    att["mounting_pitched"] = None if flat else "unknown"
    att["securing_flat"] = "unknown" if flat else None
    att["carrier"] = "unknown" if flat else None
    att["module_format"] = "unknown"
    att["technology"] = "unknown"
    att["module_count"] = None
    att["row_periodicity"] = round(strength, 2)
    att["coverage"] = round(cover, 3)
    return att


def _footprint_angle(clip):
    if clip is None or not clip.any():
        return None
    cnts, _ = cv2.findContours(clip.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), ang = cv2.minAreaRect(c)
    # direction of the long edge as a compass-like angle 0–180 (consistent with _az_of)
    v = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
    if w < h:
        v = (-v[1], v[0])
    return _az_of(v) % 180


def analyse(img: np.ndarray, mask: np.ndarray, res_m: float, clip: np.ndarray | None = None, props: dict | None = None,
            cfg: dict | None = None, year: int | None = None, review_status: str = "unreviewed") -> list[dict]:
    """all installations of one building → list of dicts with pixel outline ('outline_px') and attributes"""
    conf = settings(cfg)
    C = conf["classify"]
    m = mask.astype(bool)
    if clip is not None:
        m &= clip
    if not m.any():
        return []
    px_m = 1.0 / res_m
    gap = max(3, int(round(C["group_gap_m"] * px_m)))
    grp = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)
    fa = _footprint_angle(clip)
    eff = efficiency_for_year(year, conf)
    er = conf["module"]["efficiency_range"]
    roof_area = float(clip.sum()) * res_m * res_m if clip is not None else None
    out = []
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        pad = gap
        sl = (slice(max(0, y - pad), min(H, y + h + pad)), slice(max(0, x - pad), min(W, x + w + pad)))
        mods = m[sl] & (lab[sl] == i)
        if mods.sum() * res_m * res_m < 2.0:
            continue
        # outline: rows closed across their gaps (≤ group gap), small holes filled
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
        outline = cv2.morphologyEx(mods.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) | mods
        if clip is not None:
            outline &= clip[sl]
        # footprint edges around this field (a wide crop: the nearest building edges)
        e = 8 * gap
        sl_e = (slice(max(0, y - e), min(H, y + h + e)), slice(max(0, x - e), min(W, x + w + e)))
        att = analyse_installation(img[sl], mods, outline, res_m, conf,
                                   clip[sl_e].astype(np.uint8) if clip is not None else fa, props)
        outline_m2 = float(outline.sum()) * res_m * res_m
        f_plan = att["fill_factor"]
        tilt = att["tilt_deg"] if att["tilt_deg"] is not None else 0.0
        f_real = real_from_plan(f_plan, tilt)
        lo_f, hi_f = att["fill_factor_range"]
        att.update(outline_area_m2=round(outline_m2, 1), module_area_m2=round(float(mods.sum()) * res_m * res_m, 1),
                   module_efficiency=eff,
                   kwp_est=round(capacity_kwp(outline_m2, f_real, eff), 1),
                   kwp_low=round(capacity_kwp(outline_m2, real_from_plan(lo_f, tilt), er[0]), 1),
                   kwp_high=round(capacity_kwp(outline_m2, real_from_plan(hi_f, tilt), er[1]), 1),
                   roof_occupancy=round(outline_m2 / roof_area, 3) if roof_area else None,
                   roof_occupancy_basis="footprint" if roof_area else None,
                   review_status=review_status)
        att["kwp_low"] = min(att["kwp_low"], att["kwp_est"])
        att["kwp_high"] = max(att["kwp_high"], att["kwp_est"])
        cnts, _ = cv2.findContours(outline.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        att["outline_px"] = [[int(px + sl[1].start), int(py + sl[0].start)] for px, py in c]   # plain ints (JSON)
        out.append(att)
    out.sort(key=lambda a: -a["outline_area_m2"])
    for k, a in enumerate(out):
        a["index"] = k
    return out


def imagery_year(src) -> int | None:
    """year of the imagery, from the source (`year`) or its id (de-nw-2024, nl-pdok-hr-2025)"""
    import re
    y = getattr(src, "year", None)
    if y:
        return int(y)
    m = re.search(r"(?:^|-)(20\d\d)(?:-|$)", getattr(src, "id", "") or "")
    return int(m.group(1)) if m else None


def for_window(img, mask, win, res_m, src, owner_id: str, clip=None, props=None, cfg=None) -> list[dict]:
    """analyse() + to_wgs84() for one image window: GeoJSON features of the installations ([] if none)"""
    y = imagery_year(src)
    insts = analyse(img, mask, res_m, clip, props, cfg, year=y)
    return to_wgs84(insts, win, owner_id, {"source": getattr(src, "id", None), "year": y})


def summarise(props_list: list[dict]) -> dict:
    """installations of one building / field → kwp_est, kwp_low, kwp_high, layout (dominant ≥ 80 % or 'mixed')"""
    if not props_list:
        return {}
    by: dict = {}
    for p in props_list:
        by[p["layout"]] = by.get(p["layout"], 0) + (p.get("outline_area_m2") or 0)
    tot = sum(by.values()) or 1
    main = max(by, key=by.get)
    out = {"kwp_est": round(sum(p["kwp_est"] for p in props_list), 1),
           "kwp_low": round(sum(p["kwp_low"] for p in props_list), 1),
           "kwp_high": round(sum(p["kwp_high"] for p in props_list), 1),
           "layout": main if len(by) == 1 or by[main] >= 0.8 * tot else "mixed"}
    aged = [p for p in props_list if p.get("install_year")]
    if aged:                                    # build year of the largest dated installation
        big = max(aged, key=lambda p: p.get("outline_area_m2") or 0)
        out["install_year"] = big["install_year"]
    return out


def group_systems(props_list: list[dict]) -> list[dict]:
    """parts (fields) on one building → systems ("Anlagen"): parts that appeared on the same flights belong together.
    Build intervals are half-open (after flight `install_year_low`, at the latest on flight `install_year_high`); parts whose
    intervals overlap are one system, a part without age information joins the largest system. Sets p["system"] (1, 2, …)
    and returns per system: parts, kwp_est/low/high, layout, install_year (largest part), install_year_low/high."""
    INF = 10 ** 6
    order = sorted(range(len(props_list)), key=lambda k: -(props_list[k].get("outline_area_m2") or 0))
    systems: list = []
    undated = []
    for k in order:
        p = props_list[k]
        if p.get("install_year_basis") != "imagery":
            undated.append(k)
            continue
        lo = p.get("install_year_low")
        lo = -INF if lo is None else lo
        hi = p.get("install_year_high")
        hi = INF if hi is None else hi
        for sy in systems:
            if max(sy["lo"], lo) < min(sy["hi"], hi):        # appeared between the same two flights
                sy["lo"], sy["hi"] = max(sy["lo"], lo), min(sy["hi"], hi)
                sy["parts"].append(k)
                break
        else:
            systems.append({"lo": lo, "hi": hi, "parts": [k]})
    if undated:
        if systems:
            systems[0]["parts"] += undated
        else:
            systems.append({"lo": -INF, "hi": INF, "parts": undated})
    out = []
    for n, sy in enumerate(systems, 1):
        parts = [props_list[k] for k in sy["parts"]]
        for p in parts:
            p["system"] = n
        sm = summarise(parts)
        sm.update(system=n, parts=len(parts), outline_area_m2=round(sum(p.get("outline_area_m2") or 0 for p in parts), 1),
                  install_year_low=None if sy["lo"] == -INF else sy["lo"],
                  install_year_high=None if sy["hi"] == INF else sy["hi"])
        out.append(sm)
    return out


def map_features(feats: list[dict], owner_id: str) -> list[dict]:
    """installation features → slim features for the map layer 'Anlagen nach Bauform' (kind 'inst')"""
    out = []
    for k, f in enumerate(feats):
        p = f["properties"]
        out.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "id": owner_id, "kind": "inst", "n": k + 1, **{x: p.get(x) for x in (
                "layout", "azimuth_deg", "azimuth2_deg", "tilt_class", "fill_factor", "fill_factor_source",
                "outline_area_m2", "kwp_est", "kwp_low", "kwp_high", "install_year", "install_year_low",
                "install_year_high", "module_efficiency", "system")}}})
    return out


def to_wgs84(insts: list[dict], win, building_id: str, imagery: dict | None = None) -> list[dict]:
    """pixel outlines → GeoJSON features (WGS84) with installation_id / building_id"""
    from shapely.geometry import Polygon, mapping
    from . import geo
    feats = []
    for a in insts:
        pts = a.get("outline_px") or []
        if len(pts) < 3:
            continue
        g = Polygon([win.pixel_to_crs(x, y) for x, y in pts]).buffer(0)
        if g.is_empty:
            continue
        props = {k: v for k, v in a.items() if k not in ("outline_px", "index")}
        props.update(installation_id=f"{building_id}#i{a['index']}", building_id=building_id,
                     imagery_source=(imagery or {}).get("source"), imagery_year=(imagery or {}).get("year"))
        feats.append({"type": "Feature", "geometry": mapping(geo.reproject(g, win.crs, "EPSG:4326")), "properties": props})
    return feats


# ---------------------------------------------------------------------------------------------------------------------
# calibration against registered capacities (MaStR via PV-Map: props.pv_kwp)

def calibrate(rows: list[dict], conf: dict | None = None, min_kwp: float = 3.0, max_ratio: float = 4.0) -> dict:
    """rows: {"kwp_register", "kwp_est", "kwp_area" (old area × factor), "installations": [properties…]}.
    Buildings with one layout only (≥ 90 % of the outline) give a fill factor estimate for that layout:
    f_plan needed = register kWp × cos β / (outline × efficiency). Returns per layout: n, median f, IQR, current
    default, and for all buildings the errors of the typology estimate and of the old area factor."""
    conf = conf or settings()
    F = conf["fill_factor"]
    per: dict = {}
    err_t, err_a, inside = [], [], 0
    for r in rows:
        reg, insts = float(r.get("kwp_register") or 0), r.get("installations") or []
        if reg < min_kwp or not insts or not r.get("kwp_est"):
            continue
        ratio = r["kwp_est"] / reg
        if not (1 / max_ratio <= ratio <= max_ratio):       # other building, roof partly hidden, register error
            continue
        err_t.append(ratio - 1)
        if r.get("kwp_area"):
            err_a.append(r["kwp_area"] / reg - 1)
        lo = sum(i.get("kwp_low") or 0 for i in insts)
        hi = sum(i.get("kwp_high") or 0 for i in insts)
        inside += lo <= reg <= hi
        tot = sum(i["outline_area_m2"] for i in insts)
        lay: dict = {}
        for i in insts:
            lay[i["layout"]] = lay.get(i["layout"], 0) + i["outline_area_m2"]
        main = max(lay, key=lay.get)
        if lay[main] < 0.9 * tot or main == "unknown":
            continue
        roof = insts[0].get("roof_shape")
        key = ("flush_flat" if roof == "flat" else "flush_pitched") if main == "flush" else main
        tilt = np.mean([i["tilt_deg"] for i in insts if i.get("tilt_deg") is not None] or [0.0])
        eff = insts[0].get("module_efficiency") or conf["module"]["efficiency"]
        f = reg * math.cos(math.radians(min(tilt, 85))) / (tot * eff)
        per.setdefault(key, []).append(f)

    def stats(e):
        if not e:
            return None
        a = np.abs(e)
        return {"n": len(e), "median_error": round(float(np.median(e)), 3), "mape": round(float(a.mean()), 3),
                "within_20pct": round(float((a <= 0.2).mean()), 3)}
    out = {"layouts": {}, "typology": stats(err_t), "area_factor": stats(err_a),
           "register_inside_range": round(inside / len(err_t), 3) if err_t else None}
    for k, v in sorted(per.items()):
        q = np.percentile(v, [25, 50, 75])
        cur = F.get(k, {})
        out["layouts"][k] = {"n": len(v), "f_median": round(float(q[1]), 3), "f_p25": round(float(q[0]), 3),
                             "f_p75": round(float(q[2]), 3), "current": cur.get("f")}
    return out


def calibrate_job(db, kwp_per_m2: float = 0.20, conf: dict | None = None) -> dict:
    """reads a job's results.sqlite (buildings with register kWp and installations) → calibrate()"""
    import json as _j
    rows = []
    cols = [d[0] for d in db.execute("SELECT * FROM results LIMIT 0").description]
    if "installations" not in cols:
        return calibrate([])
    for kwp, inst, props, area, method in db.execute(
            "SELECT kwp_est, installations, props, pv_area_m2, kwp_method FROM results WHERE status='pv'"):
        pr = _j.loads(props or "{}")
        if method != "typology" or not pr.get("pv_kwp"):
            continue
        rows.append({"kwp_register": pr["pv_kwp"], "kwp_est": kwp, "kwp_area": (area or 0) * kwp_per_m2,
                     "installations": [f["properties"] for f in _j.loads(inst or "[]")]})
    return calibrate(rows, conf)
