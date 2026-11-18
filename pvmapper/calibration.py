"""
Capacity calibration from register capacities (MaStR via PV-Map) – PVMAPPER.md "Leistung kalibrieren".

Optional and explicit: only images whose register capacity the user marked in the mask editor ("Register für Kalibrierung"),
or – when chosen for a whole dataset – every checked image with plausible register data. For each such image the checked PV
mask gives the installations (typology: outline, layout, tilt); the fill factor that would give exactly the register capacity is

    f_plan = kWp_register × cos(tilt) / (outline × efficiency)

The median per layout (at least `min_n` images) replaces the default fill factor of that layout – for the model the dataset was
trained for or the one chosen (model_dir/calibration/<profile>.json). typology.settings() applies it whenever that model is
active. Register figures that are far off the detection (ratio outside 1/3 … 3) are left out and counted, they are more likely
a wrong assignment than a fill factor.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re

import numpy as np

MIN_N = 3
MAX_RATIO = 3.0
F_BOUNDS = (0.25, 1.0)


def _path(model_dir: str, profile: str) -> str:
    return os.path.join(model_dir, "calibration", re.sub(r"[^A-Za-z0-9_.-]+", "_", profile) + ".json")


def load(model_dir: str, profile: str) -> dict | None:
    p = _path(model_dir, profile)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                                # noqa: BLE001
        return None


def save(model_dir: str, profile: str, cal: dict) -> str:
    p = _path(model_dir, profile)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1)
    return p


def remove(model_dir: str, profile: str) -> bool:
    p = _path(model_dir, profile)
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


def apply(conf: dict, cal: dict):
    """calibrated fill factors into the typology settings (the range moves with the value)"""
    F = conf["fill_factor"]
    for key, v in (cal.get("fill_factor") or {}).items():
        if key not in F or not v:
            continue
        old = float(F[key]["f"])
        new = float(v)
        F[key]["f"] = round(new, 3)
        if old > 0 and F[key].get("range"):
            F[key]["range"] = [round(min(1.0, x * new / old), 3) for x in F[key]["range"]]
        F[key]["calibrated"] = True


def register_kwp(meta: dict):
    """(kWp, mean commissioning year by kWp, source) of the register data stored with an image, or (None, None, why)"""
    R = (meta.get("register") or {}).get("buildings") or []
    hits = [b for b in R if b.get("register") and b["register"].get("kwp")]
    if not hits:
        return None, None, "kein Register"
    if len(hits) > 1:
        by_id = [b for b in hits if b.get("distance_m") is None]
        if len(by_id) != 1:
            return None, None, "mehrere Gebäude"
        hits = by_id
    r = hits[0]["register"]
    yrs = {int(y): float(k) for y, k in (r.get("by_year") or {}).items() if str(y).isdigit() and k}
    year = round(sum(y * k for y, k in yrs.items()) / sum(yrs.values())) if yrs else r.get("first_year")
    return float(r["kwp"]), year, "ok"


def item_factor(meta: dict, insts: list, conf: dict) -> dict:
    """one image: register kWp vs. the typology of its checked mask → needed fill factor (plan) for its main layout"""
    from . import typology
    reg, reg_year, why = register_kwp(meta)
    if reg is None:
        return {"skip": why}
    if not insts:
        return {"skip": "keine Maske"}
    year = meta.get("install_year") or reg_year
    eff = typology.efficiency_for_year(year, conf)
    denom, est, area = 0.0, 0.0, 0.0
    by = {}
    for a in insts:
        tilt = a.get("tilt_deg") or 0.0
        o = float(a.get("outline_area_m2") or 0)
        denom += o * eff / max(math.cos(math.radians(min(tilt, 85))), 0.087)
        est += float(a.get("kwp_est") or 0) * eff / float(a.get("module_efficiency") or eff)
        area += o
        key = a["layout"]
        if key == "flush":
            key = "flush_flat" if a.get("roof_shape") == "flat" else "flush_pitched"
        by[key] = by.get(key, 0) + o
    if denom <= 0 or est <= 0:
        return {"skip": "keine Fläche"}
    ratio = reg / est
    if not (1 / MAX_RATIO <= ratio <= MAX_RATIO):
        return {"skip": "Register passt nicht (Faktor %.1f)" % ratio, "ratio": round(ratio, 2)}
    main = max(by, key=by.get)
    return {"f_plan": reg / denom, "layout": main if by[main] >= 0.8 * area and main != "unknown" else None,
            "ratio": round(ratio, 3), "register_kwp": reg, "estimate_kwp": round(est, 1), "year": year,
            "year_basis": "Luftbild" if meta.get("install_year") else ("Register" if reg_year else "Standard")}


def ensure_register(d: str, cfg: dict, use: str = "marked") -> int:
    """register data from PV-Map for the images that will be used and do not have it yet (needs a PV-Map token)"""
    from . import annotate, pvmap
    if not cfg.get("pvmap_token"):
        return 0
    todo = [m["name"] for m in annotate.list_items(d) if m.get("effective_label") == 1 and m.get("status") == "done"
            and (use == "all" or m.get("use_register")) and not m.get("register")]
    if not todo:
        return 0
    try:
        return annotate.register_lookup(d, pvmap.Client.from_config(cfg), todo)["images"]
    except Exception:                                                # noqa: BLE001 – calibrate with what is there
        return 0


def compute(d: str, cfg: dict, use: str = "marked", min_n: int = MIN_N) -> dict:
    """all images of a dataset (annotate.ds_dir) → calibration {fill_factor: {layout: f}, layouts: {…stats}, …}.
    use = 'marked' (only images marked in the editor) | 'all' (every checked PV image with register data)"""
    from . import annotate, typology
    conf = typology.settings({**cfg, "kwp_calibration": False})
    per, ratios, skipped, used = {}, [], {}, []
    for m in annotate.list_items(d):
        if m.get("effective_label") != 1 or m.get("status") != "done":
            continue
        if use == "marked" and not m.get("use_register"):
            continue
        meta = annotate.read_meta(d, m["name"])
        t = annotate.typology_of(d, m["name"], meta, None, {**cfg, "kwp_calibration": False})
        r = item_factor(meta, t.get("_insts") or [], conf)
        if "skip" in r:
            skipped[r["skip"]] = skipped.get(r["skip"], 0) + 1
            continue
        ratios.append(r["ratio"])
        used.append({"id": meta.get("id"), **{k: r[k] for k in ("layout", "ratio", "register_kwp", "estimate_kwp", "year", "year_basis")}})
        if r["layout"]:
            per.setdefault(r["layout"], []).append(r["f_plan"])
    F = conf["fill_factor"]
    layouts, ff = {}, {}
    for k, v in sorted(per.items()):
        q = np.percentile(v, [25, 50, 75])
        layouts[k] = {"n": len(v), "f_median": round(float(q[1]), 3), "f_p25": round(float(q[0]), 3),
                      "f_p75": round(float(q[2]), 3), "default": F.get(k, {}).get("f")}
        if len(v) >= min_n and k in F:
            ff[k] = round(float(min(max(q[1], F_BOUNDS[0]), F_BOUNDS[1])), 3)
    return {"created": _dt.datetime.now().isoformat(timespec="minutes"), "use": use, "n_images": len(used),
            "fill_factor": ff, "layouts": layouts, "skipped": skipped, "min_n": min_n,
            "ratio_median": round(float(np.median(ratios)), 3) if ratios else None, "images": used[:500]}
