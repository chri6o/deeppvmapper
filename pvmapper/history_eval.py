"""
Test: does the aerial image time series make the decision "PV yes/no" more accurate? (19.11.2026)

For every reviewed building of a dataset the current model scores the current image and up to `max_years` older flights of
the same window (age.timeline: NRW historical DOP, resolution ≤ age_max_res_m). Per image and classification threshold the
number of PV pixels in the footprint is stored, so decision rules over the series can be compared offline against the
reviewed labels (history_eval.json). No model is changed, nothing is written to PV-Map.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from . import age, annotate, geo, runtime, training
from .sources import blank_fraction

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def pick_years(tl: list, cur_year: int, n: int) -> list:
    """the n most recent older flights, at least one year older than the current image"""
    older = [t for t in tl if t[0] < cur_year]
    return older[-n:] if n > 0 else older


def npx_by_threshold(probs, pix) -> dict:
    out = {}
    for t in THRESHOLDS:
        sel = [pix[i] for i in np.flatnonzero(probs >= t)]
        out[str(t)] = int(len(np.unique(np.concatenate(sel)))) if sel else 0
    return out


def run(job, cfg: dict):
    p = job.params
    from .jobs import Job
    det = runtime.detector(cfg, p.get("profile") or cfg["profile"])
    src_job = Job.load(cfg["jobs_dir"], p.get("dataset_from") or "20261118-222231-56f6")
    ds = annotate.ds_dir(src_job.dir)
    metas, _ = training.reviewed_items(ds)
    if p.get("limit"):
        metas = metas[: int(p["limit"])]
    cat = runtime.catalog(cfg)
    work = os.path.join(job.dir, "images")
    os.makedirs(work, exist_ok=True)
    n_years = int(p.get("max_years", 6))
    job.log(f"{len(metas)} geprüfte Gebäude aus {src_job.id}; je Gebäude bis zu {n_years} ältere Befliegungen")

    def prepare(m):
        """fetch the older flights of one building into the work folder (same window, same size as the current image)"""
        crs, minx, miny, maxx, maxy, w, h = m["window"]
        win = geo.Window(crs, minx, miny, maxx, maxy, int(w), int(h))
        base = cat.sources.get(m["source"])
        if base is None:
            return m, []
        try:
            tl = age.timeline(base, win, cfg)
            cy = age.current_year(base, win, tl)
        except Exception as e:                                       # noqa: BLE001
            return m, [{"error": str(e)[:120]}]
        got = [{"year": cy, "res": None, "date": None, "name": m["name"], "current": True}]
        for y, date, res, s in pick_years(tl, cy, n_years):
            name = f"{m['name']}__{y}"
            path = os.path.join(work, name + ".jpg")
            if not os.path.isfile(path):
                try:
                    img = s.fetch(win)
                except Exception:                                    # noqa: BLE001
                    continue
                if img.shape[:2] != (int(h), int(w)) or blank_fraction(img) > 0.3:
                    continue
                Image.fromarray(img).save(path, quality=92)
                shutil.copy(os.path.join(ds, m["name"] + ".fp.png"), os.path.join(work, name + ".fp.png"))
            got.append({"year": y, "res": res, "date": date, "name": name, "current": False})
        return m, got

    job.total = len(metas) * 2
    job.done = 0
    series = []
    with ThreadPoolExecutor(int(cfg.get("fetch_workers", 4))) as pool:
        for m, got in pool.map(prepare, metas):
            job.check_cancel()
            series.append((m, got))
            job.step(1, **({"ohne_historie": 1} if len(got) <= 1 else {}))
    out = {}
    for m, got in series:
        job.check_cancel()
        imgs = [g for g in got if "name" in g]
        metas_img = [{"id": g["name"], "name": g["name"], "_dir": ds if g["current"] else work} for g in imgs]
        sc = training.score_buildings(det, ds, metas_img)
        out[m["id"]] = {"label": int(m["label"]), "kwp": m.get("pv_kwp"), "kwp_detected": m.get("kwp_detected"),
                        "sub": m.get("sub"), "val": bool(m.get("val")),
                        "series": [{"year": g["year"], "res": g["res"], "date": g["date"], "current": g["current"],
                                    "max_prob": float(sc[g["name"]][0].max()) if len(sc[g["name"]][0]) else 0.0,
                                    "npx": npx_by_threshold(*sc[g["name"]])} for g in imgs]}
        job.step(1)
    with open(os.path.join(job.dir, "history_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"profile": p.get("profile") or cfg["profile"], "resolution_m": det.p.resolution_m,
                   "min_pv_area_m2": float(cfg["min_pv_area_m2"]), "thresholds": THRESHOLDS, "items": out}, f)
    yrs = [len(v["series"]) - 1 for v in out.values()]
    job.counters["gebaeude"] = len(out)
    job.log(f"Fertig: {len(out)} Gebäude, im Mittel {np.mean(yrs) if yrs else 0:.1f} ältere Bilder je Gebäude "
            "(history_eval.json).")


# ---------------------------------------------------------------------------------------------------------------------
# Test: near infrared as a check of the detections (19.11.2026). Modules absorb NIR (dark), tiles, slate, bitumen and vegetation
# reflect it; the NIR brightness inside the detected PV is compared with the rest of the roof.

def nir_stats(nir, fp, sel, rgb=None) -> dict:
    """NIR (0–255) inside the detected pixels `sel` (flat indices) and on the rest of the footprint"""
    flat = nir.reshape(-1).astype(np.float32)
    fpi = np.flatnonzero(fp.reshape(-1))
    if len(sel) == 0 or len(fpi) == 0:
        return {}
    rest = np.setdiff1d(fpi, sel, assume_unique=False)
    pv = float(np.median(flat[sel]))
    roof = float(np.median(flat[rest])) if len(rest) >= 50 else None
    out = {"nir_pv": round(pv, 1), "nir_roof": round(roof, 1) if roof is not None else None,
           "nir_ratio": round(pv / roof, 3) if roof else None, "n_px": int(len(sel))}
    if rgb is not None:                                          # brightness in the visible image for comparison
        g = rgb.reshape(-1, 3).astype(np.float32).mean(axis=1)
        out["vis_pv"] = round(float(np.median(g[sel])), 1)
        out["nir_vis"] = round(pv / max(1.0, out["vis_pv"]), 3)
    return out


def run_nir(job, cfg: dict):
    p = job.params
    from dataclasses import replace
    from .jobs import Job
    det = runtime.detector(cfg, p.get("profile") or cfg["profile"])
    src_job = Job.load(cfg["jobs_dir"], p.get("dataset_from") or "20261118-222231-56f6")
    ds = annotate.ds_dir(src_job.dir)
    metas, _ = training.reviewed_items(ds)
    cat = runtime.catalog(cfg)
    layer = p.get("nir_layer", "nw_dop_nir")
    work = os.path.join(job.dir, "nir")
    os.makedirs(work, exist_ok=True)
    job.log(f"{len(metas)} geprüfte Gebäude; Infrarot aus Ebene {layer}")

    def fetch(m):
        base = cat.sources.get(m["source"])
        if base is None or not base.url:
            return m, None
        crs, minx, miny, maxx, maxy, w, h = m["window"]
        win = geo.Window(crs, minx, miny, maxx, maxy, int(w), int(h))
        path = os.path.join(work, m["name"] + ".png")
        if os.path.isfile(path):
            return m, np.asarray(Image.open(path).convert("L"))
        try:
            ns = replace(base, id=base.id + "-nir", layers=layer)
            ns.user_agent = base.user_agent
            img = ns.fetch(win)
        except Exception:                                        # noqa: BLE001
            return m, None
        if img.shape[:2] != (int(h), int(w)):
            return m, None
        g = img[..., 0] if img.ndim == 3 else img
        Image.fromarray(g).save(path)
        return m, g

    job.total = len(metas)
    job.done = 0
    out = {}
    with ThreadPoolExecutor(int(cfg.get("fetch_workers", 4))) as pool:
        got = list(pool.map(fetch, metas))
    for m, nir in got:
        job.check_cancel()
        if nir is None:
            job.step(1, ohne_infrarot=1)
            continue
        img, fp, _ = training.load_arrays(ds, m)
        probs, pix = training.score_buildings(det, ds, [m])[m["id"]]
        rec = {"label": int(m["label"]), "kwp": m.get("pv_kwp"), "kwp_detected": m.get("kwp_detected"), "val": bool(m.get("val")),
               "max_prob": float(probs.max()) if len(probs) else 0.0, "by_threshold": {}}
        for t in THRESHOLDS:
            sel = [pix[i] for i in np.flatnonzero(probs >= t)]
            idx = np.unique(np.concatenate(sel)) if sel else np.array([], int)
            rec["by_threshold"][str(t)] = nir_stats(nir, fp, idx, img)
        out[m["id"]] = rec
        job.step(1)
    with open(os.path.join(job.dir, "nir_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"layer": layer, "min_pv_area_m2": float(cfg["min_pv_area_m2"]), "resolution_m": det.p.resolution_m,
                   "thresholds": THRESHOLDS, "items": out}, f)
    job.counters["gebaeude"] = len(out)
    job.log(f"Fertig: {len(out)} Gebäude mit Infrarot (nir_eval.json).")
