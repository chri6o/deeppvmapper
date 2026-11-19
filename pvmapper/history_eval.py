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
