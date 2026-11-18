"""
Start a building check again ("Neu starten…" in the results tab) – e.g. after retraining, with another model.

"Fortsetzen" (resume) continues a cancelled job and skips the buildings already done; a restart is a new job that computes everything
again. Which buildings:
  all   the same input as before (file, PostGIS query, PV-Map list). A PV-Map work-list run (sync) has no fixed input: its restart takes
        the buildings this run checked (the rest stays in PV-Map's work list anyway).
  pv    only the buildings this run found with PV
  none  only the buildings this run found without PV
Buildings from the results are written to `restart_input.geojson` in the new job's folder (footprint and properties as read before).
A restart of a PV-Map run sends its results to PV-Map again and computes even with the same aerial image (`force`).
"""

from __future__ import annotations

import json
import os
import sqlite3

from shapely import wkt as shp_wkt
from shapely.geometry import Point, mapping

from .jobs import Job

ONLY = {"all": None, "pv": "has_pv = 1", "none": "has_pv = 0"}
KEEP = ("input", "id_field", "query", "layer", "sources", "postproc", "cross_check", "typology", "age_check", "roof_potential",
        "save_chips", "seg_tta", "push")


def _results_input(old: Job, path: str, where: str | None) -> int:
    """the buildings of an earlier run (results.sqlite) as GeoJSON → number of buildings"""
    db = sqlite3.connect(os.path.join(old.dir, "results.sqlite"))
    try:
        rows = db.execute("SELECT id, footprint_wkt, lon, lat, props FROM results WHERE status != 'fehler'"
                          + (f" AND {where}" if where else "")).fetchall()
    finally:
        db.close()
    feats = []
    for bid, fp, lon, lat, props in rows:
        g = shp_wkt.loads(fp) if fp else (Point(lon, lat) if lon is not None and lat is not None else None)
        if g is None:
            continue
        p = json.loads(props or "{}")
        p["id"] = bid
        feats.append({"type": "Feature", "id": bid, "geometry": mapping(g), "properties": p})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    return len(feats)


def restart(old: Job, jobs_dir: str, profile: str | None = None, only: str = "all") -> Job:
    if old.kind not in ("buildings", "sync"):
        raise ValueError("Neu starten geht für „Gebäude prüfen“ und den PV-Map-Abgleich.")
    if only not in ONLY:
        raise ValueError("only: all, pv oder none")
    if old.status in ("running", "queued"):
        raise ValueError("Der Lauf läuft noch – erst abbrechen.")
    p = old.params or {}
    params = {k: p[k] for k in KEEP if k in p}
    params["profile"] = profile or p.get("profile")
    params["restart_of"] = old.id
    params["restart_only"] = only
    from_queue = old.kind == "sync" or str(p.get("input") or "").startswith("pvmapqueue:")
    job = Job(jobs_dir, "buildings", params)
    if only != "all" or from_queue:
        path = os.path.join(job.dir, "restart_input.geojson")
        n = _results_input(old, path, ONLY[only])
        if not n:
            import shutil
            shutil.rmtree(job.dir, ignore_errors=True)
            raise ValueError("Keine Gebäude für diesen Neustart.")
        params.update(input=path, id_field="id")
        params.pop("query", None); params.pop("layer", None)
        if from_queue:   # results go back to PV-Map, computed again even with the same aerial image
            params.update(push=True, force=True)
    if not params.get("input"):
        raise ValueError("Für diesen Lauf ist die Gebäudeliste nicht mehr bekannt.")
    job.params = params
    job.save()
    return job
