"""
Sample review of detections ("Stichprobe prüfen", PVMAPPER.md): how many of the PV hits of a run are right, and which confusers make the
false ones (dormers, shadows, skylights, solar thermal, dark roofing …).

1. sample(job, n): n random PV hits of a run (seeded, reproducible) → a dataset (training job with prepare_only) marked `sample`.
2. In the mask editor every image gets a cause. A false hit ("Gaube", "Schatten", …) moves the detected areas to the "no PV" layer and
   sets the label to "no PV": the error becomes a hard negative for the next training and goes to PV-Map as checked by hand.
3. summary(dataset): precision with a 95 % Wilson interval and the share of each cause.
"""

from __future__ import annotations

import math
import os
import random
import sqlite3
import time

from . import annotate

CAUSES = {
    "richtig": ("richtig", True),
    "teilweise": ("teilweise richtig (Fläche korrigieren)", True),
    "gaube": ("Gaube", False),
    "schatten": ("Schatten", False),
    "dachfenster": ("Dachfenster / Lichtkuppel", False),
    "solarthermie": ("Solarthermie", False),
    "dach_dunkel": ("dunkle Dachfläche", False),
    "sonstiges": ("sonstiges", False),
}


def sample_ids(job_dir: str, n: int = 100, seed: int | None = None) -> list[str]:
    db = sqlite3.connect(os.path.join(job_dir, "results.sqlite"))
    try:
        ids = [r[0] for r in db.execute("SELECT id FROM results WHERE has_pv = 1 AND status != 'fehler' ORDER BY id")]
    finally:
        db.close()
    rnd = random.Random(seed if seed is not None else int(time.time()))
    return rnd.sample(ids, min(int(n), len(ids)))


def set_cause(d: str, name: str, cause: str, note: str | None = None) -> dict:
    """the verdict for one image; a false hit becomes a hard negative (areas → 'no PV' layer, label 0)"""
    if cause not in CAUSES:
        raise ValueError("cause: " + ", ".join(CAUSES))
    meta = annotate.read_meta(d, name)
    ok = CAUSES[cause][1]
    prev = meta.get("review") or {}
    if ok:
        if prev.get("moved") is not None:   # a false hit called right after all: the areas go back to the PV layer
            meta = annotate.save_annotation(d, name, prev["moved"], 1, "done" if cause == "richtig" else "open", "user",
                                            negatives=prev.get("neg_before") or [])
        else:
            meta = annotate.save_annotation(d, name, None, 1, "done" if cause == "richtig" else "open", "user")
        moved = neg_before = None
    elif prev.get("moved") is not None:     # another cause for a false hit: the areas are moved already
        meta = annotate.save_annotation(d, name, None, 0, "done", "user")
        moved, neg_before = prev["moved"], prev.get("neg_before")
    else:
        moved = annotate.current_polygons(d, name) or []
        neg_before = annotate.current_polygons(d, name, "neg") or []
        meta = annotate.save_annotation(d, name, [], 0, "done", "user", negatives=neg_before + moved)
    meta["review"] = {"cause": cause, "ok": ok, "note": (note or "")[:300] or None, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      **({"moved": moved, "neg_before": neg_before} if moved is not None else {})}
    annotate.write_meta(d, meta)
    return meta


def _wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def summary(d: str) -> dict:
    items = annotate.list_items(d)
    rev = [m for m in items if (m.get("review") or {}).get("cause")]
    by = {k: 0 for k in CAUSES}
    for m in rev:
        by[m["review"]["cause"]] += 1
    ok = sum(1 for m in rev if m["review"]["ok"])
    lo, hi = _wilson(ok, len(rev))
    wrong = len(rev) - ok
    return {"n": len(items), "reviewed": len(rev), "open": len(items) - len(rev), "correct": ok, "wrong": wrong,
            "precision": round(ok / len(rev), 3) if rev else None, "ci95": [lo, hi],
            "by_cause": [{"cause": k, "label": CAUSES[k][0], "ok": CAUSES[k][1], "n": v,
                          "share_of_wrong": round(v / wrong, 3) if wrong and not CAUSES[k][1] else None} for k, v in by.items()],
            "causes": {k: v[0] for k, v in CAUSES.items()}}
