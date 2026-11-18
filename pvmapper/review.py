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


# -- thresholds for 'possible' from the reviewed sample ----------------------------------------------------------------

AREA_STEPS = [6, 10, 15, 20, 30, 50]          # m²
SHARE_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, None]
PROB_STEPS = [0.6, 0.7, 0.8, 0.9, None]


def _features(d: str, results_db: str) -> list[dict]:
    """reviewed items of a sample with the run's values (area, share of the roof, score, cross-check)"""
    rev = [m for m in annotate.list_items(d) if (m.get("review") or {}).get("cause")]
    if not rev or not os.path.exists(results_db):
        return []
    db = sqlite3.connect(results_db)
    db.row_factory = sqlite3.Row
    cols = {r[1] for r in db.execute("PRAGMA table_info(results)")}
    out = []
    try:
        for m in rev:
            r = db.execute("SELECT * FROM results WHERE id = ?", [m["id"]]).fetchone()
            if r is None:
                continue
            area, fp = r["pv_area_m2"] or 0, r["footprint_m2"]
            share = r["pv_share"] if "pv_share" in cols and r["pv_share"] is not None else (area / fp if fp else None)
            out.append({"id": m["id"], "ok": bool(m["review"]["ok"]), "cause": m["review"]["cause"], "area": area,
                        "share": share, "prob": r["max_prob"] or 0, "check": r["check"] if "check" in cols else None})
    finally:
        db.close()
    return out


def rule_check(d: str, results_db: str, max_true_loss: float = 0.05) -> dict:
    """grid of rules 'area < A and share < S and score < P' → share of false hits caught and of right hits lost;
    best = most false hits caught with at most `max_true_loss` of the right hits (in the sample) lost"""
    xs = _features(d, results_db)
    ok, bad = [x for x in xs if x["ok"]], [x for x in xs if not x["ok"]]
    if not ok or not bad:
        return {"n": len(xs), "richtig": len(ok), "falsch": len(bad), "regeln": [], "beste": None,
                "hinweis": "Zu wenige geprüfte Treffer: es braucht richtige und falsche."}

    def hit(x, a, s, p):
        return x["area"] < a and (s is None or (x["share"] is not None and x["share"] < s)) and (p is None or x["prob"] < p)
    rules = []
    for a in AREA_STEPS:
        for s in SHARE_STEPS:
            for p in PROB_STEPS:
                caught = sum(hit(x, a, s, p) for x in bad)
                lost = sum(hit(x, a, s, p) for x in ok)
                rules.append({"max_area_m2": a, "max_share": s, "max_prob": p, "falsch_erfasst": caught,
                              "anteil_falsch": round(caught / len(bad), 3), "richtig_verloren": lost,
                              "anteil_richtig": round(lost / len(ok), 3)})
    rules.sort(key=lambda r: (-r["falsch_erfasst"], r["richtig_verloren"], r["max_area_m2"]))
    good = [r for r in rules if r["anteil_richtig"] <= max_true_loss and r["falsch_erfasst"] > 0]
    best = good[0] if good else None
    prec_before = len(ok) / len(xs)
    prec_after = None
    if best:
        keep_ok = len(ok) - best["richtig_verloren"]
        keep_bad = len(bad) - best["falsch_erfasst"]
        prec_after = round(keep_ok / max(keep_ok + keep_bad, 1), 3)
    by_cause: dict[str, int] = {}
    if best:
        for x in bad:
            if hit(x, best["max_area_m2"], best["max_share"], best["max_prob"]):
                by_cause[x["cause"]] = by_cause.get(x["cause"], 0) + 1
    return {"n": len(xs), "richtig": len(ok), "falsch": len(bad), "genauigkeit_vorher": round(prec_before, 3),
            "genauigkeit_nachher": prec_after, "beste": best, "erfasst_nach_ursache": by_cause,
            "regeln": rules[:15], "max_verlust": max_true_loss,
            "werte": {"richtig": _summ(ok), "falsch": _summ(bad)}}


def _summ(xs):
    import statistics as st

    def med(k):
        v = [x[k] for x in xs if x[k] is not None]
        return round(st.median(v), 3) if v else None
    return {"flaeche_m2": med("area"), "anteil_dach": med("share"), "score": med("prob")}


def rule_effect(results_db: str, rule: dict) -> dict:
    """what a rule would change in a whole run: PV hits that become 'possible'"""
    db = sqlite3.connect(results_db)
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(results)")}
        rows = db.execute("SELECT pv_area_m2, footprint_m2, max_prob" + (", pv_share" if "pv_share" in cols else ", NULL")
                          + " FROM results WHERE has_pv = 1").fetchall()
    finally:
        db.close()
    n = 0
    for area, fp, prob, share in rows:
        share = share if share is not None else (area / fp if fp else None)
        if (area or 0) < rule["max_area_m2"] and (rule.get("max_share") is None or (share is not None and share < rule["max_share"])) \
                and (rule.get("max_prob") is None or (prob or 0) < rule["max_prob"]):
            n += 1
    return {"pv_treffer": len(rows), "wuerden_moeglich": n, "anteil": round(n / len(rows), 3) if rows else None}
