"""Roof search: detections outside the footprint, rule for 'possible', thresholds from the sample review."""

import json
import os
import sqlite3

import numpy as np
import pytest
from shapely.geometry import box

from pvmapper import building_mode, config, geo, review, annotate


def _win():
    # 100 m × 100 m at 0.5 m around a point in Berlin (EPSG:25833)
    c = geo.reproject(box(13.40, 52.52, 13.4001, 52.5201).centroid, "EPSG:4326", "EPSG:25833")
    return geo.Window("EPSG:25833", c.x - 50, c.y - 50, c.x + 50, c.y + 50, 200, 200)


def _fp(win, x0, y0, x1, y1):
    """footprint in pixel coordinates of the window → WGS84"""
    ulx, xr, _, uly, _, yr = win.geotransform
    g = box(ulx + x0 * xr, uly + y1 * yr, ulx + x1 * xr, uly + y0 * yr)
    return geo.reproject(g, "EPSG:25833", "EPSG:4326")


def test_detection_mostly_outside_is_dropped_inside_kept():
    win = _win()
    fp = _fp(win, 60, 60, 140, 140)                                 # 40 m × 40 m building
    m = np.zeros((200, 200), np.uint8)
    m[80:100, 80:120] = 1                                            # on the roof
    m[60:70, 142:190] = 1                                            # carport next to it, touches the eaves strip
    m[150:170, 100:150] = 1                                          # field south of the building, overlapping 1 px row?
    cfg = dict(config.DEFAULTS)
    out, gone = building_mode.roof_filter(m, fp, win, m.shape, cfg)
    assert out[80:100, 80:120].all()
    assert not out[60:70, 150:190].any() and not out[150:170, 100:150].any()
    assert len(gone) == 2 and all(g[0] == "außerhalb des Grundrisses" for g in gone)
    # a roof array reaching 1 m over the eaves stays, clipped to footprint + 0.5 m
    m2 = np.zeros((200, 200), np.uint8)
    m2[100:120, 100:142] = 1
    out2, gone2 = building_mode.roof_filter(m2, fp, win, m2.shape, cfg)
    assert not gone2 and out2[100:120, 100:140].all() and not out2[100:120, 142:].any()
    # switched off per setting: unchanged
    assert building_mode.roof_filter(m, None, win, m.shape, cfg)[0] is m


def test_weak_hit_rule():
    cfg = dict(config.DEFAULTS)
    row = {"has_pv": 1, "pv_area_m2": 8, "pv_share": 0.01, "max_prob": 0.7}
    assert not building_mode.weak_hit(row, cfg)                      # off by default
    cfg["roof_weak_rule"] = {"max_area_m2": 15, "max_share": 0.05, "max_prob": 0.8}
    assert building_mode.weak_hit(row, cfg)
    assert not building_mode.weak_hit({**row, "pv_area_m2": 40}, cfg)
    assert not building_mode.weak_hit({**row, "pv_share": 0.2}, cfg)
    assert not building_mode.weak_hit({**row, "max_prob": 0.95}, cfg)


def test_rule_check_from_sample(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    db = sqlite3.connect(tmp_path / "results.sqlite")
    db.execute("CREATE TABLE results (id TEXT PRIMARY KEY, has_pv INTEGER, pv_area_m2 REAL, footprint_m2 REAL, max_prob REAL, "
               "pv_share REAL, \"check\" TEXT)")
    rows = []
    for i in range(30):                                              # right hits: larger, confident
        rows.append((f"ok{i}", 1, 40 + i * 5, 800, 0.9 + (i % 5) * 0.015, None, "ohne", "richtig"))
    rows.append(("ok_small", 1, 9, 6000, 0.75, None, "ohne", "richtig"))          # small real system on a large school
    for i in range(12):                                              # false hits: small, low score
        rows.append((f"bad{i}", 1, 4 + i * 0.5, 400, 0.6 + i * 0.01, None, "ohne", ["gaube", "schatten", "dachfenster"][i % 3]))
    for bid, hp, a, fp, pr, sh, ck, cause in rows:
        db.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?)", (bid, hp, a, fp, pr, sh, ck))
        ok = cause == "richtig"
        json.dump({"id": bid, "name": bid, "label": int(ok), "review": {"cause": cause, "ok": ok}}, open(d / f"{bid}.json", "w"))
    db.commit()
    db.close()
    r = review.rule_check(str(d), str(tmp_path / "results.sqlite"), max_true_loss=0.05)
    b = r["beste"]
    assert r["falsch"] == 12 and r["richtig"] == 31
    assert b["falsch_erfasst"] == 12 and b["richtig_verloren"] <= 1
    assert r["genauigkeit_nachher"] > r["genauigkeit_vorher"]
    assert set(r["erfasst_nach_ursache"]) == {"gaube", "schatten", "dachfenster"}
    eff = review.rule_effect(str(tmp_path / "results.sqlite"), b)
    assert eff["pv_treffer"] == 43 and eff["wuerden_moeglich"] >= 12


def test_threshold_grid_reaches_below_0_2(monkeypatch):
    from pvmapper import training
    metas = [{"id": f"p{i}", "label": 1, "label_kind": "det_pv"} for i in range(10)] + \
            [{"id": f"n{i}", "label": 0, "label_kind": "neg_high"} for i in range(10)]
    # real installations score 0.08–0.15 after training on hard negatives, false ones below 0.05
    scores = {m["id"]: (np.array([0.08 + 0.007 * i if m["label"] else 0.03]), None) for i, m in enumerate(metas)}
    monkeypatch.setattr(training, "predict", lambda sc, t, min_px: {k: int(v[0].max() >= t) for k, v in sc.items()})
    monkeypatch.setattr(training, "metrics", lambda ms, pred, a: {"alle": _f1(ms, pred)})
    thr, f1 = training.tune_threshold(metas, scores, 0)
    assert thr < 0.2 and f1 == 1.0


def _f1(ms, pred):
    tp = sum(1 for m in ms if m["label"] and pred[m["id"]])
    fp = sum(1 for m in ms if not m["label"] and pred[m["id"]])
    fn = sum(1 for m in ms if m["label"] and not pred[m["id"]])
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {"f1": 2 * p * r / (p + r) if p + r else 0}


def test_patch_threshold_for_small_installations():
    from pvmapper import training
    res = 0.1                                                      # 1 px = 0.01 m²
    assert training.patch_pv_threshold_px(10_000, res) == pytest.approx(300)     # 100 m² array: 3 m² per patch as before
    assert training.patch_pv_threshold_px(400, res) == pytest.approx(200)        # 4 m² array: half of it (2 m²) is enough
    assert training.patch_pv_threshold_px(40, res) == pytest.approx(50)          # tiny: at least 0.5 m²
