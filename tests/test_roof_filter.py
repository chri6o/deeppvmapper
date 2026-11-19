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


def test_reviewed_items_accept_checked(tmp_path):
    import json
    from pvmapper import training
    def item(name, **kw):
        m = {"id": name, "name": name, "label": 1, "label_kind": "det_pv", "status": "open", "has_mask": True,
             "mask_source": "model", "val": False, **kw}
        (tmp_path / f"{name}.json").write_text(json.dumps(m))
    item("big_ok", check="bestaetigt", kwp_detected=80, pv_share=0.4)
    item("big_single", check="nur_aktuell", kwp_detected=80)
    item("small_ok", check="bestaetigt", kwp_detected=10)
    item("odd_share", check="bestaetigt", kwp_detected=90, pv_share=1.4)
    item("reviewed", status="done", label_override=0, kwp_detected=5)
    got, n = training.reviewed_items(str(tmp_path))
    assert [m["id"] for m in got] == ["reviewed"] and n == 4 and got[0]["label"] == 0
    got, n = training.reviewed_items(str(tmp_path), accept_checked={"min_kwp": 30})
    ids = {m["id"] for m in got}
    assert ids == {"big_ok", "reviewed"} and n == 3
    big = next(m for m in got if m["id"] == "big_ok")
    assert big["label"] == 1 and big["has_mask"] is False and big["_accepted"]


def test_history_eval_helpers():
    import numpy as np
    from pvmapper import history_eval as he
    tl = [(2010, None, 0.2, "a"), (2016, None, 0.1, "b"), (2020, None, 0.1, "c"), (2024, None, 0.1, "d")]
    assert [t[0] for t in he.pick_years(tl, 2024, 2)] == [2016, 2020]
    probs = np.array([0.1, 0.5, 0.9])
    pix = [np.array([1, 2]), np.array([2, 3, 4]), np.array([4, 5])]
    n = he.npx_by_threshold(probs, pix)
    assert n["0.2"] == 4 and n["0.85"] == 2 and n["0.95"] == 0


def test_nir_stats():
    import numpy as np
    from pvmapper import history_eval as he
    nir = np.full((10, 10), 200, np.uint8); nir[:5] = 40          # modules dark in the infrared, roof bright
    fp = np.ones((10, 10), bool)
    sel = np.arange(50)
    s = he.nir_stats(nir, fp, sel, np.full((10, 10, 3), 80, np.uint8))
    assert s["nir_pv"] == 40 and s["nir_roof"] == 200 and s["nir_ratio"] == 0.2 and s["nir_vis"] == 0.5
    assert he.nir_stats(nir, fp, np.array([], int)) == {}


def test_seg_report(tmp_path):
    import numpy as np, torch
    from PIL import Image
    from types import SimpleNamespace
    from pvmapper import training
    img = np.zeros((64, 64, 3), np.uint8); fp = np.zeros((64, 64), np.uint8); fp[8:56, 8:56] = 255
    mask = np.zeros((64, 64), np.uint8); mask[10:30, 10:30] = 255
    Image.fromarray(img).save(tmp_path / "a.jpg"); Image.fromarray(fp).save(tmp_path / "a.fp.png"); Image.fromarray(mask).save(tmp_path / "a.mask.png")

    class Det:                                                     # draws the left half of every patch as PV
        p = SimpleNamespace(patch_px=32); batch_size = 4
        _to_tensor = staticmethod(lambda ps: torch.zeros((len(ps), 3, 32, 32)))
        _pad = staticmethod(lambda x, n: x)
        _bucket = staticmethod(lambda k, cap: k)
        @staticmethod
        def _segment(x):
            m = torch.zeros((x.shape[0], 32, 32), dtype=torch.bool); m[:, :, :16] = True
            return m
    r = training.seg_report(Det(), str(tmp_path), [{"name": "a", "has_mask": True}])
    assert r["iou"] is not None and 0 < r["iou"] < 1 and r["flaeche"] > 1 and r["in_kein_pv"] is None


def test_shape_features():
    import numpy as np, cv2
    from pvmapper import history_eval as he
    fp = np.zeros((200, 200), bool); fp[20:180, 20:180] = True
    m = np.zeros((200, 200), np.uint8); m[120:160, 40:120] = 1                 # a rectangle on the south half
    f = he.shape_features(m.astype(bool), fp, 0.1)
    assert f["n_comp"] == 1 and f["rect_largest"] > 0.95 and f["north_share"] == 0 and abs(f["area_m2"] - 32) < 0.5
    t = np.zeros((200, 200), np.uint8); cv2.fillPoly(t, [np.array([[40, 30], [140, 30], [40, 90]])], 1)   # triangle, north
    g = he.shape_features(t.astype(bool), fp, 0.1)
    assert g["rect_largest"] < 0.7 and g["north_share"] == 1
    assert he.shape_features(np.zeros((200, 200), bool), fp, 0.1) == {}


def test_north_share():
    import numpy as np
    from pvmapper import building_mode as bm
    fp = np.zeros((100, 100), bool); fp[10:90, 10:90] = True
    m = np.zeros((100, 100), bool); m[15:30, 20:40] = True
    assert bm.north_share(m, fp) == 1.0 and bm.north_share(None, fp) is None and bm.north_share(np.zeros_like(m), fp) is None
