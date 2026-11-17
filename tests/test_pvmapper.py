"""Offline tests for pvmapper (no model weights, no network).

Run:  python -m pytest tests/test_pvmapper.py -q
"""

import dataclasses
import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pvmapper import building_mode, buildings, config, geo, inference, models, runtime, sources  # noqa: E402
from pvmapper.jobs import Job  # noqa: E402


# --------------------------------------------------------------- helpers

class BrightCls(nn.Module):
    """'PV' = patch contains bright-blue pixels."""
    def forward(self, x):
        blue = (x[:, 2] > 0.8).float().mean(dim=(1, 2)) * 50
        return torch.stack([-blue, blue], 1)


class BrightSeg(nn.Module):
    def forward(self, x):
        return {"out": ((x[:, 2:3] > 0.8).float() * 20 - 10)}


def fake_profile():
    p = models.PROFILES["nrw"]
    return dataclasses.replace(p, patch_px=64, cls_input_px=64, norm_mean=(0, 0, 0), norm_std=(1, 1, 1),
                               cls_threshold=0.5, seg_threshold=0.5)


def fake_detector(*_a, **_k):
    return inference.Detector(fake_profile(), BrightCls(), BrightSeg(), "cpu", batch_size=8, overlap=0.25)


# --------------------------------------------------------------- unit tests

def test_patch_grid_covers_image():
    for w, h, s, ov in [(100, 64, 64, 0), (300, 200, 64, 0.25), (64, 64, 64, 0.5), (65, 130, 64, 0)]:
        cov = np.zeros((h, w), bool)
        for x, y in inference.patch_grid(w, h, s, ov):
            cov[y:y + s, x:x + s] = True
        assert cov.all()


def test_window_resolution_utm_and_mercator():
    g = Point(13.4, 52.5).buffer(0.0003)
    w1 = geo.window_around(g, "EPSG:25833", 0.1, 5)
    assert abs((w1.maxx - w1.minx) / w1.width - 0.1) < 1e-9
    w2 = geo.window_around(g, "EPSG:3857", 0.1, 5)
    # same number of pixels in both CRS → same ground resolution
    assert abs(w1.width - w2.width) <= 2


def test_size_classes_and_range():
    cls = config.DEFAULTS["size_classes"]
    assert config.size_class(5, cls) == "S"
    assert config.size_class(30, cls) == "M"
    assert config.size_class(5000, cls) == "XL"
    cfg = dict(config.DEFAULTS, kwp_min=100, kwp_max=None)
    assert not config.in_kwp_range(50, cfg) and config.in_kwp_range(150, cfg)


def test_blank_fraction():
    assert sources.blank_fraction(np.full((10, 10, 3), 255, np.uint8)) == 1.0
    assert sources.blank_fraction(np.full((10, 10, 3), 120, np.uint8)) == 0.0


def test_catalog_routing():
    cat = sources.Catalog()
    ids = [s.id for s in cat.candidates(13.40, 52.52)]       # Berlin
    assert ids[0] == "de-be" and "de-bb" in ids
    assert cat.candidates(5.12, 52.09)[0].id == "nl-pdok-hr"   # Utrecht
    assert [s.id for s in cat.candidates(16.37, 48.21, ["AT"])] == ["at-basemap"]


def test_readers(tmp_path):
    gj = tmp_path / "b.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[13, 52], [13.001, 52], [13.001, 52.001], [13, 52]]]},
         "properties": {"osm_id": 7, "name": "Schule"}}]}))
    b = list(buildings.read_buildings(str(gj)))
    assert b[0].id == "7" and b[0].geometry.geom_type == "Polygon"
    csvp = tmp_path / "b.csv"
    csvp.write_text("id;lat;lon;name\nA;52,5;13,4;Rathaus\nB;52.6;13.5;Schule\n")
    b = list(buildings.read_buildings(str(csvp)))
    assert [x.id for x in b] == ["A", "B"] and b[0].geometry.x == pytest.approx(13.4)


def test_mask_to_polygons_area():
    win = geo.Window("EPSG:25833", 390000, 5800000, 390010, 5800010, 100, 100)   # 0.1 m/px
    m = np.zeros((100, 100), np.uint8)
    m[20:60, 30:80] = 1                                    # 4 m × 5 m = 20 m²
    polys = inference.mask_to_polygons(m, win)
    assert len(polys) == 1
    assert geo.ground_area_m2(polys[0]) == pytest.approx(20, rel=0.08)


def test_building_job_end_to_end(tmp_path, monkeypatch):
    # two buildings; synthetic imagery has a bright-blue 'panel' only on the first
    cx, cy = 13.40, 52.52
    b1 = box(cx, cy, cx + 0.0003, cy + 0.0002)
    b2 = box(cx + 0.001, cy, cx + 0.0013, cy + 0.0002)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": b.__geo_interface__, "properties": {"id": i}}
        for i, b in (("hasPV", b1), ("noPV", b2))]}
    inp = tmp_path / "in.geojson"
    inp.write_text(json.dumps(fc))

    class FakeSrc:
        id = "fake"

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = np.full((win.height, win.width, 3), 90, np.uint8)
        if geom.intersects(b1):
            h, w = img.shape[:2]
            img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), fetch_workers=2)
    job = Job(cfg["jobs_dir"], "buildings", {"input": str(inp), "save_chips": "positive"})
    job.run(lambda j: building_mode.run(j, cfg))
    assert job.status == "done", job.error
    assert job.counters["pv"] == 1 and job.counters["kein_pv"] == 1
    rows = open(os.path.join(job.dir, "buildings.csv"), encoding="utf-8").read()
    assert "hasPV,pv" in rows and "noPV,kein_pv" in rows
    assert os.path.isfile(os.path.join(job.dir, "chips", "hasPV.jpg"))
    # resume: nothing left to do, same result
    job2 = Job.load(cfg["jobs_dir"], job.id)
    job2.run(lambda j: building_mode.run(j, cfg))
    assert job2.status == "done" and job2.done == 2


# --------------------------------------------------------------- PV-Map connector + training

def test_pvmap_input_parsing_and_paging(monkeypatch):
    from pvmapper import pvmap
    assert pvmap.parse_input("pvmap:cc=DE&sub=Education&x=1") == {"cc": "DE", "sub": "Education"}
    with pytest.raises(pvmap.PVMapError):
        pvmap.parse_input("pvmap:sub=Education")
    pages = {"": (["a", "b"], "b"), "b": (["c"], None)}
    calls = []

    def fake_get(self, path, params):
        calls.append((path, dict(params)))
        if path == "/detect/count":
            return {"total": 3, "rows": []}
        ids, nxt = pages[params.get("after", "")]
        feats = [{"type": "Feature", "id": i, "geometry": box(13, 52, 13.0002, 52.0002).__geo_interface__,
                  "properties": {"id": i, "sub": "Education", "pv_conf": None}} for i in ids]
        return {"features": feats, "next": nxt}

    monkeypatch.setattr(pvmap.Client, "_get", fake_get)
    cfg = {"pvmap_url": "http://x", "pvmap_token": "pvm_t"}
    got = list(buildings.read_buildings("pvmap:cc=DE&sub=Education", cfg=cfg))
    assert [b.id for b in got] == ["a", "b", "c"] and got[0].props["sub"] == "Education"
    assert buildings.count_buildings("pvmap:cc=DE", cfg=cfg) == 3
    assert all(c[1].get("cc") == "DE" for c in calls)


class TinyCls(nn.Module):
    """trainable stand-in for InceptionV3 (parameter names match the fine-tuned blocks)"""
    def __init__(self):
        super().__init__()
        self.Mixed_7a = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(torch.relu(self.Mixed_7a(x)).mean(dim=(2, 3)))


class TinySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(3, 1, 1)

    def forward(self, x):                      # rule-based mask (+ a trainable term so the optimizer has work)
        return {"out": (x[:, 2:3] > 0.8).float() * 20 - 10 + 0.01 * self.classifier(x)}


def test_training_job_end_to_end(tmp_path, monkeypatch):
    from pvmapper import training
    rng = np.random.default_rng(0)
    feats = []
    for i in range(48):
        lab = i % 2
        x0, y0 = 13.0 + i * 0.002, 52.0
        g = box(x0, y0, x0 + 0.0002, y0 + 0.00015)
        pv = box(x0 + 0.00005, y0 + 0.00004, x0 + 0.00012, y0 + 0.0001) if lab else None
        feats.append({"type": "Feature", "id": f"b{i}", "geometry": g.__geo_interface__,
                      "properties": {"label": lab, "label_kind": ("pos_high" if lab else ("neg_weak" if i % 4 else "neg_high")),
                                     "sub": "Education" if i % 3 else "Offices", "size_bin": "10-30" if lab else "none",
                                     "pv_kwp": 20 if lab else None,
                                     "pv_geometry": pv.__geo_interface__ if pv is not None else None}})
    lf = tmp_path / "labels.geojson"
    lf.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    class FakeSrc:
        id = "fake"

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = rng.integers(70, 110, (win.height, win.width, 3)).astype(np.uint8)
        idx = int(round((geom.centroid.x - 13.0) / 0.002))
        if idx % 2:
            h, w = img.shape[:2]
            img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    def fake_det(cfg, profile_id=None):
        return inference.Detector(fake_profile(), TinyCls(), TinySeg(), "cpu", batch_size=8, overlap=0.0)

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_det)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "models"), fetch_workers=2)
    job = Job(cfg["jobs_dir"], "train", {"labels_file": str(lf), "epochs": 4, "seg_epochs": 1, "min_seg_samples": 5,
                                          "name": "TEST tiny", "val_share": 0.3, "save_anyway": True, "lr": 2e-2})
    job.run(lambda j: training.run(j, cfg))
    assert job.status == "done", job.error
    rep = json.load(open(os.path.join(job.dir, "report.json"), encoding="utf-8"))
    assert rep["n_val"] >= 10 and rep["segmentation_trained"] and "nach_art" in rep["nachher"]
    assert set(rep["nachher"]["nach_label"]) <= {"pos_high", "neg_high", "neg_weak"}
    assert rep["nachher"]["nur_sichere_labels"]["f1"] >= 0.8, rep["nachher"]["nur_sichere_labels"]
    profs = models.custom_profiles(cfg["model_dir"])
    assert "custom:test_tiny" in profs and profs["custom:test_tiny"].cls_threshold == rep["cls_threshold"]
    assert os.path.isfile(os.path.join(cfg["model_dir"], profs["custom:test_tiny"].cls_file))


def test_annotation_ai_and_training_with_user_masks(tmp_path, monkeypatch):
    """prepare-only dataset → masks drawn by a user and suggested by AI (mocked HTTP) → training reuses the dataset"""
    from pvmapper import annotate, ai_label, training
    torch.manual_seed(0)
    feats = []
    for i in range(40):
        lab = i % 2
        x0 = 13.0 + i * 0.002
        feats.append({"type": "Feature", "id": f"b{i}", "geometry": box(x0, 52.0, x0 + 0.0002, 52.00015).__geo_interface__,
                      "properties": {"label": lab, "label_kind": "pos_high" if lab else "neg_high", "sub": "Education",
                                     "size_bin": "10-30" if lab else "none", "pv_kwp": 20 if lab else None}})
    lf = tmp_path / "l.geojson"
    lf.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    rng = np.random.default_rng(1)

    class FakeSrc:
        id = "fake"

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = rng.integers(70, 110, (win.height, win.width, 3)).astype(np.uint8)
        if int(round((geom.centroid.x - 13.0) / 0.002)) % 2:
            h, w = img.shape[:2]
            img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector",
                        lambda cfg, pid=None: inference.Detector(fake_profile(), TinyCls(), TinySeg(), "cpu", batch_size=8, overlap=0.0))
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "models"), fetch_workers=2,
               anthropic_api_key="k", openai_api_key="k")
    prep = Job(cfg["jobs_dir"], "train", {"labels_file": str(lf), "prepare_only": True})
    prep.run(lambda j: training.run(j, cfg))
    assert prep.status == "done" and prep.counters["datensatz"] == 40
    d = annotate.ds_dir(prep.dir)
    items = annotate.list_items(d)
    pos = [m for m in items if m["label"] == 1]
    assert annotate.stats(items)["positive_without_mask"] == len(pos)

    # user draws on half of the positives (the blue square), corrects one label, skips one image
    for m in pos[:10]:
        w, h = annotate.image_size(d, m["name"])
        annotate.save_annotation(d, m["name"], [[[w / 3, h / 3], [2 * w / 3, h / 3], [2 * w / 3, 2 * h / 3], [w / 3, 2 * h / 3]]])
    neg = [m for m in items if m["label"] == 0]
    annotate.save_annotation(d, neg[0]["name"], [], label=1, status="done")
    annotate.save_annotation(d, neg[1]["name"], None, status="skip")
    meta = annotate.read_meta(d, pos[0]["name"])
    assert meta["has_mask"] and meta["mask_source"] == "user" and annotate.usable_mask(meta)
    assert annotate.effective_label(annotate.read_meta(d, neg[0]["name"])) == 1
    assert len(annotate.current_polygons(d, pos[0]["name"])) == 1
    # "no PV" marks (a dormer-like square in a corner) on a positive and on a negative image
    for m in (pos[1], neg[2]):
        annotate.save_annotation(d, m["name"], None, negatives=[[[2, 2], [30, 2], [30, 30], [2, 30]]])
    assert annotate.read_meta(d, neg[2]["name"])["has_neg"] and len(annotate.current_polygons(d, neg[2]["name"], "neg")) == 1
    assert annotate.stats(annotate.list_items(d))["neg_marked"] == 2

    # AI suggestion (mocked Anthropic and OpenAI answers): candidates mode picks candidate 1
    class Resp:
        def __init__(self, j): self.status_code, self._j, self.text = 200, j, json.dumps(j)
        def json(self): return self._j
    answer = json.dumps({"pv_visible": True, "confidence": 0.9, "candidates": {"1": "pv", "2": "dormer"},
                         "extra_polygons": [], "note": "panels"})
    def fake_post(url, json=None, headers=None, timeout=None):
        if "anthropic" in url:
            assert json["messages"][0]["content"][0]["type"] == "image"
            return Resp({"content": [{"type": "text", "text": answer}], "usage": {"input_tokens": 900, "output_tokens": 40}})
        return Resp({"choices": [{"message": {"content": answer}}], "usage": {"prompt_tokens": 800, "completion_tokens": 30}})
    monkeypatch.setattr(ai_label.requests, "post", fake_post)
    det = runtime.detector(cfg)
    r = ai_label.label_item(cfg, d, pos[11]["name"], "openai", "candidates", det, 0.1)
    assert r["pv_visible"] and len(r["polygons"]) == 1 and r["mode"] == "candidates"
    aij = Job(cfg["jobs_dir"], "ai_masks", {"dataset": prep.id, "provider": "anthropic", "mode": "candidates", "limit": 5})
    aij.run(lambda j: ai_label.run(j, cfg))
    assert aij.status == "done", aij.error
    # the relabelled negative has no candidates → polygons mode; the mocked answer has none → "no PV per AI"
    assert aij.counters["maske"] + aij.counters["keine_pv_laut_ki"] == 5 and aij.counters["tokens_in"] == 4500
    ai_meta = [m for m in annotate.list_items(d) if m.get("mask_source") == "ai"]
    assert len(ai_meta) == aij.counters["maske"] >= 1 and not annotate.usable_mask(ai_meta[0], "reviewed") and annotate.usable_mask(ai_meta[0], "all")

    # training with the dataset: user masks (10) count, unreviewed AI masks do not; the skipped image is left out
    tj = Job(cfg["jobs_dir"], "train", {"dataset_from": prep.id, "epochs": 2, "seg_epochs": 1, "min_seg_samples": 3,
                                         "save_anyway": True, "name": "TEST ds", "lr": 2e-2})
    tj.run(lambda j: training.run(j, cfg))
    assert tj.status == "done", tj.error
    rep = json.load(open(os.path.join(tj.dir, "report.json"), encoding="utf-8"))
    assert rep["segmentation_trained"] and rep["n_masks_train"] <= 10
    assert rep["n_train"] + rep["n_val"] <= 39
    for a in rep["abweichungen"]:
        assert os.path.isfile(os.path.join(tj.dir, "chips", a["chip"])) and {"vorher", "nachher", "label"} <= set(a)
    assert any("„Kein PV“-Markierungen" in line for line in tj.log_lines)



def test_web_password_gate(tmp_path):
    import base64 as b64
    from fastapi.testclient import TestClient
    from pvmapper.web.app import create_app
    cfgp = tmp_path / "pvmapper.yml"
    config.save(dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "m"), web_password="geheim"), str(cfgp))
    c = TestClient(create_app(str(cfgp)))
    assert c.get("/api/sources").status_code == 401
    bad = {"Authorization": "Basic " + b64.b64encode(b"x:falsch").decode()}
    good = {"Authorization": "Basic " + b64.b64encode(b"egal:geheim").decode()}
    assert c.get("/api/sources", headers=bad).status_code == 401
    r = c.get("/api/sources", headers=good)
    assert r.status_code == 200 and len(r.json()) >= 20
    assert c.get("/api/info", headers=good).json()["config"]["web_password"] == config.MASK


def test_positives_only_dataset_gets_negatives_at_training(tmp_path, monkeypatch):
    from pvmapper import pvmap, training
    torch.manual_seed(0)
    rng = np.random.default_rng(2)

    def feat(i, lab):
        x0 = 13.0 + i * 0.002
        return {"type": "Feature", "id": f"{'p' if lab else 'n'}{i}",
                "geometry": box(x0, 52.0, x0 + 0.0002, 52.00015).__geo_interface__,
                "properties": {"label": lab, "label_kind": "pos_high" if lab else "neg_weak", "sub": "Education",
                               "size_bin": "10-30" if lab else "none", "pv_kwp": 20 if lab else None}}
    lf = tmp_path / "pos.geojson"
    lf.write_text(json.dumps({"type": "FeatureCollection", "features": [feat(i, 1) for i in range(0, 60, 2)]}))

    class FakeSrc:
        id = "fake"

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = rng.integers(70, 110, (win.height, win.width, 3)).astype(np.uint8)
        if int(round((geom.centroid.x - 13.0) / 0.002)) % 2 == 0:
            h, w = img.shape[:2]
            img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    calls = []
    def fake_training(self, bins=None, per_stratum=100, negatives=100, seed="1", **flt):
        calls.append((per_stratum, negatives, flt))
        return {"features": [feat(i, 0) for i in range(1, 61, 2)]}

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(pvmap.Client, "training", fake_training)
    monkeypatch.setattr(runtime, "detector",
                        lambda cfg, pid=None: inference.Detector(fake_profile(), TinyCls(), TinySeg(), "cpu", batch_size=8, overlap=0.0))
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "models"), fetch_workers=2)
    prep = Job(cfg["jobs_dir"], "train", {"labels_file": str(lf), "prepare_only": True, "cc": "DE", "sub": "Education"})
    prep.run(lambda j: training.run(j, cfg))
    assert prep.status == "done" and prep.counters["datensatz"] == 30
    tj = Job(cfg["jobs_dir"], "train", {"dataset_from": prep.id, "epochs": 2, "save_anyway": True, "name": "TEST pos", "lr": 2e-2})
    tj.run(lambda j: training.run(j, cfg))
    assert tj.status == "done", tj.error
    assert calls and calls[0][0] == 0 and calls[0][2] == {"cc": "DE", "sub": "Education"}
    rep = json.load(open(os.path.join(tj.dir, "report.json"), encoding="utf-8"))
    assert rep["n_train"] + rep["n_val"] == 60 and "neg_weak" in rep["sample"]


# --------------------------------------------------------------- post-processing and cross-check

def _roof(h=200, w=300, v=150):
    img = np.full((h, w, 3), v, np.uint8)
    clip = np.zeros((h, w), bool)
    clip[10:190, 10:290] = True
    return img, clip


def test_postproc_removes_dormer_row_and_bright_blobs():
    from pvmapper import postproc
    img, clip = _roof()
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(6):                                           # six small, darker, aligned blobs: dormers
        x = 30 + i * 35
        img[120:135, x:x + 12] = (110, 115, 125)
        mask[120:135, x:x + 12] = 1
    img[30:60, 30:70] = (160, 160, 160)                          # as bright as the roof, not bluer
    mask[30:60, 30:70] = 1
    out, info = postproc.clean(img, mask, 0.1, clip)
    assert out.sum() == 0
    reasons = {r for r, _ in info["removed"]}
    assert "Reihe gleichartiger Dachaufbauten" in reasons and "kein Kontrast zum Dach" in reasons


def test_postproc_completes_module_row_and_finds_lookalike():
    from pvmapper import postproc
    img, clip = _roof()
    img[40:70, 30:200] = (40, 55, 90)                            # long module row
    img[120:150, 60:130] = (40, 55, 90)                          # second field, missed by the model
    img[50:60, 120:140] = (200, 205, 210)                        # glare inside the row
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:70, 30:90] = 1                                       # model found only the left part
    out, info = postproc.clean(img, mask, 0.1, clip)
    assert out[40:70, 30:200].mean() > 0.95                      # row completed, glare hole filled
    assert out[120:150, 60:130].mean() > 0.9                     # look-alike field added
    assert out[:, 220:].sum() == 0 and info["added_m2"] > 30


def test_postproc_keeps_field_on_roof_of_same_colour():
    from pvmapper import postproc
    img, clip = _roof(v=60)                                      # dark roof: no colour completion
    img[40:70, 30:90] = (45, 50, 80)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:70, 30:90] = 1
    out, _ = postproc.clean(img, mask, 0.1, clip)
    assert abs(int(out.sum()) - 1800) < 300


def test_fuse_with_second_image():
    from pvmapper import postproc
    a = np.zeros((100, 100), np.uint8)
    a[10:30, 10:40] = 1                                          # field seen in both images
    a[60:80, 60:80] = 1                                          # only in the first (shadow artefact)
    b = np.zeros_like(a)
    b[10:30, 10:70] = 1                                          # second image sees the whole row
    out, st = postproc.fuse(a, b, 0.1)
    assert st == "teilweise" and out[10:30, 10:70].all() and not out[60:80, 60:80].any()
    out, st = postproc.fuse(a, np.zeros_like(a), 0.1)
    assert st == "nur_aktuell" and out.sum() == a.sum()
    assert postproc.fuse(a, None, 0.1)[1] == "ohne"


def test_fetch_second_skips_identical_and_blank(monkeypatch):
    cat = sources.Catalog()
    src = cat.get("de-nw")
    assert src.alt and all(cat.get(a).crs == src.crs for a in src.alt)
    win = geo.window_around(box(7.1, 50.73, 7.1005, 50.7303), src.crs, 0.1, 3, 64)
    prim = np.full((win.height, win.width, 3), 100, np.uint8)
    answers = {src.alt[0]: prim.copy(), src.alt[1]: np.full_like(prim, 255), src.alt[2]: np.full_like(prim, 130)}
    monkeypatch.setattr(sources.Source, "fetch", lambda self, w: answers[self.id])
    img, s2 = sources.fetch_second(cat, src, win, prim)
    assert s2.id == src.alt[2] and img.mean() == 130


def test_building_job_with_cross_check(tmp_path, monkeypatch):
    cx, cy = 13.40, 52.52
    b1 = box(cx, cy, cx + 0.0003, cy + 0.0002)
    inp = tmp_path / "in.geojson"
    inp.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": b1.__geo_interface__, "properties": {"id": "x"}}]}))

    class FakeSrc:
        id = "fake"
        alt = ["fake-old"]

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = np.full((win.height, win.width, 3), 90, np.uint8)
        h, w = img.shape[:2]
        img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    class Old:
        id = "fake-old"

    def fake_second(cat, src, win, prim):
        img = prim.copy()
        img[:, : img.shape[1] // 2] = 90                         # older image: left half of the field not built yet
        return img, Old()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(sources, "fetch_second", fake_second)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), fetch_workers=1)
    job = Job(cfg["jobs_dir"], "buildings", {"input": str(inp), "save_chips": "positive"})
    job.run(lambda j: building_mode.run(j, cfg))
    assert job.status == "done", job.error
    assert job.counters["pv"] == 1 and job.counters["pruefung_bestaetigt"] == 1
    import sqlite3
    row = sqlite3.connect(os.path.join(job.dir, "results.sqlite")).execute('SELECT "check", source2 FROM results').fetchone()
    assert row == ("bestaetigt", "fake-old")


def test_delete_old_training_data(tmp_path):
    from fastapi.testclient import TestClient
    from pvmapper import annotate
    from pvmapper.web.app import create_app
    jobs = tmp_path / "jobs"
    cfgp = tmp_path / "pvmapper.yml"
    config.save(dict(config.DEFAULTS, jobs_dir=str(jobs), model_dir=str(tmp_path / "m")), str(cfgp))
    made = []
    for i, hand in enumerate((False, True)):
        j = Job(str(jobs), "train", {"prepare_only": True, "name": f"TEST{i}"})
        j.status = "done"
        j.save()
        d = annotate.ds_dir(j.dir)
        os.makedirs(d)
        meta = {"name": "b", "id": "b", "label": 1, "mask_source": "user" if hand else None, "has_mask": hand}
        json.dump(meta, open(os.path.join(d, "b.json"), "w"))
        made.append(j.id)
    keep = Job(str(jobs), "buildings", {"input": "x"})
    keep.status = "done"
    keep.save()
    c = TestClient(create_app(str(cfgp)))
    prev = c.get("/api/cleanup").json()
    hm = {p["id"]: p["hand_masks"] for p in prev}
    assert hm == {made[0]: 0, made[1]: 1}
    r = c.post("/api/cleanup", json={"ids": made}).json()
    assert len(r["deleted"]) == 2 and not any(os.path.isdir(os.path.join(jobs, i)) for i in made)
    assert [j["id"] for j in c.get("/api/jobs").json()] == [keep.id]
    assert c.delete(f"/api/jobs/{keep.id}").status_code == 200 and not os.path.isdir(keep.dir)


def test_check_results_to_dataset_review_and_train(tmp_path, monkeypatch):
    """'check buildings' results → selected buildings become a dataset (detector masks as suggestions) →
    only reviewed items count in training"""
    from pvmapper import annotate, training
    torch.manual_seed(0)
    feats = []
    for i in range(12):
        x0 = 13.0 + i * 0.002
        feats.append({"type": "Feature", "geometry": box(x0, 52.0, x0 + 0.0003, 52.0002).__geo_interface__,
                      "properties": {"id": f"TEST{i}", "sub": "Education"}})
    inp = tmp_path / "in.geojson"
    inp.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    rng = np.random.default_rng(2)

    class FakeSrc:
        id = "fake"

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = rng.integers(80, 100, (win.height, win.width, 3)).astype(np.uint8)
        if int(round((geom.centroid.x - 13.0) / 0.002)) % 2 == 0:
            h, w = img.shape[:2]
            img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "models"), fetch_workers=2)
    chk = Job(cfg["jobs_dir"], "buildings", {"input": str(inp), "save_chips": "none"})
    chk.run(lambda j: building_mode.run(j, cfg))
    assert chk.status == "done" and chk.counters["pv"] == 6

    pick = ["TEST0", "TEST2", "TEST4", "TEST1"]                   # three detections with PV, one without
    prep = Job(cfg["jobs_dir"], "train", {"from_job": chk.id, "ids": pick, "prepare_only": True})
    prep.run(lambda j: training.run(j, cfg))
    assert prep.status == "done", prep.error
    d = annotate.ds_dir(prep.dir)
    items = {m["id"]: m for m in annotate.list_items(d)}
    assert set(items) == set(pick)
    m0 = items["TEST0"]
    assert m0["label_kind"] == "det_pv" and m0["has_mask"] and m0["mask_source"] == "model"
    assert not annotate.usable_mask(m0) and items["TEST1"]["label"] == 0

    # review: TEST0 accepted (saved), TEST2 was a false positive (label flipped), TEST4 left open
    annotate.save_annotation(d, "TEST0", annotate.current_polygons(d, "TEST0"))
    annotate.save_annotation(d, "TEST2", [], label=0, negatives=[[[5, 5], [40, 5], [40, 40], [5, 40]]])
    m0 = annotate.read_meta(d, "TEST0")
    assert m0["mask_source"] == "user" and annotate.usable_mask(m0)

    monkeypatch.setattr(runtime, "detector",
                        lambda cfg, pid=None: inference.Detector(fake_profile(), TinyCls(), TinySeg(), "cpu", batch_size=8, overlap=0.0))
    monkeypatch.setattr(training, "load_sample", lambda job, cfg: [])
    tj = Job(cfg["jobs_dir"], "train", {"dataset_from": prep.id, "epochs": 1, "auto_negatives": False,
                                         "save_anyway": True, "name": "TEST det", "val_share": 0.0})
    tj.run(lambda j: training.run(j, cfg))
    # only the two reviewed images count (TEST1 and TEST4 are still open) – too few to train, which is the point
    assert any("2 übernommene Erkennungen noch nicht geprüft" in line for line in tj.log_lines)
    assert tj.status == "failed" and "Training 2" in tj.error


def test_negative_filter_for_detection_datasets(tmp_path):
    from pvmapper import training
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"))
    chk = Job(cfg["jobs_dir"], "buildings", {"input": "pvmap:cc=DE&nuts3=DEA22&sub=Education&pv=none"})
    chk.save()
    f = training._negative_filter(cfg, {"from_job": chk.id}, [])
    assert f == {"cc": "DE", "nuts3": "DEA22", "sub": "Education"}
    f = training._negative_filter(cfg, {"from_job": "gone"}, [{"source": "nl-pdok-hr"}, {"source": "de-nw"}, {"source": "nl-pdok-hr"}])
    assert f == {"cc": "NL"}
    with pytest.raises(RuntimeError):
        training._negative_filter(cfg, {}, [])


def test_review_chips_for_disagreements(tmp_path):
    from PIL import Image as PImage
    from pvmapper import training
    d = tmp_path / "ds"
    d.mkdir()
    val = []
    for i, lab in enumerate((1, 0, 0)):
        name = f"TEST{i}"
        PImage.fromarray(np.full((40, 50, 3), 120, np.uint8)).save(d / f"{name}.jpg")
        PImage.fromarray(np.full((40, 50), 255, np.uint8)).save(d / f"{name}.fp.png")
        val.append({"id": name, "name": name, "label": lab, "label_kind": "det_pv" if lab else "neg_weak", "sub": "Education"})
    pix = [np.arange(0, 400)]
    scores_old = {"TEST0": (np.array([0.9]), pix), "TEST1": (np.array([0.1]), pix), "TEST2": (np.array([0.1]), pix)}
    scores_new = {"TEST0": (np.array([0.9]), pix), "TEST1": (np.array([0.8]), pix), "TEST2": (np.array([0.1]), pix)}
    po, pn = training.predict(scores_old, 0.5, 10), training.predict(scores_new, 0.5, 10)
    job = Job(str(tmp_path / "jobs"), "train", {})
    out = training.review_chips(job, str(d), val, scores_old, 0.5, scores_new, 0.5, po, pn)
    assert [a["id"] for a in out] == ["TEST1"] and out[0]["nachher"] and not out[0]["vorher"]
    assert os.path.isfile(os.path.join(job.dir, "chips", out[0]["chip"]))


def test_corner_points_for_snapping(tmp_path):
    from PIL import Image as PImage
    from pvmapper import annotate
    img = np.full((120, 160, 3), 150, np.uint8)
    img[30:70, 40:110] = (40, 55, 90)
    PImage.fromarray(img).save(tmp_path / "x.jpg")
    pts = annotate.corner_points(str(tmp_path), "x")
    for cx_, cy in ((40, 30), (109, 30), (109, 69), (40, 69)):
        assert min(abs(x - cx_) + abs(y - cy) for x, y in pts) < 3


# --------------------------------------------------------------- sync with PV-Map (work list, results)

def test_sync_job_sends_results_and_skips_unchanged_images(tmp_path, monkeypatch):
    from pvmapper import pvmap, sync
    cx, cy = 13.40, 52.52
    blds = {"TESTpv": box(cx, cy, cx + 0.0003, cy + 0.0002), "TESTnone": box(cx + 0.001, cy, cx + 0.0013, cy + 0.0002),
            "TESTsame": box(cx + 0.002, cy, cx + 0.0023, cy + 0.0002)}

    class FakeSrc:
        id = "fake"
        alt = []

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = np.full((win.height, win.width, 3), 90, np.uint8)
        if geom.intersects(blds["TESTpv"]):
            h, w = img.shape[:2]
            img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    # the fingerprint of TESTsame's image is known from an earlier check with the same model
    win = geo.window_around(blds["TESTsame"], "EPSG:25833", 0.1, 3.0, 64)
    same_hash = sources.image_hash(np.full((win.height, win.width, 3), 90, np.uint8))
    calls = {"queue": 0, "sent": [], "finish": 0, "q": None}

    def queue(self, limit=1000, **q):
        calls["queue"] += 1
        calls["q"] = q
        if calls["queue"] > 1:
            return {"features": []}
        feats = []
        for bid, g in blds.items():
            scan = {"img_hash": same_hash, "model": "nrw", "result": "none"} if bid == "TESTsame" else None
            feats.append({"type": "Feature", "id": bid, "geometry": g.__geo_interface__,
                          "properties": {"id": bid, "sub": "Education", "prio": 4 if scan else 1, "scan": scan}})
        return {"features": feats}

    monkeypatch.setattr(pvmap.Client, "queue", queue)
    monkeypatch.setattr(pvmap.Client, "queue_count", lambda self, **q: {"open": {"1": 2, "4": 1}})
    monkeypatch.setattr(pvmap.Client, "send_results", lambda self, items, run=None, model=None: calls["sent"].extend(items) or {"unknown": 0})
    monkeypatch.setattr(pvmap.Client, "finish", lambda self: calls.__setitem__("finish", calls["finish"] + 1) or {})
    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), fetch_workers=1, pvmap_token="pvm_x", sync_filter="cc=DE&nuts3=DEA2",
               typology_push=True)
    job = Job(cfg["jobs_dir"], "sync", {})
    job.run(lambda j: sync.run(j, cfg))
    assert job.status == "done", job.error
    sent = {x["id"]: x for x in calls["sent"]}
    assert sent["TESTsame"]["result"] == "unchanged" and sent["TESTnone"]["result"] == "none"
    assert sent["TESTpv"]["result"] == "pv" and sent["TESTpv"]["tier"] == 2 and sent["TESTpv"]["polygons"]["type"] in ("Polygon", "MultiPolygon")
    assert sent["TESTpv"]["img_hash"] and calls["finish"] == 1
    inst = sent["TESTpv"]["installations"]          # typology for PV-Map (migration 137): installations with geometry
    assert inst and inst[0]["geometry"]["type"] == "Polygon" and inst[0]["installation_id"] == "TESTpv#i0"
    assert sent["TESTpv"]["kwp_low"] <= sent["TESTpv"]["kwp"] <= sent["TESTpv"]["kwp_high"]
    assert "installations" not in sent["TESTnone"]
    json.dumps(calls["sent"])                        # the payload must be plain JSON
    assert calls["q"]["nuts3"] == "DEA2" and calls["q"]["model"] == "nrw"
    assert job.counters["unveraendert"] == 1 and job.counters["gesendet"] == 3


def test_sync_window_and_scheduler(tmp_path, monkeypatch):
    import datetime as dt
    from pvmapper import sync
    tz = dt.timezone(dt.timedelta(hours=1))
    cfg = {"sync_start": "22:00", "sync_end": "06:00"}
    assert sync.window(cfg, dt.datetime(2026, 11, 16, 23, 30, tzinfo=tz))[1].hour == 6
    assert sync.window(cfg, dt.datetime(2026, 11, 17, 5, 59, tzinfo=tz))[0].day == 16
    assert sync.window(cfg, dt.datetime(2026, 11, 17, 12, 0, tzinfo=tz)) is None
    monkeypatch.setenv("PVMAPPER_NO_SCHEDULER", "1")
    from pvmapper.web.app import create_app
    cfgp = tmp_path / "pvmapper.yml"
    config.save(dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "m"),
                     sync_enabled=True, pvmap_token="pvm_x"), str(cfgp))
    app = create_app(str(cfgp))
    started = []
    import pvmapper.web.app as webapp
    monkeypatch.setattr(webapp.JobManager, "submit", lambda self, job: started.append(job))
    night = dt.datetime(2026, 11, 16, 23, 0, tzinfo=tz)
    assert app.state.sync_tick(night) is not None and app.state.sync_tick(night + dt.timedelta(minutes=5)) is None
    assert app.state.sync_tick(dt.datetime(2026, 11, 17, 14, 0, tzinfo=tz)) is None
    assert app.state.sync_tick(dt.datetime(2026, 11, 17, 22, 30, tzinfo=tz)) is not None, "next night: again"
    assert started[0].kind == "sync" and started[0].params["until"].startswith("2026-11-17T06:00")


def test_review_result_to_wgs84(tmp_path):
    from PIL import Image as PImage
    from pvmapper import annotate
    d = tmp_path
    win = geo.window_around(box(13.4, 52.52, 13.4003, 52.5202), "EPSG:25833", 0.1, 3, 64)
    PImage.fromarray(np.full((win.height, win.width, 3), 100, np.uint8)).save(d / "b.jpg")
    PImage.fromarray(np.zeros((win.height, win.width), np.uint8)).save(d / "b.fp.png")
    meta = {"id": "B1", "name": "b", "label": 1, "status": "open",
            "window": [win.crs, win.minx, win.miny, win.maxx, win.maxy, win.width, win.height]}
    json.dump(meta, open(d / "b.json", "w"))
    m = annotate.save_annotation(str(d), "b", [[[10, 10], [110, 10], [110, 60], [10, 60]]])     # 100×50 px at 10 cm = 50 m²
    r = annotate.review_result(str(d), "b", m)
    assert r["result"] == "pv" and r["reviewed"] and r["tier"] == 1 and abs(r["area_m2"] - 50) < 3
    lon, lat = r["polygons"]["coordinates"][0][0]
    assert 13.39 < lon < 13.41 and 52.51 < lat < 52.53
    m = annotate.save_annotation(str(d), "b", [], label=0)
    assert annotate.review_result(str(d), "b", m) == {"id": "B1", "result": "none", "reviewed": True}


def test_confident_fields_survive_colour_rules_and_gate_below_threshold():
    """bright, silvery modules (as light as the roof) are kept when the model is sure; a roof whose best patch scores just
    below the classification threshold is still segmented (lower bound = threshold × neighbor_factor)"""
    from pvmapper import postproc
    img, clip = _roof(v=150)
    img[40:90, 30:150] = (154, 156, 158)                         # silvery modules, no contrast to the roof
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:90, 30:150] = 1
    out, info = postproc.clean(img, mask, 0.1, clip)
    assert out.sum() == 0 and info["removed"][0][0] == "kein Kontrast zum Dach"
    prob = (mask * 0.93).astype(np.float16)
    out, _ = postproc.clean(img, mask, 0.1, clip, prob=prob)
    assert out[40:90, 30:150].mean() > 0.95

    class HalfCls(nn.Module):                                    # every patch scores 0.45, threshold 0.5
        def forward(self, x):
            return torch.stack([torch.zeros(x.shape[0]), torch.full((x.shape[0],), -0.2)], 1)
    prof = fake_profile()
    imgb = np.full((128, 128, 3), 90, np.uint8)
    imgb[40:90, 40:90] = (40, 120, 250)
    for nf, expect in ((0.0, 0), (0.5, 1)):
        det = inference.Detector(prof, HalfCls(), BrightSeg(), "cpu", batch_size=8, overlap=0.0, neighbor_factor=nf, tta=False)
        r = det.run([("x", imgb)])["x"]
        assert (r.mask.sum() > 0) == bool(expect) and abs(r.max_prob - 0.45) < 0.01
        assert r.prob.dtype == np.float16 and (r.prob > 0.5).sum() == r.mask.sum()


def test_nacharbeit_and_append_to_dataset(tmp_path, monkeypatch):
    """incomplete/missed detections are listed for rework and can be added to an existing dataset (iterative training);
    items remember the model that produced them"""
    from pvmapper import annotate, training
    assert training._nacharbeit("kein_pv", 0, {"pv_kwp": 30}, 0.2) == "verpasst"
    assert training._nacharbeit("pv", 8, {"pv_kwp": 30}, 0.9) == "unvollständig"
    assert training._nacharbeit("pv", 25, {"pv_kwp": 30}, 0.9) is None
    assert training._nacharbeit("kein_pv", 0, {}, 0.6) == "verdacht"
    assert training._nacharbeit("kein_pv", 0, {}, 0.1) is None
    cx, cy = 13.40, 52.52
    feats = [{"type": "Feature", "geometry": box(cx + i * 0.001, cy, cx + i * 0.001 + 0.0003, cy + 0.0002).__geo_interface__,
              "properties": {"id": f"TEST{i}", "sub": "Education", "pv_kwp": 20 if i < 2 else None}} for i in range(4)]
    inp = tmp_path / "in.geojson"
    inp.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    class FakeSrc:
        id = "fake"
        alt = []

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        return np.full((win.height, win.width, 3), 90, np.uint8), win, FakeSrc()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), fetch_workers=1, profile="nrw")
    chk = Job(cfg["jobs_dir"], "buildings", {"input": str(inp), "save_chips": "none"})
    chk.run(lambda j: building_mode.run(j, cfg))
    first = Job(cfg["jobs_dir"], "train", {"from_job": chk.id, "ids": ["TEST0"], "prepare_only": True, "name": "TEST it"})
    first.run(lambda j: training.run(j, cfg))
    more = Job(cfg["jobs_dir"], "train", {"from_job": chk.id, "ids": ["TEST1", "TEST2"], "prepare_only": True, "append_to": first.id})
    more.run(lambda j: training.run(j, cfg))
    assert more.status == "done", more.error
    items = {m["id"]: m for m in annotate.list_items(annotate.ds_dir(first.dir))}
    assert set(items) == {"TEST0", "TEST1", "TEST2"}
    assert items["TEST1"]["source_model"] == "nrw" and items["TEST1"]["nacharbeit"] == "verpasst" and items["TEST2"]["nacharbeit"] is None


# --------------------------------------------------------------- installation typology

def test_typology_formulas_and_exports():
    from pvmapper import typology as T
    # prompt example: 100 m² outline, real fill factor 0.52, 22 % modules → 11.44 kWp
    assert T.capacity_kwp(100, 0.52, 0.22) == pytest.approx(11.44)
    # rows: 1.72 m modules at 15°, pitch 3.0 m; 1.13 m wide with 2 cm gaps
    f = T.fill_factor_pitch(1.72, 15, 3.0, 1.13, 0.02)
    assert f == pytest.approx(1.72 * np.cos(np.radians(15)) / 3.0 * 1.13 / 1.15)
    assert T.real_from_plan(0.5, 60) == pytest.approx(1.0)
    assert T.tilt_class(12) == "5-20" and T.tilt_class(90) == "90" and T.tilt_class_legacy("5-20") == "<20"
    ew = {"layout": "east_west", "azimuth_deg": 272, "azimuth2_deg": 92, "tilt_class": "5-20", "module_count": 120}
    assert T.osm_tags(ew)["generator:orientation"] == "W;E"
    m = T.mastr_fields(ew)
    assert m["Hauptausrichtung"] == "Ost" and m["Nebenausrichtung"] == "West"
    assert m["Neigungswinkel der Nebenausrichtung"] == "5-20" and m["Anzahl der Module"] == 120
    assert T.validate({**ew, "layout": "zigzag"})


def _rows_image(period_px, two_level, n_rows=10, row_px=None, angle=0):
    """synthetic flat roof (gray 150) with module rows along x; east-west: two brightness levels per period"""
    H, W = 260, 300
    img = np.full((H, W, 3), 150, np.uint8)
    mask = np.zeros((H, W), bool)
    row_px = row_px or period_px
    y0 = 20
    for r in range(n_rows):
        y = y0 + r * period_px
        if two_level:
            img[y:y + row_px // 2, 30:270] = (50, 60, 110)
            img[y + row_px // 2:y + row_px, 30:270] = (90, 100, 150)
        else:
            img[y:y + row_px, 30:270] = (50, 60, 110)
        mask[y:y + row_px, 30:270] = True
    clip = np.zeros((H, W), bool)
    clip[5:H - 5, 5:W - 5] = True
    return img, mask, clip


def test_typology_classifies_east_west_mono_and_flush():
    from pvmapper import typology as T
    res = 0.1
    # east-west: rows along x → faces point north/south… turn the picture: rows along y (faces east and west)
    img, mask, clip = _rows_image(22, True)
    img, mask, clip = np.ascontiguousarray(img.transpose(1, 0, 2)), mask.T.copy(), clip.T.copy()
    ew = T.analyse(img, mask, res, clip, {"roof:shape": "flat"})
    assert ew and ew[0]["layout"] == "east_west", ew[0]
    assert {round(ew[0]["azimuth_deg"] / 90) % 4, round(ew[0]["azimuth2_deg"] / 90) % 4} == {1, 3}
    assert ew[0]["fill_factor_source"] == "derived_pitch"
    # mono rows: 1 m of modules every 3 m → tilted towards the sun, fill factor from the row geometry
    img, mask, clip = _rows_image(30, False, n_rows=7, row_px=10)
    mono = T.analyse(img, mask, res, clip, {"roof:shape": "flat"})
    assert mono and mono[0]["layout"] == "tilted_mono", mono[0]
    assert 0.25 <= mono[0]["fill_factor"] <= 0.45
    # one dense field without rows: flush
    img, mask, clip = _rows_image(200, False, n_rows=1, row_px=200)
    fl = T.analyse(img, mask, res, clip, {"roof:shape": "gabled"})
    assert fl[0]["layout"] == "flush" and fl[0]["fill_factor_source"] == "type_default"
    assert fl[0]["kwp_low"] <= fl[0]["kwp_est"] <= fl[0]["kwp_high"]
    assert not T.validate(fl[0]) and not T.validate(ew[0]) and not T.validate(mono[0])


def test_typology_in_building_job_and_calibration(tmp_path, monkeypatch):
    from pvmapper import typology as T
    from pvmapper.jobs import open_db
    cx, cy = 13.40, 52.52
    b1 = box(cx, cy, cx + 0.0003, cy + 0.0002)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": b1.__geo_interface__, "properties": {"id": "TEST1", "pv_kwp": 30}}]}
    inp = tmp_path / "in.geojson"
    inp.write_text(json.dumps(fc))

    class FakeSrc:
        id = "fake-2024"

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = np.full((win.height, win.width, 3), 90, np.uint8)
        h, w = img.shape[:2]
        img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), fetch_workers=2, cross_check=False)
    job = Job(cfg["jobs_dir"], "buildings", {"input": str(inp)})
    job.run(lambda j: building_mode.run(j, cfg))
    assert job.status == "done", job.error
    db = open_db(os.path.join(job.dir, "results.sqlite"))
    kwp, lo, hi, lay, meth, inst = db.execute(
        "SELECT kwp_est, kwp_low, kwp_high, layout, kwp_method, installations FROM results").fetchone()
    assert meth == "typology" and lo <= kwp <= hi and lay
    feats = json.loads(inst)
    assert feats[0]["properties"]["installation_id"] == "TEST1#i0"
    assert feats[0]["properties"]["imagery_year"] == 2024
    assert feats[0]["geometry"]["type"] == "Polygon"
    gj = json.load(open(os.path.join(job.dir, "installations.geojson"), encoding="utf-8"))
    assert gj["features"][0]["properties"]["osm"]["generator:source"] == "solar"
    # web: installation outlines as their own layer in the results
    from fastapi.testclient import TestClient
    from pvmapper.web.app import create_app
    cfgp = tmp_path / "pvmapper.yml"
    config.save(dict(cfg, model_dir=str(tmp_path / "m")), str(cfgp))
    res = TestClient(create_app(str(cfgp))).get(f"/api/jobs/{job.id}/results?status=&limit=50").json()
    inst = res["installations"]["features"]
    assert inst and inst[0]["properties"]["kind"] == "inst" and inst[0]["properties"]["id"] == "TEST1"
    assert inst[0]["properties"]["layout"] and inst[0]["geometry"]["type"] == "Polygon"
    assert res["rows"][0]["kwp_low"] <= res["rows"][0]["kwp_est"] <= res["rows"][0]["kwp_high"]
    cal = T.calibrate_job(db)
    assert cal["typology"]["n"] == 1 and sum(v["n"] for v in cal["layouts"].values()) <= 1
    # switched off: the old area factor
    cfg2 = dict(cfg, typology=False)
    job2 = Job(cfg["jobs_dir"], "buildings", {"input": str(inp)})
    job2.run(lambda j: building_mode.run(j, cfg2))
    db2 = open_db(os.path.join(job2.dir, "results.sqlite"))
    assert db2.execute("SELECT kwp_method FROM results").fetchone()[0] == "area_factor"


def test_calibrate_fits_fill_factor():
    from pvmapper import typology as T
    rows = []
    for k in range(5):
        # flat-roof flush field 100 m², register 19.32 kWp at 21 % → f_plan 0.92
        rows.append({"kwp_register": 19.32, "kwp_est": 19.0 + k * 0.2, "kwp_area": 20.0,
                     "installations": [{"layout": "flush", "roof_shape": "flat", "outline_area_m2": 100, "tilt_deg": 0,
                                        "module_efficiency": 0.21, "kwp_low": 17, "kwp_high": 21}]})
    out = T.calibrate(rows)
    assert out["layouts"]["flush_flat"]["f_median"] == pytest.approx(0.92, abs=0.01)
    assert out["layouts"]["flush_flat"]["current"] == 0.92
    assert out["register_inside_range"] == 1.0 and out["typology"]["n"] == 5


def test_typology_in_area_probe_and_editor(tmp_path, monkeypatch):
    """the typology wherever installations are found: area jobs, probe, mask editor (and hand-checked results for PV-Map)"""
    from PIL import Image as PImage
    from fastapi.testclient import TestClient
    from pvmapper import annotate, area_mode
    from pvmapper.web.app import create_app

    class FakeSrc:
        id = "fake-2023"
        name = "Fake"
        attribution = ""

    def fake_fetch(cat, geom, res, buf, sel, min_px=0, snap_px=1, max_blank=0.5):
        win = geo.window_around(geom, "EPSG:25833", res, buf, min_px)
        img = np.full((win.height, win.width, 3), 90, np.uint8)
        h, w = img.shape[:2]
        img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (40, 120, 250)
        return img, win, FakeSrc()

    monkeypatch.setattr(sources, "fetch_for_geometry", fake_fetch)
    monkeypatch.setattr(runtime, "detector", fake_detector)
    monkeypatch.setattr(models, "get_profile", lambda *_a, **_k: fake_profile())
    cfg = dict(config.DEFAULTS, jobs_dir=str(tmp_path / "jobs"), model_dir=str(tmp_path / "m"), fetch_workers=1,
               area_cell_px=256)
    # area job: installations get layout, kWp range, outlines
    job = Job(cfg["jobs_dir"], "area", {"bbox": [13.4000, 52.5200, 13.4004, 52.5203]})
    job.run(lambda j: area_mode.run(j, cfg))
    assert job.status == "done", job.error
    fc = json.load(open(os.path.join(job.dir, "installations.geojson"), encoding="utf-8"))
    pr = fc["features"][0]["properties"]
    assert pr["kwp_method"] == "typology" and pr["layout"] and pr["kwp_low"] <= pr["kwp_est"] <= pr["kwp_high"]
    cfgp = tmp_path / "pvmapper.yml"
    config.save(cfg, str(cfgp))
    c = TestClient(create_app(str(cfgp)))
    res = c.get(f"/api/jobs/{job.id}/results").json()
    assert res["installations"]["features"] and res["installations"]["features"][0]["properties"]["kind"] == "inst"
    # probe
    p = c.post("/api/probe", json={"lat": 52.52, "lon": 13.40, "radius_m": 15}).json()
    assert p["kwp_method"] == "typology" and p["installations"]["features"] and p["kwp_low"] <= p["kwp_est"] <= p["kwp_high"]
    # mask editor: typology of the drawn (unsaved) mask, pixel outlines
    d = tmp_path / "ds"
    d.mkdir()
    win = geo.window_around(box(13.4, 52.52, 13.4003, 52.5202), "EPSG:25833", 0.1, 3, 64)
    img = np.full((win.height, win.width, 3), 150, np.uint8)
    PImage.fromarray(img).save(d / "b.jpg")
    fp = np.zeros((win.height, win.width), np.uint8)
    fp[5:-5, 5:-5] = 255
    PImage.fromarray(fp).save(d / "b.fp.png")
    meta = {"id": "TESTB1", "name": "b", "label": 1, "status": "open", "source": "de-nw-2024",
            "window": [win.crs, win.minx, win.miny, win.maxx, win.maxy, win.width, win.height]}
    json.dump(meta, open(d / "b.json", "w"))
    poly = [[[20, 20], [180, 20], [180, 120], [20, 120]]]                                   # 160 × 100 px = 160 m²
    t = annotate.typology_of(str(d), "b", meta, poly, cfg)
    assert t["summary"]["kwp_est"] > 0 and t["installations"][0]["outline_px"]
    assert abs(t["installations"][0]["outline_area_m2"] - 160) < 10
    m = annotate.save_annotation(str(d), "b", poly)
    r = annotate.review_result(str(d), "b", m, 0.2, dict(cfg, typology_push=True))
    assert r["kwp"] == t["summary"]["kwp_est"] and r["installations"][0]["review_status"] == "confirmed_manual"
    assert r["installations"][0]["geometry"]["type"] == "Polygon" and r["installations"][0]["imagery_year"] == 2024


# --------------------------------------------------------------- installation age from older flights

NRW_INFO = """GetFeatureInfo results:

Layer 'nw_hist_dop_info'
  Feature 1: 
    Bildflugdatum = '2023-05-04'
    Bodenauflösung Originalbild [m/Pixel] = '0.1'
  Feature 2: 
    Bildflugdatum = '2016-08-23'
    Bodenauflösung Originalbild [m/Pixel] = '0.1'
  Feature 3: 
    Bildflugdatum = '2003-07-14'
    Bodenauflösung Originalbild [m/Pixel] = '0.312500000000000'
  Feature 4: 
    Bildflugdatum = '1998-06-01'
    Bodenauflösung Originalbild [m/Pixel] = '0.4'
  Feature 5: 
    Bildflugdatum = '2009-00-00'
    Bodenauflösung Originalbild [m/Pixel] = '0.2'
"""


def test_age_nrw_metadata_and_year_math():
    from pvmapper import age, typology as T
    assert age.parse_nrw_info(NRW_INFO) == [(1998, "1998-06-01", 0.4), (2003, "2003-07-14", 0.3125),
                                            (2009, None, 0.2), (2016, "2016-08-23", 0.1), (2023, "2023-05-04", 0.1)]
    assert str(age.parse_flight_date("Kachelname = '32375_5628'\n    Bildflugdatum = '01.07.2025'")) == "2025-07-01"
    conf = T.settings()
    p = {"kwp_est": 21.5, "kwp_low": 19.0, "kwp_high": 23.0, "module_efficiency": 0.215}
    age._apply_year(p, 2012, 2010, 2013, conf)               # built 2010–2013 → 16 % instead of 21.5 %
    assert p["module_efficiency"] == 0.16 and p["kwp_est"] == 16.0
    assert p["kwp_low"] <= p["kwp_est"] <= p["kwp_high"] < 21.5
    assert p["install_year_low"] == 2010 and p["install_year_high"] == 2013


def test_age_bisection_finds_build_year(monkeypatch):
    """fake flights 2010–2022: modules from 2017 on → built between 2016 and 2017; kWp follows the efficiency"""
    from pvmapper import age, typology as T
    win = geo.window_around(box(13.4, 52.52, 13.4003, 52.5202), "EPSG:25833", 0.1, 3, 64)
    H, W = win.height, win.width

    def picture(with_pv):
        img = np.full((H, W, 3), 150, np.uint8)
        if with_pv:
            img[H // 3: 2 * H // 3, W // 3: 2 * W // 3] = (40, 120, 250)
        return img

    fetched = []

    class Flight:
        def __init__(self, y):
            self.y = y

        def fetch(self, w):
            fetched.append(self.y)
            return picture(self.y >= 2017)

    years = [2010, 2012, 2014, 2016, 2017, 2019, 2021, 2022]
    monkeypatch.setattr(age, "timeline", lambda src, w, cfg=None: [(y, f"{y}-05-01", 0.1, Flight(y)) for y in years])

    class Src:
        id = "fake-2024"
        history = {"kind": "fake"}

    img = picture(True)
    mask = (img[:, :, 2] > 200)
    cfg = dict(config.DEFAULTS)
    feats = T.for_window(img, mask, win, 0.1, Src(), "TESTA", None, {}, cfg)
    k0 = feats[0]["properties"]["kwp_est"]
    det = fake_detector()
    n = age.estimate([{"img": img, "win": win, "src": Src(), "mask": mask, "feats": feats, "res_m": 0.1}], det, cfg)
    p = feats[0]["properties"]
    assert n == 1 and (p["install_year_low"], p["install_year_high"]) == (2016, 2017), p.get("age_flights")
    assert p["install_year"] in (2016, 2017) and p["module_efficiency"] == 0.18
    assert p["kwp_est"] == pytest.approx(k0 * 0.18 / 0.215, abs=0.2)
    assert len(set(fetched)) <= 4                            # bisection, not every flight
    assert all(f["verdict"] in ("present", "absent") for f in p["age_flights"])
    # a dark patch looks like modules in 2012, but 2014 and 2016 are clearly empty: the later empty flights win
    years[:] = [2010, 2012, 2014, 2016, 2017, 2019, 2021, 2022]

    class Patchy(Flight):
        def fetch(self, w):
            return picture(self.y >= 2017 or self.y == 2012)
    monkeypatch.setattr(age, "timeline", lambda src, w, cfg=None: [(y, None, 0.1, Patchy(y)) for y in years])
    feats = T.for_window(img, mask, win, 0.1, Src(), "TESTC", None, {}, cfg)
    age.estimate([{"img": img, "win": win, "src": Src(), "mask": mask, "feats": feats, "res_m": 0.1}], det, cfg)
    p = feats[0]["properties"]
    assert (p["install_year_low"], p["install_year_high"]) == (2016, 2017), p["age_flights"]
    # only unclear flights below the first one with modules: an upper bound, no build year, kWp unchanged
    feats = T.for_window(img, mask, win, 0.1, Src(), "TESTD", None, {}, cfg)
    orig_verdict = age.verdict
    k0 = feats[0]["properties"]["kwp_est"]
    monkeypatch.setattr(age, "verdict", lambda ev, cfg=None: "present" if ev.get("coverage", 0) > 0.5 else "unclear")
    age.estimate([{"img": img, "win": win, "src": Src(), "mask": mask, "feats": feats, "res_m": 0.1}], det, cfg)
    p = feats[0]["properties"]
    assert p["install_year"] is None and p["install_year_high"] == 2012 and p["kwp_est"] == k0
    monkeypatch.setattr(age, "verdict", orig_verdict)
    # already on the oldest flight: "before 2010"
    years[:] = [2018, 2020]
    feats = T.for_window(img, mask, win, 0.1, Src(), "TESTB", None, {}, cfg)
    monkeypatch.setattr(age, "timeline", lambda src, w, cfg=None: [(y, None, 0.25, Flight(2030)) for y in years])
    age.estimate([{"img": img, "win": win, "src": Src(), "mask": mask, "feats": feats, "res_m": 0.1}], det, cfg)
    p = feats[0]["properties"]
    assert p["install_year_low"] is None and p["install_year_high"] == 2018 and p["install_year"] == 2017


def test_age_evidence_verdicts():
    from pvmapper import age
    H, W = 120, 160
    roof = np.full((H, W, 3), 150, np.uint8)
    cur = roof.copy()
    cur[40:80, 50:110] = (40, 60, 120)
    mods = np.zeros((H, W), bool)
    mods[40:80, 50:110] = True
    # old flight without modules, model finds nothing
    ev = age.evidence(cur, roof, mods, np.zeros_like(mods), 0.1, 0.1)
    assert age.verdict(ev) == "absent", ev
    # coarse old flight (31 cm) with modules: model misses them, contrast and pattern still there
    import cv2
    old = cv2.GaussianBlur(cur, (0, 0), 1.5)
    ev = age.evidence(cur, old, mods, np.zeros_like(mods), 0.1, 0.3125)
    assert age.verdict(ev) == "present", ev
