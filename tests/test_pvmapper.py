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
