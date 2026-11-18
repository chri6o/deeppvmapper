"""Ground-mounted PV, step 2: baseline (docs/freiflaechen.md). DB tests need PostGIS: PVMAPPER_GROUND_TEST_DSN."""

import json
import os

import pytest
from shapely.geometry import Polygon, box

from pvmapper.ground import inventory as inv
from pvmapper.ground import store

CFG = {"ground_min_ha": 0.5, "ground_max_building_share": 0.2}
DSN = os.environ.get("PVMAPPER_GROUND_TEST_DSN", "postgresql://postgres@/pvmapper_ground_test?host=/tmp&port=5433")
D = 1 / 111320                         # ≈ 1 m in degrees of latitude


def sq(x, y, m):
    """square of edge m metres at lon x / lat y (lon degrees stretched for ~52°N)"""
    dx = m * D / 0.6157
    return box(x, y, x + dx, y + m * D)


def feat(geom, **tags):
    return json.dumps({"type": "Feature", "geometry": json.loads(json.dumps(geom.__geo_interface__)), "properties": tags})


# -- building overlap (> 20 % → discarded) ---------------------------------------------------------------------------

@pytest.mark.parametrize("share,discard", [(0.19, False), (0.21, True), (1.0, True)])
def test_building_share_rule(share, discard):
    fence = sq(17.0, 52.0, 100)                              # 1 ha
    w = (fence.bounds[2] - fence.bounds[0]) * share
    bld = [box(fence.bounds[0], fence.bounds[1], fence.bounds[0] + w, fence.bounds[3])]
    assert inv.building_share(fence, bld) == pytest.approx(share, abs=0.01)
    rows, c = inv.osm_rows([{"key": "w1", "tags": {"power": "plant", "plant:source": "solar"}, "geom": fence}], bld, "PL", "t", CFG)
    assert (rows[0]["pruefstatus"] == "verworfen") is discard
    assert (rows[0].get("verwurf_grund") == "gebaeude") is discard


def test_roof_and_small_and_parsing():
    lines = [feat(sq(17, 52, 100), **{"@type": "way", "@id": 1, "power": "plant", "plant:source": "solar",
                                      "plant:output:electricity": "1,2 MWp", "start_date": "2021-05"}),
             feat(sq(17.1, 52, 100), **{"@type": "way", "@id": 2, "power": "generator", "generator:source": "solar",
                                        "location": "roof"}),
             feat(sq(17.2, 52, 50), **{"@type": "way", "@id": 3, "power": "plant", "plant:source": "solar"}),   # 0.25 ha
             feat(sq(17.3, 52, 100), **{"@type": "way", "@id": 4, "power": "plant", "plant:source": "wind"})]
    feats = inv.osm_features(lines)
    assert [f["key"] for f in feats] == ["w1", "w3"]                     # roof and wind gone
    rows, c = inv.osm_rows(feats, [], "PL", "t", CFG)
    assert [r["id"] for r in rows] == ["ff-pl-osm-w1"] and c["zu_klein"] == 1
    r = rows[0]
    assert r["quellen"]["osm_leistung_kwp"] == 1200 and r["baujahr_schaetzung"] == 2021
    assert r["split"] in ("train", "test") and r["flaeche_zaun_m2"] == pytest.approx(10000, rel=0.02)
    assert inv.parse_power("5000 kW") == 5000 and inv.parse_power("750000 W") == 750
    assert inv.parse_power("yes") is None and inv.parse_power("12") is None and inv.parse_power("1.5 GW") == 1.5e6


def test_generators_become_module_rows_and_duplicates_merge():
    plant = sq(17, 52, 200)                                              # 4 ha
    x0, y0, x1, y1 = plant.bounds
    rows_geo = [box(x0 + (x1 - x0) * 0.1, y0 + (y1 - y0) * (0.1 + 0.1 * i), x1 - (x1 - x0) * 0.1, y0 + (y1 - y0) * (0.14 + 0.1 * i))
                for i in range(8)]
    feats = [{"key": "w10", "tags": {"power": "plant", "plant:source": "solar"}, "geom": plant},
             {"key": "r11", "tags": {"power": "plant", "plant:source": "solar"}, "geom": plant.buffer(D)}]
    feats += [{"key": f"w{20 + i}", "tags": {"power": "generator", "generator:source": "solar"}, "geom": g}
              for i, g in enumerate(rows_geo)]
    rows, c = inv.osm_rows(feats, [], "PL", "t", CFG)
    assert len(rows) == 1 and c["dubletten"] == 1
    r = rows[0]
    assert set(r["quellen"]["osm"]) == {"w10", "r11"} or set(r["quellen"]["osm"]) == {"r11", "w10"}
    assert r["geom_module"] is not None and r["quellen"]["osm_generatoren"] == 8
    assert 0.2 < r["flaeche_module_m2"] / r["flaeche_zaun_m2"] < 0.4


def test_whole_plant_generator_is_not_module_geometry():
    plant = sq(17, 52, 100)
    feats = [{"key": "w1", "tags": {"power": "plant", "plant:source": "solar"}, "geom": plant},
             {"key": "w2", "tags": {"power": "generator", "generator:source": "solar"}, "geom": plant.buffer(-D)}]
    rows, _ = inv.osm_rows(feats, [], "PL", "t", CFG)
    assert rows[0]["geom_module"] is None and rows[0]["quellen"]["osm_module_flaechig"]


def test_loose_generator_rows_cluster():
    rows_geo = [sq(17, 52 + i * 12 * D, 100).intersection(box(0, 52 + i * 12 * D, 90, 52 + i * 12 * D + 5 * D)) for i in range(10)]
    feats = [{"key": f"w{i}", "tags": {"power": "generator", "generator:source": "solar"}, "geom": g} for i, g in enumerate(rows_geo)]
    rows, _ = inv.osm_rows(feats, [], "PL", "t", CFG)
    assert len(rows) == 1 and len(rows[0]["quellen"]["osm"]) == 10 and rows[0]["geom_module"] is not None


def test_tzsam_merge_and_new():
    osm, _ = inv.osm_rows([{"key": "w1", "tags": {"power": "plant", "plant:source": "solar"}, "geom": sq(17, 52, 100)}],
                          [], "PL", "t", CFG)
    recs = [{"cluster_id": 7, "capacity_mw": 0.9, "constructed_before": "2019-08-01T10:00", "geom": sq(17, 52, 100).buffer(2 * D)},
            {"cluster_id": 8, "capacity_mw": 2.0, "constructed_before": "2022-03-05", "geom": sq(17.5, 52, 150)},
            {"cluster_id": 9, "capacity_mw": 0.1, "geom": sq(17.6, 52, 40)}]
    new, c = inv.tzsam_rows(recs, None, osm, [], "PL", "t", CFG, "zenodo:1")
    assert c["tzsam_zusammengefuehrt"] == 1 and c["tzsam_neu"] == 1 and c["zu_klein"] == 1
    q = osm[0]["quellen"]
    assert q["tzsam"] == ["7"] and q["lizenz_nc"] and q["tzsam_kwp_schaetzung"] == 900 and osm[0]["baujahr_schaetzung"] == 2019
    assert new[0]["id"] == "ff-pl-tz-8" and new[0]["first_seen"] == "2022-03-01" and new[0]["quellen"]["lizenz_nc"]


def test_tzsam_large_cluster_does_not_swallow_small_plant():
    small = sq(17, 52, 150)                                              # 2.25 ha
    osm, _ = inv.osm_rows([{"key": "w1", "tags": {"power": "plant", "plant:source": "solar"}, "geom": small}], [], "PL", "t", CFG)
    big = sq(17, 52, 900)                                                # 81 ha cluster that contains the small plant
    new, c = inv.tzsam_rows([{"cluster_id": 5, "capacity_mw": 83, "geom": big}], None, osm, [], "PL", "t", CFG, "z")
    q = osm[0]["quellen"]
    assert "tzsam_kwp_schaetzung" not in q and q["tzsam_teilweise"] == ["5"]
    assert len(new) == 1 and new[0]["quellen"]["ohne_osm_teil"] and new[0]["flaeche_zaun_m2"] == pytest.approx(81e4 - 2.25e4, rel=0.03)


def test_split_is_spatial_and_stable():
    a, b = sq(17.0, 52.0, 50), sq(17.0 + 20 * D / 0.6157, 52.0, 50)       # 20 m apart: same 10 km block
    assert inv.split_of(a) == inv.split_of(b) == inv.split_of(a)
    blocks = {inv.split_of(sq(15 + i * 0.2, 52, 50)) for i in range(40)}
    assert blocks == {"train", "test"}


def test_read_poly():
    g = inv.read_poly("x\n1\n  0 0\n  0 1\n  1 1\n  1 0\nEND\n!2\n  0.2 0.2\n  0.2 0.4\n  0.4 0.4\n  0.4 0.2\nEND\nEND\n")
    assert g.area == pytest.approx(1 - 0.04)


# -- store -------------------------------------------------------------------------------------------------------------

@pytest.fixture()
def con():
    psycopg = pytest.importorskip("psycopg")
    try:
        admin = psycopg.connect(DSN.replace("/pvmapper_ground_test", "/postgres"), autocommit=True)
    except Exception:                                                  # noqa: BLE001
        pytest.skip("no PostGIS test server")
    admin.execute("DROP DATABASE IF EXISTS pvmapper_ground_test")
    admin.execute("CREATE DATABASE pvmapper_ground_test")
    c = store.connect({"ground_db": DSN})
    store.migrate(c)
    yield c
    c.close()
    admin.execute("DROP DATABASE IF EXISTS pvmapper_ground_test")
    admin.close()


def _row(i="ff-pl-osm-w1", **kw):
    r = {"id": i, "land": "PL", "region": "t", "geom_zaun": sq(17, 52, 100), "flaeche_zaun_m2": 10000, "status": "in_betrieb",
         "herkunft": "bestand_osm", "split": "train", "quellen": {"osm": ["w1"]}, "pruefstatus": "kandidat", "typ": "unbekannt"}
    r.update(kw)
    return r


def test_store_upsert_idempotent_and_protects_reviews(con):
    assert store.upsert(con, [_row()]) == {"neu": 1, "aktualisiert": 0, "geschuetzt": 0}
    assert store.upsert(con, [_row()])["aktualisiert"] == 1
    con.execute("UPDATE freiflaeche SET pruefstatus='manuell_geprueft', typ='tracker' WHERE id='ff-pl-osm-w1'")
    n = store.upsert(con, [_row(typ="unbekannt", quellen={"tzsam": ["7"]})])
    assert n["geschuetzt"] == 1
    st, typ, q = con.execute("SELECT pruefstatus, typ, quellen FROM freiflaeche").fetchone()
    assert (st, typ) == ("manuell_geprueft", "tracker") and q == {"osm": ["w1"], "tzsam": ["7"]}
    fc = store.features(con)
    assert fc["features"][0]["geometry"]["type"] == "MultiPolygon" and fc["features"][0]["properties"]["typ"] == "tracker"
    assert store.stats(con)["rows"][0]["n"] == 1
    store.mark_imported(con, "osm:t", "v1", 1)
    assert store.imported(con, "osm:t", "v1") and not store.imported(con, "osm:t", "v2")


def test_store_rejects_unknown_type(con):
    import psycopg
    with pytest.raises(psycopg.errors.CheckViolation):
        store.upsert(con, [_row(typ="quer")])


def test_inventory_job_end_to_end(con, tmp_path, monkeypatch):
    """job with mocked downloads: OSM plant + TZ-SAM extra row stored, re-run keeps ids"""
    from pvmapper.jobs import Job, runner

    class R:
        text = "r\n1\n  16 51\n  16 54\n  20 54\n  20 51\nEND\nEND\n"
        def raise_for_status(self): pass
    monkeypatch.setattr(inv.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(inv, "geofabrik_latest", lambda p: ("https://x/w-261117.osm.pbf", "w-261117.osm.pbf"))
    monkeypatch.setattr(inv, "download", lambda url, path, timeout=60: open(path, "wb").write(b"pbf"))
    fence = sq(17, 52, 100)
    monkeypatch.setattr(inv, "extract_osm", lambda pbf, d, log=print: (
        [{"key": "w1", "tags": {"power": "plant", "plant:source": "solar"}, "geom": fence}], []))
    monkeypatch.setattr(inv, "tzsam_file", lambda cfg, d, log=print: ("tz.gpkg", "zenodo:1"))
    monkeypatch.setattr(inv, "read_tzsam", lambda path, bbox: [{"cluster_id": 8, "capacity_mw": 2.0, "geom": sq(18, 52, 150)}])
    cfg = {**CFG, "ground_db": DSN, "jobs_dir": str(tmp_path)}
    for _ in range(2):
        job = Job(str(tmp_path), "ground_inventory", {"region": "pl-wielkopolskie"})
        runner("ground_inventory")(job, cfg)
    ids = [r[0] for r in con.execute("SELECT id FROM freiflaeche ORDER BY id")]
    assert ids == ["ff-pl-osm-w1", "ff-pl-tz-8"] and job.counters["test"] + 0 >= 0
    assert any("2 aktualisiert" in ln for ln in job.log_lines)


# -- step 3: coarse search ---------------------------------------------------------------------------------------------

from pvmapper.ground import coarse  # noqa: E402


def write_tif(path, bands, x0, y0, res=10.0, epsg=32633):
    import numpy as np
    import tifffile
    arr = np.stack(bands).astype("uint8")
    extra = [(33550, "d", 3, (res, res, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, x0, y0, 0.0)),
             (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg))]
    tifffile.imwrite(path, arr if len(bands) > 1 else arr[0], extratags=extra, planarconfig="separate" if len(bands) > 1 else None)


def test_tiles_window_skip():
    reg = box(16.9, 52.0, 17.3, 52.25)                                   # ~27 × 28 km
    ts = coarse.tiles(reg, 20)
    assert 4 <= len(ts) <= 9 and all(t["id"].startswith("E") for t in ts)
    assert sum(t["km2"] for t in ts) == pytest.approx(geo_area(reg) / 1e6, rel=0.02)
    assert coarse.window("2026-10") == ("2026-08-01", "2026-11-01")
    assert coarse.window("2026-01") == ("2025-11-01", "2026-02-01")
    assert coarse.window("2026-12") == ("2026-10-01", "2027-01-01")
    assert coarse.skipped("2026-01", {}) and not coarse.skipped("2026-06", {})
    import datetime as dt
    assert coarse.last_month(dt.date(2026, 1, 5)) == "2025-12"


def geo_area(g):
    from pvmapper import geo
    return geo.ground_area_m2(g)


def test_geotiff_and_polygons(tmp_path):
    import numpy as np
    pv = np.zeros((200, 200), np.uint8)
    pv[20:80, 30:130] = 1                                                # 60 × 100 px = 60 ha at 10 m
    pv[40:50, 60:70] = 0                                                 # hole 1 ha
    pv[150:152, 150:152] = 1                                             # noise
    wc = np.full((200, 200), 40, np.uint8)
    wc[20:80, 30:80] = 10
    f = str(tmp_path / "t.tif")
    write_tif(f, [pv, wc], 500000.0, 5800000.0)
    arr, aff, crs = coarse.read_geotiff(f)
    assert arr.shape == (2, 200, 200) and crs == "EPSG:32633" and aff == (500000.0, 10.0, 5800000.0, -10.0)
    ps = coarse.polygons(arr[0], aff, crs)
    assert len(ps) == 1
    assert geo_area(ps[0]) == pytest.approx(60e4 - 1e4, rel=0.03)
    import cv2
    m = np.zeros((200, 200), np.uint8)
    m[20:80, 30:130] = 1
    wcf = arr[1].astype("float32")
    wcf[20:25, 30:130] = float("nan")
    assert coarse.worldcover_shares(wcf, m)["baum"] > 0.4
    sh = coarse.worldcover_shares(arr[1], m)
    assert sh["baum"] == pytest.approx(0.5, abs=0.01) and sh["acker"] == pytest.approx(0.5, abs=0.01)


def test_match_known_new_and_buildings(con):
    known = _row(geom_zaun=sq(17, 52, 200), flaeche_zaun_m2=40000, split="test")
    store.upsert(con, [known])
    new = sq(17.5, 52, 150)
    housing = sq(17.8, 52, 100)
    n = coarse.match(con, [sq(17, 52, 190), new, housing, sq(18, 52, 30)], "2026-10", "PL", "t", CFG,
                     [housing.buffer(-D)], [None, {"wasser": 0.9}, None, None])
    assert n == {"flaechen": 3, "zu_klein": 1, "gebaeude": 1, "bekannt": 1, "neu": 1}
    assert con.execute("SELECT zeitreihe FROM freiflaeche WHERE id='ff-pl-osm-w1'").fetchone()[0] is None   # see coverage()
    r = con.execute("SELECT herkunft, first_seen::text, typ, methode, quellen FROM freiflaeche WHERE herkunft='suche'").fetchone()
    assert r[:4] == ("suche", "2026-10-01", "floating", "s2_polygon") and r[4]["worldcover_2021"] == {"wasser": 0.9}
    # seen again next month: same id, confidence from the time series
    coarse.match(con, [new.buffer(D)], "2026-11", "PL", "t", CFG, [], None)
    coarse.confidence(con, "t")
    rows = con.execute("SELECT id, konfidenz, zeitreihe FROM freiflaeche WHERE herkunft='suche'").fetchall()
    assert len(rows) == 1 and rows[0][1] == 1.0 and set(rows[0][2]) == {"2026-10", "2026-11"}


def test_coverage_per_plant_on_the_raster(con):
    """a park split into many blobs: coverage from all PV pixels in its fence; tile edge without data ignored"""
    import numpy as np
    from pvmapper import geo as g
    fence = sq(17, 52, 400)                                               # 16 ha
    store.upsert(con, [_row(geom_zaun=fence, flaeche_zaun_m2=160000, split="test")])
    c = g.reproject(fence.centroid, "EPSG:4326", "EPSG:32633")
    x0, y0 = round(c.x) - 1000, round(c.y) + 1000
    aff, crs = (x0, 10.0, y0, -10.0), "EPSG:32633"
    pv = np.zeros((200, 200), np.float32)
    pv[80:120, 80:120] = 1
    pv[90:110, 90:110] = 0                                                # holes: one plant, several fragments
    pv[:, 190:] = 1                                                       # phantom strip at the edge without data
    wc = np.full((200, 200), 40, np.float32)
    wc[:, 190:] = np.nan
    bbox = list(box(*g.reproject(box(x0, y0 - 2000, x0 + 2000, y0), crs, "EPSG:4326").bounds).bounds)
    valid = coarse.valid_area(pv, wc, aff, crs, bbox)
    assert not valid[:, 195].any() and valid[100, 100]
    masks, n = coarse.coverage(con, np.where(valid, pv, 0), valid, aff, crs, bbox, "2026-10")
    z = con.execute("SELECT zeitreihe FROM freiflaeche WHERE id='ff-pl-osm-w1'").fetchone()[0]
    assert n == 1 and z["2026-10"]["s2"] == pytest.approx(0.75, abs=0.05)
    assert coarse.recall(con, "t", "2026-10") == {"n": 1, "hits": 1}
    assert coarse.recall(con, "t", "2026-09") == {"n": 0, "hits": 0}          # not looked at in that month


def test_scan_job_with_mocked_openeo(con, tmp_path, monkeypatch):
    import numpy as np
    from pvmapper.jobs import Job, runner
    base = tmp_path / "_ground" / "pl-wielkopolskie"
    base.mkdir(parents=True)
    (base / "region.poly").write_text("r\n1\n  16.9 52.0\n  16.9 52.1\n  17.1 52.1\n  17.1 52.0\nEND\nEND\n")
    calls = []

    def fake_tiles(conn, todo, start, end, d, cfg, job, state, save_state):
        calls.append((len(todo), start, end))
        from pvmapper import geo as g
        for t in todo:
            pv = np.zeros((100, 100), np.uint8)
            c = g.reproject(box(*t["bbox"]).centroid, "EPSG:4326", "EPSG:32633")
            pv[10:40, 10:60] = 1                                         # 15 ha
            write_tif(os.path.join(d, t["id"] + ".tif"), [pv], round(c.x) - 500, round(c.y) + 500)
            state.setdefault("done", []).append(t["id"])
    monkeypatch.setattr(coarse, "connect", lambda cfg, log=print: object())
    monkeypatch.setattr(coarse, "run_tiles", fake_tiles)
    cfg = {**CFG, "ground_db": DSN, "jobs_dir": str(tmp_path)}
    job = Job(str(tmp_path), "ground_scan", {"region": "pl-wielkopolskie", "month": "2026-01"})
    runner("ground_scan")(job, cfg)
    assert not calls and any("Winterfenster" in ln for ln in job.log_lines)
    job = Job(str(tmp_path), "ground_scan", {"region": "pl-wielkopolskie", "month": "2026-09", "dry_run": True})
    runner("ground_scan")(job, cfg)
    assert not calls and any("Credits" in ln for ln in job.log_lines)
    job = Job(str(tmp_path), "ground_scan", {"region": "pl-wielkopolskie", "month": "2026-09", "max_tiles": 2})
    runner("ground_scan")(job, cfg)
    assert calls == [(2, "2026-07-01", "2026-10-01")]
    assert con.execute("SELECT count(*) FROM freiflaeche WHERE herkunft='suche'").fetchone()[0] == 2
    assert job.counters["neu"] == 2
    con.execute("UPDATE freiflaeche SET first_seen='2026-09-01', zeitreihe='{\"2026-09\":{\"s2\":1}}' WHERE herkunft='suche'")
    store.upsert(con, [_row("ff-pl-s2-202609-phantom", herkunft="suche", region="pl-wielkopolskie", first_seen="2026-09-01", zeitreihe={"2026-09": {"s2": 1}},
                            geom_zaun=box(*coarse.tiles(inventory_poly(base), 20)[0]["bbox"]).centroid.buffer(0.001))])
    job = Job(str(tmp_path), "ground_scan", {"region": "pl-wielkopolskie", "month": "2026-09", "max_tiles": 2})
    runner("ground_scan")(job, cfg)
    ids = [r[0] for r in con.execute("SELECT id FROM freiflaeche WHERE herkunft='suche'")]
    assert "ff-pl-s2-202609-phantom" not in ids and len(ids) == 2           # rerun of the month replaced its candidates


def inventory_poly(base):
    from pvmapper.ground import inventory as gi
    return gi.read_poly((base / "region.poly").read_text())


def test_cpu_lane_and_monthly_tick(tmp_path, monkeypatch):
    import datetime as dt
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PVMAPPER_NO_SCHEDULER", "1")
    from pvmapper.web import app as webapp
    p = tmp_path / "pvmapper.yml"
    p.write_text(f"jobs_dir: {tmp_path}/jobs\nground_scan_enabled: true\nground_pilot: pl-wielkopolskie\n")
    a = webapp.create_app(str(p))
    TestClient(a)
    started = []
    monkeypatch.setattr(webapp.JobManager, "submit", lambda self, job: started.append(job))
    tick = a.state.ground_tick
    assert tick(dt.datetime(2026, 11, 3, 2)) is None                     # before the scan day
    assert tick(dt.datetime(2026, 11, 6, 14)) is None                    # daytime
    j = tick(dt.datetime(2026, 11, 6, 2))
    assert j.kind == "ground_scan" and j.params["month"] == "2026-10"
    assert tick(dt.datetime(2026, 11, 7, 2)) is None                     # once per month
    assert tick(dt.datetime(2027, 2, 6, 2)) is None                      # January window skipped
    assert "ground_scan" in webapp.CPU_KINDS


def test_review_by_hand_survives_imports(con):
    store.upsert(con, [_row(), _row("ff-pl-osm-w2", quellen={"osm": ["w2"]})])
    assert store.review(con, "ff-pl-osm-w1", "type", typ="tracker")["typ"] == "tracker"
    store.upsert(con, [_row()])                                              # re-import: type set by hand stays
    assert con.execute("SELECT typ, pruefstatus FROM freiflaeche WHERE id='ff-pl-osm-w1'").fetchone() == ("tracker", "kandidat")
    store.review(con, "ff-pl-osm-w2", "discard", reason="gewaechshaus")
    n = store.upsert(con, [_row("ff-pl-osm-w2", quellen={"osm": ["w2"], "neu": 1})])
    assert n["geschuetzt"] == 1
    r = con.execute("SELECT pruefstatus, verwurf_grund, jsonb_array_length(historie) FROM freiflaeche WHERE id='ff-pl-osm-w2'").fetchone()
    assert r == ("verworfen", "manuell:gewaechshaus", 1)
    store.review(con, "ff-pl-osm-w1", "confirm")
    assert store.features(con, "id = %s", ["ff-pl-osm-w1"])["features"][0]["properties"]["historie"][-1]["aktion"] == "confirm"
    assert store.review(con, "ff-pl-osm-w2", "reset")["pruefstatus"] == "kandidat"
    with pytest.raises(ValueError):
        store.review(con, "ff-pl-osm-w1", "discard")                       # reason required
    with pytest.raises(KeyError):
        store.review(con, "gibt-es-nicht", "confirm")


def test_rescan_does_not_resurrect_discarded_candidate(con):
    g = sq(17.5, 52, 150)
    coarse.match(con, [g], "2026-09", "PL", "t", CFG, [], None)
    fid = con.execute("SELECT id FROM freiflaeche WHERE herkunft='suche'").fetchone()[0]
    store.review(con, fid, "discard", reason="keine_pv")
    n = coarse.match(con, [g.buffer(D)], "2026-10", "PL", "t", CFG, [], None)
    assert n["neu"] == 0 and n["bekannt"] == 1
    assert con.execute("SELECT count(*) FROM freiflaeche WHERE herkunft='suche'").fetchone()[0] == 1
