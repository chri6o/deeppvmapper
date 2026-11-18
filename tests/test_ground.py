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
