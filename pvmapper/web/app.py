"""Local web UI (FastAPI). Start with: python -m pvmapper web"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import shutil
import sqlite3
import threading
import time

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from shapely import wkt as shp_wkt
from shapely.geometry import Point, mapping

from .. import area_mode, building_mode, buildings as bld, config as cfgmod
from .. import ai_label, annotate, geo, inference, models, runtime, postproc, sources, typology
from ..jobs import Job, runner

STATIC = os.path.join(os.path.dirname(__file__), "static")


CPU_KINDS = ("ground_inventory", "ground_scan", "ground_capacity", "ground_register", "ground_history")     # no GPU, mostly waiting for downloads/openEO: own lane


class JobManager:
    """Runs one GPU job at a time (the GPU is the shared resource); ground-mounted PV jobs run one at a time in a second lane."""

    def __init__(self, get_cfg):
        self.get_cfg = get_cfg
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        self.q_cpu: queue.Queue = queue.Queue()
        self.current: Job | None = None
        self.current_cpu: Job | None = None
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._worker, args=(True,), daemon=True).start()

    def load_existing(self):
        jd = self.get_cfg()["jobs_dir"]
        if not os.path.isdir(jd):
            return
        for jid in sorted(os.listdir(jd)):
            if os.path.isfile(os.path.join(jd, jid, "job.json")):
                try:
                    job = Job.load(jd, jid)
                    if job.status in ("running", "queued"):
                        job.status = "cancelled"        # server was stopped mid-run
                    self.jobs[jid] = job
                except Exception:                       # noqa: BLE001
                    pass

    def submit(self, job: Job):
        job.status = "queued"
        job.save()
        self.jobs[job.id] = job
        (self.q_cpu if job.kind in CPU_KINDS else self.q).put(job)

    def _worker(self, cpu: bool = False):
        q = self.q_cpu if cpu else self.q
        while True:
            job = q.get()
            if job._cancel.is_set():
                job.status = "cancelled"
                job.save()
                continue
            setattr(self, "current_cpu" if cpu else "current", job)
            cfg = self.get_cfg()
            fn = runner(job.kind)
            job.run(lambda j: fn(j, cfg))
            setattr(self, "current_cpu" if cpu else "current", None)


def _coords(g):
    """all (x, y) of a GeoJSON geometry"""
    def walk(c):
        if c and isinstance(c[0], (int, float)):
            yield (c[0], c[1])
        else:
            for x in c or []:
                yield from walk(x)
    return list(walk((g or {}).get("coordinates")))


def _png_b64(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def create_app(config_path: str = "pvmapper.yml") -> FastAPI:
    app = FastAPI(title="pvmapper")
    state = {"cfg": cfgmod.load(config_path)}

    def cfg():
        return state["cfg"]

    os.makedirs(cfg()["jobs_dir"], exist_ok=True)
    upload_dir = os.path.join(cfg()["jobs_dir"], "_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    mgr = JobManager(cfg)
    mgr.load_existing()

    # night sync with PV-Map: one sync job per time window, started when the GPU is free (PV-Map docs/pvmapper-ergebnisse.md)
    def sync_tick(now=None):
        from .. import sync
        c = cfg()
        if not c.get("sync_enabled") or not c.get("pvmap_token"):
            return None
        w = sync.window(c, now)
        if not w:
            return None
        key = w[0].isoformat()
        if state.get("sync_window") == key:
            return None                                          # already started in this window
        if mgr.current is not None or not mgr.q.empty():
            return None                                          # another job has the GPU: try again in a minute
        state["sync_window"] = key
        job = Job(c["jobs_dir"], "sync", {"filter": c.get("sync_filter") or "cc=DE", "until": w[1].isoformat(), "auto": True})
        mgr.submit(job)
        return job

    def ground_tick(now=None):
        """monthly coarse search for ground-mounted PV: once per month from `ground_scan_day`, at night (01–05 h)"""
        import datetime as _dt
        from ..ground import coarse
        c = cfg()
        if not c.get("ground_scan_enabled"):
            return None
        now = now or _dt.datetime.now()
        if now.day < int(c.get("ground_scan_day") or 5) or not 1 <= now.hour < 5:
            return None
        month = coarse.last_month(now.date())
        key = f"{c.get('ground_pilot')}:{month}"
        if state.get("ground_scan") == key or coarse.skipped(month, c) or mgr.current_cpu is not None:
            return None
        state["ground_scan"] = key
        job = Job(c["jobs_dir"], "ground_scan", {"region": c.get("ground_pilot"), "month": month, "auto": True})
        mgr.submit(job)
        return job

    def sync_loop():
        import time as _t
        while True:
            _t.sleep(60)
            for tick in (sync_tick, ground_tick):
                try:
                    tick()
                except Exception:                                # noqa: BLE001
                    pass
    if not os.environ.get("PVMAPPER_NO_SCHEDULER"):
        threading.Thread(target=sync_loop, name="sync", daemon=True).start()
    app.state.sync_tick = sync_tick
    app.state.ground_tick = ground_tick

    @app.middleware("http")
    async def password_gate(request, call_next):
        """optional HTTP basic auth for use in the local network (config: web_password; any user name)"""
        pw = cfg().get("web_password") or ""
        if pw:
            import secrets as _s
            h = request.headers.get("authorization", "")
            ok = False
            if h.lower().startswith("basic "):
                try:
                    ok = _s.compare_digest(base64.b64decode(h[6:]).decode().split(":", 1)[1], pw)
                except Exception:                                # noqa: BLE001
                    ok = False
            if not ok:
                from fastapi import Response
                return Response("Passwort erforderlich", status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="PV-Mapper", charset="UTF-8"'})
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/favicon.ico")
    def favicon():
        from fastapi import Response
        return Response(status_code=204)

    # -- settings / info --------------------------------------------------------

    @app.get("/api/info")
    def info():
        prof = []
        for p in models.all_profiles(cfg()["model_dir"]).values():
            have = all(os.path.isfile(os.path.join(cfg()["model_dir"], f if f.startswith("custom/") else os.path.basename(f)))
                       for f in (p.cls_file, p.seg_file))
            prof.append({"id": p.id, "label": p.label, "resolution_m": p.resolution_m,
                         "licence": p.licence, "downloaded": have,
                         "cls_threshold": p.cls_threshold, "seg_threshold": p.seg_threshold})
        return {"device": models.device_info(), "profiles": prof, "config": cfgmod.public(cfg()),
                "config_path": os.path.abspath(config_path)}

    @app.post("/api/config")
    def set_config(new: dict = Body(...)):
        c = dict(cfg())
        for k, v in new.items():
            if k in cfgmod.DEFAULTS and not (k in cfgmod.SECRET_KEYS and v == cfgmod.MASK):
                c[k] = v
        cfgmod.save(c, config_path)
        state["cfg"] = cfgmod.load(config_path)
        return cfgmod.public(state["cfg"])

    @app.post("/api/models/download")
    def download_models(body: dict = Body(default={})):
        pid = body.get("profile") or cfg()["profile"]
        p = models.get_profile(pid, cfg()["model_dir"])
        paths = [models.weight_path(cfg()["model_dir"], f) for f in (p.cls_file, p.seg_file)]
        return {"ok": True, "paths": paths}

    # -- sources / probe ------------------------------------------------------------

    @app.get("/api/sources")
    def list_sources():
        cat = runtime.catalog(cfg())
        return [{**d, "url": s.url, "layers": s.layers, "crs": s.crs, "max_zoom": s.max_zoom}
                for d, s in zip(cat.as_dicts(), cat.sources.values())]

    @app.post("/api/probe")
    def probe(body: dict = Body(...)):
        """Fetch imagery around a point and (optionally) run the models on it."""
        lat, lon = float(body["lat"]), float(body["lon"])
        radius = float(body.get("radius_m", 40))
        c = cfg()
        prof = models.get_profile(body.get("profile") or c["profile"], c["model_dir"])
        cat = runtime.catalog(c)
        g = Point(lon, lat)
        utm = f"EPSG:{geo.utm_epsg(lon, lat)}"
        area = geo.reproject(geo.reproject(g, "EPSG:4326", utm).buffer(radius, cap_style=3), utm, "EPSG:4326")
        sel = [body["source"]] if body.get("source") and body["source"] != "auto" else None
        t0 = time.time()
        try:
            img, win, src = sources.fetch_for_geometry(cat, area, prof.resolution_m, 0, sel,
                                                       min_px=prof.patch_px)
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(400, f"Kein Bild: {e}")
        t_fetch = time.time() - t0
        out = {"source": src.id, "source_name": src.name, "attribution": src.attribution,
               "fetch_s": round(t_fetch, 2), "px": [img.shape[1], img.shape[0]]}
        mask = None
        if body.get("detect", True):
            det = runtime.detector(c, prof.id)
            t1 = time.time()
            r = det.run([("p", img)])["p"]
            mask = r.mask
            if c.get("postproc", True):
                mask, _ = postproc.clean(img, mask, prof.resolution_m, None, dict(c, postproc_complete=False), prob=r.prob)
            polys = inference.mask_to_polygons(mask, win)
            area_m2 = sum(geo.ground_area_m2(p) for p in polys)
            kwp = round(area_m2 * float(c["kwp_per_m2"]), 1)
            tf, sm = [], {}
            if polys and c.get("typology", True):
                try:
                    tf = typology.for_window(img, mask, win, prof.resolution_m, src, "Probe", None, None, c)
                    if tf and c.get("age_check", True) and body.get("age", True):
                        from .. import age
                        age.estimate([{"img": img, "win": win, "src": src, "mask": mask, "feats": tf,
                                       "res_m": prof.resolution_m}], det, c, int(c.get("fetch_workers") or 4))
                    sm = typology.summarise([f["properties"] for f in tf])
                except Exception as e:                           # noqa: BLE001
                    out["typology_error"] = f"{type(e).__name__}: {e}"[:200]
            out.update(detect_s=round(time.time() - t1, 2), max_prob=round(r.max_prob, 3),
                       n_patches=r.n_patches, n_positive=r.n_positive, pv_area_m2=round(area_m2, 1),
                       kwp_est=sm.get("kwp_est", kwp), kwp_low=sm.get("kwp_low"), kwp_high=sm.get("kwp_high"),
                       layout=sm.get("layout"), kwp_method="typology" if sm else "area_factor",
                       installations={"type": "FeatureCollection", "features": typology.map_features(tf, "Probe")},
                       polygons={"type": "FeatureCollection",
                                 "features": [{"type": "Feature", "geometry": mapping(p), "properties": {}}
                                              for p in polys]})
        out["image"] = _png_b64(runtime.render_chip(img, mask, win, max_w=1100))
        return out

    # -- uploads ------------------------------------------------------------------------

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        name = runtime.safe_name(os.path.basename(file.filename or "upload"))
        path = os.path.join(upload_dir, time.strftime("%Y%m%d-%H%M%S-") + name)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return inspect_input({"input": path})

    @app.post("/api/inspect")
    def inspect_input(body: dict = Body(...)):
        """Counts records and returns a small preview of a building input."""
        kw = dict(id_field=body.get("id_field") or None, query=body.get("query") or None,
                  layer=body.get("layer") or None, cfg=cfg())
        if str(body["input"]).startswith("pvmap:"):
            from .. import pvmap
            try:
                cl = pvmap.Client.from_config(cfg())
                flt = pvmap.parse_input(body["input"])
                cnt = cl.count(**flt)
                page = cl._get("/detect/buildings", {**flt, "limit": 300})["features"]
            except Exception as e:                               # noqa: BLE001
                raise HTTPException(400, f"PV-Map: {e}")
            xs = [c for f in page for c in _coords(f["geometry"])]
            bbox = [min(x for x, _ in xs), min(y for _, y in xs), max(x for x, _ in xs), max(y for _, y in xs)] if xs else None
            return {"input": body["input"], "count": cnt["total"], "breakdown": cnt["rows"],
                    "fields": list((page[0]["properties"] if page else {}).keys()), "bbox": bbox,
                    "preview": {"type": "FeatureCollection", "features": [
                        {"type": "Feature", "geometry": f["geometry"], "properties": {"id": f["id"]}} for f in page]}}
        try:
            n, preview, keys = 0, [], []
            minx = miny = 1e9
            maxx = maxy = -1e9
            for b in bld.read_buildings(body["input"], **kw):
                n += 1
                x0, y0, x1, y1 = b.geometry.bounds
                minx, miny, maxx, maxy = min(minx, x0), min(miny, y0), max(maxx, x1), max(maxy, y1)
                if len(preview) < 300:
                    preview.append({"type": "Feature", "geometry": mapping(b.geometry),
                                    "properties": {"id": b.id}})
                if not keys:
                    keys = list(b.props.keys())
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(400, f"Eingabe nicht lesbar: {e}")
        return {"input": body["input"], "count": n, "fields": keys,
                "bbox": [minx, miny, maxx, maxy] if n else None,
                "preview": {"type": "FeatureCollection", "features": preview}}

    # -- PV-Map connector ---------------------------------------------------------------------

    @app.post("/api/pvmap/strata")
    def pvmap_strata(body: dict = Body(...)):
        from .. import pvmap
        flt = {k: body.get(k) for k in pvmap.FILTER_KEYS if body.get(k)}
        if not flt.get("cc"):
            raise HTTPException(400, "Bitte ein Land wählen.")
        try:
            return pvmap.Client.from_config(cfg()).strata(bins=body.get("bins") or None, **flt)
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(400, f"PV-Map: {e}")

    # -- ground-mounted PV (docs/freiflaechen.md) ----------------------------------------------------------------------

    def _gcon():
        from ..ground import store as gstore
        try:
            con = gstore.connect(cfg())
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(503, f"Freiflächen-Datenbank nicht erreichbar ({e}). Einmal „sudo pvmapper-update“ ausführen.")
        gstore.migrate(con)
        return con

    @app.get("/api/ground/regions")
    def ground_regions():
        from ..ground import inventory as ginv
        return {"pilot": cfg().get("ground_pilot"), "regions": ginv.regions(cfg())}

    @app.get("/api/ground/stats")
    def ground_stats(region: str | None = None):
        from ..ground import store as gstore
        with _gcon() as con:
            return gstore.stats(con, region)

    @app.get("/api/ground/features")
    def ground_features(region: str | None = None, bbox: str | None = None, pruefstatus: str | None = None,
                        herkunft: str | None = None, limit: int = 20000):
        from ..ground import store as gstore
        w, a = ["TRUE"], []
        if region:
            w.append("region = %s"); a.append(region)
        if pruefstatus:
            w.append("pruefstatus = ANY(%s)"); a.append(pruefstatus.split(","))
        if herkunft:
            w.append("herkunft = ANY(%s)"); a.append(herkunft.split(","))
        if bbox:
            x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
            w.append("geom_zaun && ST_MakeEnvelope(%s,%s,%s,%s,4326)"); a += [x0, y0, x1, y1]
        with _gcon() as con:
            return gstore.features(con, " AND ".join(w), a, min(int(limit), 50000))

    @app.get("/api/ground/zubau")
    def ground_zubau(region: str | None = None, cc: str | None = None, group: str = "none", confirmed: bool = False,
                     format: str = "json"):
        from ..ground import history as ghist
        with _gcon() as con:
            z = ghist.zubau(con, region, cc, group, confirmed)
            if region:
                z["register"] = ghist.register_quarters(con, region)
        if format == "csv":
            import csv as _csv
            buf = io.StringIO()
            wr = _csv.writer(buf, delimiter=";")
            wr.writerow(["quartal", "segment", "anlagen", "dc_mw_min", "dc_mw_max", "ac_mw_min", "ac_mw_max"])
            for e in z["reihe"]:
                wr.writerow([e["quartal"], e["segment"], e["n"], e["dc_mw_min"], e["dc_mw_max"], e["ac_mw_min"], e["ac_mw_max"]])
            return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                            headers={"Content-Disposition": f"attachment; filename=zubau_{region or cc or 'alle'}_{group}.csv"})
        return z

    @app.get("/api/ground/register/gaps")
    def ground_gaps(region: str):
        from ..ground import register as greg
        with _gcon() as con:
            return {"region": region, "orte": greg.gaps(con, region)}

    @app.get("/api/ground/calibration/{region}")
    def ground_calibration(region: str):
        from ..ground import inventory as ginv
        p = os.path.join(ginv.workdir(cfg(), region), "calibration.json")
        if not os.path.exists(p):
            raise HTTPException(404, "Noch keine Kalibrierung für diese Region")
        with open(p) as f:
            return json.load(f)

    @app.post("/api/ground/{fid}/review")
    def ground_review(fid: str, body: dict = Body(...)):
        from ..ground import store as gstore
        with _gcon() as con:
            try:
                return gstore.review(con, fid, body.get("action"), body.get("reason"), body.get("typ"), body.get("note"))
            except KeyError:
                raise HTTPException(404, "Fläche nicht gefunden")
            except ValueError as e:
                raise HTTPException(400, str(e))

    @app.get("/api/roofshape/models")
    def roofshape_models():
        from .. import roofshape
        return roofshape.list_models(cfg())

    @app.get("/api/jobs/{jid}/roofimg/{bid}")
    def roofshape_image(jid: str, bid: str):
        path = os.path.join(_job(jid).dir, "dataset", os.path.basename(bid) + ".jpg")
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/jobs/{jid}/report")
    def job_report(jid: str):
        path = os.path.join(_job(jid).dir, "report.json")
        if not os.path.isfile(path):
            raise HTTPException(404, "noch kein Bericht")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # -- datasets & masks ---------------------------------------------------------------------

    def _ds(jid):
        d = annotate.ds_dir(_job(jid).dir)
        if not os.path.isdir(d):
            raise HTTPException(404, "Dieser Auftrag hat keinen Datensatz")
        return d

    def _item(jid, name):
        d = _ds(jid)
        name = os.path.basename(name)
        if not os.path.isfile(os.path.join(d, name + ".json")):
            raise HTTPException(404, "Bild nicht im Datensatz")
        return d, name

    @app.get("/api/datasets")
    def datasets():
        out = []
        for j in mgr.jobs.values():
            d = annotate.ds_dir(j.dir)
            if j.kind == "train" and os.path.isdir(d) and not j.params.get("dataset_from"):
                items = annotate.list_items(d)
                if items:
                    out.append({"id": j.id, "name": j.params.get("name") or j.params.get("cc") or "Labels",
                                "prepare_only": bool(j.params.get("prepare_only")), "status": j.status,
                                "profile": j.params.get("profile"), "stats": annotate.stats(items)})
        return sorted(out, key=lambda x: x["id"], reverse=True)

    @app.get("/api/datasets/{jid}/items")
    def dataset_items(jid: str):
        items = annotate.list_items(_ds(jid))
        keep = ("id", "name", "label", "effective_label", "label_kind", "sub", "size_bin", "pv_kwp", "has_mask",
                "mask_source", "status", "val", "source", "ai", "use_register")
        return {"stats": annotate.stats(items), "items": [{**{k: m.get(k) for k in keep}, "has_register": bool(
            m.get("register") and any(b.get("register") for b in m["register"].get("buildings") or []))} for m in items]}

    @app.get("/api/datasets/{jid}/items/{name}")
    def dataset_item(jid: str, name: str):
        d, name = _item(jid, name)
        m = annotate.read_meta(d, name)
        w, h = annotate.image_size(d, name)
        return {"meta": m, "width": w, "height": h, "footprint": annotate.footprint_polygons(d, name),
                "polygons": annotate.current_polygons(d, name), "negatives": annotate.current_polygons(d, name, "neg"),
                "effective_label": annotate.effective_label(m)}

    @app.get("/api/datasets/{jid}/items/{name}/image")
    def dataset_image(jid: str, name: str):
        d, name = _item(jid, name)
        return FileResponse(os.path.join(d, name + ".jpg"), headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/datasets/{jid}/items/{name}")
    def dataset_save(jid: str, name: str, body: dict = Body(...)):
        d, name = _item(jid, name)
        st = body.get("status", "done")
        if st not in ("open", "done", "skip"):
            raise HTTPException(400, "status: open, done oder skip")
        meta = annotate.save_annotation(d, name, body.get("polygons"), body.get("label"), st, "user",
                                        negatives=body.get("negatives"))
        if st == "done" and cfg().get("sync_push_reviews", True) and cfg().get("pvmap_token"):
            threading.Thread(target=_push_review, args=(d, name, meta), daemon=True).start()
        return meta

    def _push_review(d, name, meta):
        """a mask checked by hand → PV-Map as a reviewed result (confirmed PV or no PV); buildings PV-Map does not know are ignored there"""
        try:
            from .. import pvmap
            item = annotate.review_result(d, name, meta, float(cfg().get("kwp_per_m2") or 0.2), cfg())
            if item:
                pvmap.Client.from_config(cfg()).send_results([item], run="review", model="review")
        except Exception:                                        # noqa: BLE001
            pass

    @app.get("/api/sync/status")
    def sync_status():
        """open buildings in the PV-Map work list per priority, the night window and the last sync jobs"""
        from .. import pvmap, sync
        c = cfg()
        out = {"enabled": bool(c.get("sync_enabled")), "start": c.get("sync_start"), "end": c.get("sync_end"),
               "filter": c.get("sync_filter"), "in_window": bool(sync.window(c)),
               "jobs": [j.as_dict() | {"log": None} for j in sorted(mgr.jobs.values(), key=lambda j: j.id, reverse=True)
                        if j.kind == "sync"][:5]}
        try:
            q = pvmap.parse_queue("pvmapqueue:" + (c.get("sync_filter") or "cc=DE"))
            q.pop("limit", None)
            out["queue"] = pvmap.Client.from_config(c).queue_count(model=(c.get("sync_profile") or c.get("profile")),
                                                                    recheck_days=int(c.get("sync_recheck_days") or 365), **q)
        except Exception as e:                                   # noqa: BLE001
            out["queue_error"] = str(e)[:300]
        return out

    @app.post("/api/sync/start")
    def sync_start(body: dict = Body(default={})):
        """start a sync now (optionally limited to `hours` or `max` buildings)"""
        import datetime as _dt
        p = {"filter": body.get("filter") or cfg().get("sync_filter") or "cc=DE"}
        if body.get("hours"):
            p["until"] = (_dt.datetime.now().astimezone() + _dt.timedelta(hours=float(body["hours"]))).isoformat()
        if body.get("max"):
            p["max"] = int(body["max"])
        job = Job(cfg()["jobs_dir"], "sync", p)
        mgr.submit(job)
        return job.as_dict()

    @app.post("/api/datasets/{jid}/register")
    def dataset_register(jid: str, body: dict = Body(default={})):
        """register data (MaStR via PV-Map) for all images of a dataset or the given ones (`names`)"""
        from .. import pvmap
        job = _job(jid)
        d = annotate.ds_dir(job.dir)
        if not os.path.isdir(d):
            raise HTTPException(404, "Datensatz nicht gefunden")
        try:
            cl = pvmap.Client.from_config(cfg())
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(400, str(e))
        return annotate.register_lookup(d, cl, body.get("names"), float(body.get("radius_m") or 30))

    @app.post("/api/datasets/{jid}/items/{name}/flags")
    def dataset_flags(jid: str, name: str, body: dict = Body(...)):
        """per image: use its register capacity for the capacity calibration (`use_register`)"""
        d, name = _item(jid, name)
        meta = annotate.read_meta(d, name)
        if "use_register" in body:
            meta["use_register"] = bool(body["use_register"])
        with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return {"use_register": meta.get("use_register", False)}

    @app.post("/api/datasets/{jid}/calibrate")
    def dataset_calibrate(jid: str, body: dict = Body(default={})):
        """capacity calibration from the register capacities of a dataset (use: marked | all); save=True stores it for
        `profile` (default: the dataset's model) and every later run with that model uses it"""
        from .. import calibration
        job = _job(jid)
        d = annotate.ds_dir(job.dir)
        if not os.path.isdir(d):
            raise HTTPException(404, "Datensatz nicht gefunden")
        c = cfg()
        use = body.get("use") or "marked"
        if use not in ("marked", "all"):
            raise HTTPException(400, "use: marked oder all")
        prof = body.get("profile") or job.params.get("profile") or c.get("profile")
        if body.get("lookup", True):
            calibration.ensure_register(d, c, use)
        cal = calibration.compute(d, {**c, "profile": prof}, use)
        cal["profile"] = prof
        if body.get("save") and cal["fill_factor"]:
            cal["path"] = calibration.save(c["model_dir"], prof, cal)
        return cal

    @app.get("/api/calibration/{profile}")
    def calibration_get(profile: str):
        from .. import calibration
        return calibration.load(cfg()["model_dir"], profile) or {}

    @app.delete("/api/calibration/{profile}")
    def calibration_delete(profile: str):
        from .. import calibration
        return {"removed": calibration.remove(cfg()["model_dir"], profile)}

    @app.get("/api/geocode")
    def geocode(q: str, limit: int = 6):
        """place / region search (OSM Nominatim): name, bbox and boundary (simplified) of matching places"""
        import requests as _rq
        q = (q or "").strip()
        if len(q) < 2:
            return []
        try:
            r = _rq.get("https://nominatim.openstreetmap.org/search", timeout=15,
                        params={"q": q, "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0.0003,
                                "limit": max(1, min(int(limit), 10)), "accept-language": "de"},
                        headers={"User-Agent": "PV-Mapper (https://github.com/chri6o/deeppvmapper)"})
            r.raise_for_status()
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(502, f"Ortssuche nicht erreichbar: {type(e).__name__}")
        out = []
        for x in r.json():
            s_, n_, w_, e_ = [float(v) for v in x["boundingbox"]]
            g = x.get("geojson") if (x.get("geojson") or {}).get("type") in ("Polygon", "MultiPolygon") else None
            out.append({"name": x.get("name") or x.get("display_name", "").split(",")[0], "label": x.get("display_name"),
                        "kind": x.get("addresstype") or x.get("type"), "bbox": [w_, s_, e_, n_], "geometry": g,
                        "lat": float(x["lat"]), "lon": float(x["lon"])})
        return out

    @app.post("/api/osm/buildings")
    def osm_buildings(body: dict = Body(...)):
        """OSM building footprints in a bbox [w, s, e, n] or a polygon (GeoJSON geometry); with `inside` only those whose
        centre lies in the polygon. At most 4 km² per request."""
        from shapely.geometry import box as _box, shape as _shape
        from .. import osm
        if body.get("geometry"):
            region = _shape(body["geometry"])
        elif body.get("bbox"):
            a, b_, c_, d_ = [float(x) for x in body["bbox"]]
            region = _box(min(a, c_), min(b_, d_), max(a, c_), max(b_, d_))
        else:
            raise HTTPException(400, "bbox oder geometry fehlt")
        if geo.ground_area_m2(region) > 4e6:
            raise HTTPException(400, "Gebiet zu groß (höchstens 4 km²) – bitte näher heranzoomen oder kleiner zeichnen.")
        try:
            feats = osm.buildings(region, quick=True)
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(502, str(e))
        if body.get("geometry") and body.get("inside", True):
            feats = [f for f in feats if region.contains(_shape(f["geometry"]).representative_point())]
        return {"type": "FeatureCollection", "features": feats[:5000], "truncated": len(feats) > 5000,
                "attribution": "© OpenStreetMap contributors, ODbL"}

    @app.post("/api/roofs/start")
    def roofs_start(body: dict = Body(...)):
        """roof potential (and PV check) for buildings picked on the map: the features are stored as an input file and a
        'check buildings' job with roof_potential starts"""
        feats = (body.get("features") or [])[:20000]
        if not feats:
            raise HTTPException(400, "Keine Gebäude ausgewählt")
        path = os.path.join(upload_dir, time.strftime("%Y%m%d-%H%M%S-") + f"dachpotenzial-{len(feats)}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
        params = {"input": path, "roof_potential": body.get("roof_potential", True), "id_field": "id",
                  "sources": body.get("sources") or "auto", "profile": body.get("profile") or cfg().get("profile"),
                  "save_chips": body.get("save_chips") or "all",
                  **{k: body[k] for k in ("typology", "age_check", "cross_check") if body.get(k) is not None}}
        job = Job(cfg()["jobs_dir"], "buildings", params)
        mgr.submit(job)
        return job.as_dict()

    @app.post("/api/probe/dataset")
    def probe_dataset(body: dict = Body(...)):
        """the probe cut-out with its detected PV as an image of a training dataset (new dataset or appended)"""
        from shapely.geometry import shape, mapping
        from shapely.ops import unary_union
        lat, lon, radius = float(body["lat"]), float(body["lon"]), float(body.get("radius_m", 40))
        utm = f"EPSG:{geo.utm_epsg(lon, lat)}"
        area = geo.reproject(geo.reproject(Point(lon, lat), "EPSG:4326", utm).buffer(radius, cap_style=3), utm, "EPSG:4326")
        polys = [shape(f["geometry"]) for f in ((body.get("polygons") or {}).get("features") or [])]
        pv = unary_union(polys) if polys else None
        bid = f"PROBE-{lat:.5f}_{lon:.5f}"
        c = cfg()
        params = {"prepare_only": True, "profile": body.get("profile") or c.get("profile"),
                  "sources": [body["source"]] if body.get("source") and body["source"] != "auto" else None,
                  "name": body.get("name") or f"Probe {lat:.4f} {lon:.4f}", "append_to": body.get("append_to") or None,
                  "features": [{"type": "Feature", "id": bid, "geometry": mapping(area), "properties": {
                      "id": bid, "label": int(pv is not None), "label_kind": "det_pv" if pv is not None else "det_none",
                      "pv_geometry": mapping(pv) if pv is not None else None, "mask_source": "model",
                      "kwp_detected": body.get("kwp_est"), "source_model": body.get("profile") or c.get("profile"),
                      "extra": {"lon": lon, "lat": lat, "install_year": body.get("install_year"), "layout": body.get("layout")}}}]}
        if params["append_to"]:
            tgt = _job(params["append_to"])
            if not os.path.isdir(annotate.ds_dir(tgt.dir)):
                raise HTTPException(400, "Ziel-Datensatz nicht gefunden")
        job = Job(c["jobs_dir"], "train", params)
        mgr.submit(job)
        return job.as_dict()

    @app.post("/api/datasets/{jid}/items/{name}/typology")
    def dataset_typology(jid: str, name: str, body: dict = Body(default={})):
        """installation typology of the mask currently drawn in the editor (not saved)"""
        d, name = _item(jid, name)
        t = annotate.typology_of(d, name, annotate.read_meta(d, name), body.get("polygons"), cfg())
        return {"summary": t["summary"], "installations": t["installations"]}

    @app.get("/api/datasets/{jid}/items/{name}/corners")
    def dataset_corners(jid: str, name: str):
        d, name = _item(jid, name)
        return {"points": annotate.corner_points(d, name)}

    @app.post("/api/datasets/{jid}/items/{name}/suggest")
    def dataset_suggest(jid: str, name: str, body: dict = Body(default={})):
        d, name = _item(jid, name)
        src = "ai" if body.get("provider") else "model"
        det = runtime.detector(cfg(), _job(jid).params.get("profile"))
        if src == "model":
            return {"source": "model", **annotate.suggest_from_model(det, d, name, body.get("cls_threshold", 0.3))}
        try:
            r = ai_label.label_item(cfg(), d, name, body["provider"], body.get("mode", "candidates"), det,
                                    det.p.resolution_m)
        except ai_label.AIError as e:
            raise HTTPException(400, str(e))
        return {"source": "ai", **r}

    # -- jobs ------------------------------------------------------------------------------

    @app.post("/api/jobs")
    def create_job(body: dict = Body(...)):
        kind = body.get("kind")
        params = body.get("params") or {}
        if kind not in ("buildings", "area", "train", "ai_masks", "sync", "roofshape", "ground_inventory", "ground_scan", "ground_capacity", "ground_register", "ground_history"):
            raise HTTPException(400, "unknown job kind")
        if kind == "buildings" and not params.get("input"):
            raise HTTPException(400, "Keine Gebäudedatei gewählt")
        if kind == "train" and params.get("from_job"):
            src = _job(params["from_job"])
            if src.kind not in ("buildings", "sync", "area"):
                raise HTTPException(400, "Nur Ergebnisse von „Gebäude prüfen“ oder „Fläche auswerten“ lassen sich übernehmen.")
            params.setdefault("profile", src.params.get("profile") or cfg().get("profile"))   # the model that found these
            params.setdefault("sources", src.params.get("sources"))
            params["prepare_only"] = True
            if params.get("append_to"):
                tgt = _job(params["append_to"])
                if not os.path.isdir(annotate.ds_dir(tgt.dir)):
                    raise HTTPException(400, "Ziel-Datensatz nicht gefunden")
        if kind == "roofshape" and not (params.get("bbox") or params.get("geometry")):
            raise HTTPException(400, "Gebiet fehlt: Rechteck/Polygon in der Karte zeichnen oder Begrenzung eintragen (NRW).")
        if kind == "area":
            try:
                params["estimate"] = area_mode.estimate(params, cfg())
            except Exception as e:                               # noqa: BLE001
                raise HTTPException(400, str(e))
        job = Job(cfg()["jobs_dir"], kind, params)
        mgr.submit(job)
        return job.as_dict()

    @app.post("/api/area/estimate")
    def estimate(body: dict = Body(...)):
        try:
            return area_mode.estimate(body, cfg())
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(400, str(e))

    @app.get("/api/jobs")
    def list_jobs():
        out = [{k: v for k, v in j.as_dict().items() if k not in ("log",)} for j in mgr.jobs.values()]
        return sorted(out, key=lambda d: d["id"], reverse=True)

    def _job(jid) -> Job:
        if jid not in mgr.jobs:
            raise HTTPException(404, "Job unbekannt")
        return mgr.jobs[jid]

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return _job(jid).as_dict()

    def _delete_job(job: Job) -> dict:
        import shutil
        if job.status in ("running", "queued") or mgr.current is job:
            raise HTTPException(409, "Auftrag läuft noch – erst abbrechen.")
        hand = 0
        d = annotate.ds_dir(job.dir)
        if os.path.isdir(d):
            hand = sum(1 for m in annotate.list_items(d) if m.get("mask_source") == "user" or m.get("has_neg"))
        shutil.rmtree(job.dir, ignore_errors=True)
        mgr.jobs.pop(job.id, None)
        return {"id": job.id, "kind": job.kind, "hand_masks": hand}

    @app.get("/api/cleanup")
    def cleanup_preview(kinds: str = "train,ai_masks"):
        """what 'delete old training data' would remove (trained models in model_dir/custom stay)"""
        ks = set(kinds.split(","))
        out = []
        for j in mgr.jobs.values():
            if j.kind in ks and j.status not in ("running", "queued"):
                d = annotate.ds_dir(j.dir)
                items = annotate.list_items(d) if os.path.isdir(d) else []
                out.append({"id": j.id, "kind": j.kind, "name": j.params.get("name") or "",
                            "images": len(items),
                            "hand_masks": sum(1 for m in items if m.get("mask_source") == "user" or m.get("has_neg"))})
        return sorted(out, key=lambda x: x["id"])

    @app.post("/api/cleanup")
    def cleanup(body: dict = Body(...)):
        ids = body.get("ids") or []
        return {"deleted": [_delete_job(_job(i)) for i in ids if i in mgr.jobs]}

    @app.delete("/api/jobs/{jid}")
    def delete_job(jid: str):
        return _delete_job(_job(jid))

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str):
        _job(jid).cancel()
        return {"ok": True}

    @app.post("/api/jobs/{jid}/resume")
    def resume(jid: str):
        old = _job(jid)
        if old.status in ("running", "queued"):
            raise HTTPException(400, "Job läuft bereits")
        job = Job.load(cfg()["jobs_dir"], jid)
        mgr.submit(job)
        return job.as_dict()

    @app.get("/api/jobs/{jid}/files/{name}")
    def job_file(jid: str, name: str):
        path = os.path.join(_job(jid).dir, os.path.basename(name))
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path, filename=f"{jid}_{os.path.basename(name)}")

    @app.get("/api/jobs/{jid}/chips/{name}")
    def chip(jid: str, name: str):
        path = os.path.join(_job(jid).dir, "chips", os.path.basename(name))
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/jobs/{jid}/results")
    def results(jid: str, status: str = "", size_class: str = "", limit: int = 500, offset: int = 0):
        job = _job(jid)
        db_path = os.path.join(job.dir, "results.sqlite")
        if not os.path.isfile(db_path):
            return {"rows": [], "geojson": {"type": "FeatureCollection", "features": []}}
        con = sqlite3.connect(db_path)
        if job.kind != "area" and not con.execute("SELECT 1 FROM sqlite_master WHERE name='results'").fetchone():
            return {"rows": [], "total": 0, "geojson": {"type": "FeatureCollection", "features": []}}   # job just started
        if job.kind == "area":
            feats, rows = [], []
            path = os.path.join(job.dir, "installations.geojson")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    fc = json.load(f)
                feats = [ft for ft in fc["features"]
                         if not size_class or ft["properties"]["size_class"] == size_class]
                rows = [ft["properties"] for ft in feats]
            else:                                 # still running: show raw polygons
                for (w,) in con.execute("SELECT wkt FROM polys LIMIT 5000"):
                    feats.append({"type": "Feature", "geometry": mapping(shp_wkt.loads(w)), "properties": {}})
            chips = [{"cell": r[0], "chip": f"cell_{r[0]}.jpg"} for r in
                     con.execute("SELECT id FROM cells WHERE n_polys>0 LIMIT 200")]
            outl = []
            po = os.path.join(job.dir, "installation_outlines.geojson")
            if os.path.isfile(po):
                with open(po, encoding="utf-8") as f:
                    outl = json.load(f)["features"]
                if size_class:
                    keep = {f"#{r['installation_id']}" for r in rows}
                    outl = [o for o in outl if o["properties"]["id"] in keep]
            return {"rows": rows[offset:offset + limit], "total": len(rows),
                    "geojson": {"type": "FeatureCollection", "features": feats[:20000]}, "chips": chips,
                    "installations": {"type": "FeatureCollection", "features": outl[:20000]}}
        where, args = [], []
        nach = status == "nacharbeit"                            # incomplete or missed detections (batch rework)
        if status and not nach:
            where.append("status=?")
            args.append(status)
        if nach:
            where.append("status!='fehler'")
        if size_class:
            where.append("size_class=?")
            args.append(size_class)
        sql_where = (" WHERE " + " AND ".join(where)) if where else ""
        total = con.execute("SELECT count(*) FROM results" + sql_where, args).fetchone()[0]
        from .. import building_mode, training
        building_mode.ensure_columns(con)
        cur = con.execute(
            "SELECT id,status,pv_area_m2,kwp_est,size_class,in_range,max_prob,footprint_m2,source,lon,lat,"
            'error,chip,footprint_wkt,pv_wkt,"check",source2,removed,props,kwp_low,kwp_high,layout,installations,'
            'roof_type,roof_area_m2,usable_area_m2,kwp_potential,kwp_pot_low,kwp_pot_high,roof '
            'FROM results' + sql_where +
            (" ORDER BY kwp_est DESC, rowid" if nach else " ORDER BY kwp_est DESC, rowid LIMIT ? OFFSET ?"),
            args if nach else args + [limit, offset])
        allr = cur.fetchall()
        if nach:
            order = {"verpasst": 0, "unvollständig": 1, "verdacht": 2}
            tagged = []
            for r in allr:
                pr = json.loads(r[18] or "{}")
                why = training._nacharbeit(r[1], r[3], pr, r[6])
                if why:
                    tagged.append((order[why], -float(pr.get("pv_kwp") or 0), -(r[6] or 0), why, r))
            tagged.sort(key=lambda t: t[:3])
            total = len(tagged)
            allr = [t[4] for t in tagged][offset:offset + limit]
            whys = {t[4][0]: t[3] for t in tagged}
        rows, feats, insts, roofs_fc = [], [], [], []
        for r in allr:
            d = dict(zip(["id", "status", "pv_area_m2", "kwp_est", "size_class", "in_range", "max_prob",
                          "footprint_m2", "source", "lon", "lat", "error", "chip"], r[:13]))
            pr = json.loads(r[18] or "{}")
            d.update(check=r[15], source2=r[16], removed=json.loads(r[17]) if r[17] else [],
                     pv_kwp_register=pr.get("pv_kwp"), nacharbeit=whys.get(r[0]) if nach else None,
                     kwp_low=r[19], kwp_high=r[20], layout=r[21],
                     installations=[{k: f["properties"].get(k) for k in (
                         "layout", "azimuth_deg", "azimuth2_deg", "tilt_class", "fill_factor", "fill_factor_source",
                         "outline_area_m2", "kwp_est", "kwp_low", "kwp_high", "install_year", "install_year_low",
                         "install_year_high", "module_efficiency", "system")} for f in json.loads(r[22] or "[]")])
            rows.append(d)
            if r[13]:
                feats.append({"type": "Feature", "geometry": mapping(shp_wkt.loads(r[13])),
                              "properties": {"id": d["id"], "status": d["status"], "kind": "building"}})
            else:
                feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
                              "properties": {"id": d["id"], "status": d["status"], "kind": "building"}})
            if r[14]:
                feats.append({"type": "Feature", "geometry": mapping(shp_wkt.loads(r[14])),
                              "properties": {"id": d["id"], "kind": "pv"}})
            # installation outlines from the typology (rows with their gaps), own map layer
            insts += typology.map_features(json.loads(r[22] or "[]"), d["id"])
            # free roof area and obstacles (roofs.py)
            d.update(roof_type=r[23], roof_area_m2=r[24], usable_area_m2=r[25], kwp_potential=r[26], kwp_pot_low=r[27],
                     kwp_pot_high=r[28])
            if r[29]:
                rf = json.loads(r[29])
                d["roof"] = {k: rf.get(k) for k in ("roof_type_basis", "obstacles_area_m2", "shade_area_m2", "pv_area_m2",
                                                     "layout_assumed", "tilt_assumed_deg", "module_efficiency", "faces")}
                for kind in ("free", "obstacles"):
                    if rf.get(kind):
                        roofs_fc.append({"type": "Feature", "geometry": rf[kind], "properties": {
                            "id": d["id"], "kind": kind, "usable_area_m2": d["usable_area_m2"],
                            "kwp_potential": d["kwp_potential"], "kwp_pot_low": d["kwp_pot_low"],
                            "kwp_pot_high": d["kwp_pot_high"], "roof_type": d["roof_type"]}})
        return {"rows": rows, "total": total, "geojson": {"type": "FeatureCollection", "features": feats},
                "installations": {"type": "FeatureCollection", "features": insts},
                "roofs": {"type": "FeatureCollection", "features": roofs_fc}}

    @app.post("/api/jobs/{jid}/export")
    def reexport(jid: str):
        job = _job(jid)
        if job.kind != "buildings":
            raise HTTPException(400, "Nur für Gebäude-Jobs")
        building_mode.export(job)
        job.save()
        return job.as_dict()

    @app.exception_handler(Exception)
    async def on_error(request, exc):                            # noqa: ARG001
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
