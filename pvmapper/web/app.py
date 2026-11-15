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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from shapely import wkt as shp_wkt
from shapely.geometry import Point, mapping

from .. import area_mode, building_mode, buildings as bld, config as cfgmod
from .. import geo, inference, models, runtime, sources
from ..jobs import Job

STATIC = os.path.join(os.path.dirname(__file__), "static")


class JobManager:
    """Runs one job at a time (the GPU is the shared resource)."""

    def __init__(self, get_cfg):
        self.get_cfg = get_cfg
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        self.current: Job | None = None
        threading.Thread(target=self._worker, daemon=True).start()

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
        self.q.put(job)

    def _worker(self):
        while True:
            job = self.q.get()
            if job._cancel.is_set():
                job.status = "cancelled"
                job.save()
                continue
            self.current = job
            cfg = self.get_cfg()
            fn = building_mode.run if job.kind == "buildings" else area_mode.run
            job.run(lambda j: fn(j, cfg))
            self.current = None


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

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    # -- settings / info --------------------------------------------------------

    @app.get("/api/info")
    def info():
        prof = []
        for p in models.PROFILES.values():
            have = all(os.path.isfile(os.path.join(cfg()["model_dir"], os.path.basename(f)))
                       for f in (p.cls_file, p.seg_file))
            prof.append({"id": p.id, "label": p.label, "resolution_m": p.resolution_m,
                         "licence": p.licence, "downloaded": have,
                         "cls_threshold": p.cls_threshold, "seg_threshold": p.seg_threshold})
        return {"device": models.device_info(), "profiles": prof, "config": cfg(),
                "config_path": os.path.abspath(config_path)}

    @app.post("/api/config")
    def set_config(new: dict = Body(...)):
        c = dict(cfg())
        for k, v in new.items():
            if k in cfgmod.DEFAULTS:
                c[k] = v
        cfgmod.save(c, config_path)
        state["cfg"] = cfgmod.load(config_path)
        return state["cfg"]

    @app.post("/api/models/download")
    def download_models(body: dict = Body(default={})):
        pid = body.get("profile") or cfg()["profile"]
        p = models.PROFILES[pid]
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
        prof = models.PROFILES[body.get("profile") or c["profile"]]
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
            polys = inference.mask_to_polygons(r.mask, win)
            mask = r.mask
            area_m2 = sum(geo.ground_area_m2(p) for p in polys)
            out.update(detect_s=round(time.time() - t1, 2), max_prob=round(r.max_prob, 3),
                       n_patches=r.n_patches, n_positive=r.n_positive, pv_area_m2=round(area_m2, 1),
                       kwp_est=round(area_m2 * float(c["kwp_per_m2"]), 1),
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
                  layer=body.get("layer") or None)
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

    # -- jobs ------------------------------------------------------------------------------

    @app.post("/api/jobs")
    def create_job(body: dict = Body(...)):
        kind = body.get("kind")
        params = body.get("params") or {}
        if kind not in ("buildings", "area"):
            raise HTTPException(400, "kind must be buildings or area")
        if kind == "buildings" and not params.get("input"):
            raise HTTPException(400, "Keine Gebäudedatei gewählt")
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
            return {"rows": rows[offset:offset + limit], "total": len(rows),
                    "geojson": {"type": "FeatureCollection", "features": feats[:20000]}, "chips": chips}
        where, args = [], []
        if status:
            where.append("status=?")
            args.append(status)
        if size_class:
            where.append("size_class=?")
            args.append(size_class)
        sql_where = (" WHERE " + " AND ".join(where)) if where else ""
        total = con.execute("SELECT count(*) FROM results" + sql_where, args).fetchone()[0]
        cur = con.execute(
            "SELECT id,status,pv_area_m2,kwp_est,size_class,in_range,max_prob,footprint_m2,source,lon,lat,"
            "error,chip,footprint_wkt,pv_wkt FROM results" + sql_where +
            " ORDER BY kwp_est DESC, rowid LIMIT ? OFFSET ?", args + [limit, offset])
        rows, feats = [], []
        for r in cur:
            d = dict(zip(["id", "status", "pv_area_m2", "kwp_est", "size_class", "in_range", "max_prob",
                          "footprint_m2", "source", "lon", "lat", "error", "chip"], r[:13]))
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
        return {"rows": rows, "total": total, "geojson": {"type": "FeatureCollection", "features": feats}}

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
