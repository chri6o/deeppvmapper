"""
pvmapper command line.

  python -m pvmapper web                      start the local web UI (http://127.0.0.1:8765)
  python -m pvmapper device                   show GPU / ROCm / CUDA status
  python -m pvmapper models                   download model weights for the active profile
  python -m pvmapper sources [--test]         list (and live-test) imagery sources
  python -m pvmapper buildings FILE [...]     check a building list for PV
  python -m pvmapper area --bbox W S E N      wall-to-wall detection for a region
  python -m pvmapper resume JOB_ID            continue an interrupted job
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import config as cfgmod


def _cfg(args):
    cfg = cfgmod.load(args.config)
    for k in ("profile", "device", "model_dir", "jobs_dir", "batch_size", "fetch_workers",
              "kwp_min", "kwp_max", "kwp_per_m2", "min_pv_area_m2", "save_chips"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if getattr(args, "sources", None):
        cfg["sources"] = args.sources
    return cfg


def _run(kind, params, cfg):
    from .jobs import Job
    from . import area_mode, building_mode
    job = Job(cfg["jobs_dir"], kind, params)
    job.log(f"Job {job.id} in {job.dir}")
    fn = building_mode.run if kind == "buildings" else area_mode.run
    job.run(lambda j: fn(j, cfg))
    print(json.dumps({k: v for k, v in job.as_dict().items() if k != "log"}, indent=1, ensure_ascii=False))
    return 0 if job.status == "done" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pvmapper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pvmapper.yml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--profile", choices=["nrw", "ign", "google"])
        sp.add_argument("--device")
        sp.add_argument("--model-dir", dest="model_dir")
        sp.add_argument("--jobs-dir", dest="jobs_dir")
        sp.add_argument("--batch-size", dest="batch_size", type=int)
        sp.add_argument("--fetch-workers", dest="fetch_workers", type=int)
        sp.add_argument("--sources", help="auto | comma-separated source ids or country codes (DE,NL,AT)")
        sp.add_argument("--kwp-min", dest="kwp_min", type=float)
        sp.add_argument("--kwp-max", dest="kwp_max", type=float)
        sp.add_argument("--kwp-per-m2", dest="kwp_per_m2", type=float)
        sp.add_argument("--min-pv-area", dest="min_pv_area_m2", type=float)
        sp.add_argument("--chips", dest="save_chips", choices=["none", "positive", "all"])

    w = sub.add_parser("web")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8765)
    sub.add_parser("device")
    m = sub.add_parser("models")
    common(m)
    s = sub.add_parser("sources")
    s.add_argument("--test", action="store_true")
    b = sub.add_parser("buildings")
    common(b)
    b.add_argument("input", help="GeoJSON/CSV/GPKG/SHP or postgresql://… DSN")
    b.add_argument("--id-field")
    b.add_argument("--query", help="SQL for PostGIS input")
    b.add_argument("--layer")
    a = sub.add_parser("area")
    common(a)
    a.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    a.add_argument("--geojson", help="region polygon as GeoJSON file")
    a.add_argument("--buildings", help="optional building layer to match detections")
    a.add_argument("--all-detections", action="store_true",
                   help="keep detections not on a building (default: only on buildings if a layer is given)")
    r = sub.add_parser("resume")
    r.add_argument("job_id")
    r.add_argument("--jobs-dir", dest="jobs_dir")

    args = ap.parse_args(argv)

    if args.cmd == "web":
        import uvicorn
        from .web.app import create_app
        print(f"pvmapper läuft auf http://{args.host}:{args.port}")
        uvicorn.run(create_app(args.config), host=args.host, port=args.port, log_level="warning")
        return 0

    if args.cmd == "device":
        from .models import device_info
        print(json.dumps(device_info(), indent=1))
        return 0

    cfg = _cfg(args)

    if args.cmd == "models":
        from .models import PROFILES, weight_path
        prof = PROFILES[cfg["profile"]]
        for f in (prof.cls_file, prof.seg_file):
            print("→", weight_path(cfg["model_dir"], f))
        return 0

    if args.cmd == "sources":
        from . import runtime, sources as srcmod, geo
        from shapely.geometry import box
        cat = runtime.catalog(cfg)
        for s in cat.sources.values():
            line = f"{s.id:12s} {s.native_resolution_m:>5} m  {s.name}"
            if args.test:
                a, b_, c, d = s.bbox
                g = box((a + c) / 2, (b_ + d) / 2, (a + c) / 2 + 0.0005, (b_ + d) / 2 + 0.0005)
                t = time.time()
                try:
                    img = s.fetch(geo.window_around(g, s.crs, 0.2, 0, 256))
                    line += f"   OK {time.time() - t:.1f}s blank={srcmod.blank_fraction(img):.2f}"
                except Exception as e:                           # noqa: BLE001
                    line += f"   FEHLER {str(e)[:80]}"
            print(line)
        return 0

    if args.cmd == "buildings":
        return _run("buildings", {"input": args.input, "id_field": args.id_field,
                                  "query": args.query, "layer": args.layer}, cfg)

    if args.cmd == "area":
        params = {"buildings": args.buildings, "only_on_buildings": not args.all_detections}
        if args.bbox:
            params["bbox"] = args.bbox
        elif args.geojson:
            with open(args.geojson, encoding="utf-8") as f:
                params["geometry"] = json.load(f)
        else:
            ap.error("area needs --bbox or --geojson")
        return _run("area", params, cfg)

    if args.cmd == "resume":
        from .jobs import Job
        from . import area_mode, building_mode
        cfg = cfgmod.load(args.config)
        jd = args.jobs_dir or cfg["jobs_dir"]
        job = Job.load(jd, args.job_id)
        fn = building_mode.run if job.kind == "buildings" else area_mode.run
        job.run(lambda j: fn(j, cfg))
        return 0 if job.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
