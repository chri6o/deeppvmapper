"""Settings with defaults; persisted as YAML (pvmapper.yml in the working dir)."""

from __future__ import annotations

import copy
import os

import yaml

DEFAULTS = {
    "model_dir": "models",
    "jobs_dir": "jobs",
    "profile": "nrw",                 # nrw | ign | google
    "device": "auto",                 # auto | cuda | cpu | mps
    "batch_size": 32,                 # patches per GPU batch (16 GB VRAM: 32–64)
    "cls_threshold": None,            # None = profile default
    "seg_threshold": None,
    "patch_overlap": 0.25,            # 0 = no overlap; 0.25–0.5 finds arrays cut by patch borders
    "fetch_workers": 8,               # parallel imagery downloads
    "buffer_m": 3.0,                  # margin around building footprints
    "point_radius_m": 20.0,           # search radius when only a point is given
    "min_pv_area_m2": 3.0,            # smaller detections are treated as noise
    "kwp_per_m2": 0.20,               # rough installed capacity per m² of detected module area
    "kwp_min": None,                  # keep only buildings/installations in this kWp range
    "kwp_max": None,
    "size_classes": [                 # by estimated kWp, upper bound exclusive
        {"name": "S", "label": "bis 30 kWp", "max": 30},
        {"name": "M", "label": "30–100 kWp", "max": 100},
        {"name": "L", "label": "100–750 kWp", "max": 750},
        {"name": "XL", "label": "ab 750 kWp", "max": None},
    ],
    "save_chips": "positive",         # none | positive | all — JPEG previews for review
    "sources": "auto",                # 'auto' or list of source ids / country codes
    "sources_extra": None,            # path to an additional sources YAML
    "area_cell_px": 2560,             # area mode: image cell edge length in pixels
    "pvmap_url": "https://pv-map.solar",  # PV-Map connector (building lists, training samples)
    "pvmap_token": "",                # API token from PV-Map: Verwaltung → Datenquellen → API-Tokens
    "anthropic_api_key": "",          # AI mask suggestions (Claude)
    "anthropic_model": "claude-sonnet-5-5",
    "openai_api_key": "",             # AI mask suggestions (ChatGPT)
    "openai_model": "gpt-5.1",
    "ai_masks": "reviewed",           # training uses AI masks: reviewed (default) | all | none
    "ai_prompt_hint": "",             # extra instruction appended to the AI prompt (e.g. regional roof types)
    "seg_tta": True,                  # average segmentation over flipped images
    "precision": "auto",              # auto = half precision (fp16) on the GPU, fp32 on the CPU; fp32 forces full precision
    "neighbor_factor": 0.2,           # segment every patch scoring ≥ this × classification threshold: the classifier misses
                                      # many PV roofs (scores of 0.2–0.3 on obvious arrays), the segmentation decides
    "cross_check": True,              # second aerial image (other year) as cross-check, where the source has one
    "postproc": True,                 # mask post-processing (see pvmapper/postproc.py; postproc_* thresholds)
    "sync_enabled": False,            # night sync with PV-Map: fetch the work list, check, send results (PV-Map docs/pvmapper-ergebnisse.md)
    "sync_start": "22:00",            # time window, local time
    "sync_end": "06:00",
    "sync_filter": "cc=DE",           # which buildings: cc, nuts3 (any NUTS level, e.g. DEA2), lau, sub, cls
    "sync_profile": None,             # model for the sync (None = default model)
    "sync_recheck_days": 365,         # look at a building's aerial image again after … days (new image → new check)
    "sync_confirm_score": 0.9,        # cross-check confirmed and score ≥ this → "confirmed" (tier 1) in PV-Map
    "sync_chunk": 500,                # buildings per request to the work list
    "sync_push_reviews": True,        # masks checked in the editor go to PV-Map as "checked by hand"
    "typology": True,                 # classify every detected installation (layout, azimuth, tilt class, fill factor) and
                                      # estimate kWp from outline × fill factor × efficiency (pvmapper/typology.py)
    "typology_file": None,            # own typology.yml (defaults, thresholds); None = the one shipped with pvmapper
    "typology_push": False,           # send the installations to PV-Map (PV-Map migration 137 and later, docs/typology.md)
    "config_version": 2,
    "web_password": "",               # if set: the web UI asks for this password (any user name)
    "user_agent": "pvmapper/0.1 (+https://github.com/chri6o/deeppvmapper)",
}


SECRET_KEYS = ("pvmap_token", "anthropic_api_key", "openai_api_key", "web_password")
MASK = "••••••"


def public(cfg: dict) -> dict:
    """config for the UI: secrets replaced by a placeholder"""
    return {k: (MASK if k in SECRET_KEYS and v else v) for k, v in cfg.items()}


def load(path: str = "pvmapper.yml") -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            stored = yaml.safe_load(f) or {}
        cfg.update(stored)
        if int(stored.get("config_version") or 1) < 2:
            # saved settings carry the old defaults: move them to the new ones (unless changed by hand)
            if cfg.get("neighbor_factor") == 0.5:
                cfg["neighbor_factor"] = DEFAULTS["neighbor_factor"]
            cfg["config_version"] = DEFAULTS["config_version"]
    return cfg


def save(cfg: dict, path: str = "pvmapper.yml"):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def size_class(kwp: float, classes) -> str:
    for c in classes:
        if c.get("max") is None or kwp < float(c["max"]):
            return c["name"]
    return classes[-1]["name"] if classes else ""


def in_kwp_range(kwp: float, cfg: dict) -> bool:
    lo, hi = cfg.get("kwp_min"), cfg.get("kwp_max")
    return (lo is None or kwp >= float(lo)) and (hi is None or kwp <= float(hi))
