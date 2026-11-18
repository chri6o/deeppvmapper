"""
Night sync with PV-Map (PV-Map docs/pvmapper-ergebnisse.md): fetch the work list, check the buildings, send every result back.

The work list brings never-checked buildings first, then "possible" detections made with an older model, then buildings whose last
look at the aerial image is older than `sync_recheck_days`; for those the image fingerprint decides whether anything is computed
(same image → "unchanged"). Runs as a 'buildings' job with input `pvmapqueue:` and push=True, until the time window ends.
"""

from __future__ import annotations

import datetime as dt
from urllib.parse import parse_qsl, urlencode

from . import building_mode, runtime


def queue_input(cfg: dict, flt: str, model: str) -> str:
    q = {k: v for k, v in parse_qsl((flt or "").lstrip("?").replace("pvmap:", "")) if v}
    q.update(model=model, recheck_days=str(int(cfg.get("sync_recheck_days") or 365)), limit=str(int(cfg.get("sync_chunk") or 500)))
    return "pvmapqueue:" + urlencode(q)


def run(job, cfg: dict):
    p = job.params
    flt = p.get("filter") or cfg.get("sync_filter") or "cc=DE"
    prof = runtime.detector(cfg, p.get("profile") or cfg.get("sync_profile") or None).p
    job.params.update(input=queue_input(cfg, flt, prof.id), push=True, profile=prof.id,
                      save_chips=p.get("save_chips", "positive"))
    job.log((f"PV-Map-Prüfaufträge, Modell {prof.id}" if p.get("requests") else f"PV-Map-Abgleich: Arbeitsliste {flt}, Modell {prof.id}")
            + (f", bis {dt.datetime.fromisoformat(p['until']).strftime('%H:%M')}" if p.get("until") else "")
            + (f", höchstens {p['max']} Gebäude" if p.get("max") else "") + ".")
    building_mode.run(job, cfg)


def window(cfg: dict, now: dt.datetime | None = None):
    """(start, end) of the current night window if `now` lies in it, else None. Times 'HH:MM', local time; end may be after midnight."""
    now = now or dt.datetime.now().astimezone()
    try:
        sh, sm = (int(x) for x in str(cfg.get("sync_start") or "22:00").split(":"))
        eh, em = (int(x) for x in str(cfg.get("sync_end") or "06:00").split(":"))
    except ValueError:
        return None
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    if end <= start:                                   # over midnight
        if now >= start:
            end += dt.timedelta(days=1)
        else:
            start -= dt.timedelta(days=1)
    return (start, end) if start <= now < end else None
