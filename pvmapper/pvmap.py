"""
PV-Map connector (https://pv-map.solar, API docs: PV-Map docs/pvmapper-connector.md).

  - building lists:   GET /api/detect/buildings  (paged GeoJSON, footprints + PV state from registers/layers)
  - counts:           GET /api/detect/count
  - training sample:  GET /api/detect/training   (labels pos_high/pos_medium/neg_high/neg_weak, stratified by
                                                   building type × size of the existing PV, optional PV polygons)
  - work list:        GET /api/detect/queue      (next buildings by priority, leased; PV-Map docs/pvmapper-ergebnisse.md)
  - results:          POST /api/detect/results   (every check, also "no PV"; /results/finish at the end of a run)

Authentication: API token created in PV-Map (Verwaltung → Datenquellen → API-Tokens), `pvm_…`.
Configure `pvmap_url` and `pvmap_token` in pvmapper.yml / the settings tab, or env PVMAP_URL / PVMAP_TOKEN.

As a building input, use  `pvmap:cc=DE&lau=DE_11000000&sub=Education&pv=none`  wherever a file path is accepted;
`pvmapqueue:cc=DE&nuts3=DEA2&model=…` reads the work list instead (sync job).
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode

import requests

FILTER_KEYS = ("cc", "nuts3", "lau", "sub", "cls", "tag", "bbox", "pv", "include_d")
DEFAULT_URL = "https://pv-map.solar"


class PVMapError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: int = 120):
        self.url = (url or os.environ.get("PVMAP_URL") or DEFAULT_URL).rstrip("/")
        self.token = token or os.environ.get("PVMAP_TOKEN") or ""
        self.timeout = timeout
        self.s = requests.Session()

    @classmethod
    def from_config(cls, cfg: dict | None):
        cfg = cfg or {}
        return cls(cfg.get("pvmap_url"), cfg.get("pvmap_token"))

    def _get(self, path: str, params: dict, method: str = "GET", body=None):
        if not self.token:
            raise PVMapError("Kein PV-Map-Token konfiguriert (Einstellungen → PV-Map).")
        p = {k: v for k, v in params.items() if v not in (None, "", [])}
        r = self.s.request(method, f"{self.url}/api{path}", params=p, json=body, timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.token}", "Accept-Language": "de"})
        if r.status_code != 200:
            try:
                msg = r.json().get("detail")
            except Exception:                                    # noqa: BLE001
                msg = r.text[:300]
            if r.status_code == 401 and "basic" in r.headers.get("WWW-Authenticate", "").lower():
                msg = ("Die PV-Map-Seite verlangt ein Seiten-Passwort (Vorschau-Modus) – der Konnektor-Pfad "
                       "/api/detect/ muss davon ausgenommen sein (PV-Map infra/Caddyfile).")
            raise PVMapError(f"PV-Map {r.status_code}: {msg}")
        return r.json()

    # -- API ------------------------------------------------------------------

    def count(self, **filters) -> dict:
        return self._get("/detect/count", filters)

    def buildings(self, page_size: int = 2000, **filters):
        """Yields GeoJSON features page by page."""
        after = ""
        while True:
            d = self._get("/detect/buildings", {**filters, "limit": page_size, "after": after})
            yield from d["features"]
            if not d.get("next"):
                break
            after = d["next"]

    def training(self, bins=None, per_stratum: int = 100, negatives: int = 100, seed: str = "1", **filters) -> dict:
        params = {**filters, "per_stratum": per_stratum, "negatives": negatives, "seed": seed}
        if bins:
            params["bins"] = ",".join(str(b) for b in bins)
        return self._get("/detect/training", params)

    def strata(self, bins=None, **filters) -> dict:
        params = {**filters, "strata_only": "true"}
        if bins:
            params["bins"] = ",".join(str(b) for b in bins)
        return self._get("/detect/training", params)


    # -- work list and results (PV-Map docs/pvmapper-ergebnisse.md) ---------------------------------------

    def queue(self, limit: int = 1000, model: str | None = None, recheck_days: int = 365, **filters) -> dict:
        return self._get("/detect/queue", {**filters, "limit": limit, "model": model, "recheck_days": recheck_days})

    def queue_count(self, model: str | None = None, recheck_days: int = 365, **filters) -> dict:
        return self._get("/detect/queue/count", {**filters, "model": model, "recheck_days": recheck_days})

    def send_results(self, items: list, run: str | None = None, model: str | None = None) -> dict:
        return self._get("/detect/results", {}, "POST", {"run": run, "model": model, "items": items})

    def finish(self) -> dict:
        return self._get("/detect/results/finish", {}, "POST", {})


QUEUE_KEYS = FILTER_KEYS + ("model", "recheck_days", "limit")


def parse_queue(s: str) -> dict:
    """'pvmapqueue:cc=DE&nuts3=DEA2&model=custom:x' → filters + model/recheck_days/limit"""
    q = (s.split(":", 1)[1] if ":" in s else "").lstrip("?")
    d = {k: v for k, v in parse_qsl(q) if k in QUEUE_KEYS}
    d.pop("pv", None)
    if not d.get("cc"):
        raise PVMapError("PV-Map-Arbeitsliste braucht mindestens cc=… (Ländercode)")
    return d


def parse_input(s: str) -> dict:
    """'pvmap:cc=DE&sub=Education' → {'cc': 'DE', 'sub': 'Education'}"""
    q = s.split(":", 1)[1] if ":" in s else ""
    q = q.lstrip("?")
    d = {k: v for k, v in parse_qsl(q) if k in FILTER_KEYS}
    if not d.get("cc"):
        raise PVMapError("PV-Map-Eingabe braucht mindestens cc=… (Ländercode)")
    return d


def make_input(**filters) -> str:
    return "pvmap:" + urlencode({k: v for k, v in filters.items() if k in FILTER_KEYS and v not in (None, "")})
