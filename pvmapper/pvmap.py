"""
PV-Map connector (https://pv-map.solar, API docs: PV-Map docs/pvmapper-connector.md).

  - building lists:   GET /api/detect/buildings  (paged GeoJSON, footprints + PV state from registers/layers)
  - counts:           GET /api/detect/count
  - training sample:  GET /api/detect/training   (labels pos_high/pos_medium/neg_high/neg_weak, stratified by
                                                   building type × size of the existing PV, optional PV polygons)

Authentication: API token created in PV-Map (Verwaltung → Datenquellen → API-Tokens), `pvm_…`.
Configure `pvmap_url` and `pvmap_token` in pvmapper.yml / the settings tab, or env PVMAP_URL / PVMAP_TOKEN.

As a building input, use  `pvmap:cc=DE&lau=DE_11000000&sub=Education&pv=none`  wherever a file path is accepted.
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

    def _get(self, path: str, params: dict):
        if not self.token:
            raise PVMapError("Kein PV-Map-Token konfiguriert (Einstellungen → PV-Map).")
        p = {k: v for k, v in params.items() if v not in (None, "", [])}
        r = self.s.get(f"{self.url}/api{path}", params=p, timeout=self.timeout,
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
