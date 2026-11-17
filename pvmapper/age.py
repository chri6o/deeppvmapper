"""
Installation age from older aerial images (PVMAPPER.md "Anlagenalter").

For every installation found on the current image, older flights of the same window are checked: does the outline already
show modules? A bisection over the available years finds the first flight with modules; the installation was built between
the last flight without and the first flight with modules. The build year sets the module efficiency and with it the
capacity estimate (typology.efficiency_for_year) – old installations have far less kWp per m² than new ones.

Which older flights exist comes from the imagery source (`history` in sources.yml):

  kind: nrw_info   Geobasis NRW – GetFeatureInfo on the metadata layer lists every flight at the point with date and
                   resolution; one layer per year (nw_hist_dop_<year>)
  kind: years      a fixed list of year layers (PDOK: 2016–2020 at 25 cm, from 2021 at 8 cm)

Evidence per installation and year (`evidence`):
  coverage    share of the current module pixels that the model also finds in the old image
  contrast    contrast modules ↔ surrounding roof in the old image relative to the current one (≈1 with modules, ≈0 without)
  similarity  correlation of the brightness pattern inside the outline between old and current image (blurred to the
              coarser resolution)
Verdict: present / absent / unclear; unclear years are skipped in the bisection.
"""

from __future__ import annotations

import datetime as _dt
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import geo, typology
from .sources import Source, blank_fraction

_SRC_CACHE: dict = {}
_INFO_CACHE: dict = {}
_LOCK = threading.Lock()

DEFAULTS = {
    "age_max_res_m": 0.4,          # older flights coarser than this are not used (modules are not recognisable)
    "age_min_year": 2000,          # no PV on public roofs worth dating before this
    "age_present_cov": 0.35,       # model finds ≥ this share of the current modules → present
    "age_absent_cov": 0.20,
}


def _cfg(cfg):
    return {**DEFAULTS, **{k: v for k, v in (cfg or {}).items() if k in DEFAULTS and v is not None}}


# ---------------------------------------------------------------------------------------------------------------------
# which older flights exist at a place

def _hist_source(base: Source, year: int, layer: str, res: float, h: dict) -> Source:
    key = (base.id, year)
    with _LOCK:
        s = _SRC_CACHE.get(key)
        if s is None:
            s = _SRC_CACHE[key] = Source(id=f"{base.id}-{year}", name=f"{base.name} · {year}", type="wms",
                                         url=h["url"], layers=layer, crs=h.get("crs") or base.crs,
                                         native_resolution_m=res, format=h.get("format", "image/jpeg"),
                                         version=h.get("version", "1.3.0"), concurrency=int(h.get("concurrency", 4)),
                                         attribution=base.attribution, license=base.license, priority=99)
            s.user_agent = base.user_agent
    return s


def _nrw_info(h: dict, base: Source, x: float, y: float, crs: str) -> list:
    """[(year, date, resolution_m)] of all historical flights at a point (Geobasis NRW metadata layer)"""
    key = (h["url"], round(x / 500), round(y / 500))                 # 500 m cells: the flights are tiled per km
    with _LOCK:
        if key in _INFO_CACHE:
            return _INFO_CACHE[key]
    p = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo", "LAYERS": h["info_layer"],
         "QUERY_LAYERS": h["info_layer"], "CRS": crs, "BBOX": f"{x - 50},{y - 50},{x + 50},{y + 50}", "WIDTH": 101,
         "HEIGHT": 101, "I": 50, "J": 50, "INFO_FORMAT": "text/plain", "FEATURE_COUNT": 500, "STYLES": ""}
    r = base._http().get(h["url"], params=p, timeout=40)
    r.raise_for_status()
    out = parse_nrw_info(r.text)
    with _LOCK:
        _INFO_CACHE[key] = out
    return out


def parse_nrw_info(text: str) -> list:
    out = {}
    for block in re.split(r"\n\s*Feature \d+:", text):
        d = re.search(r"Bildflugdatum = '(\d{4})-(\d{2})-(\d{2})'", block)
        rr = re.search(r"Bodenaufl\S* Originalbild[^=]*= '([\d.]+)'", block)
        if not d:
            continue
        y = int(d.group(1))
        try:
            date = _dt.date(y, int(d.group(2)), int(d.group(3))).isoformat()
        except ValueError:                                           # e.g. '2009-00-00': only the year is known
            date = None
        res = float(rr.group(1)) if rr else 1.0
        if y not in out or res < out[y][2]:
            out[y] = (y, date, res)
    return sorted(out.values())


def timeline(base: Source, win: geo.Window, cfg: dict | None = None) -> list:
    """older flights usable at this window, oldest first: [(year, date or None, resolution_m, Source)]"""
    c = _cfg(cfg)
    h = getattr(base, "history", None) or {}
    if not h:
        return []
    crs = h.get("crs") or base.crs
    cx, cy = (win.minx + win.maxx) / 2, (win.miny + win.maxy) / 2
    if crs != win.crs:
        from shapely.geometry import Point
        p = geo.reproject(Point(cx, cy), win.crs, crs)
        cx, cy = p.x, p.y
    rows = []
    if h.get("kind") == "nrw_info":
        for y, date, res in _nrw_info(h, base, cx, cy, crs):
            rows.append((y, date, res, h["layer"].format(year=y)))
    elif h.get("kind") == "years":
        for y, spec in sorted((int(k), v) for k, v in (h.get("layers") or {}).items()):
            layer, res = (spec, float(h.get("resolution_m", 0.25))) if isinstance(spec, str) else (spec["layer"], float(spec["res"]))
            rows.append((y, None, res, layer))
    out = []
    for y, date, res, layer in rows:
        if res > float(c["age_max_res_m"]) or y < int(c["age_min_year"]):
            continue
        out.append((y, date, res, _hist_source(base, y, layer, res, h)))
    return out


def parse_flight_date(text: str):
    """first Bildflugdatum in a GetFeatureInfo answer (dd.mm.yyyy or yyyy-mm-dd) → date or None"""
    m = re.search(r"Bildflugdatum = '(\d{2})\.(\d{2})\.(\d{4})'", text)
    if m:
        return _dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.search(r"Bildflugdatum = '(\d{4})-(\d{2})-(\d{2})'", text)
    return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def current_year(base: Source, win: geo.Window, tl: list) -> int:
    """year of the current image: flight date from the source's metadata layer (history.current_info), the year in the
    source id (de-nw-2024 …), else today"""
    y = typology.imagery_year(base)
    if y:
        return y
    ci = (getattr(base, "history", None) or {}).get("current_info")
    if ci:
        cx, cy = (win.minx + win.maxx) / 2, (win.miny + win.maxy) / 2
        key = ("cur", ci["url"], round(cx / 500), round(cy / 500))
        with _LOCK:
            hit = _INFO_CACHE.get(key)
        if hit is None:
            try:
                p = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo", "LAYERS": ci["layer"],
                     "QUERY_LAYERS": ci["layer"], "CRS": win.crs, "BBOX": f"{cx - 50},{cy - 50},{cx + 50},{cy + 50}",
                     "WIDTH": 101, "HEIGHT": 101, "I": 50, "J": 50, "INFO_FORMAT": "text/plain", "FEATURE_COUNT": 5,
                     "STYLES": ""}
                r = base._http().get(ci["url"], params=p, timeout=40)
                d = parse_flight_date(r.text) if r.status_code == 200 else None
                hit = d.year if d else 0
            except Exception:                                        # noqa: BLE001
                hit = 0
            with _LOCK:
                _INFO_CACHE[key] = hit
        if hit:
            return hit
    return _dt.date.today().year


# ---------------------------------------------------------------------------------------------------------------------
# evidence for one installation in one older image

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _ring(mask, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool) & ~cv2.dilate(mask.astype(np.uint8),
                                                                              np.ones((3, 3), np.uint8)).astype(bool)


def evidence(cur_img, old_img, mods, old_mask, res_cur: float, res_old: float, roof=None) -> dict:
    """mods: current module pixels of one installation; old_mask: model mask on the old image (same window).
    roof: pixels of the roof around the installation (footprint), for the contrast."""
    n = int(mods.sum())
    if n < 20:
        return {"verdict": "unclear"}
    cov = float((old_mask & mods).sum()) / n
    sig = max(0.0, (res_old / max(res_cur, 1e-3) - 1.0) * 0.6)
    gc, go = _gray(cur_img), _gray(old_img)
    if sig > 0.3:
        gc = cv2.GaussianBlur(gc, (0, 0), sig)
        go = cv2.GaussianBlur(go, (0, 0), sig)
    ring = _ring(mods, max(4, int(round(1.5 / res_cur))))
    if roof is not None:
        ring &= roof
    if ring.sum() < 20:
        ring = _ring(mods, max(4, int(round(1.5 / res_cur))))
    c_cur = float(np.median(gc[ring]) - np.median(gc[mods]))
    c_old = float(np.median(go[ring]) - np.median(go[mods]))
    con = c_old / c_cur if abs(c_cur) >= 8 else None
    a, b = gc[mods], go[mods]
    a, b = a - a.mean(), b - b.mean()
    sim = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
    return {"coverage": round(cov, 2), "contrast": None if con is None else round(con, 2), "similarity": round(sim, 2)}


def verdict(ev: dict, cfg: dict | None = None) -> str:
    c = _cfg(cfg)
    if "coverage" not in ev:
        return "unclear"
    cov, con, sim = ev["coverage"], ev.get("contrast"), ev.get("similarity", 0)
    if cov >= c["age_present_cov"]:
        return "present"
    if con is not None and con >= 0.6 and sim >= 0.45:
        return "present"
    if cov < c["age_absent_cov"] and (con is None or con < 0.3) and sim < 0.35:
        return "absent"
    return "unclear"


# ---------------------------------------------------------------------------------------------------------------------
# bisection over the flights, many buildings at once

class _Inst:
    def __init__(self, feat, mods, n_years):
        self.f, self.mods = feat, mods
        self.lo, self.hi = -1, n_years                 # index of last flight without / first flight with modules
        self.n = n_years
        self.contradicted = False
        self.skip: set = set()
        self.seen: dict = {}                            # year index → evidence

    def next_index(self):
        cand = [i for i in range(self.lo + 1, self.hi) if i not in self.skip and i not in self.seen]
        if not cand:
            return None
        return cand[len(cand) // 2]

    def record(self, i, ev, v):
        self.seen[i] = {**ev, "verdict": v}
        if v == "unclear":
            self.skip.add(i)
        # modules do not disappear and come back: a later flight without modules outweighs an earlier "present"
        # (dark roof patches, other installations), so the bounds follow from all verdicts so far
        absent = [k for k, x in self.seen.items() if x["verdict"] == "absent"]
        self.lo = max(absent) if absent else -1
        present = [k for k, x in self.seen.items() if x["verdict"] == "present" and k > self.lo]
        self.hi = min(present) if present else self.n
        self.contradicted = any(x["verdict"] == "present" and k < self.lo for k, x in self.seen.items())


def estimate(items: list, det, cfg: dict | None = None, workers: int = 4, job=None) -> int:
    """items: [{"img", "win", "src", "mask", "feats", "clip"(optional), "res_m"}]; feats are the GeoJSON installation
    features of typology.for_window and get the build year added (install_year, install_year_low/high, age_flights) and
    their kWp recalculated. Returns the number of dated installations."""
    from . import runtime
    from shapely.geometry import shape
    c = _cfg(cfg)
    conf = typology.settings(cfg)
    work = []
    for it in items:
        if not it.get("feats"):
            continue
        try:
            tl = timeline(it["src"], it["win"], cfg)
        except Exception as e:                                       # noqa: BLE001 – metadata service down: no age
            if job:
                job.log(f"Luftbild-Historie nicht abrufbar: {type(e).__name__}: {e}"[:200])
            tl = []
        cy = current_year(it["src"], it["win"], tl)
        tl = [t for t in tl if t[0] < cy]
        if not tl:
            continue
        insts = []
        for f in it["feats"]:
            out = runtime.clip_mask(shape(f["geometry"]), it["win"], it["img"].shape, buffer_m=0) > 0
            mods = out & it["mask"].astype(bool)
            if mods.sum() >= 20:
                insts.append(_Inst(f, mods, len(tl)))
        if insts:
            work.append({"it": it, "tl": tl, "insts": insts, "cy": cy, "cache": {}})
    if not work:
        return 0

    def fetch(w, i):
        y, date, res, s = w["tl"][i]
        try:
            img = s.fetch(w["it"]["win"])
        except Exception:                                            # noqa: BLE001
            return w, i, None
        if img.shape != w["it"]["img"].shape or blank_fraction(img) > 0.3:
            return w, i, None
        return w, i, img

    for _round in range(8):
        todo = []
        for w in work:
            need = {inst.next_index() for inst in w["insts"]} - {None} - set(w["cache"])
            # one flight per building and round: the one most installations need
            if need:
                todo.append((w, min(need, key=lambda i: -sum(1 for x in w["insts"] if x.next_index() == i))))
        if not todo:
            break
        with ThreadPoolExecutor(max(1, workers)) as pool:
            got = list(pool.map(lambda t: fetch(*t), todo))
        ok = [(w, i, img) for w, i, img in got if img is not None]
        res = det.run([(k, img) for k, (_, _, img) in enumerate(ok)]) if ok else {}
        for w, i, img in got:
            w["cache"][i] = None
        for k, (w, i, img) in enumerate(ok):
            w["cache"][i] = (img, res[k].mask.astype(bool))
        for w in work:
            for inst in w["insts"]:
                i = inst.next_index()
                if i is None or i not in w["cache"]:
                    continue
                hit = w["cache"][i]
                if hit is None:
                    inst.record(i, {}, "unclear")
                    continue
                img, m = hit
                ev = evidence(w["it"]["img"], img, inst.mods, m, w["it"]["res_m"], w["tl"][i][2], w["it"].get("clip"))
                inst.record(i, ev, verdict(ev, c))
        if job is not None:
            job.check_cancel()

    n = 0
    for w in work:
        tl, cy = w["tl"], w["cy"]
        for inst in w["insts"]:
            if not inst.seen:
                continue
            p = inst.f["properties"]
            after = tl[inst.lo] if inst.lo >= 0 else None             # last flight without modules
            before = tl[inst.hi] if inst.hi < len(tl) else None       # first flight with modules
            y_lo = after[0] if after else None                         # built after this flight (same year possible)
            y_hi = before[0] if before else cy
            flights = [{"year": tl[i][0], "date": tl[i][1], "res_m": tl[i][2], **ev} for i, ev in sorted(inst.seen.items())]
            if after is None and inst.hi > 0:
                # no flight clearly without modules (only unclear ones below): just an upper bound, efficiency unchanged
                p.update(install_year=None, install_year_low=None, install_year_high=y_hi, install_year_basis="imagery",
                         age_flights=flights)
                continue
            if after is None:
                y_est = before[0] - 1          # modules on the oldest usable flight: older than that, how much is open
            else:
                y_est = int(round((y_lo + y_hi) / 2))
            _apply_year(p, y_est, y_lo, y_hi, conf)
            p.update(install_year_basis="imagery", age_flights=flights)
            if inst.contradicted:
                p["age_note"] = "contradicting flights"
            n += 1
    return n


def _apply_year(p: dict, y_est: int, y_lo: int | None, y_hi: int, conf: dict):
    """build year → efficiency and kWp of one installation (kWp is proportional to the efficiency)"""
    e_old = float(p.get("module_efficiency") or conf["module"]["efficiency"])
    er = conf["module"]["efficiency_range"]
    e_est = typology.efficiency_for_year(y_est, conf)
    e_lo = typology.efficiency_for_year(y_lo if y_lo else int(min(conf["module"]["efficiency_by_year"])), conf)
    e_hi = typology.efficiency_for_year(y_hi, conf)
    if p.get("kwp_est") is not None:
        p["kwp_est"] = round(p["kwp_est"] * e_est / e_old, 1)
    if p.get("kwp_low") is not None:
        p["kwp_low"] = round(min(p["kwp_low"] / er[0] * e_lo, p["kwp_est"]), 1)
    if p.get("kwp_high") is not None:
        p["kwp_high"] = round(max(p["kwp_high"] / er[1] * e_hi, p["kwp_est"]), 1)
    p.update(install_year=y_est, install_year_low=y_lo, install_year_high=y_hi, module_efficiency=e_est)
