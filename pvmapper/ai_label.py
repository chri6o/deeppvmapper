"""
AI-assisted PV masks via Claude (Anthropic) or ChatGPT (OpenAI) vision models.

Why two modes:
  Large language/vision models recognise solar panels on aerial photos well, but they are weak at returning
  pixel-exact outlines. So the default mode lets them *judge* instead of *draw*:

  candidates  The segmentation model proposes regions with a low threshold; they are drawn on the image as
              numbered outlines. The AI answers which numbers are PV panels (and may add missing areas as
              polygons). Outlines stay pixel-accurate, the decision comes from an independent model.
  polygons    The AI returns PV polygons in pixel coordinates directly (rougher; useful when the segmentation
              model finds nothing).

Every AI mask is stored with mask_source='ai' and reviewed=false. Training uses AI masks only after a user
has checked them in the mask editor (setting ai_masks='reviewed', default) – unless explicitly set to 'all'.
Evaluation always uses register / reported labels, never AI output.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from . import annotate

PROVIDERS = {
    "anthropic": {"label": "Claude (Anthropic)", "url": "https://api.anthropic.com/v1/messages",
                  "default_model": "claude-sonnet-5-5"},
    "openai": {"label": "ChatGPT (OpenAI)", "url": "https://api.openai.com/v1/chat/completions",
               "default_model": "gpt-5.1"},
}

RULES = """What counts as PV: the MODULE FIELDS themselves – dark blue/black rectangular panels with a fine grid
of cells, arranged in rows or blocks, often with thin gaps between rows. Mark ONLY these module surfaces.

Do NOT mark (common confusions – check each region against this list):
- DORMERS: small gabled or flat structures on pitched roofs, usually several in a regular row, each with a lit and
  a shaded side and a window; their dark roof surfaces look like panels but are NOT PV
- roof windows / skylights / light domes / glass strips: single small dark or reflective rectangles, often in a
  regular grid on flat roofs, usually ≤ 1.5 m wide
- solar thermal collectors (few large panels with visible tubes or a different, more uniform surface)
- the roof outline or the whole roof, roof edges, the building footprint (YELLOW line), parapets, roof hatches,
  ventilation units, chimneys, gravel or dark bitumen/slate roofing, shadows, green roofs, cars, the ground
- anything outside the YELLOW footprint

A PV field usually covers only PART of a roof. A polygon that follows the roof or footprint outline is wrong.
Draw tight polygons around each contiguous block of modules (one polygon per block; a rotated rectangle with
4 corners is ideal for a regular block). Include the WHOLE block: panels in bright sunlight, with glare or in
partial shade belong to the block, and so do the narrow gaps between the panels of a row."""

PROMPT_CANDIDATES = """You see an orthophoto (aerial image, north up) of one building. Image size: {w}×{h} pixels,
about {res} m per pixel. The building footprint is outlined in YELLOW. Candidate regions proposed by a detector
are outlined in CYAN and numbered.
{expect}
""" + RULES + """

Task: classify EVERY numbered candidate as one of: "pv", "dormer", "skylight", "solar_thermal", "roof",
"shadow", "other". Only "pv" candidates are kept. If PV modules are visible that no "pv" candidate covers
(or only partly), add tight polygons around those whole blocks in "extra_polygons" (pixel coordinates of this
image, x right, y down). {hint}

Answer ONLY with JSON:
{{"pv_visible": true|false, "confidence": 0.0-1.0, "candidates": {{"1": "pv", "2": "dormer", ...}}, "extra_polygons": [[[x,y],...],...], "note": "short"}}"""

PROMPT_POLYGONS = """You see an orthophoto (aerial image, north up) of one building. Image size: {w}×{h} pixels,
about {res} m per pixel. The building footprint is outlined in YELLOW. A light grid with labels every 100 px
helps with coordinates (x to the right, y downwards, origin top-left).
{expect}
""" + RULES + """

Task: return tight polygons around every block of PV modules on the outlined building, in pixel coordinates of
this image. Before answering, check every polygon against the list above (dormers and skylights are the most
common mistakes) and label it; only "pv" polygons are used. {hint}

Answer ONLY with JSON:
{{"pv_visible": true|false, "confidence": 0.0-1.0, "polygons": [{{"kind": "pv"|"dormer"|"skylight"|"solar_thermal"|"other", "points": [[x,y],...]}}, ...], "note": "short"}}"""


class AIError(RuntimeError):
    pass


import threading  # noqa: E402

_DET_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:                                            # noqa: BLE001
        return ImageFont.load_default()


def render_for_ai(img: np.ndarray, footprint, candidates=None, grid=False, max_side=1024, crop=True):
    """Crops to the footprint (+ margin) and enlarges, so modules are well visible; draws footprint (yellow),
    numbered candidates (cyan) and optionally a coordinate grid.
    Returns (JPEG bytes, scale, (ox, oy)): image px = answer px / scale + offset."""
    h, w = img.shape[:2]
    ox = oy = 0
    if crop and footprint:
        xs = [x for r in footprint for x, _ in r]
        ys = [y for r in footprint for _, y in r]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mx = max(40, 0.15 * (x1 - x0))
        my = max(40, 0.15 * (y1 - y0))
        ox, oy = int(max(0, x0 - mx)), int(max(0, y0 - my))
        ex, ey = int(min(w, x1 + mx)), int(min(h, y1 + my))
        if ex - ox > 20 and ey - oy > 20:
            img = img[oy:ey, ox:ex]
        else:
            ox = oy = 0
    im = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
    scale = max_side / max(im.size)
    scale = min(scale, 4.0)
    if abs(scale - 1.0) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 350)
    if grid:
        f = _font(max(10, im.width // 70))
        for x in range(0, im.width, 100):
            d.line([(x, 0), (x, im.height)], fill=(255, 255, 255), width=1)
            d.text((x + 2, 2), str(x), fill=(255, 255, 255), font=f)
        for y in range(0, im.height, 100):
            d.line([(0, y), (im.width, y)], fill=(255, 255, 255), width=1)
            d.text((2, y + 2), str(y), fill=(255, 255, 255), font=f)
    T = lambda x, y: ((x - ox) * scale, (y - oy) * scale)
    for ring in footprint or []:
        pts = [T(x, y) for x, y in ring]
        d.line(pts + pts[:1], fill=(255, 220, 0), width=lw)
    f = _font(max(14, im.width // 45))
    for i, ring in enumerate(candidates or [], 1):
        pts = [T(x, y) for x, y in ring]
        d.line(pts + pts[:1], fill=(0, 230, 255), width=lw)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        d.rectangle([cx - 12, cy - 12, cx + 12, cy + 12], fill=(0, 0, 0))
        d.text((cx - 8, cy - 11), str(i), fill=(0, 230, 255), font=f)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), scale, (ox, oy)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise AIError(f"keine JSON-Antwort: {text[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise AIError(f"JSON nicht lesbar: {e}: {m.group(0)[:200]}")


def ask(provider: str, api_key: str, model: str, prompt: str, jpeg: bytes, timeout: int = 90,
        session=None) -> tuple[dict, dict]:
    """Sends one image + prompt; returns (parsed JSON, usage)."""
    if provider not in PROVIDERS:
        raise AIError(f"unbekannter Anbieter {provider}")
    if not api_key:
        raise AIError(f"Kein API-Schlüssel für {PROVIDERS[provider]['label']} (Einstellungen).")
    model = model or PROVIDERS[provider]["default_model"]
    b64 = base64.b64encode(jpeg).decode()
    http = session or requests
    if provider == "anthropic":
        body = {"model": model, "max_tokens": 1500, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt}]}]}
        r = http.post(PROVIDERS[provider]["url"], json=body, timeout=timeout,
                      headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        if r.status_code != 200:
            raise AIError(f"Anthropic {r.status_code}: {r.text[:300]}")
        j = r.json()
        text = "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
        u = j.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}
    else:
        body = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}]}],
            "response_format": {"type": "json_object"}}
        r = http.post(PROVIDERS[provider]["url"], json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"})
        if r.status_code != 200:
            raise AIError(f"OpenAI {r.status_code}: {r.text[:300]}")
        j = r.json()
        text = j["choices"][0]["message"].get("content") or ""
        u = j.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}
    return _parse_json(text), usage


# ---------------------------------------------------------------------------
# one dataset item
# ---------------------------------------------------------------------------

def _clean_polys(polys, scale, w, h, min_pts=3, off=(0, 0)):
    out = []
    for p in polys or []:
        try:
            pts = [[min(max(float(x) / scale + off[0], 0), w - 1), min(max(float(y) / scale + off[1], 0), h - 1)]
                   for x, y in p]
        except (TypeError, ValueError):
            continue
        if len(pts) >= min_pts and cv2.contourArea(np.array(pts, np.float32)) >= 12:
            out.append([[round(x, 1), round(y, 1)] for x, y in pts])
    return out


def _pv_only(polys):
    """polygons may come as [[x,y],…] or {"kind": …, "points": [[x,y],…]}; keep PV only"""
    out = []
    for p in polys or []:
        if isinstance(p, dict):
            if str(p.get("kind", "pv")).lower() == "pv":
                out.append(p.get("points") or [])
        else:
            out.append(p)
    return out


def _expect_text(meta, res_m, scale, kwp_per_m2):
    kwp = (meta or {}).get("pv_kwp")
    if not kwp:
        return "", None
    m2 = float(kwp) / kwp_per_m2
    px = m2 / (res_m / scale) ** 2
    return (f"Hint: the PV register lists about {float(kwp):.0f} kWp for this building, i.e. roughly {m2:.0f} m² of "
            f"modules ≈ {px:.0f} px² in this image. Use this to judge the size of the module area (it is usually "
            f"much smaller than the roof)."), m2


def refine_to_modules(img: np.ndarray, polys: list, expect_px: float | None = None) -> list:
    """Shrinks AI polygons to the dark, bluish pixels typical of PV modules inside them. Only used when the
    result is plausible (15–90 % of the polygon and, if the register size is known, 0.4–2.5× of it);
    otherwise the AI polygon is kept as it is (dark modules on dark roofs cannot be separated by colour)."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    V = f.max(-1)
    blue = f[..., 2] - f[..., 0]
    out = []
    for p in polys:
        m = annotate.polygons_to_mask([p], w, h).astype(bool)
        if m.sum() < 30:
            continue
        med = float(np.median(V[m]))
        medb = float(np.median(blue[m]))
        # darker than the polygon's median, or clearly bluer (modules in sunlight / with glare)
        cand = (((V < med * 0.85) & (blue > 4)) | (blue > medb + 10)) & m
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        share = cand.sum() / m.sum()
        ok = 0.15 <= share <= 0.9
        if ok and expect_px:
            ok = 0.4 * expect_px <= cand.sum() <= 2.5 * expect_px
        out += annotate.mask_to_px_polygons(cand, min_area=40, eps=1.5) if ok else [p]
    return out


def _postprocess(img, polys, fp, res_m, expect_px, cfg):
    """same rules as for the detector (postproc.py): footprint clip, dormer rows, contrast, completion of rows,
    rectangles. Falls back to the AI polygons if the result is empty or implausible."""
    if not polys or not (cfg or {}).get("postproc", True):
        return polys
    from . import postproc
    h, w = img.shape[:2]
    m = annotate.polygons_to_mask(polys, w, h)
    clip = None
    if fp:
        clip = cv2.dilate(annotate.polygons_to_mask(fp, w, h),
                          np.ones((2 * int(1.5 / res_m) + 1,) * 2, np.uint8)).astype(bool)
    out, _ = postproc.clean(img, m, res_m, clip, cfg)
    a = int(out.sum())
    if a == 0:
        return polys
    if expect_px and not (0.3 * expect_px <= a <= 3 * expect_px) and abs(a - expect_px) > abs(int(m.sum()) - expect_px):
        return polys                                     # further from the register size than before
    return annotate.mask_to_px_polygons(out, min_area=40, eps=1.5)


def _area_px(polys, w, h):
    return int(annotate.polygons_to_mask(polys, w, h).sum()) if polys else 0


def label_item(cfg: dict, d: str, name: str, provider: str, mode: str = "candidates", det=None,
               resolution_m: float = 0.1) -> dict:
    """AI suggestion for one image. Returns {pv_visible, confidence, polygons, note, usage, mode, model}."""
    img = np.asarray(Image.open(f"{d}/{name}.jpg").convert("RGB"))
    h, w = img.shape[:2]
    meta = annotate.read_meta(d, name)
    fp = annotate.footprint_polygons(d, name)
    fp_px = _area_px(fp, w, h)
    key = cfg.get(f"{provider}_api_key") or ""
    model = cfg.get(f"{provider}_model") or PROVIDERS[provider]["default_model"]
    kwp_m2 = float(cfg.get("kwp_per_m2") or 0.2)
    hint = (cfg.get("ai_prompt_hint") or "").strip()
    cands = []
    if mode == "candidates" and det is not None:
        with _DET_LOCK:                                          # one GPU user at a time (threads share the model)
            cands = annotate.suggest_from_model(det, d, name, cls_threshold=0.05)["polygons"]
        # candidates that are basically the whole roof are useless to judge
        cands = [c for c in cands if not fp_px or _area_px([c], w, h) < 0.85 * fp_px]
        if not cands:
            mode = "polygons"                                    # nothing to judge: let the AI draw
    if mode == "candidates" and cands:
        jpeg, scale, off = render_for_ai(img, fp, cands)
        expect, exp_m2 = _expect_text(meta, resolution_m, scale, kwp_m2)
        exp_px = exp_m2 / resolution_m ** 2 if exp_m2 else None
        ans, usage = ask(provider, key, model, PROMPT_CANDIDATES.format(
            w=int(w * scale), h=int(h * scale), res=round(resolution_m / scale, 3), expect=expect, hint=hint), jpeg)
        picked = []
        kinds = ans.get("candidates") if isinstance(ans.get("candidates"), dict) else {}
        nums = [k for k, v in kinds.items() if str(v).lower() == "pv"] or ans.get("pv_candidates") or []
        for n in nums:
            try:
                i = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(cands):
                picked.append(cands[i])
        extra = refine_to_modules(img, _clean_polys(ans.get("extra_polygons"), scale, w, h, off=off), exp_px)
        polys = picked + extra
    else:
        jpeg, scale, off = render_for_ai(img, fp, grid=True)
        expect, exp_m2 = _expect_text(meta, resolution_m, scale, kwp_m2)
        exp_px = exp_m2 / resolution_m ** 2 if exp_m2 else None
        ans, usage = ask(provider, key, model, PROMPT_POLYGONS.format(
            w=int(w * scale), h=int(h * scale), res=round(resolution_m / scale, 3), expect=expect, hint=hint), jpeg)
        polys = refine_to_modules(img, _clean_polys(_pv_only(ans.get("polygons")), scale, w, h, off=off), exp_px)
    if not ans.get("pv_visible"):
        polys = []
    polys = _postprocess(img, polys, fp, resolution_m, exp_px, cfg)
    # plausibility: a mask covering (nearly) the whole footprint is a roof outline, not modules
    area = _area_px(polys, w, h)
    flag = None
    if polys and fp_px and area > 0.9 * fp_px:
        flag = "Fläche ≈ ganzes Dach – bitte prüfen"
    elif polys and exp_m2 and area * resolution_m ** 2 > 3 * exp_m2:
        flag = f"Fläche {area * resolution_m ** 2:.0f} m² deutlich größer als laut Register (~{exp_m2:.0f} m²)"
    note = str(ans.get("note") or "")[:300]
    return {"pv_visible": bool(ans.get("pv_visible")), "confidence": ans.get("confidence"), "polygons": polys,
            "note": (flag + " · " if flag else "") + note, "usage": usage, "mode": mode, "model": model,
            "provider": provider, "n_candidates": len(cands), "flag": flag}


# ---------------------------------------------------------------------------
# batch job
# ---------------------------------------------------------------------------

def run(job, cfg: dict):
    """Job kind 'ai_masks': AI suggestions for the positives of a dataset that have no mask yet."""
    from concurrent.futures import ThreadPoolExecutor
    from .jobs import Job
    from . import runtime
    p = job.params
    src_job = Job.load(cfg["jobs_dir"], p["dataset"])
    d = annotate.ds_dir(src_job.dir)
    provider = p.get("provider", "anthropic")
    mode = p.get("mode", "candidates")
    items = [m for m in annotate.list_items(d)
             if annotate.effective_label(m) == 1 and m.get("status") != "skip"
             and (not m.get("has_mask") or (p.get("overwrite_ai") and m.get("mask_source") == "ai"))]
    limit = int(p.get("limit") or 0)
    if limit:
        items = items[:limit]
    job.total, job.done = len(items), 0
    job.log(f"KI-Masken mit {PROVIDERS[provider]['label']} ({cfg.get(provider + '_model') or PROVIDERS[provider]['default_model']}), "
            f"Modus {mode}, {len(items)} Bilder ohne Maske.")
    job.save()
    det = None
    if mode == "candidates":
        det = runtime.detector(cfg, src_job.params.get("profile"))
    res_m = det.p.resolution_m if det else 0.1
    tokens = {"in": 0, "out": 0}

    def one(m):
        if job._cancel.is_set():
            return m, None, "abgebrochen"
        try:
            return m, label_item(cfg, d, m["name"], provider, mode, det, res_m), None
        except Exception as e:                                   # noqa: BLE001
            return m, None, str(e)[:300]

    from concurrent.futures import as_completed
    t_start = time.time()
    with ThreadPoolExecutor(int(p.get("concurrency", 3))) as pool:
        futs = [pool.submit(one, m) for m in items]
        for k, fut in enumerate(as_completed(futs)):            # progress as soon as any image is done
            m, r, err = fut.result()
            if k == 0:
                job.log(f"Erstes Bild nach {time.time() - t_start:.0f} s fertig.")
            job.check_cancel()
            if err:
                job.step(1, fehler=1)
                if job.counters["fehler"] <= 10:
                    job.log(f"{m['id']}: {err}")
                continue
            tokens["in"] += (r["usage"].get("input_tokens") or 0)
            tokens["out"] += (r["usage"].get("output_tokens") or 0)
            ai = {k: r[k] for k in ("provider", "model", "mode", "confidence", "note", "pv_visible", "n_candidates", "flag")}
            ai.update(reviewed=False, at=time.strftime("%Y-%m-%d %H:%M:%S"))
            if r["polygons"]:
                annotate.save_annotation(d, m["name"], r["polygons"], status="open", source="ai", ai=ai)
                job.step(1, maske=1)
            else:
                meta = annotate.read_meta(d, m["name"])
                meta["ai"] = ai
                annotate.write_meta(d, meta)
                job.step(1, keine_pv_laut_ki=1)
    job.counters["tokens_in"], job.counters["tokens_out"] = tokens["in"], tokens["out"]
    job.log(f"Fertig. Tokens: {tokens['in']} ein, {tokens['out']} aus. Masken jetzt im Reiter „Masken“ prüfen "
            "(Filter „KI ungeprüft“).")
