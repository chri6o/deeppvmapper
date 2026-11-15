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

PROMPT_CANDIDATES = """You see an orthophoto (aerial image, north up) of one building. Image size: {w}×{h} pixels,
about {res} m per pixel. The building footprint is outlined in YELLOW. Candidate regions proposed by a
detector are outlined in CYAN and numbered.

Task: decide which candidates are rooftop SOLAR PV MODULES (photovoltaic panels: dark blue/black rectangular
cells in regular rows; not skylights, windows, roof hatches, shadows, dark roofing, solar thermal tubes or
cars). If clearly visible PV modules on the outlined building are not covered by any candidate, add them as
polygons in pixel coordinates of this image.

Answer ONLY with JSON:
{{"pv_visible": true|false, "confidence": 0.0-1.0, "pv_candidates": [numbers], "extra_polygons": [[[x,y],...],...], "note": "short"}}"""

PROMPT_POLYGONS = """You see an orthophoto (aerial image, north up) of one building. Image size: {w}×{h} pixels,
about {res} m per pixel. The building footprint is outlined in YELLOW. A light grid with labels every 100 px
helps with coordinates.

Task: outline every area of rooftop SOLAR PV MODULES (photovoltaic panels: dark blue/black rectangular cells in
regular rows; not skylights, windows, shadows, dark roofing, solar thermal tubes) on the outlined building as
polygons in pixel coordinates of this image (x to the right, y downwards, origin top-left). Follow the outer
edge of each contiguous block of modules; one polygon per block.

Answer ONLY with JSON:
{{"pv_visible": true|false, "confidence": 0.0-1.0, "polygons": [[[x,y],...],...], "note": "short"}}"""


class AIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:                                            # noqa: BLE001
        return ImageFont.load_default()


def render_for_ai(img: np.ndarray, footprint, candidates=None, grid=False, max_side=1024):
    """Draws footprint (yellow), numbered candidates (cyan) and optionally a coordinate grid.
    Returns (JPEG bytes, scale) – coordinates in the AI answer are divided by scale."""
    im = Image.fromarray(img).convert("RGB")
    scale = min(1.0, max_side / max(im.size)) if max(im.size) > max_side else min(2.0, 768 / max(im.size))
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 300)
    if grid:
        f = _font(max(10, im.width // 70))
        step = 100
        for x in range(0, im.width, step):
            d.line([(x, 0), (x, im.height)], fill=(255, 255, 255), width=1)
            d.text((x + 2, 2), str(x), fill=(255, 255, 255), font=f)
        for y in range(0, im.height, step):
            d.line([(0, y), (im.width, y)], fill=(255, 255, 255), width=1)
            d.text((2, y + 2), str(y), fill=(255, 255, 255), font=f)
    for ring in footprint or []:
        d.line([(x * scale, y * scale) for x, y in ring + ring[:1]], fill=(255, 220, 0), width=lw)
    f = _font(max(14, im.width // 45))
    for i, ring in enumerate(candidates or [], 1):
        pts = [(x * scale, y * scale) for x, y in ring]
        d.line(pts + pts[:1], fill=(0, 230, 255), width=lw)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        d.rectangle([cx - 12, cy - 12, cx + 12, cy + 12], fill=(0, 0, 0))
        d.text((cx - 8, cy - 11), str(i), fill=(0, 230, 255), font=f)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), scale


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


def ask(provider: str, api_key: str, model: str, prompt: str, jpeg: bytes, timeout: int = 120,
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

def _clean_polys(polys, scale, w, h, min_pts=3):
    out = []
    for p in polys or []:
        try:
            pts = [[min(max(float(x) / scale, 0), w - 1), min(max(float(y) / scale, 0), h - 1)] for x, y in p]
        except (TypeError, ValueError):
            continue
        if len(pts) >= min_pts and cv2.contourArea(np.array(pts, np.float32)) >= 12:
            out.append([[round(x, 1), round(y, 1)] for x, y in pts])
    return out


def label_item(cfg: dict, d: str, name: str, provider: str, mode: str = "candidates", det=None,
               resolution_m: float = 0.1) -> dict:
    """AI suggestion for one image. Returns {pv_visible, confidence, polygons, note, usage, mode, model}."""
    img = np.asarray(Image.open(f"{d}/{name}.jpg").convert("RGB"))
    h, w = img.shape[:2]
    fp = annotate.footprint_polygons(d, name)
    key = cfg.get(f"{provider}_api_key") or ""
    model = cfg.get(f"{provider}_model") or PROVIDERS[provider]["default_model"]
    cands = []
    if mode == "candidates" and det is not None:
        cands = annotate.suggest_from_model(det, d, name, cls_threshold=0.05)["polygons"]
        if not cands:
            mode = "polygons"                                    # nothing to judge: let the AI draw
    if mode == "candidates" and cands:
        jpeg, scale = render_for_ai(img, fp, cands)
        ans, usage = ask(provider, key, model, PROMPT_CANDIDATES.format(w=int(w * scale), h=int(h * scale),
                                                                         res=round(resolution_m / scale, 3)), jpeg)
        picked = []
        for n in ans.get("pv_candidates") or []:
            try:
                i = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(cands):
                picked.append(cands[i])
        polys = picked + _clean_polys(ans.get("extra_polygons"), scale, w, h)
    else:
        jpeg, scale = render_for_ai(img, fp, grid=True)
        ans, usage = ask(provider, key, model, PROMPT_POLYGONS.format(w=int(w * scale), h=int(h * scale),
                                                                       res=round(resolution_m / scale, 3)), jpeg)
        polys = _clean_polys(ans.get("polygons"), scale, w, h)
    if not ans.get("pv_visible"):
        polys = []
    return {"pv_visible": bool(ans.get("pv_visible")), "confidence": ans.get("confidence"), "polygons": polys,
            "note": str(ans.get("note") or "")[:300], "usage": usage, "mode": mode, "model": model,
            "provider": provider, "n_candidates": len(cands)}


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

    with ThreadPoolExecutor(int(p.get("concurrency", 3))) as pool:
        for m, r, err in pool.map(one, items):
            job.check_cancel()
            if err:
                job.step(1, fehler=1)
                if job.counters["fehler"] <= 10:
                    job.log(f"{m['id']}: {err}")
                continue
            tokens["in"] += (r["usage"].get("input_tokens") or 0)
            tokens["out"] += (r["usage"].get("output_tokens") or 0)
            ai = {k: r[k] for k in ("provider", "model", "mode", "confidence", "note", "pv_visible", "n_candidates")}
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
