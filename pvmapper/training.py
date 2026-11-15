"""
Fine-tuning with the existing PV stock (PV-Map training sample or a local labelled file).

Steps of a training job
  1. Sample    labelled buildings from PV-Map (/api/detect/training): positives from registers/reports/layers,
               negatives from reports ('no PV') and 'no PV known' (weak), stratified by building type × PV size.
  2. Imagery   one image per building (footprint + buffer) from the open-data sources; PV polygons, where PV-Map
               has them, are rasterised to masks. Cached under dataset/.
  3. Split     train / validation by building id (deterministic).
  4. Baseline  the base model on the validation set → metrics per label, building type, size class.
  5. Clean     weak negatives the base model is very sure about (likely unregistered PV) are dropped from training.
  6. Train     classifier with multiple-instance learning: a building's label applies to the *max* over its patches,
               so small and large roofs work alike. Segmentation head only if enough PV polygons exist.
  7. Tune      classification threshold for the best F1 on validation; kWp per m² calibrated per size class
               (register kWp / detected area).
  8. Evaluate  the new model on the same validation set, side by side with the baseline → report.json.
  9. Register  weights + profile JSON in model_dir/custom/ → selectable as model profile "custom:<name>".
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from shapely.geometry import shape

from . import config as cfgmod
from . import annotate, geo, inference, models, runtime, sources


# ---------------------------------------------------------------------------
# 1. sample
# ---------------------------------------------------------------------------

def load_sample(job, cfg) -> list[dict]:
    p = job.params
    if p.get("labels_file"):                                   # local GeoJSON with properties.label (0/1)
        with open(p["labels_file"], encoding="utf-8") as f:
            fc = json.load(f)
        feats = fc["features"]
        for ft in feats:
            pr = ft.setdefault("properties", {})
            pr.setdefault("label_kind", "pos_high" if int(pr.get("label", 0)) else "neg_high")
            pr.setdefault("size_bin", "?")
            pr.setdefault("sub", pr.get("sub") or "?")
        return feats
    from . import pvmap
    cl = pvmap.Client.from_config(cfg)
    flt = {k: p.get(k) for k in pvmap.FILTER_KEYS if p.get(k)}
    d = cl.training(bins=p.get("bins") or None, per_stratum=int(p.get("per_stratum", 100)),
                    negatives=int(p.get("negatives", 100)), seed=str(p.get("seed", "1")), **flt)
    feats = d["features"]
    if not p.get("use_weak_negatives", True):
        feats = [f for f in feats if f["properties"]["label_kind"] != "neg_weak"]
    if not p.get("use_medium_positives", True):
        feats = [f for f in feats if f["properties"]["label_kind"] != "pos_medium"]
    if p.get("min_kwp"):
        lo = float(p["min_kwp"])
        feats = [f for f in feats if f["properties"]["label_kind"].startswith("neg")
                 or (f["properties"].get("pv_kwp") or 0) >= lo]
    return feats


def is_val(bid: str, share: float) -> bool:
    h = int(hashlib.md5(str(bid).encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000 < share


# ---------------------------------------------------------------------------
# 2. imagery + masks
# ---------------------------------------------------------------------------

def _rasterize(geom_wgs84, win: geo.Window) -> np.ndarray:
    m = Image.new("L", (win.width, win.height), 0)
    d = ImageDraw.Draw(m)
    for ring in runtime.geom_to_px(geom_wgs84, win):
        d.polygon([tuple(p) for p in ring], fill=1)
    return np.asarray(m, dtype=np.uint8)


def build_dataset(job, cfg, feats, prof) -> list[dict]:
    ds_dir = os.path.join(job.dir, "dataset")
    os.makedirs(ds_dir, exist_ok=True)
    cat = runtime.catalog(cfg)
    src_filter = runtime.source_filter(cfg, job.params.get("sources"))
    buffer_m = float(cfg["buffer_m"])

    def one(ft):
        pr = ft["properties"]
        bid = str(ft.get("id") or pr.get("id"))
        name = runtime.safe_name(bid)
        meta_path = os.path.join(ds_dir, name + ".json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        g = shape(ft["geometry"])
        if g.geom_type == "Point":
            c = g
            utm = f"EPSG:{geo.utm_epsg(c.x, c.y)}"
            g = geo.reproject(geo.reproject(c, "EPSG:4326", utm).buffer(15), utm, "EPSG:4326")
        try:
            img, win, src = sources.fetch_for_geometry(cat, g, prof.resolution_m, buffer_m, src_filter,
                                                       min_px=prof.patch_px)
        except Exception as e:                                   # noqa: BLE001
            return {"id": bid, "error": str(e)[:200]}
        Image.fromarray(img).save(os.path.join(ds_dir, name + ".jpg"), quality=92)
        fp = _rasterize(g, win)
        Image.fromarray(fp * 255).save(os.path.join(ds_dir, name + ".fp.png"))
        has_mask = False
        if pr.get("pv_geometry"):
            m = _rasterize(shape(pr["pv_geometry"]), win)
            if m.any():
                Image.fromarray(m * 255).save(os.path.join(ds_dir, name + ".mask.png"))
                has_mask = True
        meta = {"id": bid, "name": name, "label": int(pr.get("label", 0)), "label_kind": pr.get("label_kind"),
                "sub": pr.get("sub") or "?", "size_bin": pr.get("size_bin") or "?", "pv_kwp": pr.get("pv_kwp"),
                "source": src.id, "has_mask": has_mask, "mask_source": "pvmap" if has_mask else None,
                "status": "open", "window": [win.crs, win.minx, win.miny, win.maxx, win.maxy,
                                                                     win.width, win.height],
                "val": is_val(bid, float(job.params.get("val_share", 0.25)))}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return meta

    out = []
    job.total = len(feats)
    job.done = 0
    with ThreadPoolExecutor(int(cfg["fetch_workers"])) as pool:
        for meta in pool.map(one, feats):
            job.check_cancel()
            if "error" in meta:
                job.step(1, bild_fehlt=1)
            else:
                out.append(meta)
                job.step(1)
    return out


def load_arrays(ds_dir, meta):
    ds_dir = meta.get("_dir") or ds_dir
    img = np.asarray(Image.open(os.path.join(ds_dir, meta["name"] + ".jpg")).convert("RGB"))
    fp = np.asarray(Image.open(os.path.join(ds_dir, meta["name"] + ".fp.png"))) > 0
    mask = None
    if meta.get("has_mask"):
        mask = np.asarray(Image.open(os.path.join(ds_dir, meta["name"] + ".mask.png"))) > 0
    return img, fp, mask


# ---------------------------------------------------------------------------
# 4./8. evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_buildings(det: inference.Detector, ds_dir, metas):
    """Per building: patch probabilities and (for every patch) the segmented PV pixels inside the footprint,
    so any classification threshold can be evaluated afterwards without re-running the models."""
    size = det.p.patch_px
    out = {}
    for m in metas:
        img, fp, _ = load_arrays(ds_dir, m)
        h, w = img.shape[:2]
        grid = inference.patch_grid(w, h, size, det.overlap)
        patches = [img[y:y + size, x:x + size] for x, y in grid]
        probs, pix = [], []
        for b in range(0, len(patches), det.batch_size):
            x = det._to_tensor(patches[b:b + det.batch_size])
            n = x.shape[0]
            xp = det._pad(x, det._bucket(n, det.batch_size))
            probs += det._classify(xp).float().cpu().tolist()[:n]
            segs = det._segment(xp).cpu().numpy()[:n]
            for (px, py), sm in zip(grid[b:b + det.batch_size], segs):
                hit = np.zeros((h, w), bool)
                hit[py:py + size, px:px + size] = sm[:min(size, h - py), :min(size, w - px)]
                pix.append(np.flatnonzero(hit & fp))
        out[m["id"]] = (np.array(probs), pix)
    return out


def predict(scores, threshold, min_px):
    """building id → (has_pv, PV pixels in footprint) for a classification threshold"""
    res = {}
    for bid, (probs, pix) in scores.items():
        sel = [pix[i] for i in np.flatnonzero(probs >= threshold)]
        n = len(np.unique(np.concatenate(sel))) if sel else 0
        res[bid] = (n >= min_px, n)
    return res


def metrics(metas, pred, px_area_m2):
    """overall and per group: precision/recall/F1 on strong labels, plus recall per size class, FP rate per label"""
    def block(rows):
        tp = sum(1 for m, (p, _) in rows if p and m["label"] == 1)
        fp = sum(1 for m, (p, _) in rows if p and m["label"] == 0)
        fn = sum(1 for m, (p, _) in rows if not p and m["label"] == 1)
        tn = sum(1 for m, (p, _) in rows if not p and m["label"] == 0)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": _r(prec), "recall": _r(rec), "f1": _r(f1)}
    rows = [(m, pred[m["id"]]) for m in metas if m["id"] in pred]
    strong = [r for r in rows if r[0]["label_kind"] in ("pos_high", "neg_high", "pos_medium")]
    out = {"alle": block(rows), "nur_sichere_labels": block(strong), "nach_label": {}, "nach_art": {}, "nach_groesse": {}}
    for key, fn in (("nach_label", lambda m: m["label_kind"]), ("nach_art", lambda m: m["sub"]),
                    ("nach_groesse", lambda m: m["size_bin"])):
        g = collections.defaultdict(list)
        for r in rows:
            g[fn(r[0])].append(r)
        out[key] = {k: block(v) for k, v in sorted(g.items())}
    # kWp calibration: register kWp vs detected PV area, per size class
    cal = collections.defaultdict(list)
    for m, (p, n) in rows:
        if m["label"] == 1 and p and m.get("pv_kwp") and n > 0:
            cal[m["size_bin"]].append(float(m["pv_kwp"]) / (n * px_area_m2))
    out["kwp_pro_m2"] = {k: {"n": len(v), "median": _r(float(np.median(v)))} for k, v in sorted(cal.items())}
    return out


def _r(x, d=3):
    return None if x is None else round(float(x), d)


def tune_threshold(metas, scores, min_px):
    best = (None, -1)
    strong = [m for m in metas if m["label_kind"] != "neg_weak"] or metas
    for t in np.arange(0.2, 0.96, 0.025):
        mt = metrics(strong, predict(scores, t, min_px), 1)["alle"]
        f1 = mt["f1"] or 0
        if f1 > best[1]:
            best = (round(float(t), 3), f1)
    return best


# ---------------------------------------------------------------------------
# 6. training
# ---------------------------------------------------------------------------

def _augment(x: torch.Tensor) -> torch.Tensor:
    """x: (N,3,H,W) in [0,1] — flips, 90° rotations, brightness/contrast jitter"""
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k, (-2, -1))
    b = 1 + (random.random() - 0.5) * 0.3
    c = 1 + (random.random() - 0.5) * 0.3
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean * b).clamp(0, 1)


def _patches_for(det, img, fp, mask, max_k, size):
    """candidate patches of a building: those touching the footprint; for PV polygons, those with PV first"""
    h, w = img.shape[:2]
    grid = inference.patch_grid(w, h, size, 0.0)
    cand = []
    for x, y in grid:
        f = fp[y:y + size, x:x + size].mean()
        pv = mask[y:y + size, x:x + size].mean() if mask is not None else 0
        if f > 0 or len(grid) == 1:
            cand.append((pv, f, x, y))
    cand.sort(reverse=True)
    return [(x, y) for _, _, x, y in cand[:max_k]] or [grid[0]]


def train_classifier(job, det, ds_dir, train, epochs, lr, weak_weight, max_patches):
    model = det.cls
    size = det.p.patch_px
    for name, prm in model.named_parameters():                     # fine-tune the top of the network only
        prm.requires_grad = name.startswith(("Mixed_6d", "Mixed_6e", "Mixed_7", "fc"))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(train)))
    pos = [m for m in train if m["label"] == 1]
    neg = [m for m in train if m["label"] == 0]
    job.log(f"Training Klassifikator: {len(pos)} positiv, {len(neg)} negativ, {epochs} Epochen")
    cache = {}
    for ep in range(epochs):
        model.train()
        for mod in model.modules():                               # keep BatchNorm statistics of the base model
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        order = train[:]
        random.shuffle(order)
        losses = []
        for m in order:
            job.check_cancel()
            if m["id"] not in cache:
                img, fp, mask = load_arrays(ds_dir, m)
                cache[m["id"]] = np.stack([img[y:y + size, x:x + size]
                                           for x, y in _patches_for(det, img, fp, mask, max_patches, size)])
            x = det._to_tensor(list(cache[m["id"]]))
            x = _augment(x)
            if x.shape[-1] != det.p.cls_input_px:
                x = F.interpolate(x, size=(det.p.cls_input_px,) * 2, mode="bilinear", align_corners=False)
            logits = model((x - det.mean) / det.std)
            if isinstance(logits, tuple):
                logits = logits[0]
            z = logits[:, 1] - logits[:, 0] if det.p.cls_outputs == 2 else logits[:, 0]
            z = z.max()                                            # multiple-instance: building = max over patches
            y = torch.tensor(float(m["label"]), device=z.device)
            wgt = weak_weight if m["label_kind"] == "neg_weak" else 1.0
            # balance classes
            wgt *= (len(train) / (2 * max(1, len(pos)))) if m["label"] == 1 else (len(train) / (2 * max(1, len(neg))))
            loss = F.binary_cross_entropy_with_logits(z, y) * wgt
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        job.log(f"Epoche {ep + 1}/{epochs}: Verlust {np.mean(losses):.4f}")
        job.step(1)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False


def train_segmenter(job, det, ds_dir, train, epochs, lr):
    with_mask = [m for m in train if m.get("has_mask")]
    model = det.seg
    size = det.p.patch_px
    for name, prm in model.named_parameters():
        prm.requires_grad = name.startswith(("classifier", "backbone.layer4"))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=1e-4)
    job.log(f"Training Segmentierung: {len(with_mask)} Gebäude mit PV-Polygonen, {epochs} Epochen")
    for ep in range(epochs):
        model.train()
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        losses = []
        random.shuffle(with_mask)
        for m in with_mask:
            job.check_cancel()
            img, fp, mask = load_arrays(ds_dir, m)
            for x0, y0 in _patches_for(det, img, fp, mask, 2, size):
                xi = det._to_tensor([img[y0:y0 + size, x0:x0 + size]])
                yi = torch.from_numpy(mask[y0:y0 + size, x0:x0 + size].astype(np.float32))[None, None].to(xi.device)
                if random.random() < 0.5:
                    xi, yi = xi.flip(-1), yi.flip(-1)
                if random.random() < 0.5:
                    xi, yi = xi.flip(-2), yi.flip(-2)
                out = model((xi - det.mean) / det.std)["out"]
                loss = F.binary_cross_entropy_with_logits(out, yi)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
        job.log(f"Segmentierung Epoche {ep + 1}/{epochs}: Verlust {np.mean(losses) if losses else 0:.4f}")
        job.step(1)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------

def run(job, cfg: dict):
    p = job.params
    t0 = time.time()
    random.seed(int(p.get("seed", 1)) if str(p.get("seed", 1)).isdigit() else 1)
    torch.manual_seed(1)
    base_id = p.get("profile") or cfg["profile"]
    det = runtime.detector(cfg, base_id)
    prof = det.p
    if p.get("dataset_from"):                                     # reuse a dataset (with masks drawn by users / AI)
        from .jobs import Job
        src = Job.load(cfg["jobs_dir"], p["dataset_from"])
        ds_dir = annotate.ds_dir(src.dir)
        ai_mode = p.get("ai_masks", "reviewed")
        metas = []
        for m in annotate.list_items(ds_dir):
            if m.get("status") == "skip":
                continue
            m = dict(m, label=annotate.effective_label(m))
            m["has_mask"] = annotate.usable_mask(m, ai_mode)
            metas.append(m)
        if not any(m["label"] == 0 for m in metas) and p.get("auto_negatives", True):
            # positives-only dataset (for masking): fetch negatives with the same filters for training
            from . import pvmap
            sp = src.params
            flt = {k: sp.get(k) for k in pvmap.FILTER_KEYS if sp.get(k)}
            n_neg = int(p.get("negatives") or max(50, len(metas) // 2))
            job.log(f"Datensatz hat keine Negativbeispiele – lade {n_neg} je Gebäudeart aus PV-Map …")
            d = pvmap.Client.from_config(cfg).training(per_stratum=0, negatives=n_neg,
                                                       seed=str(sp.get("seed", "1")), **flt)
            negf = [f for f in d["features"] if f["properties"]["label"] == 0
                    and (p.get("use_weak_negatives", sp.get("use_weak_negatives", True))
                         or f["properties"]["label_kind"] != "neg_weak")]
            neg_metas = build_dataset(job, cfg, negf, prof)
            own = os.path.join(job.dir, "dataset")
            for m in neg_metas:
                m["_dir"] = own
            metas += neg_metas
        kinds = collections.Counter(m["label_kind"] for m in metas)
        job.log(f"Datensatz aus Auftrag {src.id}: {len(metas)} Gebäude, "
                f"{sum(1 for m in metas if m['has_mask'])} mit nutzbarer Maske (KI-Masken: {ai_mode}).")
    else:
        job.log(f"Lade Stichprobe ({'Datei' if p.get('labels_file') else 'PV-Map'}) …")
        feats = load_sample(job, cfg)
        kinds = collections.Counter(f["properties"]["label_kind"] for f in feats)
        job.log(f"{len(feats)} Gebäude: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        if not feats:
            raise RuntimeError("Die Stichprobe ist leer – Filter prüfen.")
        job.log("Lade Luftbilder …")
        metas = build_dataset(job, cfg, feats, prof)
        ds_dir = os.path.join(job.dir, "dataset")
        if p.get("prepare_only"):
            job.counters["datensatz"] = len(metas)
            job.log(f"Datensatz angelegt: {len(metas)} Gebäude, "
                    f"{sum(1 for m in metas if m.get('has_mask'))} mit PV-Polygonen aus PV-Map. "
                    "Jetzt im Reiter „Masken“ bearbeiten, danach Training mit diesem Datensatz starten.")
            return

    # fresh copies of the networks: training must not change the cached base model
    import copy
    det = inference.Detector(prof, copy.deepcopy(det.cls), copy.deepcopy(det.seg), det.device,
                             cls_threshold=det.cls_threshold, seg_threshold=det.seg_threshold,
                             batch_size=det.batch_size, overlap=det.overlap)

    val = [m for m in metas if m["val"]]
    train = [m for m in metas if not m["val"]]
    if len(val) < 10 or len({m["label"] for m in train}) < 2:
        raise RuntimeError(f"Zu wenige Beispiele (Training {len(train)}, Validierung {len(val)}) oder nur eine Klasse.")
    px_area = prof.resolution_m ** 2
    min_px = int(float(cfg["min_pv_area_m2"]) / px_area)

    job.log(f"Ausgangsmodell '{base_id}' auf {len(val)} Validierungsgebäuden …")
    epochs = int(p.get("epochs", 6))
    seg_epochs = int(p.get("seg_epochs", 3))
    n_mask = sum(1 for m in train if m.get("has_mask"))
    do_seg = n_mask >= int(p.get("min_seg_samples", 30))
    job.total = len(val) * 2 + len(train) + epochs + (seg_epochs if do_seg else 0)
    job.done = 0
    base_scores = score_buildings(det, ds_dir, val)
    job.step(len(val))
    base_pred = predict(base_scores, det.cls_threshold, min_px)
    base_metrics = metrics(val, base_pred, px_area)

    # 5. drop weak negatives the base model is sure about
    drop_t = float(p.get("weak_negative_drop", 0.9))
    weak = [m for m in train if m["label_kind"] == "neg_weak"]
    if weak:
        ws = score_buildings(det, ds_dir, weak)
        dropped = {bid for bid, (probs, _) in ws.items() if len(probs) and probs.max() >= drop_t}
        train = [m for m in train if m["id"] not in dropped]
        job.log(f"{len(dropped)} von {len(weak)} schwachen Negativen verworfen (Ausgangsmodell ≥ {drop_t}: "
                "vermutlich nicht registrierte PV).")
    job.step(len(weak) if weak else 0)

    train_classifier(job, det, ds_dir, train, epochs, float(p.get("lr", 1e-4)),
                     float(p.get("weak_weight", 0.3)), int(p.get("max_patches", 12)))
    if do_seg:
        train_segmenter(job, det, ds_dir, train, seg_epochs, float(p.get("seg_lr", 5e-5)))
    else:
        job.log(f"Segmentierung bleibt unverändert ({n_mask} Gebäude mit PV-Polygonen; nötig "
                f"{p.get('min_seg_samples', 30)}).")

    job.log("Bewerte neues Modell …")
    det.cls_threshold = 0.0                                     # score everything, tune afterwards
    new_scores = score_buildings(det, ds_dir, val)
    job.step(len(val))
    thr, f1 = tune_threshold(val, new_scores, min_px)
    det.cls_threshold = thr
    new_pred = predict(new_scores, thr, min_px)
    new_metrics = metrics(val, new_pred, px_area)
    base_f1 = (metrics([m for m in val if m["label_kind"] != "neg_weak"] or val, base_pred, px_area)["alle"]["f1"]) or 0
    job.log(f"F1 (sichere Labels) vorher {base_f1:.3f} → nachher {f1:.3f}, Schwelle {thr}")

    name = runtime.safe_name(p.get("name") or f"{base_id}-{p.get('cc', 'xx')}-{time.strftime('%Y%m%d')}").lower()
    calib = {k: v["median"] for k, v in new_metrics["kwp_pro_m2"].items() if v["n"] >= 5}
    report = {"name": name, "base_profile": base_id, "created": time.strftime("%Y-%m-%d %H:%M"),
              "params": {k: v for k, v in p.items() if k != "pvmap_token"},
              "sample": dict(kinds), "n_train": len(train), "n_val": len(val), "n_masks_train": n_mask,
              "segmentation_trained": do_seg, "cls_threshold": thr, "kwp_per_m2_by_size": calib,
              "vorher": base_metrics, "nachher": new_metrics, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(job.dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    better = f1 >= base_f1
    if better or p.get("save_anyway"):
        path = models.save_custom(cfg["model_dir"], name, prof, det.cls, det.seg, thr, det.seg_threshold, report)
        job.log(f"Modell gespeichert als Profil 'custom:{name}' ({path}).")
        job.counters["modell"] = 1
    else:
        job.log("Das neue Modell ist auf der Validierung nicht besser – nicht gespeichert (report.json ansehen).")
    job.done = job.total
    job.counters["f1_vorher"] = round(base_f1, 3)
    job.counters["f1_nachher"] = round(f1, 3)
