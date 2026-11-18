"""
Roof-shape model (PVMAPPER.md "Dachformen"): a small image classifier for buildings without LoD2 data.

Classes: flat, mono (Pultdach), gabled (Satteldach), hipped (Walm-, Krüppelwalm-, Zeltdach), complex (Mischform, Mansarde,
Shed …). Training labels come from the official LoD2 models (lod2.py): in a region the job samples buildings per class,
fetches their aerial image and learns from crops of the footprint – no hand labelling. Each crop is turned so the long side of
the footprint is horizontal (the ridge of gable roofs runs along it), the surroundings are dimmed.

    python: job kind "roofshape" (Training tab → "Dachformen-Modell trainieren")
    model:  model_dir/roofshape/<name>.pth + <name>.json (classes, accuracy, confusion matrix)
    use:    setting `roof_model` = <name>; roofs.py asks predict() where no LoD2 face data exists
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

CLASSES = ["flat", "mono", "gabled", "hipped", "complex"]
LABEL_DE = {"flat": "Flachdach", "mono": "Pultdach", "gabled": "Satteldach", "hipped": "Walmdach", "complex": "komplex"}
SIZE = 160
_MODELS: dict = {}
_LOCK = threading.Lock()


def model_dir(cfg: dict) -> str:
    d = os.path.join(cfg.get("model_dir") or "models", "roofshape")
    os.makedirs(d, exist_ok=True)
    return d


def list_models(cfg: dict) -> list[dict]:
    d = model_dir(cfg)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    j = json.load(fh)
                out.append({"name": f[:-5], "accuracy": j.get("accuracy"), "n_train": j.get("n_train"), "created": j.get("created")})
            except Exception:                                        # noqa: BLE001
                continue
    return out


# ---------------------------------------------------------------------------------------------------------------------
# input

def crop(img: np.ndarray, fp: np.ndarray, res_m: float, margin_m: float = 2.0) -> np.ndarray | None:
    """footprint crop turned so its long side is horizontal, surroundings dimmed, SIZE × SIZE RGB (uint8)"""
    fp = fp.astype(np.uint8)
    cnts, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if h > w:
        ang += 90
        w, h = h, w
    m = margin_m / res_m
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    W, H = int(w + 2 * m), int(h + 2 * m)
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    im = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mk = cv2.warpAffine(fp * 255, M, (W, H), flags=cv2.INTER_NEAREST) > 0
    im = np.where(mk[..., None], im, (im * 0.3).astype(np.uint8))
    s = SIZE / max(W, H)
    im = cv2.resize(im, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((SIZE, SIZE, 3), np.uint8)
    y0, x0 = (SIZE - im.shape[0]) // 2, (SIZE - im.shape[1]) // 2
    out[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return out


def _net(pretrained: bool, log=None):
    import torch.nn as nn
    import torchvision
    m = None
    if pretrained:
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        except Exception as e:                                       # noqa: BLE001 – offline: train from scratch
            if log:
                log(f"Vortrainierte Gewichte nicht verfügbar ({type(e).__name__}) – trainiere ohne.")
    if m is None:
        m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def _tensor(batch: np.ndarray):
    import torch
    x = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def _augment(x: np.ndarray) -> np.ndarray:
    """flips (the long axis stays horizontal), brightness/contrast; 180° is a flip pair"""
    if random.random() < 0.5:
        x = x[:, ::-1]
    if random.random() < 0.5:
        x = x[::-1]
    a = random.uniform(0.75, 1.25)
    b = random.uniform(-20, 20)
    return np.clip(x.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# prediction

def _load(cfg: dict, name: str):
    import torch
    from . import models
    with _LOCK:
        if name in _MODELS:
            return _MODELS[name]
    d = model_dir(cfg)
    with open(os.path.join(d, name + ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    net = _net(False)
    net.load_state_dict(torch.load(os.path.join(d, name + ".pth"), map_location="cpu"))
    dev = models.pick_device(cfg.get("device", "auto"))
    net = net.to(dev).eval()
    with _LOCK:
        _MODELS[name] = (net, dev, meta)
    return _MODELS[name]


def predict_crops(cfg: dict, name: str, crops: list) -> list[tuple]:
    import torch
    net, dev, meta = _load(cfg, name)
    classes = meta.get("classes") or CLASSES
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = _tensor(np.stack(crops[i:i + 64])).to(dev)
            p = torch.softmax(net(x), 1) + torch.softmax(net(torch.flip(x, [3])), 1)      # mirror average
            p = (p / 2).cpu().numpy()
            for row in p:
                k = int(row.argmax())
                out.append((classes[k], float(row[k])))
    return out


def predict(img: np.ndarray, fp: np.ndarray, res_m: float, cfg: dict, min_p: float = 0.5) -> tuple | None:
    """(class, basis) for roofs.analyse, or None if the model is unsure / no model"""
    name = cfg.get("roof_model")
    if not name:
        return None
    c = crop(img, fp, res_m)
    if c is None:
        return None
    try:
        cls, p = predict_crops(cfg, name, [c])[0]
    except Exception:                                                # noqa: BLE001
        return None
    return (cls, f"Modell ({p:.0%})") if p >= min_p else None


# ---------------------------------------------------------------------------------------------------------------------
# training job

def _split(bid: str) -> bool:
    return int(hashlib.md5(bid.encode()).hexdigest()[:6], 16) % 5 == 0          # 20 % validation


def run(job, cfg: dict):
    """params: bbox [w,s,e,n] or geometry (GeoJSON), per_class (300), epochs (8), name, min_area_m2 (50)"""
    import torch
    import torch.nn as nn
    from shapely.geometry import box, shape
    from . import geo, lod2, models, runtime, sources
    p = job.params
    t0 = time.time()
    region = box(*p["bbox"]) if p.get("bbox") else shape(p["geometry"])
    per_class = int(p.get("per_class") or 300)
    min_area = float(p.get("min_area_m2") or 50)
    job.log("Lese LoD2-Gebäude im Gebiet (Geobasis NRW) …")
    blds = [b for b in lod2.buildings_in(region, cfg, job.log) if b["class"] and b["area_m2"] >= min_area]
    if not blds:
        raise RuntimeError("Keine LoD2-Gebäude im Gebiet – LoD2 gibt es derzeit für NRW.")
    rng = random.Random(int(p.get("seed") or 1))
    random.seed(int(p.get("seed") or 1))
    torch.manual_seed(int(p.get("seed") or 1))
    by = {}
    for b in blds:
        by.setdefault(b["class"], []).append(b)
    sample = []
    for cl in CLASSES:
        xs = by.get(cl, [])
        rng.shuffle(xs)
        sample += xs[:per_class]
    job.log("Stichprobe: " + ", ".join(f"{LABEL_DE[c]} {min(len(by.get(c, [])), per_class)} (von {len(by.get(c, []))})" for c in CLASSES))
    cat = runtime.catalog(cfg)
    src_filter = runtime.source_filter(cfg, p.get("sources"))
    res = float(p.get("resolution_m") or 0.1)
    ds = os.path.join(job.dir, "dataset")
    os.makedirs(ds, exist_ok=True)
    job.total, job.done = len(sample), 0

    def one(b):
        g = shape(b["footprint"])
        try:
            img, win, _ = sources.fetch_for_geometry(cat, g, res, 2.0, src_filter, min_px=64)
        except Exception:                                            # noqa: BLE001
            return None
        fp = runtime.clip_mask(g, win, img.shape, buffer_m=0) > 0
        c = crop(img, fp, res)
        if c is None:
            return None
        cv2.imwrite(os.path.join(ds, f"{b['id']}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return b["id"], CLASSES.index(b["class"]), b["roof_type"]

    items = []
    with ThreadPoolExecutor(int(cfg.get("fetch_workers") or 8)) as pool:
        for r in pool.map(one, sample):
            job.check_cancel()
            job.step(1, **({"bild": 1} if r else {"bild_fehlt": 1}))
            if r:
                items.append(r)
    with open(os.path.join(ds, "labels.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": i, "class": CLASSES[k], "roof_type": rt} for i, k, rt in items], f)
    train = [x for x in items if not _split(x[0])]
    val = [x for x in items if _split(x[0])]
    if len(train) < 20 or len(val) < 5:
        raise RuntimeError(f"Zu wenige Bilder ({len(train)} Training, {len(val)} Prüfung) – größeres Gebiet wählen.")
    imgs = {i: cv2.cvtColor(cv2.imread(os.path.join(ds, f"{i}.jpg")), cv2.COLOR_BGR2RGB) for i, _, _ in items}

    dev = models.pick_device(cfg.get("device", "auto"))
    net = _net(bool(p.get("pretrained", True)), job.log).to(dev)
    counts = np.bincount([k for _, k, _ in train], minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(CLASSES), 0.0), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(net.parameters(), lr=float(p.get("lr") or 3e-4), weight_decay=1e-4)
    epochs = int(p.get("epochs") or 8)
    bs = int(p.get("batch") or 32)
    job.total, job.done = epochs, 0
    best, best_state = -1.0, None

    def evaluate():
        net.eval()
        conf = np.zeros((len(CLASSES), len(CLASSES)), int)
        preds = {}
        with torch.no_grad():
            for i in range(0, len(val), 64):
                chunk = val[i:i + 64]
                x = _tensor(np.stack([imgs[c[0]] for c in chunk])).to(dev)
                pr = (torch.softmax(net(x), 1) + torch.softmax(net(torch.flip(x, [3])), 1)).cpu().numpy()
                for (bid, k, _), row in zip(chunk, pr):
                    q = int(row.argmax())
                    conf[k, q] += 1
                    preds[bid] = (q, float(row[q] / 2))
        return conf, preds

    for ep in range(epochs):
        net.train()
        rng.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), bs):
            job.check_cancel()
            chunk = train[i:i + bs]
            x = _tensor(np.stack([_augment(imgs[c[0]]) for c in chunk])).to(dev)
            y = torch.tensor([c[1] for c in chunk]).to(dev)
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(chunk)
        conf, _ = evaluate()
        acc = float(np.trace(conf)) / max(conf.sum(), 1)
        job.log(f"Epoche {ep + 1}/{epochs}: Verlust {tot / len(train):.3f}, Treffer Prüfmenge {acc:.1%}")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        job.step(1)
    net.load_state_dict(best_state)
    conf, preds = evaluate()
    acc = float(np.trace(conf)) / max(conf.sum(), 1)
    per = {CLASSES[k]: {"n": int(conf[k].sum()), "recall": round(float(conf[k, k]) / max(conf[k].sum(), 1), 3),
                        "precision": round(float(conf[k, k]) / max(conf[:, k].sum(), 1), 3)} for k in range(len(CLASSES))}
    flat_vs = conf[0, 0] + conf[1:, 1:].sum()
    wrong = [{"id": bid, "label": CLASSES[k], "pred": CLASSES[preds[bid][0]], "p": round(preds[bid][1], 2), "roof_type": rt}
             for bid, k, rt in val if preds[bid][0] != k]
    rng.shuffle(wrong)
    name = runtime.safe_name(p.get("name") or f"dach-{time.strftime('%Y%m%d-%H%M')}").lower()
    meta = {"name": name, "classes": CLASSES, "created": time.strftime("%Y-%m-%d %H:%M"), "accuracy": round(acc, 3),
            "flat_vs_pitched": round(float(flat_vs) / max(conf.sum(), 1), 3), "per_class": per,
            "confusion": conf.tolist(), "n_train": len(train), "n_val": len(val), "size": SIZE,
            "region": p.get("bbox") or "Polygon", "minutes": round((time.time() - t0) / 60, 1)}
    d = model_dir(cfg)
    torch.save(best_state, os.path.join(d, name + ".pth"))
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(os.path.join(job.dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "kind": "roofshape", "wrong": wrong[:60]}, f, ensure_ascii=False, indent=1)
    job.counters.update(treffer=round(acc, 3), modell=1)
    job.log(f"Dachformen-Modell '{name}' gespeichert: {acc:.1%} richtig, flach/geneigt {meta['flat_vs_pitched']:.1%}. "
            f"In den Einstellungen als Dachformen-Modell wählen.")
