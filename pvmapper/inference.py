"""
Patch-wise classification + segmentation on arbitrary RGB images.

`Detector.run(items)` takes a list of (key, rgb_array) at the profile's
resolution, cuts each image into patch_px × patch_px patches (edge patches
are shifted inwards so every pixel is covered), classifies all patches in
GPU batches, segments the positive ones and returns per image a full-size
binary PV mask plus the highest classification score.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import geo
from .models import Profile


def patch_grid(width: int, height: int, size: int, overlap: float = 0.0):
    """Top-left offsets of patches covering a width×height image.
    `overlap` (0–0.75) shifts patches by size*(1-overlap) so PV arrays cut by a
    patch border are also seen whole in a neighbouring patch."""
    stride = max(1, int(round(size * (1 - min(max(overlap, 0.0), 0.75)))))

    def axis(n):
        if n <= size:
            return [0]
        offs = list(range(0, n - size, stride))
        offs.append(n - size)
        return offs
    return [(x, y) for y in axis(height) for x in axis(width)]


@dataclass
class ImageResult:
    mask: np.ndarray        # (H, W) uint8 {0,1}
    max_prob: float         # highest patch classification score
    n_patches: int
    n_positive: int


class Detector:
    def __init__(self, profile: Profile, cls_model, seg_model, device: str,
                 cls_threshold: float | None = None, seg_threshold: float | None = None,
                 batch_size: int = 32, overlap: float = 0.25, tta: bool = True,
                 neighbor_factor: float = 0.5, precision: str = "auto"):
        self.p = profile
        self.cls = cls_model
        self.seg = seg_model
        self.device = device
        self.cls_threshold = profile.cls_threshold if cls_threshold is None else cls_threshold
        self.seg_threshold = profile.seg_threshold if seg_threshold is None else seg_threshold
        self.batch_size = batch_size
        self.overlap = overlap
        self.tta = tta                      # average segmentation over flips (less dependent on light direction)
        self.neighbor_factor = neighbor_factor  # in images with PV, also segment patches scoring ≥ factor×threshold
        # half precision on the GPU: about twice the throughput on RDNA3/Ampere, results practically unchanged
        self.fp16 = precision == "fp16" or (precision == "auto" and str(device).startswith("cuda"))
        self.stats = {"patches": 0, "segmented": 0, "gpu_s": 0.0}
        self.mean = torch.tensor(profile.norm_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(profile.norm_std, device=device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------

    def _to_tensor(self, patches: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(patches)).to(self.device)        # N,H,W,3 uint8
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        return x

    @staticmethod
    def _pad(x: torch.Tensor, n: int) -> torch.Tensor:
        """Pads a batch to n items. Constant batch shapes matter on AMD ROCm: MIOpen searches and compiles
        kernels per input shape, so every new batch size would cost seconds to minutes."""
        if x.shape[0] >= n:
            return x
        return torch.cat([x, x.new_zeros((n - x.shape[0],) + tuple(x.shape[1:]))])

    @staticmethod
    def _bucket(k: int, cap: int) -> int:
        b = 1
        while b < k:
            b *= 2
        return min(b, max(cap, k))

    def _amp(self):
        import contextlib
        if self.fp16 and not torch.is_grad_enabled():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        with self._amp():
            return self._classify_impl(x).float()

    def _classify_impl(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.p.cls_input_px:
            x = F.interpolate(x, size=(self.p.cls_input_px,) * 2, mode="bilinear",
                              align_corners=False, antialias=True)
        logits = self.cls((x - self.mean) / self.std)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.p.cls_outputs == 2:
            return F.softmax(logits, dim=1)[:, 1]
        return torch.sigmoid(logits[:, 0])

    @torch.no_grad()
    def _segment(self, x: torch.Tensor) -> torch.Tensor:
        with self._amp():
            return self._segment_impl(x)

    def _segment_impl(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.mean) / self.std
        prob = torch.sigmoid(self.seg(xn)["out"][:, 0].float())
        if self.tta:
            # flips change where shadows and glare fall relative to the network's receptive field
            prob = prob + torch.sigmoid(self.seg(xn.flip(-1))["out"][:, 0].float()).flip(-1)
            prob = prob + torch.sigmoid(self.seg(xn.flip(-2))["out"][:, 0].float()).flip(-2)
            prob = prob / 3
        return prob >= self.seg_threshold

    # ------------------------------------------------------------------

    def run(self, items: list[tuple[object, np.ndarray]]) -> dict:
        size = self.p.patch_px
        index = []           # (key, x, y)
        patches = []
        results = {}
        for key, img in items:
            h, w = img.shape[:2]
            if h < size or w < size:                      # pad tiny images
                pad = np.zeros((max(h, size), max(w, size), 3), np.uint8)
                pad[:h, :w] = img
                img = pad
            grid = patch_grid(img.shape[1], img.shape[0], size, self.overlap)
            results[key] = ImageResult(np.zeros((h, w), np.uint8), 0.0, len(grid), 0)
            for x, y in grid:
                index.append((key, x, y, h, w))
                patches.append(img[y:y + size, x:x + size])

        import time as _t
        t0 = _t.time()
        bs = self.batch_size
        probs = np.zeros(len(patches), np.float32)
        for b in range(0, len(patches), bs):                     # pass 1: classify all patches
            x = self._to_tensor(patches[b:b + bs])
            n = x.shape[0]
            probs[b:b + n] = self._classify(self._pad(x, bs)).float().cpu().numpy()[:n]
        for (key, *_), p in zip(index, probs):
            r = results[key]
            r.max_prob = max(r.max_prob, float(p))
        # pass 2: segment positive patches; in images with a positive patch also the borderline ones, so a
        # module field reaching into a weaker neighbouring patch is segmented completely
        low = self.cls_threshold * self.neighbor_factor if self.neighbor_factor else self.cls_threshold
        sel = [i for i, (key, *_) in enumerate(index)
               if probs[i] >= self.cls_threshold
               or (probs[i] >= low and results[key].max_prob >= self.cls_threshold)]
        for b in range(0, len(sel), bs):
            ids = sel[b:b + bs]
            x = self._to_tensor([patches[i] for i in ids])
            masks = self._segment(self._pad(x, self._bucket(len(ids), bs))).cpu().numpy()[:len(ids)]
            for i, m in zip(ids, masks):
                key, px, py, h, w = index[i]
                r = results[key]
                r.n_positive += int(probs[i] >= self.cls_threshold)
                hh, ww = min(size, h - py), min(size, w - px)
                if hh <= 0 or ww <= 0:
                    continue
                region = r.mask[py:py + hh, px:px + ww]
                np.maximum(region, m[:hh, :ww].astype(np.uint8), out=region)
        self.stats["patches"] += len(patches)
        self.stats["segmented"] += len(sel)
        self.stats["gpu_s"] += _t.time() - t0
        return results


# ---------------------------------------------------------------------------
# Mask → polygons
# ---------------------------------------------------------------------------

def mask_to_polygons(mask: np.ndarray, win: geo.Window, min_px: int = 20,
                     simplify_m: float = 0.15):
    """Binary mask in window pixel space → list of WGS84 shapely polygons."""
    from shapely.geometry import Polygon
    if mask.sum() == 0:
        return []
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    if hierarchy is None:
        return polys
    hierarchy = hierarchy[0]
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1 or cv2.contourArea(cnt) < min_px or len(cnt) < 3:
            continue                                             # holes / tiny blobs
        holes = [contours[j] for j in range(len(contours)) if hierarchy[j][3] == i
                 and len(contours[j]) >= 3]

        def to_crs(c):
            pts = c.reshape(-1, 2).astype(float)
            return [win.pixel_to_crs(px, py) for px, py in pts]

        poly = Polygon(to_crs(cnt), [to_crs(hh) for hh in holes]).buffer(0)
        if poly.is_empty:
            continue
        k = geo.scale_factor(win.crs, *geo.reproject(poly.centroid, win.crs, "EPSG:4326").coords[0])
        poly = poly.simplify(simplify_m * k)
        polys.append(geo.reproject(poly, win.crs, "EPSG:4326"))
    return polys
