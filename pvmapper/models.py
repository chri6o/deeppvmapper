"""
Model profiles and loading.

A profile bundles a classification network (InceptionV3, "is there PV in
this patch?") and a segmentation network (DeepLabV3-ResNet101, "which pixels
are PV?") with the input geometry they were trained on.

Profiles
  nrw    Mayer et al. (2022), 3D-PV-Locator — trained on 10 cm DOP from
         North Rhine-Westphalia. Closest match for German/Dutch/Austrian
         orthophotos. MIT licence. Default.
  ign    Kasmi et al. — BDAPPV fine-tune on French IGN 20 cm imagery. CC-BY 4.0.
  google Kasmi et al. — BDAPPV fine-tune on Google imagery. Imagery licence
         CC-BY-NC 4.0: non-commercial use only.

Weights are downloaded from Hugging Face (gabrielkasmi/bdappv-models) into
`model_dir` on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

HF_REPO = "gabrielkasmi/bdappv-models"


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    cls_file: str
    seg_file: str
    resolution_m: float        # ground metres per pixel the models expect
    patch_px: int              # patch edge length fed to segmentation
    cls_input_px: int          # classification input size (InceptionV3: 299)
    cls_outputs: int           # 2 = softmax[:,1], 1 = sigmoid
    norm_mean: tuple
    norm_std: tuple
    cls_threshold: float
    seg_threshold: float
    licence: str


IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

PROFILES = {
    "nrw": Profile("nrw", "NRW 10 cm (3D-PV-Locator) — empfohlen für DE/NL/AT",
                   "train/pretrained/inceptionv3_weights.tar",
                   "train/pretrained/deeplabv3_weights.tar",
                   0.1, 320, 299, 2, *HALF, 0.68, 0.5, "MIT"),
    "ign": Profile("ign", "IGN 20 cm (BDAPPV) — Frankreich",
                   "inception_ign_best.pth", "deeplab_ign_best.pth",
                   0.2, 400, 299, 1, *IMAGENET, 0.5, 0.5, "CC-BY 4.0"),
    "google": Profile("google", "Google (BDAPPV) — nur nicht-kommerziell",
                      "inception_google_best.pth", "deeplab_google_best.pth",
                      0.2, 400, 299, 1, *IMAGENET, 0.5, 0.5, "CC-BY-NC 4.0 (imagery)"),
}


def weight_path(model_dir: str, filename: str, download: bool = True) -> str:
    local = os.path.join(model_dir, os.path.basename(filename))
    if os.path.isfile(local):
        return local
    if not download:
        raise FileNotFoundError(local)
    from huggingface_hub import hf_hub_download
    os.makedirs(model_dir, exist_ok=True)
    tmp = hf_hub_download(HF_REPO, filename, local_dir=model_dir)
    if os.path.abspath(tmp) != os.path.abspath(local):
        os.replace(tmp, local)
    return local


def _state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return obj


def build_classifier(profile: Profile, path: str) -> nn.Module:
    from torchvision.models import inception_v3
    model = inception_v3(weights=None, aux_logits=True, init_weights=False)
    model.fc = nn.Linear(model.fc.in_features, profile.cls_outputs)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, profile.cls_outputs)
    state = _state_dict(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("AuxLogits")]
    if real_missing:
        raise RuntimeError(f"classifier weights incomplete: {real_missing[:5]}")
    model.aux_logits = False
    model.AuxLogits = None
    return model.eval()


def build_segmenter(profile: Profile, path: str) -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet101
    model = deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=False)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    state = {k: v for k, v in _state_dict(path).items() if not k.startswith("aux_classifier")}
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"segmenter weights incomplete: {missing[:5]}")
    return model.eval()


def pick_device(preference: str = "auto") -> str:
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():          # NVIDIA CUDA *and* AMD ROCm builds
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info() -> dict:
    info = {"torch": torch.__version__, "device": pick_device(), "gpu": None,
            "backend": "cpu", "vram_gb": None}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1),
                    backend="rocm" if getattr(torch.version, "hip", None) else "cuda")
    elif info["device"] == "mps":
        info.update(gpu="Apple MPS", backend="mps")
    return info


def load(profile_id: str, model_dir: str, device: str = "auto"):
    profile = PROFILES[profile_id]
    dev = pick_device(device)
    cls = build_classifier(profile, weight_path(model_dir, profile.cls_file)).to(dev)
    seg = build_segmenter(profile, weight_path(model_dir, profile.seg_file)).to(dev)
    return profile, cls, seg, dev
