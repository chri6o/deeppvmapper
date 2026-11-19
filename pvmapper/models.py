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

import dataclasses
import json
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
    channels: int = 3          # 3 = RGB; 4 = RGB + near infrared (test models, 20.11.2026 – not yet usable in aerial-image runs)


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
    if filename.startswith("custom/") or os.path.isabs(filename):
        path = filename if os.path.isabs(filename) else os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
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


FIRST_CONV = {"cls": ("Conv2d_1a_3x3", "conv"), "seg": ("backbone", "conv1")}


def _first_conv(model, kind):
    a, b = FIRST_CONV[kind]
    return getattr(getattr(model, a), b)


def widen_first_conv(model, kind: str, channels: int = 4):
    """a 4th input channel (near infrared) with zero weights: the widened network gives exactly the results of the RGB network
    until training teaches it to use the infrared"""
    a, b = FIRST_CONV[kind]
    old = _first_conv(model, kind)
    if old.in_channels == channels:
        return model
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(getattr(model, a), b, new.to(old.weight.device))
    return model


def _widen_state(state: dict, key: str, channels: int):
    w = state.get(key)
    if w is not None and w.shape[1] < channels:
        z = torch.zeros((w.shape[0], channels - w.shape[1]) + tuple(w.shape[2:]), dtype=w.dtype)
        state[key] = torch.cat([w, z], 1)
    return state


def build_classifier(profile: Profile, path: str) -> nn.Module:
    from torchvision.models import inception_v3
    model = inception_v3(weights=None, aux_logits=True, init_weights=False)
    model.fc = nn.Linear(model.fc.in_features, profile.cls_outputs)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, profile.cls_outputs)
    state = _state_dict(path)
    if profile.channels != 3:
        widen_first_conv(model, "cls", profile.channels)
        _widen_state(state, "Conv2d_1a_3x3.conv.weight", profile.channels)
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
    if profile.channels != 3:
        widen_first_conv(model, "seg", profile.channels)
        _widen_state(state, "backbone.conv1.weight", profile.channels)
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


# ---------------------------------------------------------------------------
# custom profiles (fine-tuned with pvmapper.training): model_dir/custom/<name>.profile.json
# ---------------------------------------------------------------------------

def custom_profiles(model_dir: str) -> dict:
    out = {}
    d = os.path.join(model_dir, "custom")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".profile.json"):
            continue
        try:
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                j = json.load(fh)
            pr = j["profile"]
            pr["norm_mean"], pr["norm_std"] = tuple(pr["norm_mean"]), tuple(pr["norm_std"])
            out[pr["id"]] = Profile(**pr)
        except Exception:                                        # noqa: BLE001
            continue
    return out


def all_profiles(model_dir: str) -> dict:
    return {**PROFILES, **custom_profiles(model_dir)}


def get_profile(profile_id: str, model_dir: str) -> Profile:
    P = all_profiles(model_dir)
    if profile_id not in P:
        raise KeyError(f"unbekanntes Modellprofil '{profile_id}'")
    return P[profile_id]


def save_custom(model_dir: str, name: str, base: Profile, cls, seg, cls_threshold: float,
                seg_threshold: float, report: dict) -> str:
    d = os.path.join(model_dir, "custom")
    os.makedirs(d, exist_ok=True)
    cls_f, seg_f = f"custom/{name}_cls.pth", f"custom/{name}_seg.pth"
    torch.save({k: v.cpu() for k, v in cls.state_dict().items()}, os.path.join(model_dir, cls_f))
    torch.save({k: v.cpu() for k, v in seg.state_dict().items()}, os.path.join(model_dir, seg_f))
    pr = dataclasses.asdict(base)
    pr.update(id=f"custom:{name}", label=f"Eigenes Modell: {name} (Basis {base.id})", cls_file=cls_f, seg_file=seg_f,
              cls_threshold=float(cls_threshold), seg_threshold=float(seg_threshold),
              licence=f"{base.licence} (fine-tuned)")
    path = os.path.join(d, f"{name}.profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"profile": pr, "report": report}, f, ensure_ascii=False, indent=1)
    return path


def load(profile_id: str, model_dir: str, device: str = "auto"):
    profile = get_profile(profile_id, model_dir)
    dev = pick_device(device)
    cls = build_classifier(profile, weight_path(model_dir, profile.cls_file)).to(dev)
    seg = build_segmenter(profile, weight_path(model_dir, profile.seg_file)).to(dev)
    return profile, cls, seg, dev
