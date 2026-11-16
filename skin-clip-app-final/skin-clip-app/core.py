"""Model code shared by the Streamlit app and retrain.py.

The fine-tuned model = OpenAI CLIP image encoder (last N transformer blocks tuned)
+ a cosine-similarity classification head + a calibration temperature.
Only the tuned tensors are stored in the checkpoint (small, fp16); the frozen part
is re-loaded from the original CLIP weights, so the file stays well under 100 MB.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import concepts as cx

DEFAULT_CLASSES = ["melanocytic nevus", "melanoma", "benign keratosis", "basal cell carcinoma",
                   "actinic keratosis", "vascular lesion", "dermatofibroma"]
DEFAULT_CODES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
RISK_TIER = {"mel": "Malignant", "bcc": "Malignant", "akiec": "Pre-malignant",
             "nv": "Benign", "bkl": "Benign", "vasc": "Benign", "df": "Benign"}
ZERO_SHOT_PROMPT = "a clinical dermatological photo of {}"
MODEL_DIR = Path(__file__).parent / "model"


# --------------------------------------------------------------------------- network
class SkinCLIPNet(nn.Module):
    """CLIP visual tower + cosine classifier head (weights initialised from text embeddings)."""

    def __init__(self, visual: nn.Module, head_weight: torch.Tensor, logit_scale: float = math.log(100.0)):
        super().__init__()
        self.visual = visual
        self.head = nn.Parameter(head_weight.detach().clone().float())
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual(x).float(), dim=-1)

    def logits_from_features(self, f: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.head, dim=-1)
        return self.logit_scale.clamp(max=math.log(100.0)).exp() * f @ w.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_features(self.features(x))

    # -- which parameters are fine-tuned ------------------------------------------------
    def set_trainable(self, n_blocks: int) -> list[str]:
        """Freeze the visual tower except the last `n_blocks` transformer blocks, ln_post and proj.
        n_blocks=0 -> backbone fully frozen (head only). Returns names of trainable visual params."""
        for p in self.visual.parameters():
            p.requires_grad = False
        if n_blocks > 0:
            for blk in self.visual.transformer.resblocks[-n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            self.visual.ln_post.requires_grad_(True)
            self.visual.proj.requires_grad_(True)
        return [n for n, p in self.visual.named_parameters() if p.requires_grad]

    def tuned_visual_state(self, n_blocks: int) -> dict[str, torch.Tensor]:
        names = set(self.set_trainable(n_blocks)) if n_blocks > 0 else set()
        sd = self.visual.state_dict()
        return {k: v.detach().cpu().half() for k, v in sd.items() if k in names}


def load_base_clip(backbone: str):
    """Original OpenAI CLIP in fp32 on CPU (clip.load keeps fp16 weights on GPU, which breaks fine-tuning)."""
    import clip
    model, preprocess = clip.load(backbone, device="cpu", jit=False)
    return model.float().eval(), preprocess


@torch.no_grad()
def zero_shot_head(clip_model, classes: list[str], prompt: str = ZERO_SHOT_PROMPT) -> torch.Tensor:
    import clip
    t = clip_model.encode_text(clip.tokenize([prompt.format(c) for c in classes])).float()
    return F.normalize(t, dim=-1)


def save_checkpoint(net: SkinCLIPNet, path, *, n_blocks: int, backbone: str, classes, codes,
                    temperature: float, version: str, extra: dict | None = None) -> None:
    ckpt = {
        "visual_tuned": net.tuned_visual_state(n_blocks),
        "head_weight": net.head.detach().cpu().float(),
        "logit_scale": float(net.logit_scale.detach().cpu()),
        "temperature": float(temperature),
        "backbone": backbone,
        "classes": list(classes),
        "codes": list(codes),
        "n_blocks": int(n_blocks),
        "version": version,
    }
    if extra:
        ckpt["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def build_net_from_checkpoint(ckpt_path) -> tuple[SkinCLIPNet, dict, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base, preprocess = load_base_clip(ckpt["backbone"])
    missing, unexpected = base.visual.load_state_dict(
        {k: v.float() for k, v in ckpt["visual_tuned"].items()}, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    net = SkinCLIPNet(base.visual, ckpt["head_weight"], ckpt["logit_scale"]).eval()
    return net, ckpt, preprocess


# --------------------------------------------------------------------------- input gate
GATE_POSITIVE = ["a dermoscopy image of a skin lesion", "a close-up clinical photo of a skin lesion",
                 "a dermatoscope photo of a mole on human skin", "a close-up photo of human skin with a lesion"]
GATE_NEGATIVE = ["a photo of a dog", "a photo of a cat", "a photo of an animal", "a photo of a person's face",
                 "a photo of a person", "a landscape photo", "a photo of food", "a photo of a car",
                 "a photo of a building", "a photo of an everyday object", "a screenshot of a computer screen",
                 "a scanned text document", "a cartoon or illustration", "a photo of a plant or flower"]


class BaseCLIP:
    """The ORIGINAL (not fine-tuned) CLIP image tower, kept in bfloat16 to limit memory (~180 MB), plus prompt banks.
    Used for (a) the skin-image gate and (b) scoring the dermoscopic symptoms (concepts).

    The gate is needed because the 7-class classifier cannot reject images by itself: softmax always distributes
    100% over the known diseases, even for a photo of a dog."""

    def __init__(self, backbone: str = "ViT-B/32", device: str = "cpu"):
        import clip
        base, self.preprocess = load_base_clip(backbone)
        banks = {"gate": GATE_POSITIVE + GATE_NEGATIVE, "concepts": cx.PROMPTS}
        with torch.no_grad():
            self.text = {k: F.normalize(base.encode_text(clip.tokenize(v)).float(), dim=-1).to(device)
                         for k, v in banks.items()}
        self.n_pos = len(GATE_POSITIVE)
        self.visual = self._to_bf16(base.visual).eval()       # text tower is freed with `base`
        self.device = device
        self.visual.to(device)

    @staticmethod
    def _to_bf16(visual: nn.Module) -> nn.Module:
        """Same selective conversion as clip.model.convert_weights (LayerNorm must stay fp32)."""
        def conv(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                m.weight.data = m.weight.data.to(torch.bfloat16)
                if m.bias is not None:
                    m.bias.data = m.bias.data.to(torch.bfloat16)
            if isinstance(m, nn.MultiheadAttention):
                for n in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight", "in_proj_bias",
                          "bias_k", "bias_v"):
                    t = getattr(m, n, None)
                    if t is not None:
                        t.data = t.data.to(torch.bfloat16)
        visual.apply(conv)
        for n in ("proj", "class_embedding", "positional_embedding"):
            t = getattr(visual, n, None)
            if t is not None:
                t.data = t.data.to(torch.bfloat16)
        return visual

    @torch.inference_mode()
    def image_features(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device).to(torch.bfloat16)
        return F.normalize(self.visual(x).float(), dim=-1)

    @torch.inference_mode()
    def skin_score(self, img: Image.Image) -> float:
        p = torch.softmax(100.0 * self.image_features(img) @ self.text["gate"].T, dim=-1)[0]
        return float(p[: self.n_pos].sum())

    @torch.inference_mode()
    def concept_scores(self, img: Image.Image) -> np.ndarray:
        """Raw cosine similarity of the image to every symptom prompt, shape (n_concepts,)."""
        return (self.image_features(img) @ self.text["concepts"].T)[0].cpu().numpy().astype(np.float64)


SkinGate = BaseCLIP      # backwards-compatible name


# --------------------------------------------------------------------------- inference
@dataclass
class Prediction:
    probs: dict          # code -> probability (calibrated)
    top_code: str
    top_name: str
    confidence: float
    margin: float        # top1 - top2 probability
    risk_tier: str
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class SkinCLIP:
    """Inference wrapper. Falls back to the zero-shot baseline if no checkpoint is available."""

    def __init__(self, ckpt_path: str | Path | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(ckpt_path) if ckpt_path else MODEL_DIR / "clip_skin_ft.pt"
        meta_path = ckpt_path.parent / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if ckpt_path.exists():
            self.net, ckpt, self.preprocess = build_net_from_checkpoint(ckpt_path)
            self.classes, self.codes = ckpt["classes"], ckpt["codes"]
            self.temperature = float(ckpt.get("temperature", 1.0))
            self.version = ckpt.get("version", "ft")
            self.is_finetuned = True
        else:  # baseline fallback so the UI is still demo-able before training has been run
            base, self.preprocess = load_base_clip(self.metadata.get("backbone", "ViT-B/32"))
            self.classes, self.codes = DEFAULT_CLASSES, DEFAULT_CODES
            self.net = SkinCLIPNet(base.visual, zero_shot_head(base, self.classes)).eval()
            self.temperature, self.version, self.is_finetuned = 1.0, "baseline-zero-shot", False
        self.net.to(self.device)
        self.gate_enabled = os.environ.get("SKIN_GATE", "1") != "0"
        self.concepts_enabled = os.environ.get("CONCEPTS", "1") != "0"
        self.base = None            # original CLIP, shared by the skin gate and the symptom extractor
        if self.gate_enabled or self.concepts_enabled:
            self.base = BaseCLIP(self.metadata.get("backbone", "ViT-B/32"), self.device)
        self.concept_stats = cx.load_stats(ckpt_path.parent / "concept_stats.json") if self.concepts_enabled else None
        self.replay_dir = ckpt_path.parent / "replay"

    def skin_score(self, img: Image.Image) -> float | None:
        """Probability (0-1) that the image is a skin lesion photo; None if the gate is disabled."""
        return self.base.skin_score(img) if (self.base is not None and self.gate_enabled) else None

    # ------------------------------------------------------------------ symptoms (concepts)
    def ensure_concept_stats(self, max_images: int = 400) -> bool:
        """Reference mean/std of every concept score, computed once from model/replay/ and cached in
        model/concept_stats.json. Returns False if no reference images exist (z-scores then start at 0)."""
        if not self.concepts_enabled or self.concept_stats is not None:
            return self.concept_stats is not None
        import pandas as pd
        man = self.replay_dir / "manifest.csv"
        if not man.exists():
            return False
        files = pd.read_csv(man)["file"].tolist()[:max_images]
        raw = []
        for f in files:
            p = self.replay_dir / f
            if p.exists():
                raw.append(self.base.concept_scores(Image.open(p).convert("RGB")))
        if len(raw) < 10:
            return False
        self.concept_stats = cx.make_stats(np.stack(raw))
        try:
            cx.save_stats(self.concept_stats, self.replay_dir.parent / "concept_stats.json")
        except OSError:
            pass                     # read-only model dir: keep the stats in memory only
        return True

    def extract_concepts(self, img: Image.Image) -> tuple[np.ndarray, bool] | None:
        """(z-scores aligned with concepts.CONCEPT_IDS, calibrated?) or None if the symptom editor is disabled.
        Not calibrated -> no reference statistics, every slider starts at 0 (physician edits still work)."""
        if not self.concepts_enabled or self.base is None:
            return None
        if self.concept_stats is None:
            return np.zeros(len(cx.CONCEPT_IDS)), False
        return cx.to_z(self.base.concept_scores(img.convert("RGB")), self.concept_stats), True

    @torch.inference_mode()
    def _logits(self, img: Image.Image, tta: bool) -> torch.Tensor:
        views = [img]
        if tta:
            views += [img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM)]
        x = torch.stack([self.preprocess(v) for v in views]).to(self.device)
        return self.net(x).mean(dim=0)

    def predict(self, img: Image.Image, tta: bool = True) -> Prediction:
        img = img.convert("RGB")
        p = torch.softmax(self._logits(img, tta) / self.temperature, dim=-1).cpu().numpy()
        order = np.argsort(-p)
        k = int(order[0])
        return Prediction(
            probs={c: float(p[i]) for i, c in enumerate(self.codes)},
            top_code=self.codes[k], top_name=self.classes[k], confidence=float(p[k]),
            margin=float(p[k] - p[int(order[1])]), risk_tier=RISK_TIER.get(self.codes[k], "n/a"),
            model_version=self.version)
