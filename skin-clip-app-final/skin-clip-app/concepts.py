"""Symptom (dermoscopic concept) layer for ClipSkinCancer.  Pure numpy: no torch / CLIP needed to import.

Flow
----
1. core.py scores every concept on the image with the ORIGINAL CLIP (zero-shot text prompt) -> raw similarity.
2. `to_z` standardises the raw scores with reference statistics (mean/std over the replay images) -> z-score.
   0 = dataset average, positive = more pronounced, negative = absent.
3. The physician edits the z-scores with sliders.
4. `rediagnose` shifts the model's log-probabilities by  GAIN * W @ (z_edited - z_extracted).
   Untouched sliders contribute nothing, so with no edits the result equals the model's own prediction.
5. `recommend_check` picks the concepts that best separate the model's rank-1 and rank-2 disease
   (shown with an orange frame in the UI).

W (`PRIOR_WEIGHTS`) is a clinical-knowledge prior (ABCD / pattern analysis), NOT learned from data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CODES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
Z_MIN, Z_MAX, Z_STEP = -3.0, 3.0, 0.1
GAIN = 0.8              # logit shift per 1.0 z-unit at weight 1.0
RECOMMEND_K = 4         # how many sliders get the orange frame
RECOMMEND_MIN_DIFF = 0.3

# group id -> concepts (id, label shown in the UI, CLIP prompt)
GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("structure", [
        ("asymmetric_shape", "asymmetric shape", "a dermoscopy image of a skin lesion with an asymmetric shape"),
        ("symmetric_round", "symmetric round", "a dermoscopy image of a symmetric round skin lesion"),
        ("irregular_border", "irregular border", "a dermoscopy image of a skin lesion with an irregular border"),
        ("sharp_smooth_border", "sharp smooth border", "a dermoscopy image of a skin lesion with a sharp smooth border"),
        ("multiple_colors", "multiple colors", "a dermoscopy image of a skin lesion with multiple colors"),
        ("uniform_color", "uniform color", "a dermoscopy image of a skin lesion with a uniform single color"),
        ("atypical_network", "atypical network", "a dermoscopy image with an atypical pigment network"),
        ("regular_network", "regular network", "a dermoscopy image with a regular honeycomb pigment network"),
        ("blue_white_veil", "blue-white veil", "a dermoscopy image with a blue-white veil"),
        ("dots_globules", "dots & globules", "a dermoscopy image with dots and globules"),
        ("streaks_pseudopods", "streaks/pseudopods", "a dermoscopy image with radial streaks and pseudopods"),
        ("regression_scar", "regression/scar-like", "a dermoscopy image with regression structures and scar-like areas"),
    ]),
    ("color", [
        ("dark_brown_black", "dark brown/black", "a skin lesion with dark brown and black color"),
        ("light_brown_tan", "light brown/tan", "a skin lesion with light brown and tan color"),
        ("pink_red", "pink/red color", "a skin lesion with pink and red color"),
        ("blue_gray", "blue-gray color", "a skin lesion with blue-gray color"),
    ]),
    ("surface", [
        ("scaly_surface", "scaly surface", "a skin lesion with a scaly surface"),
        ("redness_erythema", "redness/erythema", "a skin lesion with redness and erythema"),
        ("arborizing_vessels", "arborizing vessels", "a dermoscopy image with arborizing branching blood vessels"),
        ("red_purple_lagoons", "red/purple lagoons", "a dermoscopy image with red and purple vascular lagoons"),
        ("milia_like_cysts", "milia-like cysts", "a dermoscopy image with milia-like cysts"),
        ("comedo_like_openings", "comedo-like openings", "a dermoscopy image with comedo-like openings"),
        ("ulcer_crust", "ulcer/crust", "a skin lesion with an ulcer and crust"),
        ("central_white_patch", "central white patch", "a dermoscopy image with a central white patch"),
        ("waxy_stuck_on", "waxy stuck-on look", "a skin lesion with a waxy stuck-on appearance"),
    ]),
]
CONCEPT_IDS = [cid for _, cs in GROUPS for cid, _, _ in cs]
LABELS = {cid: lab for _, cs in GROUPS for cid, lab, _ in cs}
PROMPTS = [p for _, cs in GROUPS for _, _, p in cs]

# concept -> {disease code: weight in [-1.4, 1.4]}
PRIOR_WEIGHTS: dict[str, dict[str, float]] = {
    "asymmetric_shape":     {"mel": 1.0, "bcc": 0.2, "akiec": 0.2, "bkl": 0.1, "nv": -0.6, "df": -0.3},
    "symmetric_round":      {"nv": 0.8, "df": 0.5, "vasc": 0.3, "bkl": 0.1, "mel": -0.9},
    "irregular_border":     {"mel": 1.0, "bcc": 0.3, "akiec": 0.3, "nv": -0.6},
    "sharp_smooth_border":  {"nv": 0.5, "bkl": 0.6, "df": 0.3, "vasc": 0.3, "mel": -0.6},
    "multiple_colors":      {"mel": 1.0, "bcc": 0.2, "bkl": 0.1, "nv": -0.3},
    "uniform_color":        {"nv": 0.7, "df": 0.3, "vasc": 0.2, "mel": -0.8},
    "atypical_network":     {"mel": 1.0, "nv": -0.4},
    "regular_network":      {"nv": 1.0, "df": 0.4, "mel": -0.8},
    "blue_white_veil":      {"mel": 1.2, "nv": -0.3},
    "dots_globules":        {"nv": 0.6, "mel": 0.4},
    "streaks_pseudopods":   {"mel": 1.0, "nv": -0.2},
    "regression_scar":      {"mel": 0.8, "bkl": 0.4, "df": 0.4, "nv": -0.4},
    "dark_brown_black":     {"mel": 0.7, "nv": 0.3, "bkl": 0.2, "bcc": 0.2, "vasc": -0.3},
    "light_brown_tan":      {"bkl": 0.7, "df": 0.5, "nv": 0.4, "akiec": 0.3, "mel": -0.1},
    "pink_red":             {"bcc": 0.7, "vasc": 0.6, "akiec": 0.5, "df": 0.4, "mel": 0.1, "nv": -0.4, "bkl": -0.2},
    "blue_gray":            {"mel": 0.8, "bcc": 0.5, "nv": 0.1},
    "scaly_surface":        {"akiec": 1.0, "bkl": 0.4, "bcc": 0.2, "mel": -0.2, "nv": -0.3},
    "redness_erythema":     {"akiec": 0.6, "bcc": 0.4, "vasc": 0.4, "nv": -0.3},
    "arborizing_vessels":   {"bcc": 1.3, "mel": -0.3, "nv": -0.3},
    "red_purple_lagoons":   {"vasc": 1.3, "mel": 0.2, "nv": -0.4},
    "milia_like_cysts":     {"bkl": 1.2, "nv": 0.1, "mel": -0.3},
    "comedo_like_openings": {"bkl": 1.0, "nv": 0.2, "mel": -0.3},
    "ulcer_crust":          {"bcc": 0.8, "akiec": 0.7, "mel": 0.4, "bkl": 0.2, "nv": -0.4},
    "central_white_patch":  {"df": 1.3, "mel": 0.2, "nv": -0.2},
    "waxy_stuck_on":        {"bkl": 1.2, "nv": -0.2, "mel": -0.3},
}
assert set(PRIOR_WEIGHTS) == set(CONCEPT_IDS)


def weight_matrix(codes: list[str] | None = None) -> np.ndarray:
    """(n_concepts, n_classes) matrix aligned with CONCEPT_IDS x codes."""
    codes = codes or CODES
    return np.array([[PRIOR_WEIGHTS[c].get(k, 0.0) for k in codes] for c in CONCEPT_IDS], dtype=np.float64)


# ----------------------------------------------------------------------------- statistics
def make_stats(raw: np.ndarray) -> dict:
    """raw: (n_images, n_concepts) similarity scores of reference images -> {'mean': [...], 'std': [...]}"""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(CONCEPT_IDS) or raw.shape[0] < 2:
        raise ValueError(f"need (n>=2, {len(CONCEPT_IDS)}) raw scores, got {raw.shape}")
    return {"concepts": CONCEPT_IDS, "n": int(raw.shape[0]), "mean": raw.mean(0).tolist(),
            "std": np.maximum(raw.std(0), 1e-6).tolist()}


def save_stats(stats: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats))


def load_stats(path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s if s.get("concepts") == CONCEPT_IDS else None      # stale file (concept list changed) -> rebuild


def to_z(raw: np.ndarray, stats: dict) -> np.ndarray:
    z = (np.asarray(raw, dtype=np.float64) - np.array(stats["mean"])) / np.array(stats["std"])
    return np.clip(z, Z_MIN, Z_MAX)


# ----------------------------------------------------------------------------- re-diagnosis
def rediagnose(base_probs: dict, z_extracted: np.ndarray, z_edited: np.ndarray, codes: list[str] | None = None,
               gain: float = GAIN) -> dict:
    """New probabilities after the physician edited the symptom sliders (dict code -> prob)."""
    codes = codes or list(base_probs)
    p = np.array([max(base_probs[c], 1e-9) for c in codes], dtype=np.float64)
    delta = np.asarray(z_edited, dtype=np.float64) - np.asarray(z_extracted, dtype=np.float64)
    logit = np.log(p) + gain * (delta @ weight_matrix(codes))
    logit -= logit.max()
    q = np.exp(logit)
    q /= q.sum()
    return {c: float(v) for c, v in zip(codes, q)}


def recommend_check(base_probs: dict, z_extracted: np.ndarray, codes: list[str] | None = None,
                    k: int = RECOMMEND_K) -> list[str]:
    """Concept ids that best separate the rank-1 and rank-2 disease (largest |w1 - w2|), preferring concepts
    whose extracted value is near 0 (the extractor is least sure there)."""
    codes = codes or list(base_probs)
    order = sorted(codes, key=lambda c: -base_probs[c])
    if len(order) < 2:
        return []
    i1, i2 = codes.index(order[0]), codes.index(order[1])
    W = weight_matrix(codes)
    diff = np.abs(W[:, i1] - W[:, i2])
    z = np.abs(np.asarray(z_extracted, dtype=np.float64))
    score = diff * (1.0 - 0.5 * np.minimum(z, 2.0) / 2.0)
    idx = [i for i in np.argsort(-score) if diff[i] >= RECOMMEND_MIN_DIFF][:k]
    return [CONCEPT_IDS[i] for i in idx]


def top_code(probs: dict) -> str:
    return max(probs, key=probs.get)
