"""python -m tests.test_concepts   (from the project root). Pure numpy: needs neither torch nor CLIP weights."""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import concepts as cx

codes = cx.CODES
n = len(cx.CONCEPT_IDS)
idx = {c: i for i, c in enumerate(cx.CONCEPT_IDS)}
base = {"nv": 0.45, "mel": 0.30, "bkl": 0.10, "bcc": 0.07, "akiec": 0.04, "vasc": 0.02, "df": 0.02}

# layout matches the UI spec: 12 + 4 + 9 concepts, unique ids, a prior weight row for each
assert [len(c) for _, c in cx.GROUPS] == [12, 4, 9] and len(set(cx.CONCEPT_IDS)) == n == 25
assert len(cx.PROMPTS) == n and cx.weight_matrix(codes).shape == (n, 7)

# 1) no edit -> identical to the model's probabilities
z0 = np.zeros(n)
same = cx.rediagnose(base, z0, z0.copy(), codes)
assert all(abs(same[c] - base[c]) < 1e-9 for c in codes)

# 2) probabilities always sum to 1
rng = np.random.default_rng(0)
for _ in range(50):
    q = cx.rediagnose(base, z0, rng.uniform(-3, 3, n), codes)
    assert abs(sum(q.values()) - 1) < 1e-9 and all(v >= 0 for v in q.values())

# 3) clinically sensible directions
def edit(**kw):
    z = z0.copy()
    for k, v in kw.items():
        z[idx[k]] = v
    return cx.rediagnose(base, z0, z, codes)
assert edit(blue_white_veil=3, irregular_border=2)["mel"] > base["mel"] + 0.2          # melanoma features -> mel up
assert edit(blue_white_veil=3, irregular_border=2)["nv"] < base["nv"]
assert cx.top_code(edit(arborizing_vessels=3, pink_red=2)) == "bcc"                     # BCC hallmark flips the top
assert edit(scaly_surface=3)["akiec"] > base["akiec"]
assert edit(milia_like_cysts=3, waxy_stuck_on=3)["bkl"] > base["bkl"]
assert edit(regular_network=-3)["nv"] < base["nv"]                                      # removing a feature works

# 3b) the shift is relative to the EXTRACTED value: slider at its extracted value = no change, whatever the value
z_ext = rng.uniform(-2, 2, n)
back = cx.rediagnose(base, z_ext, z_ext.copy(), codes)
assert all(abs(back[c] - base[c]) < 1e-9 for c in codes)

# 4) recommended sliders separate rank 1 (nv) and rank 2 (mel)
rec = cx.recommend_check(base, z0, codes)
W = cx.weight_matrix(codes)
assert 1 <= len(rec) <= cx.RECOMMEND_K and len(set(rec)) == len(rec)
for cid in rec:
    assert abs(W[idx[cid], codes.index("nv")] - W[idx[cid], codes.index("mel")]) >= cx.RECOMMEND_MIN_DIFF
assert "regular_network" in rec or "atypical_network" in rec or "asymmetric_shape" in rec
# a slider that is extreme already (CLIP is sure) ranks below an equally discriminative one near 0
z_hi = z0.copy(); z_hi[idx["asymmetric_shape"]] = 3
r1 = cx.recommend_check(base, z0, codes, k=25); r2 = cx.recommend_check(base, z_hi, codes, k=25)
assert r2.index("asymmetric_shape") > r1.index("asymmetric_shape")

# 5) stats -> z-scores
raw = rng.normal(0.25, 0.02, (60, n)) + np.linspace(0, 0.05, n)
stats = cx.make_stats(raw)
z = cx.to_z(raw, stats)
assert abs(z.mean()) < 1e-2 and abs(z.std(0).mean() - 1) < 0.02   # tiny drift: |z| > 3 is clipped
assert cx.to_z(raw[:1] + 10, stats).max() == cx.Z_MAX                                   # clipped to the slider range
with tempfile.TemporaryDirectory() as d:
    cx.save_stats(stats, Path(d) / "model" / "concept_stats.json")
    assert cx.load_stats(Path(d) / "model" / "concept_stats.json")["n"] == 60
    assert cx.load_stats(Path(d) / "missing.json") is None
    bad = dict(stats, concepts=["x"]); cx.save_stats(bad, Path(d) / "old.json")
    assert cx.load_stats(Path(d) / "old.json") is None                                  # stale file is ignored
print("test_concepts: ALL PASSED")
