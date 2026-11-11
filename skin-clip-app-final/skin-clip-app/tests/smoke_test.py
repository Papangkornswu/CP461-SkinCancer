"""python -m tests.smoke_test   (from the project root). Uses random CLIP weights: checks plumbing, not accuracy."""
import json, os, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image

from tests import fake_clip
fake_load = fake_clip.install()
# make the fake model deterministic so base weights match between "training" and "serving"
_orig = fake_load
_cache = {}
import clip
def det_load(name, device="cpu", jit=False, **kw):
    if "m" not in _cache:
        _cache["m"] = _orig(name, "cpu")
    import copy
    m, tf = copy.deepcopy(_cache["m"][0]), _cache["m"][1]
    return m.to(device), tf
clip.load = det_load

import core
from feedback_store import FeedbackStore

tmp = Path(tempfile.mkdtemp())
model_dir = tmp / "model"; model_dir.mkdir()
codes, classes = core.DEFAULT_CODES, core.DEFAULT_CLASSES

# 1) build + save + reload checkpoint
base, _ = core.load_base_clip("ViT-B/32")
net = core.SkinCLIPNet(base.visual, core.zero_shot_head(base, classes))
net.set_trainable(2)
with torch.no_grad():
    for p in net.visual.transformer.resblocks[-1].parameters():
        p.add_(0.01 * torch.randn_like(p))          # pretend we fine-tuned
core.save_checkpoint(net, model_dir / "clip_skin_ft.pt", n_blocks=2, backbone="ViT-B/32", classes=classes,
                     codes=codes, temperature=1.3, version="test-v1")
size_mb = (model_dir / "clip_skin_ft.pt").stat().st_size / 1e6
print(f"checkpoint {size_mb:.1f} MB")
net.eval()
img = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))
x = clip_pre = det_load("x")[1](img).unsqueeze(0)
ref = torch.softmax(net(x)[0] / 1.3, -1)
sk = core.SkinCLIP(model_dir / "clip_skin_ft.pt", device="cpu")
p_no_tta = sk.predict(img, tta=False)
assert sk.is_finetuned and sk.version == "test-v1"
assert abs(sum(p_no_tta.probs.values()) - 1) < 1e-4
np.testing.assert_allclose([p_no_tta.probs[c] for c in codes], ref.detach().numpy(), atol=2e-3)   # fp16 rounding
pred = sk.predict(img)
print("pred:", pred.top_code, f"{pred.confidence:.3f}", pred.risk_tier)

# 2) baseline fallback
sk0 = core.SkinCLIP(tmp / "nonexistent" / "x.pt", device="cpu")
assert not sk0.is_finetuned and sk0.predict(img).model_version == "baseline-zero-shot"

# 3) feedback store
st = FeedbackStore(tmp / "data")
for i in range(6):
    im = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    d = pred.to_dict()
    if i % 3 == 0:
        st.add_review(img=im, prediction=d, reviewer_id="dr-x", decision="approved", label_basis="Dermoscopic review",
                      reviewer_certainty="High")
    elif i % 3 == 1:
        st.add_review(img=im, prediction=d, reviewer_id="dr-x", decision="corrected",
                      final_label=[c for c in codes if c != d["top_code"]][0], label_basis="Histopathology-confirmed",
                      reviewer_certainty="High")
    else:
        st.add_review(img=im, prediction=d, reviewer_id="dr-x", decision="unusable")
for bad in [dict(decision="corrected", final_label=pred.top_code), dict(decision="approved", reviewer_id=" ")]:
    kw = dict(img=im, prediction=pred.to_dict(), reviewer_id="dr-x"); kw.update(bad)
    try: st.add_review(**kw); raise SystemExit("validation failed to reject " + str(bad))
    except ValueError: pass
df = st.reviews(); man = st.training_manifest()
assert len(df) == 6 and len(man) == 4, (len(df), len(man))
assert all(Path(p).exists() for p in man["abs_path"])
assert "exif" not in Image.open(man["abs_path"][0]).info
print("store ok:", len(df), "reviews,", len(man), "trainable")
assert len(st.export_zip()) > 100

# 4) retrain end-to-end on the tiny set (needs replay set)
rp = model_dir / "replay"; rp.mkdir()
rows = []
for k, c in enumerate(codes):
    for j in range(4):
        fn = f"{c}_{j}.jpg"; Image.fromarray((np.random.rand(96, 96, 3) * 255).astype("uint8")).save(rp / fn); rows.append((fn, c))
import pandas as pd
pd.DataFrame(rows, columns=["file", "label"]).to_csv(rp / "manifest.csv", index=False)
(model_dir / "metadata.json").write_text(json.dumps({"version": "test-v1", "backbone": "ViT-B/32"}))
import subprocess
code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
        "from tests import fake_clip; fake_clip.install(); import clip, copy; _o=clip.load; c={};\n"
        "def l(n, device='cpu', jit=False, **k):\n"
        "    c.setdefault('m', _o(n, 'cpu')); return copy.deepcopy(c['m'][0]).to(device), c['m'][1]\n"
        "clip.load = l\n"
        "import sys; sys.argv=['retrain.py','--data-dir',%r,'--ckpt',%r,'--replay-dir',%r,'--out-dir',%r,'--epochs','1',"
        "'--min-feedback','2','--batch-size','4','--promote']\n"
        "import retrain; retrain.main()") % (str(tmp / "data"), str(model_dir / "clip_skin_ft.pt"), str(rp), str(model_dir / "cand"))
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
print(r.stdout[-900:], r.stderr[-1500:] if r.returncode else "")
assert r.returncode == 0 and list((model_dir / "cand").glob("*.pt"))
print("\nALL SMOKE TESTS PASSED")
