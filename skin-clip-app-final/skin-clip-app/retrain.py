"""Continual fine-tuning from physician feedback.

    python retrain.py --data-dir data --ckpt model/clip_skin_ft.pt            # train a candidate + report
    python retrain.py --data-dir data --ckpt model/clip_skin_ft.pt --promote  # also replace the live model if it passes

What it does
  1. Loads the latest approved/corrected review per image (label = physician's final label, weighted by label basis).
  2. Mixes it with a small replay set of original HAM10000 images (model/replay/, exported by the notebook)
     so the model does not forget what it already knows.
  3. Fine-tunes the same layers as in the notebook with a small learning rate.
  4. Evaluates old vs. new on held-out feedback AND held-out replay images. The candidate passes only if it
     improves (or ties) on feedback and loses no more than --max-regression on the replay hold-out.
  5. Writes a versioned candidate checkpoint + retrain_report.json. --promote swaps it in (old one is archived).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from core import MODEL_DIR, build_net_from_checkpoint, save_checkpoint
from feedback_store import FeedbackStore

MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
EVAL_TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
                      T.ColorJitter(0.15, 0.15, 0.1, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])


class ImgDS(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.y, self.w, self.tf = df["path"].tolist(), df["y"].to_numpy(), df["w"].to_numpy(np.float32), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.y[i]), float(self.w[i])


@torch.no_grad()
def predict_logits(net, df, device, bs=64):
    net.eval()
    out = []
    for x, _, _ in DataLoader(ImgDS(df, EVAL_TF), batch_size=bs, shuffle=False, num_workers=2):
        out.append(net(x.to(device)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, net.head.shape[0])


def accuracy(logits, y):
    return float((logits.argmax(1).numpy() == y).mean()) if len(y) else float("nan")


def fit_temperature(logits, y, init):
    if len(y) < 50:
        return init
    logt = torch.tensor(float(np.log(init)), requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=100)
    yt = torch.as_tensor(y)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logt.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logt.exp().clamp(0.2, 5.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ckpt", default=str(MODEL_DIR / "clip_skin_ft.pt"))
    ap.add_argument("--replay-dir", default=str(MODEL_DIR / "replay"))
    ap.add_argument("--out-dir", default=str(MODEL_DIR / "candidates"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-6, help="backbone LR (head uses 20x)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-feedback", type=int, default=30, help="refuse to train on fewer usable reviews")
    ap.add_argument("--min-certainty", choices=["Low", "Medium", "High"], default=None)
    ap.add_argument("--max-regression", type=float, default=0.02)
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ckpt, _ = build_net_from_checkpoint(a.ckpt)
    classes, codes, n_blocks = ckpt["classes"], ckpt["codes"], int(ckpt["n_blocks"])
    code2i = {c: i for i, c in enumerate(codes)}

    fb = FeedbackStore(a.data_dir).training_manifest(a.min_certainty)
    if len(fb) < a.min_feedback:
        raise SystemExit(f"Only {len(fb)} usable reviews (< --min-feedback {a.min_feedback}). Collect more first.")
    fb = pd.DataFrame({"path": fb["abs_path"], "y": fb["final_label"].map(code2i), "w": fb["weight"]})

    rp_csv = Path(a.replay_dir) / "manifest.csv"
    if rp_csv.exists():
        rp = pd.read_csv(rp_csv)
        rp = pd.DataFrame({"path": [str(Path(a.replay_dir) / f) for f in rp["file"]],
                           "y": rp["label"].map(code2i), "w": 1.0})
    else:
        print("WARNING: no replay set found: the candidate cannot be checked for forgetting.")
        rp = pd.DataFrame(columns=["path", "y", "w"])

    def split(df, frac=0.2):
        idx = rng.permutation(len(df))
        n_ho = max(1, int(len(df) * frac)) if len(df) >= 5 else 0
        return df.iloc[idx[n_ho:]].reset_index(drop=True), df.iloc[idx[:n_ho]].reset_index(drop=True)

    fb_tr, fb_ho = split(fb)
    rp_tr, rp_ho = split(rp)
    net.to(device)
    before = {"feedback_holdout_acc": accuracy(predict_logits(net, fb_ho, device), fb_ho["y"].to_numpy()),
              "replay_holdout_acc": accuracy(predict_logits(net, rp_ho, device), rp_ho["y"].to_numpy())}
    print("before:", before)

    net.set_trainable(n_blocks)
    bb = [p for p in net.visual.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr}, {"params": [net.head, net.logit_scale], "lr": a.lr * 20}],
                            weight_decay=0.05)
    steps = a.epochs * int(np.ceil(2 * len(fb_tr) / a.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 20], total_steps=max(steps, 1),
                                                pct_start=0.15)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        # every epoch: all feedback + an equal number of replay images
        mix = pd.concat([fb_tr, rp_tr.sample(min(len(rp_tr), len(fb_tr)), random_state=a.seed + ep)]) \
            if len(rp_tr) else fb_tr
        mix = mix.sample(frac=1, random_state=a.seed + ep).reset_index(drop=True)
        losses = []
        for x, y, w in DataLoader(ImgDS(mix, TRAIN_TF), batch_size=a.batch_size, shuffle=False, num_workers=2):
            loss = (F.cross_entropy(net(x.to(device)), y.to(device), label_smoothing=0.05, reduction="none")
                    * w.to(device)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            losses.append(loss.item())
        print(f"epoch {ep+1}/{a.epochs} loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)")

    ho = pd.concat([fb_ho, rp_ho]).reset_index(drop=True)
    ho_logits = predict_logits(net, ho, device)
    temperature = fit_temperature(ho_logits, ho["y"].to_numpy(), float(ckpt.get("temperature", 1.0)))
    after = {"feedback_holdout_acc": accuracy(predict_logits(net, fb_ho, device), fb_ho["y"].to_numpy()),
             "replay_holdout_acc": accuracy(predict_logits(net, rp_ho, device), rp_ho["y"].to_numpy())}
    print("after: ", after, "temperature:", round(temperature, 3))

    ok_fb = np.isnan(before["feedback_holdout_acc"]) or after["feedback_holdout_acc"] >= before["feedback_holdout_acc"]
    ok_rp = np.isnan(before["replay_holdout_acc"]) or \
        after["replay_holdout_acc"] >= before["replay_holdout_acc"] - a.max_regression
    passed = bool(ok_fb and ok_rp)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    version = f"{str(ckpt.get('version', 'ft')).split('+fb')[0]}+fb{stamp}"
    out_dir = Path(a.out_dir)
    cand = out_dir / f"clip_skin_ft_{stamp}.pt"
    save_checkpoint(net, cand, n_blocks=n_blocks, backbone=ckpt["backbone"], classes=classes, codes=codes,
                    temperature=temperature, version=version,
                    extra={"n_feedback_train": len(fb_tr), "before": before, "after": after})
    nn_ = lambda d: {k: (None if v != v else v) for k, v in d.items()}   # NaN -> null (valid JSON)
    before, after = nn_(before), nn_(after)
    report = {"candidate": str(cand), "version": version, "passed": passed, "before": before, "after": after,
              "n_feedback_train": len(fb_tr), "n_feedback_holdout": len(fb_ho), "n_replay_train": len(rp_tr),
              "n_replay_holdout": len(rp_ho), "temperature": temperature}
    (out_dir / f"retrain_report_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if a.promote and passed:
        live = Path(a.ckpt)
        archive = live.parent / "archive"
        archive.mkdir(exist_ok=True)
        shutil.copy2(live, archive / f"{live.stem}_{stamp}_prev.pt")
        shutil.copy2(cand, live)
        meta_p = live.parent / "metadata.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            meta["version"] = version
            meta.setdefault("retrain_history", []).append({"stamp": stamp, **{k: report[k] for k in ("before", "after")}})
            meta_p.write_text(json.dumps(meta, indent=2))
        print(f"Promoted -> {live}. Restart the app to load it.")
    elif a.promote:
        print("Candidate did NOT pass the regression gate: live model left unchanged.")


if __name__ == "__main__":
    main()
