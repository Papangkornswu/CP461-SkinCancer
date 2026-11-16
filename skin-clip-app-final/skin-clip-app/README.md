# ClipSkinCancer

Fine-tuned CLIP (HAM10000, 7 classes) behind a Streamlit UI, with **editable dermoscopic symptoms** (z-score sliders that re-diagnose live)
and a **physician approval workflow** whose data feeds continual retraining.

```
ICBM_Skin_CLIP_FineTune.ipynb   train + evaluate (baseline vs fine-tuned), exports model/
app.py                          Streamlit UI (analyze · physician review · analytics · model card)
core.py                         model definition / inference / image gate / symptom extraction (shared with retrain.py)
concepts.py                     25 symptoms, z-scoring, re-diagnosis, "check first" recommendation (pure numpy)
i18n.py                         English / Thai UI strings
feedback_store.py               SQLite + image store for reviews
retrain.py                      continual fine-tuning with a no-regression gate
Dockerfile, requirements.txt    deployment
```

## 1 · Train (notebook)
Open `ICBM_Skin_CLIP_FineTune.ipynb` on Colab/Kaggle with a GPU and *Run all*. It reproduces the original project's test set,
evaluates zero-shot CLIP, a linear probe and the fine-tuned model side by side (with a lesion-level bootstrap CI), then exports
`skin_clip_export.zip`. Unzip it into `model/`.

## 2 · Run locally
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export REVIEWER_PASSCODE='a-long-random-string'
streamlit run app.py
```

### UI options
* **Language:** EN / ไทย toggle (top right). **Theme:** Light / Dark switch. Both are remembered in the URL (`?lang=th&theme=dark`).
* **Physician review:** choosing "Correct diagnosis" and a label updates the result card and chart immediately; the review is stored on *Submit*.
  After saving, **Change my label** re-opens the form; the newer review supersedes the old one (latest review per image wins at training time).
* **Symptom editor (step 2):** the original CLIP scores 25 dermoscopic symptoms on the image; each is shown as a z-score slider
  (0 = dataset average, + = clearly present, − = absent). Moving a slider re-diagnoses instantly: the model's log-probabilities are
  shifted by `0.8 · W · (z_edited − z_extracted)`, so untouched sliders change nothing. **Orange frame** = the symptoms that best
  separate the rank-1 and rank-2 disease. *Save edits* stores before/after z-scores in the `concept_edits` table.
  `W` is a clinical-knowledge prior in `concepts.py`, **not learned**: treat the new diagnosis as a second opinion.
  Reference mean/std come from `model/replay/` and are cached in `model/concept_stats.json` on first start (delete it to rebuild).
  `CONCEPTS=0` hides the editor.
* **Image check:** non-skin images are blocked before prediction (`SKIN_GATE_MIN`, default 0.5; `SKIN_GATE=0` disables it; the original-CLIP image tower, ~200 MB RAM, is only freed if `CONCEPTS=0` too).

## 3 · Deploy
```bash
docker build -t clipskincancer .
docker run -p 8501:8501 -e REVIEWER_PASSCODE='...' -v clipskincancer-data:/app/data clipskincancer
```
* **Mount a persistent volume on `/app/data`**: it holds the feedback DB and images. Hosts with ephemeral disks
  (Streamlit Community Cloud, free Hugging Face Spaces) will lose reviews on restart.
* Put it behind HTTPS (reverse proxy / platform TLS) and keep `REVIEWER_PASSCODE` set. For real clinical use add proper
  per-user authentication (SSO) and audit logging; the shared passcode is a minimal gate.
* The checkpoint is ~tens of MB; if you commit it to GitHub use Git LFS above 100 MB.

## 4 · Feedback loop
1. Physician uploads a de-identified image → model prediction + calibrated probabilities.
2. Physician **approves**, **corrects** (picks the right diagnosis, states basis + certainty) or marks the image **unusable**.
3. Stored in `data/feedback/feedback.db` + `data/feedback/images/` (re-encoded JPEG, EXIF stripped). Duplicate images are
   de-duplicated at training time (latest review wins).
4. `python retrain.py --data-dir data --ckpt model/clip_skin_ft.pt [--promote]` fine-tunes on feedback + replay images,
   compares old/new on held-out feedback **and** held-out replay data, and promotes only if there is no regression. The old model is archived in `model/archive/`.

Treat reviewer labels as a training signal, not a ground-truth test set: keep a fixed, independently labelled gold set for release decisions.

## Privacy & safety
Research prototype, **not a medical device**. Upload only de-identified images, comply with your institution's data-protection rules
(GDPR/PDPA/HIPAA as applicable), and never use the output as the sole basis for diagnosis or treatment.
