Put the files exported by the notebook here:

    clip_skin_ft.pt     fine-tuned weights (tuned tensors + head + temperature)
    metadata.json       class list, metrics, version
    replay/             replay images + manifest.csv (used by retrain.py)

Without `clip_skin_ft.pt` the app starts in *demo mode* with the zero-shot baseline.
