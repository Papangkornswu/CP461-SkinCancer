"""SQLite-backed store for physician reviews (the continuous-improvement data set)."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    case_uid TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    image_path TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_pred TEXT NOT NULL,
    model_conf REAL NOT NULL,
    probs_json TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approved','corrected','unusable')),
    final_label TEXT,
    label_basis TEXT,
    reviewer_certainty TEXT,
    lesion_site TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_sha ON reviews(image_sha256);
"""
DECISIONS = ("approved", "corrected", "unusable")
LABEL_BASIS = ("Clinical impression", "Dermoscopic review", "Histopathology-confirmed")
CERTAINTY = ("Low", "Medium", "High")


class FeedbackStore:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.img_dir = self.root / "feedback" / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "feedback" / "feedback.db"
        with self._conn() as c:
            c.executescript(SCHEMA)
        self._schema_ready = True

    @contextmanager
    def _conn(self):
        self.img_dir.mkdir(parents=True, exist_ok=True)      # survives the data folder being wiped/remounted
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.executescript(SCHEMA)                          # idempotent (CREATE ... IF NOT EXISTS)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ images
    @staticmethod
    def image_hash(img: Image.Image) -> str:
        rgb = img.convert("RGB")
        h = hashlib.sha256()
        h.update(f"{rgb.size}".encode())
        h.update(np.asarray(rgb).tobytes())
        return h.hexdigest()

    def save_image(self, img: Image.Image) -> tuple[str, str]:
        """Re-encodes as JPEG (drops EXIF/GPS and other metadata) and returns (sha256, relative path)."""
        sha = self.image_hash(img)
        rel = f"feedback/images/{sha[:32]}.jpg"
        out = self.root / rel
        if not out.exists():
            img.convert("RGB").save(out, format="JPEG", quality=95)   # no exif= -> metadata stripped
        return sha, rel

    # ------------------------------------------------------------------ writes
    def add_review(self, *, img: Image.Image, prediction: dict, reviewer_id: str, decision: str,
                   final_label: str | None = None, label_basis: str | None = None,
                   reviewer_certainty: str | None = None, lesion_site: str | None = None,
                   notes: str | None = None) -> int:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if not reviewer_id.strip():
            raise ValueError("Reviewer ID is required")
        pred = prediction["top_code"]
        if decision == "approved":
            final_label = pred
        elif decision == "corrected":
            if not final_label or final_label == pred:
                raise ValueError("Pick a diagnosis that differs from the model's prediction")
        else:
            final_label = None
        sha, rel = self.save_image(img)
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO reviews (created_at, case_uid, image_sha256, image_path, model_version, model_pred,
                   model_conf, probs_json, reviewer_id, decision, final_label, label_basis, reviewer_certainty,
                   lesion_site, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), uuid.uuid4().hex, sha, rel,
                 prediction["model_version"], pred, float(prediction["confidence"]),
                 json.dumps(prediction["probs"]), reviewer_id.strip(), decision, final_label, label_basis,
                 reviewer_certainty, lesion_site or None, (notes or "").strip() or None))
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ reads
    def reviews(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM reviews ORDER BY id", c)
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    def training_manifest(self, min_certainty: str | None = None) -> pd.DataFrame:
        """Latest usable review per image: columns image_path (absolute), label, weight, reviewer_id..."""
        df = self.reviews()
        if df.empty:
            return df
        df = df[df["decision"].isin(["approved", "corrected"])]
        if min_certainty:
            rank = {c: i for i, c in enumerate(CERTAINTY)}
            df = df[df["reviewer_certainty"].map(rank).fillna(-1) >= rank[min_certainty]]
        df = df.sort_values("id").groupby("image_sha256", as_index=False).tail(1).copy()
        df["abs_path"] = df["image_path"].map(lambda p: str(self.root / p))
        basis_w = {"Histopathology-confirmed": 1.0, "Dermoscopic review": 0.8, "Clinical impression": 0.6}
        df["weight"] = df["label_basis"].map(basis_w).fillna(0.7)
        return df[["id", "image_sha256", "abs_path", "final_label", "model_pred", "decision", "weight",
                   "label_basis", "reviewer_certainty", "reviewer_id", "created_at"]].reset_index(drop=True)

    def export_zip(self, include_images: bool = True) -> bytes:
        buf = io.BytesIO()
        df = self.reviews()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("reviews.csv", df.to_csv(index=False))
            if include_images:
                for rel in df["image_path"].unique() if not df.empty else []:
                    p = self.root / rel
                    if p.exists():
                        z.write(p, rel)
        return buf.getvalue()
