"""Production model: train once, save a single artifact, serve predictions.

The artifact bundles everything inference needs, so the API and the agent never
touch the raw dataset:
  - the 4-class Gradient Boosting classifier (selected in train.py by CV macro F1)
  - training-set feature statistics (for per-case explanations)
  - permutation importances on the test set (global importance)
  - a standardized k-NN index over the training set (similar-case retrieval)
  - the test metrics, used as a model card
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import extract

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT = ROOT / "models" / "model.joblib"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY = {"glioma": "Glioma", "meningioma": "Meningioma",
           "pituitary": "Pituitary tumor", "notumor": "No tumor"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("glcm_", "fo_"))]


def build_artifact(features_csv: Path, out: Path) -> dict:
    df = pd.read_csv(features_csv)
    feats = feature_columns(df)
    tr, te = df[df.split == "Training"], df[df.split == "Testing"]

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=42)
    clf.fit(tr[feats], tr.label)
    pred = clf.predict(te[feats])
    metrics = {"test_accuracy": round(accuracy_score(te.label, pred), 4),
               "test_f1_macro": round(f1_score(te.label, pred, average="macro"), 4),
               "test_size": int(len(te)), "train_size": int(len(tr))}

    perm = permutation_importance(clf, te[feats], te.label, n_repeats=5,
                                  random_state=42, scoring="f1_macro", n_jobs=-1)
    importance = dict(zip(feats, np.round(perm.importances_mean, 5).tolist()))

    scaler = StandardScaler().fit(tr[feats])
    Z = scaler.transform(tr[feats]).astype(np.float32)

    artifact = {
        "classifier": clf, "features": feats, "scaler": scaler, "train_z": Z,
        "train_labels": tr.label.to_numpy(), "train_files": tr.file.to_numpy(),
        "class_means": tr.groupby("label")[feats].mean().to_dict("index"),
        "importance": importance, "metrics": metrics,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)
    return metrics


@dataclass
class Prediction:
    label: str
    display: str
    probabilities: dict[str, float]
    tumor_probability: float

    def as_dict(self) -> dict:
        return {"label": self.label, "display": self.display,
                "probabilities": self.probabilities,
                "tumor_probability": self.tumor_probability}


class RadiomicsModel:
    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT):
        a = joblib.load(artifact_path)
        self.clf = a["classifier"]
        self.features: list[str] = a["features"]
        self.scaler: StandardScaler = a["scaler"]
        self.train_labels = a["train_labels"]
        self.train_files = a["train_files"]
        self.class_means = a["class_means"]
        self.importance: dict[str, float] = a["importance"]
        self.metrics: dict = a["metrics"]
        self.knn = NearestNeighbors(n_neighbors=10).fit(a["train_z"])

    # --- feature extraction -------------------------------------------------
    def featurize(self, image_path: Path) -> dict[str, float]:
        return extract(Path(image_path))

    def _vector(self, feats: dict[str, float]) -> pd.DataFrame:
        missing = set(self.features) - feats.keys()
        if missing:
            raise ValueError(f"missing features: {sorted(missing)[:3]}...")
        return pd.DataFrame([[feats[f] for f in self.features]], columns=self.features)

    # --- inference ----------------------------------------------------------
    def predict(self, feats: dict[str, float]) -> Prediction:
        proba = self.clf.predict_proba(self._vector(feats))[0]
        probs = {c: round(float(p), 4) for c, p in zip(self.clf.classes_, proba)}
        label = max(probs, key=probs.get)
        return Prediction(label, DISPLAY[label], probs,
                          round(1 - probs.get("notumor", 0.0), 4))

    def explain(self, feats: dict[str, float], top_k: int = 5) -> list[dict]:
        """Local explanation: among the globally most important features, how far
        this case sits from the healthy-tissue profile (in training-set std units)."""
        z = self.scaler.transform(self._vector(feats))[0]
        healthy = self.scaler.transform(self._vector(self.class_means["notumor"]))[0]
        rows = []
        for i, f in enumerate(self.features):
            imp = max(self.importance.get(f, 0.0), 0.0)
            dev = float(z[i] - healthy[i])
            rows.append({"feature": f, "value": round(feats[f], 4),
                         "deviation_from_healthy_std": round(dev, 2),
                         "global_importance": round(imp, 4), "score": imp * abs(dev)})
        rows.sort(key=lambda r: r["score"], reverse=True)
        for r in rows:
            r.pop("score")
        return rows[:top_k]

    def similar(self, feats: dict[str, float], k: int = 5) -> list[dict]:
        z = self.scaler.transform(self._vector(feats))
        dist, idx = self.knn.kneighbors(z, n_neighbors=k)
        return [{"file": str(self.train_files[i]), "label": str(self.train_labels[i]),
                 "distance": round(float(d), 3)} for d, i in zip(dist[0], idx[0])]

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True)[:n]


@lru_cache(maxsize=1)
def get_model(path: str | None = None) -> RadiomicsModel:
    return RadiomicsModel(Path(path) if path else DEFAULT_ARTIFACT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=ROOT / "results" / "features.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    args = ap.parse_args()
    print(json.dumps(build_artifact(args.features, args.out), indent=2))
