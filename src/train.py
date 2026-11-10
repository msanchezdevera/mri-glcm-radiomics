"""Entrena y evalúa clasificadores sobre las features de textura.

Dos tareas:
  - binary: tumor vs sin tumor
  - multiclass: glioma, meningioma, pituitario, sin tumor
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS_ES = {"glioma": "Glioma", "meningioma": "Meningioma",
             "pituitary": "Pituitario", "notumor": "Sin tumor"}


def models() -> dict:
    return {
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True)),
        "Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=42),
    }


def run(df: pd.DataFrame, task: str, out: Path) -> dict:
    feats = [c for c in df.columns if c.startswith(("glcm_", "fo_"))]
    y_all = (df["label"] != "notumor").astype(int) if task == "binary" else df["label"]
    tr, te = df["split"] == "Training", df["split"] == "Testing"
    X_tr, X_te, y_tr, y_te = df.loc[tr, feats], df.loc[te, feats], y_all[tr], y_all[te]

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    report, best_name, best_cv = {}, None, -1.0
    for name, model in models().items():
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1).mean()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)
        auc = (roc_auc_score(y_te, proba[:, 1]) if task == "binary"
               else roc_auc_score(y_te, proba, multi_class="ovr"))
        report[name] = {"cv_f1_macro": round(cv_f1, 4),
                        "test_accuracy": round(accuracy_score(y_te, pred), 4),
                        "test_f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
                        "test_roc_auc": round(auc, 4)}
        print(f"[{task}] {name}: {report[name]}")
        if cv_f1 > best_cv:
            best_cv, best_name = cv_f1, name

    best = models()[best_name].fit(X_tr, y_tr)
    pred = best.predict(X_te)
    names = ["Sin tumor", "Tumor"] if task == "binary" else None
    if task == "multiclass":
        order = ["glioma", "meningioma", "pituitary", "notumor"]
        disp = ConfusionMatrixDisplay.from_predictions(
            y_te, pred, labels=order, display_labels=[LABELS_ES[o] for o in order],
            normalize="true", cmap="Blues", values_format=".2f")
    else:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_te, pred, display_labels=names, normalize="true", cmap="Blues", values_format=".2f")
    disp.ax_.set_title(f"{best_name} — {'tumor vs sin tumor' if task == 'binary' else '4 clases'}")
    plt.tight_layout()
    plt.savefig(out / f"confusion_{task}.png", dpi=150)
    plt.close()

    report["_best"] = best_name
    report["_classification_report"] = classification_report(y_te, pred, output_dict=True)

    if task == "binary":
        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
        imp = pd.Series(rf.feature_importances_, index=feats).sort_values().tail(12)
        plt.figure(figsize=(7, 5))
        imp.plot.barh(color="#2b6cb0")
        plt.title("Features más importantes (Random Forest)")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(out / "feature_importance.png", dpi=150)
        plt.close()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("results/features.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    df = pd.read_csv(args.features)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {task: run(df, task, args.out) for task in ("binary", "multiclass")}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
