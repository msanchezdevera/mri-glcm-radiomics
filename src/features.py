"""Extracción de características de textura (GLCM) y de primer orden en MRI cerebrales."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte

LEVELS = 32
DISTANCES = (1, 2, 4)
ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def load_brain(path: Path, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Carga la imagen, la pasa a gris, la recorta al cerebro y devuelve imagen + máscara."""
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    img = transform.resize(img, (size, size), anti_aliasing=True)
    img = img_as_ubyte(img / img.max() if img.max() > 0 else img)

    mask = img > threshold_otsu(img)
    ys, xs = np.nonzero(mask)
    if len(ys) > 100:
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return img, mask


def quantize(img: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    return (img.astype(np.float32) / 256 * levels).astype(np.uint8)


def glcm_features(img_q: np.ndarray, levels: int = LEVELS) -> dict[str, float]:
    glcm = graycomatrix(img_q, distances=DISTANCES, angles=ANGLES,
                        levels=levels, symmetric=True, normed=True)
    feats: dict[str, float] = {}
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)  # (distancias, ángulos)
        for i, d in enumerate(DISTANCES):
            feats[f"glcm_{prop}_d{d}_mean"] = float(values[i].mean())
            feats[f"glcm_{prop}_d{d}_range"] = float(np.ptp(values[i]))
    for i, d in enumerate(DISTANCES):
        p = glcm[:, :, i, :]
        ent = -np.sum(p * np.log2(p + 1e-12), axis=(0, 1))
        feats[f"glcm_entropy_d{d}_mean"] = float(ent.mean())
    return feats


def first_order_features(img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = img[mask].astype(np.float32) if mask.any() else img.ravel().astype(np.float32)
    hist, _ = np.histogram(vals, bins=64, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return {
        "fo_mean": float(vals.mean()),
        "fo_std": float(vals.std()),
        "fo_skewness": float(skew(vals)),
        "fo_kurtosis": float(kurtosis(vals)),
        "fo_p10": float(np.percentile(vals, 10)),
        "fo_p90": float(np.percentile(vals, 90)),
        "fo_entropy": float(-np.sum(hist * np.log2(hist))),
    }


def extract(path: Path) -> dict[str, float]:
    img, mask = load_brain(path)
    return {**first_order_features(img, mask), **glcm_features(quantize(img))}


def _row(item: tuple[str, str, Path]) -> dict:
    split, label, f = item
    return {"split": split, "label": label, "file": f.name, **extract(f)}


def build_table(root: Path, jobs: int = 4) -> pd.DataFrame:
    items = [(split, d.name, f)
             for split in ("Training", "Testing")
             for d in sorted((root / split).glob("*"))
             for f in sorted(d.glob("*.jpg"))]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(_row, items, chunksize=32))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/features.csv"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    df = build_table(args.data, args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} imágenes, {df.shape[1] - 3} features -> {args.out}")
