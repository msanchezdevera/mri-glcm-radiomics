"""Visually compare the GLCM of a healthy MRI slice and a tumor slice."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from .features import LEVELS, load_brain, quantize


def glcm_of(path: Path):
    img, _ = load_brain(path)
    q = quantize(img)
    g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=LEVELS,
                     symmetric=True, normed=True).mean(axis=3)
    props = {p: graycoprops(g[..., None], p).mean() for p in ("contrast", "homogeneity", "energy")}
    return img, g[:, :, 0], props


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("healthy", type=Path)
    ap.add_argument("tumor", type=Path)
    ap.add_argument("--out", type=Path, default=Path("assets/glcm_comparison.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for row, (title, path) in enumerate((("No tumor", args.healthy), ("Tumor", args.tumor))):
        img, g, props = glcm_of(path)
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"MRI — {title}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(np.log1p(g * 1e4), cmap="viridis")
        axes[row, 1].set_title("GLCM (log scale)\n" + " · ".join(f"{k} {v:.2f}" for k, v in props.items()), fontsize=9)
        axes[row, 1].set_xlabel("gray level j")
        axes[row, 1].set_ylabel("gray level i")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
