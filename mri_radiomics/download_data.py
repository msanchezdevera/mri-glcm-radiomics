"""Download the Brain Tumor MRI Dataset (Masoud Nickparvar) from Hugging Face into ./data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def fetch(rel: str, out: Path) -> None:
    dest = out / rel
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        r = requests.get(RAW + rel, timeout=60)
        if r.ok:
            dest.write_bytes(r.content)
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    files = [s["rfilename"] for s in requests.get(API, timeout=60).json()["siblings"]
             if s["rfilename"].endswith(".jpg")]
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda f: fetch(f, args.out), files))
    print(f"{len(files)} images in {args.out}")
