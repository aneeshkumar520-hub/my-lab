#!/usr/bin/env python3
"""Download the Heart Disease (Cleveland) dataset into data/raw/heart.csv.

The official source is the UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/45/heart+disease). UCI distributes the
file as `processed.cleveland.data` with its own coding (cp 1-4, thal 3/6/7,
'?' for missing). This project uses the widely-mirrored 14-column CSV version of
the same Cleveland data (cp 0-3, thal 1-3), so the script downloads that CSV from
GitHub mirrors. See README ("Data") for the manual UCI route and the caveats.

Usage:  python scripts/download_data.py [--out data/raw/heart.csv]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
]
EXPECTED_HEADER = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target"


def looks_valid(text: str) -> bool:
    lines = text.lstrip("\ufeff").strip().splitlines()
    return bool(lines) and lines[0].strip() == EXPECTED_HEADER and len(lines) > 250


def download(out: Path, mirrors: list[str] | None = None) -> Path:
    for url in mirrors or MIRRORS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # nosec - fixed https URLs
                text = resp.read().decode("utf-8")
            if looks_valid(text):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text.lstrip("\ufeff"), encoding="utf-8", newline="\n")
                print(f"OK  downloaded {len(text.splitlines()) - 1} rows from {url}")
                return out
            print(f"WARN unexpected content from {url}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - try next mirror
            print(f"WARN {url}: {exc}", file=sys.stderr)
    raise SystemExit("Could not download the dataset from any mirror. "
                     "Download it manually (see README) into data/raw/heart.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/heart.csv")
    download(ap.parse_args().out)
