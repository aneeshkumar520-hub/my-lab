"""Loading and cleaning of the Heart Disease (Cleveland / UCI) dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

EXPECTED_COLUMNS = config.FEATURES + [config.TARGET]
# Codes that are not valid in the data dictionary and in practice stand for
# "missing" (ca has only 0-3, thal only 1-3 in this encoding).
INVALID_CODES = {"ca": [4], "thal": [0]}


def load_raw(path: Path | str = config.DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV (handles a UTF-8 BOM on the header line)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(df: pd.DataFrame, invert_target: bool = True) -> pd.DataFrame:
    """Return a cleaned copy of the raw frame.

    Steps
    -----
    1. validate that all expected columns exist and keep only those;
    2. coerce everything to numeric;
    3. replace invalid category codes by NaN (imputed later, inside the model
       pipeline, so that imputation statistics come from training data only);
    4. drop exact duplicate rows;
    5. make ``target`` mean "disease present". The public mirror of the data
       stores 1 = *absence* of disease (verified: target==1 has higher max heart
       rate and fewer vessels/angina), so it is inverted by default.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for col, bad in INVALID_CODES.items():
        out.loc[out[col].isin(bad), col] = np.nan

    n_before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    log.info("dropped %d duplicate rows", n_before - len(out))

    out = out.dropna(subset=[config.TARGET])
    if invert_target:
        out[config.TARGET] = 1 - out[config.TARGET]
    out[config.TARGET] = out[config.TARGET].astype(int)
    return out


def prepare(raw_path: Path | str = config.DATA_RAW,
            out_path: Path | str = config.DATA_CLEAN) -> pd.DataFrame:
    """Load raw data, clean it and persist the cleaned dataset."""
    df = clean(load_raw(raw_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("saved %s rows to %s", len(df), out_path)
    return df


def load_clean(path: Path | str = config.DATA_CLEAN) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = prepare()
    print(d.shape, "missing cells:", int(d.isna().sum().sum()))
