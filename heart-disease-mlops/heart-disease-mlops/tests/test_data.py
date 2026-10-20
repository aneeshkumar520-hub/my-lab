import numpy as np
import pandas as pd
import pytest

from heart import config, data


def _row(**kw):
    base = {c: 0 for c in config.FEATURES + [config.TARGET]}
    base.update({"age": 50, "thal": 2, "target": 1})
    base.update(kw)
    return base


def test_clean_has_expected_columns_and_numeric_dtypes(clean_df):
    assert list(clean_df.columns) == config.FEATURES + [config.TARGET]
    assert all(pd.api.types.is_numeric_dtype(t) for t in clean_df.dtypes)


def test_target_is_binary_and_inverted():
    df = pd.DataFrame([_row(target=1), _row(target=0, age=60)])
    out = data.clean(df)
    assert out[config.TARGET].tolist() == [0, 1]       # 1 (absent in mirror) -> 0
    kept = data.clean(df, invert_target=False)
    assert kept[config.TARGET].tolist() == [1, 0]


def test_invalid_codes_become_nan():
    df = pd.DataFrame([_row(ca=4), _row(thal=0, age=61)])
    out = data.clean(df)
    assert np.isnan(out.loc[0, "ca"])
    assert np.isnan(out.loc[1, "thal"])


def test_duplicates_are_dropped():
    df = pd.DataFrame([_row(), _row(), _row(age=70)])
    assert len(data.clean(df)) == 2


def test_missing_column_raises():
    df = pd.DataFrame([_row()]).drop(columns=["chol"])
    with pytest.raises(ValueError, match="chol"):
        data.clean(df)


def test_real_dataset_matches_published_statistics(clean_df):
    # Cleveland: 303 patients, 164 healthy / 139 diseased, one duplicate row.
    assert len(clean_df) == 302
    assert clean_df[config.TARGET].value_counts().to_dict() == {0: 164, 1: 138}
    assert clean_df["ca"].isna().sum() == 4 and clean_df["thal"].isna().sum() == 2


def test_prepare_roundtrip(tmp_path, raw_df):
    out = tmp_path / "clean.csv"
    raw = tmp_path / "raw.csv"
    raw_df.to_csv(raw, index=False)
    df = data.prepare(raw, out)
    assert out.exists() and data.load_clean(out).shape == df.shape
