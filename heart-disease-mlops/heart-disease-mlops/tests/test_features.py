import numpy as np
import pandas as pd

from heart import config
from heart.features import FeatureAdder, build_preprocessor, feature_names


def test_feature_adder_values(sample):
    out = FeatureAdder().transform(pd.DataFrame([sample]))
    assert out.loc[0, "pct_max_hr"] == np.float64(150) / (220 - 63)
    assert out.loc[0, "st_depression_with_angina"] == 0.0     # exang = 0
    assert set(config.ENGINEERED) <= set(out.columns)


def test_feature_adder_does_not_mutate_input(sample):
    df = pd.DataFrame([sample])
    FeatureAdder().transform(df)
    assert "pct_max_hr" not in df.columns


def test_preprocessor_output_is_finite_even_with_missing(clean_df):
    X = clean_df[config.FEATURES].astype(float)
    assert X.isna().any().any()                     # fixture really has gaps
    Z = build_preprocessor().fit_transform(X)
    assert Z.shape[0] == len(X) and np.isfinite(np.asarray(Z)).all()


def test_numeric_columns_are_standardised(clean_df):
    pre = build_preprocessor().fit(clean_df[config.FEATURES].astype(float))
    Z = pre.transform(clean_df[config.FEATURES].astype(float))
    n_num = len(config.NUMERIC + config.ENGINEERED)
    assert np.allclose(np.asarray(Z)[:, :n_num].mean(axis=0), 0, atol=1e-8)
    assert np.allclose(np.asarray(Z)[:, :n_num].std(axis=0), 1, atol=1e-8)


def test_unseen_category_is_ignored(clean_df, sample):
    pre = build_preprocessor().fit(clean_df[config.FEATURES].astype(float))
    row = dict(sample, cp=9)                        # category never seen in training
    out = pre.transform(pd.DataFrame([row])[config.FEATURES].astype(float))
    assert np.isfinite(np.asarray(out)).all()


def test_feature_names_match_width(clean_df):
    pre = build_preprocessor().fit(clean_df[config.FEATURES].astype(float))
    names = feature_names(pre)
    assert len(names) == pre.transform(clean_df[config.FEATURES].astype(float)).shape[1]
