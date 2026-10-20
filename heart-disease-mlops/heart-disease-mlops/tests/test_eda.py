from heart import eda


def test_eda_writes_all_figures(clean_df, tmp_path):
    files = eda.run_all(clean_df, tmp_path)
    names = {f.name for f in files}
    assert {"class_balance.png", "histograms.png", "correlation_heatmap.png",
            "categorical_counts.png", "boxplots.png", "missing_values.png"} <= names
    assert all(f.stat().st_size > 5_000 for f in files)
