"""Exploratory data analysis plots (used by the notebook and by `python -m heart.eda`)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config, data  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C78A8", 1: "#E4572E"}


def _save(fig, name: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / name, dpi=150, bbox_inches="tight")
    return outdir / name


def class_balance(df, outdir=config.FIGURES_DIR):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    counts = df[config.TARGET].value_counts().sort_index()
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values,
                  color=[PALETTE[0], PALETTE[1]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n({v / len(df):.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    _save(fig, "class_balance.png", outdir)
    return fig


def histograms(df, outdir=config.FIGURES_DIR):
    cols = config.NUMERIC
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue=config.TARGET, bins=25, kde=True,
                     palette=PALETTE, ax=ax, element="step", stat="count")
        ax.set_title(col)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Distributions of continuous features by class (1 = disease)", y=1.01)
    fig.tight_layout()
    _save(fig, "histograms.png", outdir)
    return fig


def categorical_counts(df, outdir=config.FIGURES_DIR):
    cols = config.CATEGORICAL
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.ravel(), cols):
        sns.countplot(data=df, x=col, hue=config.TARGET, palette=PALETTE, ax=ax)
        ax.set_title(col)
        ax.legend_.remove() if ax.get_legend() else None
    fig.suptitle("Categorical features by class (blue = no disease, red = disease)", y=1.01)
    fig.tight_layout()
    _save(fig, "categorical_counts.png", outdir)
    return fig


def correlation_heatmap(df, outdir=config.FIGURES_DIR):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation heatmap (Pearson)")
    _save(fig, "correlation_heatmap.png", outdir)
    return fig


def boxplots(df, outdir=config.FIGURES_DIR):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    for ax, col in zip(axes, config.NUMERIC):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET, palette=PALETTE,
                    legend=False, ax=ax)
        ax.set_title(col)
    fig.suptitle("Continuous features vs target (outlier check)", y=1.03)
    fig.tight_layout()
    _save(fig, "boxplots.png", outdir)
    return fig


def missingness(df, outdir=config.FIGURES_DIR):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    if len(miss):
        ax.bar(miss.index, miss.values, color="#8172B3")
        for i, v in enumerate(miss.values):
            ax.text(i, v + 0.1, str(int(v)), ha="center")
    ax.set_title("Missing values after cleaning")
    ax.set_ylabel("Rows")
    _save(fig, "missing_values.png", outdir)
    return fig


def run_all(df: pd.DataFrame | None = None, outdir=config.FIGURES_DIR):
    df = data.load_clean() if df is None else df
    for fn in (class_balance, histograms, categorical_counts, correlation_heatmap, boxplots, missingness):
        fn(df, outdir)
        plt.close("all")
    return sorted(Path(outdir).glob("*.png"))


if __name__ == "__main__":
    for p in run_all():
        print("saved", p)
