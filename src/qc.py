"""Data quality checks and filtering for the processed interday dataset."""

from pathlib import Path

from loguru import logger
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

MIN_DAILY_WEAR_TIME = 10  # Hours
MIN_DAYS_PER_SUBJECT = 30
MISSINGNESS_WARN_THRESHOLD = 0.75

# Expected median ranges — flags systematic pipeline issues, not individual outliers.
MEDIAN_RANGES = {
    "hr": (50, 100),
    "hr_resting": (45, 90),
    "temperature": (33.0, 37.0),
    "temperature_diff": (-1.5, 1.5),
    "sleep_min_total": (300, 600),
    "sleep_efficiency": (70, 100),
    "hrv_rmssd_mean": (10, 120),
    "step_count": (2_000, 25_000),
    "estrogen": (5, 150),
    "lh": (2, 80),
}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _filter_wear_time(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df["wear_valid"] = df["wear_minutes"] >= (MIN_DAILY_WEAR_TIME * 60)
    df = df[df["wear_valid"].fillna(False)].drop(columns=["wear_valid"])
    logger.info(f"Wear time filter: removed {before - len(df)} days below {MIN_DAILY_WEAR_TIME}h")
    return df.reset_index(drop=True)


def _filter_min_days(df: pd.DataFrame) -> pd.DataFrame:
    day_counts = df.groupby("id")["day_in_study"].nunique()
    valid = day_counts[day_counts >= MIN_DAYS_PER_SUBJECT].index
    before = df["id"].nunique()
    df = df[df["id"].isin(valid)].reset_index(drop=True)
    logger.info(
        f"Min days filter: removed {before - len(valid)} subjects with < {MIN_DAYS_PER_SUBJECT} days"
    )
    return df


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_missingness(df: pd.DataFrame) -> int:
    rates = df.isnull().mean().sort_values(ascending=False)
    logger.info(f"Null rates (columns with any missing):\n{rates[rates > 0].to_string()}")
    flagged = rates[rates > MISSINGNESS_WARN_THRESHOLD]
    for col, rate in flagged.items():
        logger.warning(f"High missingness: {col} is {rate:.1%} null")
    return len(flagged)


def _plot_numeric_distributions(df: pd.DataFrame, output_path: Path):
    n_cols = 4
    n_rows = int(np.ceil(len(df.columns) / n_cols))
    _, ax = plt.subplots(n_rows, n_cols, figsize=(15, 18), sharey=False)
    for i, col in enumerate(df.columns):
        r, c = i // n_cols, i % n_cols
        ax[r, c].hist(
            df[col],
            bins=30,
            histtype="stepfilled",
            facecolor=mcolors.to_rgba("purple", alpha=0.5),
            edgecolor="purple",
            linewidth=2,
        )
        ax[r, c].set_title(f"{col}")
        if c == 0:
            ax[r, c].set_ylabel("Counts")
        for axs in ax.flatten()[len(df.columns) :]:
            axs.set_visible(False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "qc_distributions.png", dpi=150)
    plt.close()


def _plot_numeric_correlations(df: pd.DataFrame, output_path: Path):
    corr = df.corr()
    n = len(corr)
    _, ax = plt.subplots(figsize=(11, 10))
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["orange", "white", "purple"])
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "qc_correlations.png", dpi=150)
    plt.close()


def _check_distributions(df: pd.DataFrame) -> int:
    numeric = df.select_dtypes("number").drop(
        columns=["id", "day_in_study", "cycle_id", "wear_minutes"], errors="ignore"
    )
    pct = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2)
    logger.info(f"Percentiles (p05 / p25 / p50 / p75 / p95):\n{pct.T.to_string()}")

    _plot_numeric_distributions(numeric, FIGURES_DIR)
    _plot_numeric_correlations(numeric, FIGURES_DIR)

    issues = 0
    for col, (lo, hi) in MEDIAN_RANGES.items():
        if col not in df.columns:
            continue
        median = df[col].median()
        if not (lo <= median <= hi):
            logger.warning(f"{col}: median {median:.2f} outside expected [{lo}, {hi}]")
            issues += 1
    return issues


def _check_coverage(df: pd.DataFrame) -> int:
    issues = 0
    n_subjects = df["id"].nunique()

    day_counts = df.groupby("id")["day_in_study"].nunique()
    logger.info(
        f"Subjects: {n_subjects} | Days per subject — "
        f"min:{day_counts.min()} median:{day_counts.median():.0f} max:{day_counts.max()}"
    )

    if "cycle_id" in df.columns:
        cycles = df.groupby("id")["cycle_id"].nunique()
        no_cycles = n_subjects - len(cycles)
        if len(cycles):
            logger.info(
                f"Cycles per subject — min:{cycles.min()} "
                f"median:{cycles.median():.0f} max:{cycles.max()}"
            )
        cycle_days = df[df["is_full_cycle"]].groupby(["id", "cycle_id"])["day_in_study"].count()
        if len(cycle_days):
            logger.info(
                f"Days per cycle per subject — min:{cycle_days.min()} "
                f"median:{cycle_days.median():.0f} max:{cycle_days.max()}"
            )
        if no_cycles:
            logger.warning(f"{no_cycles} subjects have no complete cycles")
            issues += 1

    return issues


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main() -> None:
    logger.info("Starting QC pipeline...")

    path = PROCESSED_DATA_DIR / "interday.csv"
    if not path.exists():
        logger.error(f"Interday file not found at {path} — run 'dataset.py' first")
        raise typer.Exit(code=1)

    df = pd.read_csv(path)
    logger.info(
        f"Loaded {path.name}: {df['id'].nunique()} subjects, "
        f"{len(df):,} rows, {df.shape[1]} columns"
    )

    before = len(df)
    df = df.dropna(subset=["id", "day_in_study", "phase"])
    logger.info(f"Filtering for null id, day_in_study, phase: removed {before - len(df)} days")

    df = _filter_wear_time(df)
    df = _filter_min_days(df)
    logger.info(f"After filtering: {df['id'].nunique()} subjects, {len(df):,} rows")

    issues = 0
    issues += _check_missingness(df)
    issues += _check_distributions(df)
    issues += _check_coverage(df)

    out_path = PROCESSED_DATA_DIR / "interday_qc.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Saved filtered interday: {out_path} ({len(df):,} rows)")

    if issues:
        logger.warning(f"QC finished — {issues} warning(s)")
        raise typer.Exit(code=1)
    logger.success("QC finished — no issues")


if __name__ == "__main__":
    app()
