"""Feature engineering pipeline for cycle tracking."""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

INTERDAY_INPUT = PROCESSED_DATA_DIR / "interday_qc.csv"
OUTPUT_FITBIT = PROCESSED_DATA_DIR / "interday_fitbit.csv"
OUTPUT_SELFREPORTS = PROCESSED_DATA_DIR / "interday_selfreports.csv"

SHARED_FEATURES = [
    "id",
    "day_in_study",
    "phase",
    "phase_id",
    "cycle_id",
    "is_full_cycle",
]

WEARABLE_FEATURES = [
    # "active_min_sedentary",
    "active_min_light",
    "active_min_moderate",
    "active_min_high",
    # "temperature",
    # "exercise_count",
    # "exercise_min",
    "hr",
    # "wear_minutes",
    "hrv_rmssd_mean",
    "hrv_rmssd_median",
    # "hrv_hf_mean",
    # "hrv_hf_median",
    # "hrv_lf_mean",
    # "hrv_lf_median",
    "hr_resting",
    # "sleep_onset_latency",
    "sleep_min_total",
    "sleep_min_deep",
    "sleep_min_light",
    "sleep_min_rem",
    # "sleep_min_wake",
    "sleep_efficiency",
    "step_count",
    "temperature_diff",
]

SELFREPORT_FEATURES = [
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
    "flow_volume",
]

# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle_day, cycle_pct, cycle_pct_bin, phase_dual to an interday dataframe."""
    df = df.copy().sort_values(["id", "day_in_study"]).reset_index(drop=True)
    df["phase_id"] = pd.Categorical(
        df["phase"], categories=["Menstrual", "Follicular", "Fertility", "Luteal"]
    ).codes

    conditions = [
        df["phase"] == "Menstrual",
        df["phase"] == "Fertility",
    ]
    choices = ["Follicular", "Luteal"]
    df["phase_dual"] = np.select(conditions, choices, default=df["phase"])

    df["cycle_day"] = df.groupby(["id", "cycle_id"]).cumcount()
    cycle_lengths = df.groupby(["id", "cycle_id"])["cycle_day"].max().rename("cycle_total_days")
    df = df.merge(cycle_lengths, on=["id", "cycle_id"])
    df["cycle_pct"] = ((df["cycle_day"] / df["cycle_total_days"]) * 100).round(1)
    df["cycle_pct_bin"] = pd.cut(
        df["cycle_pct"], bins=range(0, 105, 5), right=False, labels=range(0, 100, 5)
    )
    return df.drop(columns=["cycle_total_days"])


def reports_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flow_scale = [
        "Not at all",
        "Spotting / Very Light",
        "Light",
        "Somewhat Light",
        "Moderate",
        "Somewhat Heavy",
        "Heavy",
        "Very Heavy",
    ]
    df["flow_volume"] = (
        df["flow_volume"].map({v: i for i, v in enumerate(flow_scale)}).astype("Int64")
    )

    report_scale = [
        "Not at all",
        "Very Low/Little",
        "Low",
        "Moderate",
        "High",
        "Very High",
    ]
    scale_map = {v: i for i, v in enumerate(report_scale)}
    for col in SELFREPORT_FEATURES:
        if col != "flow_volume" and col in df.columns:
            df[col] = df[col].map(scale_map).astype("Int64")
    return df


def add_rolling_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    new_cols = {}
    for lag in lags:
        for col in cols:
            if col in df.columns:
                grouped = df.groupby("id")[col]
                new_cols[col + "_rm_" + str(lag)] = grouped.transform(
                    lambda x, w=lag: x.rolling(window=w).mean()
                )
                new_cols[col + "_rs_" + str(lag)] = grouped.transform(
                    lambda x, w=lag: x.rolling(window=w).std()
                )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    interday_input: Path = INTERDAY_INPUT,
    output_fitbit: Path = OUTPUT_FITBIT,
    output_selfreports: Path = OUTPUT_SELFREPORTS,
) -> None:
    logger.info("Starting feature engineering...")

    df = pd.read_csv(interday_input)
    logger.info(f"Loaded {interday_input.name}: {df['id'].nunique()} subjects, {len(df):,} rows")

    df = add_cycle_features(df)

    fitbit = df[SHARED_FEATURES + WEARABLE_FEATURES].copy()
    fitbit = add_rolling_features(fitbit, WEARABLE_FEATURES, lags=[5, 7, 14])

    selfreports = df[SHARED_FEATURES + SELFREPORT_FEATURES].copy()
    selfreports = reports_to_numeric(selfreports)
    selfreports = add_rolling_features(selfreports, SELFREPORT_FEATURES, lags=[5, 7, 14])

    output_fitbit.parent.mkdir(parents=True, exist_ok=True)
    fitbit.to_csv(output_fitbit, index=False)
    logger.success(
        f"Saved fitbit features: {output_fitbit} ({len(fitbit):,} rows, {fitbit.shape[1]} columns)"
    )

    selfreports.to_csv(output_selfreports, index=False)
    logger.success(
        f"Saved selfreport features: {output_selfreports} ({len(selfreports):,} rows, {selfreports.shape[1]} columns)"
    )


if __name__ == "__main__":
    app()
