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
OUTPUT_COMBINED = PROCESSED_DATA_DIR / "interday_combined.csv"

SHARED_FEATURES = [
    "id",
    "day_in_study",
    "phase",
    "phase_label",
    "fertility_label",
    "cycle_id",
    "is_full_cycle",
]

WEARABLE_FEATURES = [
    # "active_min_sedentary",
    "active_min_light",
    "active_min_moderate",
    "active_min_high",
    # "exercise_count",
    # "exercise_min",
    "hr",
    # "wear_min",
    # "wear_min_active",
    "hrv_rmssd_mean",
    # "hrv_rmssd_median",
    "hrv_hf_mean",
    # "hrv_hf_median",
    "hrv_lf_mean",
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
    "temperature",
    "temperature_diff",
    "respiratory_rate",
    "vo2_max",
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

FEATURE_GROUPS: dict[str, list[str]] = {
    "Heart": ["hr", "hr_resting", "hrv_rmssd_mean", "hrv_hf_mean", "hrv_lf_mean", "lf_hf_ratio"],
    "Sleep": [
        "sleep_min_total",
        "sleep_min_deep",
        "sleep_min_light",
        "sleep_min_rem",
        "sleep_efficiency",
    ],
    "Activity": ["active_min_light", "active_min_moderate", "active_min_high", "step_count"],
    "Body": ["temperature", "temperature_diff", "respiratory_rate", "vo2_max"],
    "Menstrual": ["cramps", "flow_volume", "sorebreasts"],
    "Stomach": ["appetite", "foodcravings", "indigestion", "bloating"],
    "Symptoms": ["headaches", "fatigue", "sleepissue", "moodswing", "stress", "exerciselevel"],
    "Position": ["sin", "cos"],
}

# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["id", "day_in_study"]).reset_index(drop=True)
    df["phase_label"] = pd.Categorical(
        df["phase"], categories=["Menstrual", "Follicular", "Fertility", "Luteal"]
    ).codes

    df["fertility_label"] = (df["phase"] == "Fertility").astype("Int64")

    # conditions = [
    #     df["phase"] == "Menstrual",
    #     df["phase"] == "Fertility",
    # ]
    # choices = ["Follicular", "Luteal"]
    # df["phase_dual"] = np.select(conditions, choices, default=df["phase"])

    # df["cycle_day"] = df.groupby(["id", "cycle_id"]).cumcount()
    # cycle_lengths = df.groupby(["id", "cycle_id"])["cycle_day"].max().rename("cycle_total_days")
    # df = df.merge(cycle_lengths, on=["id", "cycle_id"])
    # df["cycle_pct"] = ((df["cycle_day"] / df["cycle_total_days"]) * 100).round(1)
    # df["cycle_pct_bin"] = pd.cut(
    #     df["cycle_pct"], bins=range(0, 105, 5), right=False, labels=range(0, 100, 5)
    # )
    # return df.drop(columns=["cycle_total_days"])
    return df


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
            df.loc[df[col] == "Very Low", col] = "Very Low/Little"
            df[col] = df[col].map(scale_map).astype("Int64")
    return df


def impute_ffill(df: pd.DataFrame, cols: list, limit: int) -> pd.DataFrame:
    df = df.sort_values(["id", "day_in_study"])
    df[cols] = df.groupby("id")[cols].ffill(limit=limit)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    new_cols = {}
    for lag in lags:
        for col in cols:
            if col in df.columns:
                grouped = df.groupby("id")[col]
                new_cols[col + "_rm_" + str(lag)] = grouped.transform(
                    lambda x, w=lag: x.rolling(window=w, min_periods=1).mean()
                )
                new_cols[col + "_rs_" + str(lag)] = grouped.transform(
                    lambda x, w=lag: x.rolling(window=w, min_periods=1).std()
                )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_fitbit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lf_hf_ratio"] = df["hrv_lf_mean"] / df["hrv_hf_mean"]
    return df


def add_cycle_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "day_in_study"]).copy()
    mask = df["flow_volume"].ffill() >= 3
    shifted1 = df.groupby("id")["flow_volume"].ffill().shift(-1) >= 2
    consec = mask & shifted1
    consec_start = consec & ~consec.shift(1)
    df["group"] = consec_start.groupby(df["id"]).cumsum().ffill(limit=1)
    regular = df.groupby(["id", "group"]).cumcount().astype("Int64")
    reverse = 28 - df.groupby(["id", "group"]).cumcount(ascending=False).astype("Int64")
    df["cycle_day"] = regular
    df.loc[df["group"] == 0, "cycle_day"] = reverse[df["group"] == 0]

    df["sin"] = np.sin((2 * np.pi * df["cycle_day"]) / 28)
    df["cos"] = np.cos((2 * np.pi * df["cycle_day"]) / 28)
    return df.drop(columns=["group", "cycle_day"])


def z_score(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df.groupby("id")[col].transform(lambda x: (x - x.mean()) / x.std())
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    interday_input: Path = INTERDAY_INPUT,
    output_fitbit: Path = OUTPUT_FITBIT,
    output_selfreports: Path = OUTPUT_SELFREPORTS,
    output_combined: Path = OUTPUT_COMBINED,
) -> None:
    logger.info("Starting feature engineering...")

    df = pd.read_csv(interday_input)
    logger.info(f"Loaded {interday_input.name}: {df['id'].nunique()} subjects, {len(df):,} rows")

    df = add_cycle_features(df)

    fitbit = df[SHARED_FEATURES + WEARABLE_FEATURES].copy()
    fitbit = z_score(fitbit, WEARABLE_FEATURES)
    fitbit = impute_ffill(fitbit, WEARABLE_FEATURES, 3)
    fitbit = add_rolling_features(fitbit, WEARABLE_FEATURES, lags=[7, 14])
    fitbit = add_fitbit_features(fitbit)

    selfreports = df[SHARED_FEATURES + SELFREPORT_FEATURES].copy()
    selfreports = reports_to_numeric(selfreports)
    selfreports = impute_ffill(selfreports, SELFREPORT_FEATURES, 1)
    selfreports = add_rolling_features(selfreports, SELFREPORT_FEATURES, lags=[7, 14])
    selfreports = add_cycle_encoding(selfreports)

    output_fitbit.parent.mkdir(parents=True, exist_ok=True)
    fitbit.to_csv(output_fitbit, index=False)
    logger.success(
        f"Saved fitbit features: {output_fitbit} ({len(fitbit):,} rows, {fitbit.shape[1]} columns)"
    )

    selfreports.to_csv(output_selfreports, index=False)
    logger.success(
        f"Saved selfreport features: {output_selfreports} ({len(selfreports):,} rows, {selfreports.shape[1]} columns)"
    )

    combined = pd.merge(fitbit, selfreports, on=SHARED_FEATURES)
    combined.to_csv(output_combined, index=False)
    logger.success(
        f"Saved selfreport features: {output_combined} ({len(combined):,} rows, {combined.shape[1]} columns)"
    )


if __name__ == "__main__":
    app()
