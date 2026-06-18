"""Data loading and preprocessing pipeline for cycle tracking."""

import ast

from loguru import logger
import pandas as pd
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

STUDY_INTERVAL = 2022

MERGE_KEYS = ["id", "day_in_study"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def deduplicate_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df.drop_duplicates(subset=["id", "study_interval", "day_in_study", "timestamp"])
    return df.drop_duplicates()


def _make_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime column from day_in_study + HH:MM:SS timestamp."""
    df = df.copy()
    ref = pd.Timestamp("2022-01-01")
    df["datetime"] = (
        ref + pd.to_timedelta(df["day_in_study"] - 1, unit="D") + pd.to_timedelta(df["timestamp"])
    )
    return df


# ---------------------------------------------------------------------------
# Interday loaders
# ---------------------------------------------------------------------------


def _identify_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle_id column; increments each time the Menstrual phase begins."""
    df = df.copy()
    is_menstrual = df["phase"] == "Menstrual"
    prev_is_menstrual = is_menstrual.groupby(df["id"]).shift(1).fillna(False).astype(bool)
    cycle_start = is_menstrual & ~prev_is_menstrual
    df["cycle_id"] = cycle_start.groupby(df["id"]).cumsum()
    df["is_full_cycle"] = df.groupby(["id", "cycle_id"])["phase"].transform(
        lambda x: {"Menstrual", "Follicular", "Fertility", "Luteal"}.issubset(x)
    )
    return df


def _load_active_minutes() -> pd.DataFrame:
    logger.info("Processing active_minutes.csv")
    df = pd.read_csv(RAW_DATA_DIR / "active_minutes.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = df[["id", "day_in_study", "sedentary", "lightly", "moderately", "very"]].rename(
        columns={
            "sedentary": "active_min_sedentary",
            "lightly": "active_min_light",
            "moderately": "active_min_moderate",
            "very": "active_min_high",
        }
    )
    # Transform to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Check for implausible values
    check = [
        "active_min_sedentary",
        "active_min_light",
        "active_min_moderate",
        "active_min_high",
    ]
    for col in check:
        df.loc[(df[col] < 0), col] = pd.NA
    return df


def _load_computed_temperature() -> pd.DataFrame:
    logger.info("Processing computed_temperature.csv")
    df = pd.read_csv(RAW_DATA_DIR / "computed_temperature.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = deduplicate_by_timestamp(df)
    df = df[
        [
            "id",
            "sleep_start_day_in_study",
            "temperature_samples",
            "nightly_temperature",
        ]
    ].rename(
        columns={
            "sleep_start_day_in_study": "day_in_study",
            "nightly_temperature": "temperature",
        }
    )
    # Transform to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Check for implausible values
    df.loc[(df["temperature"] < 25) | (df["temperature"] > 45), "temperature"] = pd.NA
    # Same-day duplicates arise from nap + overnight sessions; keep the better-sampled one
    df = df.sort_values("temperature_samples", ascending=False).drop_duplicates(
        subset=["id", "day_in_study"], keep="first"
    )
    return df.drop(columns=["temperature_samples"])


def _load_demographic_vo2_max() -> pd.DataFrame:
    logger.info("Processing demographic_vo2_max.csv")
    df = pd.read_csv(RAW_DATA_DIR / "demographic_vo2_max.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = df[["id", "day_in_study", "demographic_vo2_max"]].rename(
        columns={"demographic_vo2_max": "vo2_max"}
    )
    df["vo2_max"] = pd.to_numeric(df["vo2_max"], errors="coerce")
    df.loc[(df["vo2_max"] < 10) | (df["vo2_max"] > 80), "vo2_max"] = pd.NA
    return df


def _load_exercise() -> pd.DataFrame:
    logger.info("Processing exercise.csv")
    df = pd.read_csv(RAW_DATA_DIR / "exercise.csv")
    df = (
        df[df["study_interval"] == STUDY_INTERVAL]
        .copy()
        .rename(columns={"start_day_in_study": "day_in_study"})
    )
    deduplicate_cols = [
        "id",
        "study_interval",
        "day_in_study",
        "start_timestamp",
        "activitytypeid",
        "duration",
    ]
    df = df.drop_duplicates(subset=deduplicate_cols)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    # Raw duration is in milliseconds; divide by 60_000 to get minutes
    df["duration"] = df["duration"] / 60000.0
    df.loc[(df["duration"] < 0), "duration"] = pd.NA
    return (
        df.groupby(["id", "day_in_study"])
        .agg(exercise_count=("duration", "count"), exercise_min=("duration", "sum"))
        .reset_index()
    )


def _load_heart_rate() -> pd.DataFrame:
    logger.info("Processing heart_rate.csv")
    df = pd.read_csv(RAW_DATA_DIR / "heart_rate.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = deduplicate_by_timestamp(df)
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df.loc[(df["bpm"] < 30) | (df["bpm"] > 220), "bpm"] = pd.NA

    df = _make_datetime(df)
    # Resample to 5-min bins; non-null bin count × 5 later estimates minutes of device wear
    df = (
        df.set_index("datetime")
        .groupby(["id", "day_in_study"])["bpm"]
        .resample("5min")
        .mean()
        .reset_index()
    )
    df = (
        df.groupby(["id", "day_in_study"])["bpm"]
        .agg(hr="mean", wear_minutes="count")
        .reset_index()
    )
    df["wear_minutes"] = df["wear_minutes"] * 5
    return df


def _load_heart_rate_variability() -> pd.DataFrame:
    logger.info("Processing heart_rate_variability_details.csv")
    df = pd.read_csv(RAW_DATA_DIR / "heart_rate_variability_details.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = deduplicate_by_timestamp(df)
    numeric_cols = ["rmssd", "high_frequency", "low_frequency"]
    # Transform to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Check for implausible values
    check = ["rmssd"]
    for col in check:
        df.loc[(df[col] <= 0), col] = pd.NA
    df = (
        df.groupby(["id", "day_in_study"])
        .agg(
            rmssd_mean=("rmssd", "mean"),
            rmssd_median=("rmssd", "median"),
            hf_mean=("high_frequency", "mean"),
            hf_median=("high_frequency", "median"),
            lf_mean=("low_frequency", "mean"),
            lf_median=("low_frequency", "median"),
        )
        .reset_index()
        .rename(
            columns={
                "rmssd_mean": "hrv_rmssd_mean",
                "rmssd_median": "hrv_rmssd_median",
                "hf_median": "hrv_hf_median",
                "hf_mean": "hrv_hf_mean",
                "lf_median": "hrv_lf_median",
                "lf_mean": "hrv_lf_mean",
            }
        )
    )
    return df


def _load_hormones_and_selfreports() -> pd.DataFrame:
    logger.info("Processing hormones_and_selfreport.csv")
    df = pd.read_csv(RAW_DATA_DIR / "hormones_and_selfreport.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = df.drop(columns=["study_interval", "is_weekend", "pdg"])
    # Transform to numeric
    for col in ["estrogen", "lh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Check for implausible values
    for col in ["estrogen", "lh"]:
        df.loc[(df[col] <= 0), col] = pd.NA
    return df


def _load_respiratory_rate() -> pd.DataFrame:
    logger.info("Processing respiratory_rate_summary.csv")
    df = pd.read_csv(RAW_DATA_DIR / "respiratory_rate_summary.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = deduplicate_by_timestamp(df)
    df = df[["id", "day_in_study", "full_sleep_breathing_rate"]].rename(
        columns={"full_sleep_breathing_rate": "respiratory_rate"}
    )
    df["respiratory_rate"] = pd.to_numeric(df["respiratory_rate"], errors="coerce")
    df.loc[(df["respiratory_rate"] < 4) | (df["respiratory_rate"] > 30), "respiratory_rate"] = (
        pd.NA
    )
    return df


def _load_resting_heart_rate() -> pd.DataFrame:
    logger.info("Processing resting_heart_rate.csv")
    df = pd.read_csv(RAW_DATA_DIR / "resting_heart_rate.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = df[["id", "day_in_study", "value"]].rename(columns={"value": "hr_resting"})
    df["hr_resting"] = pd.to_numeric(df["hr_resting"], errors="coerce")
    df.loc[(df["hr_resting"] < 30) | (df["hr_resting"] > 140), "hr_resting"] = pd.NA
    return df


def _load_sleep() -> pd.DataFrame:
    logger.info("Processing sleep.csv")
    df = pd.read_csv(RAW_DATA_DIR / "sleep.csv")
    df = df[(df["study_interval"] == STUDY_INTERVAL) & (df["mainsleep"])].copy()
    # "stages" type has per-stage breakdown; "classic" (older devices) only records totals
    stages_mask = df["type"] == "stages"

    def _parse_summary(s):
        try:
            return ast.literal_eval(s).get("summary", {})
        except (ValueError, SyntaxError):
            return {}

    summaries = df.loc[stages_mask, "levels"].map(_parse_summary)
    stage_cols = (
        pd.json_normalize(summaries.tolist())
        .reindex(columns=["deep.minutes", "light.minutes", "rem.minutes", "wake.minutes"])
        .rename(
            columns={
                "deep.minutes": "minutes_deep",
                "light.minutes": "minutes_light",
                "rem.minutes": "minutes_rem",
                "wake.minutes": "minutes_wake",
            }
        )
        .set_index(summaries.index)
    )
    df = df.join(stage_cols)
    keep = [
        "id",
        "sleep_start_day_in_study",
        "minutestofallasleep",
        "minutesasleep",
        "minutes_deep",
        "minutes_light",
        "minutes_rem",
        "minutes_wake",
        "efficiency",
    ]
    df = df[keep].rename(
        columns={
            "sleep_start_day_in_study": "day_in_study",
            "minutestofallasleep": "sleep_onset_latency",
            "minutesasleep": "sleep_min_total",
            "minutes_deep": "sleep_min_deep",
            "minutes_light": "sleep_min_light",
            "minutes_rem": "sleep_min_rem",
            "minutes_wake": "sleep_min_wake",
            "efficiency": "sleep_efficiency",
        }
    )
    # Transform to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Check for implausible values
    check = ["sleep_min_total", "sleep_efficiency"]
    for col in check:
        df.loc[(df[col] < 0), col] = pd.NA
    # Some days have two mainsleep records (nap + overnight); keep the longer one
    return df.sort_values("sleep_min_total", ascending=False).drop_duplicates(
        subset=["id", "day_in_study"], keep="first"
    )


def _load_steps() -> pd.DataFrame:
    logger.info("Processing steps.csv")
    df = pd.read_csv(RAW_DATA_DIR / "steps.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = deduplicate_by_timestamp(df)
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    # Per-interval cap
    df.loc[(df["steps"] < 0) | (df["steps"] > 5_000), "steps"] = pd.NA
    df = (
        df.groupby(["id", "day_in_study"])["steps"]
        .sum()
        .reset_index()
        .rename(columns={"steps": "step_count"})
    )
    # Post-aggregation cap
    df.loc[df["step_count"] > 200_000, "step_count"] = pd.NA
    return df


def _load_wrist_temperature() -> pd.DataFrame:
    logger.info("Processing wrist_temperature.csv")
    df = pd.read_csv(RAW_DATA_DIR / "wrist_temperature.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = deduplicate_by_timestamp(df)
    df = df[["id", "day_in_study", "temperature_diff_from_baseline"]].rename(
        columns={"temperature_diff_from_baseline": "temperature_diff"}
    )
    # Transform to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Check for implausible values
    df.loc[(df["temperature_diff"] < -5) | (df["temperature_diff"] > 5), "temperature_diff"] = (
        pd.NA
    )
    return df.groupby(["id", "day_in_study"])["temperature_diff"].mean().reset_index()


def build_interday() -> pd.DataFrame:
    """Assemble interday CSV from all interday sources."""
    active_min = _load_active_minutes()
    temp = _load_computed_temperature()
    vo2_max = _load_demographic_vo2_max()
    exercise = _load_exercise()
    hr = _load_heart_rate()
    hrv = _load_heart_rate_variability()
    hormones_selfreport = _load_hormones_and_selfreports()
    rhr = _load_resting_heart_rate()
    resp = _load_respiratory_rate()
    sleep = _load_sleep()
    steps = _load_steps()
    temp_diff = _load_wrist_temperature()

    interday = hormones_selfreport
    for other in [
        active_min,
        temp,
        vo2_max,
        exercise,
        hr,
        hrv,
        rhr,
        resp,
        sleep,
        steps,
        temp_diff,
    ]:
        interday = interday.merge(other, on=MERGE_KEYS, how="left")

    interday = _identify_cycles(interday)

    first_cols = ["id", "day_in_study"]
    rest = [c for c in interday.columns if c not in first_cols + ["cycle_id", "is_full_cycle"]]
    rest.insert(rest.index("phase") + 1, "cycle_id")
    rest.insert(rest.index("cycle_id") + 1, "is_full_cycle")
    return interday[first_cols + rest]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main() -> None:
    logger.info("Starting data pipeline...")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Processing interday data...")
    interday = build_interday()

    interday_path = PROCESSED_DATA_DIR / "interday.csv"
    interday.to_csv(interday_path, index=False)
    logger.success(f"Saved interday: {interday_path} ({len(interday):,} rows)")

    n_subjects = interday["id"].nunique()
    n_cycles = interday[interday["is_full_cycle"]].groupby("id")["cycle_id"].nunique().sum()
    logger.success(f"Dataset summary: {n_subjects} subjects, {n_cycles} full cycles")


if __name__ == "__main__":
    app()
