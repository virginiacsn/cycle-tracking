"""Data loading and preprocessing pipeline for cycle tracking."""

import ast

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

STUDY_INTERVAL = 2022
MIN_INTRADAY_HOURS = 18
RESAMPLE_MINUTES = 5
MIN_INTERVALS = int(MIN_INTRADAY_HOURS * 60 / RESAMPLE_MINUTES)  # 216 five-minute slots
SLOTS_PER_DAY = int(24 * 60 / RESAMPLE_MINUTES)  # 288
MAX_CONSEC_HORMONE_MISSING = 4
MAX_FRACTION_HORMONE_MISSING = 0.40
MIN_DAYS_PER_SUBJECT = 30

# Intraday files and the value columns to keep from each
INTRADAY_FILES: dict[str, tuple[str, ...]] = {
    "heart_rate": ("bpm",),
    "glucose": ("glucose_value",),
}

MERGE_KEYS = ["id", "study_interval", "day_in_study", "datetime", "time"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime column from day_in_study + HH:MM:SS timestamp."""
    df = df.copy()
    ref = pd.Timestamp("2022-01-01")
    df["datetime"] = (
        ref + pd.to_timedelta(df["day_in_study"] - 1, unit="D") + pd.to_timedelta(df["timestamp"])
    )
    return df


def _full_day_range(day: int) -> pd.DatetimeIndex:
    """96 fifteen-minute slots covering calendar day `day` (1-indexed)."""
    start = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(day) - 1)
    return pd.date_range(start, periods=SLOTS_PER_DAY, freq=f"{RESAMPLE_MINUTES}min")


def _resample_intraday(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Resample intraday df to 15-min intervals within a full 24h grid per (id, day)."""
    results = []
    groups = df.groupby(["id", "study_interval", "day_in_study"])
    for (sid, study_int, day), grp in tqdm(groups, desc="Resampling", leave=False):
        is_weekend = grp["is_weekend"].iloc[0] if "is_weekend" in grp.columns else None
        resampled = grp.set_index("datetime")[value_cols].resample(f"{RESAMPLE_MINUTES}min").mean()
        resampled = resampled.reindex(_full_day_range(day))
        resampled = resampled.reset_index().rename(columns={"index": "datetime"})
        resampled["id"] = sid
        resampled["study_interval"] = study_int
        resampled["day_in_study"] = day
        if is_weekend is not None:
            resampled["is_weekend"] = is_weekend
        resampled["time"] = resampled["datetime"].dt.strftime("%H:%M")
        resampled["datetime"] = resampled["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        results.append(resampled)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Interday loaders
# ---------------------------------------------------------------------------


def _load_hormones() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "hormones_and_selfreport.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = df.drop(columns=["pdg"], errors="ignore")
    for col in ("lh", "estrogen"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["id", "day_in_study"]).reset_index(drop=True)


def _identify_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle_id column; increments each time the Menstrual phase begins."""
    df = df.copy()
    df["cycle_id"] = 0
    for sid, grp in df.groupby("id"):
        cycle, prev, ids = 0, None, []
        for phase in grp["phase"]:
            if phase == "Menstrual" and prev != "Menstrual":
                cycle += 1
            ids.append(cycle)
            prev = phase
        df.loc[grp.index, "cycle_id"] = ids
    return df


def filter_hormone_cycles(df: pd.DataFrame) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    """Remove cycles with >= 4 consecutive missing hormone days or > 40% missing.

    Returns the filtered df and the set of valid (id, day_in_study) pairs.
    """
    df = _identify_cycles(df)
    df["_missing"] = df["lh"].isna() & df["estrogen"].isna()

    bad: set[tuple] = set()
    for (sid, cid), grp in df.groupby(["id", "cycle_id"]):
        missing = grp["_missing"].values
        n = len(missing)
        if missing.sum() / n > MAX_FRACTION_HORMONE_MISSING:
            bad.add((sid, cid))
            continue
        consec = max_consec = 0
        for m in missing:
            consec = consec + 1 if m else 0
            max_consec = max(max_consec, consec)
        if max_consec >= MAX_CONSEC_HORMONE_MISSING:
            bad.add((sid, cid))

    is_bad = df.apply(lambda r: (r["id"], r["cycle_id"]) in bad, axis=1)
    df_filtered = df[~is_bad].drop(columns=["_missing", "cycle_id"])
    logger.info(f"Removed {len(bad)} cycles for incomplete hormone data")

    valid_days: set[tuple[int, int]] = set(zip(df_filtered["id"], df_filtered["day_in_study"]))
    return df_filtered, valid_days


def _parse_sleep_stages(row: pd.Series) -> pd.Series:
    try:
        lvl = ast.literal_eval(row["levels"])
        summary = lvl.get("summary", {})
    except (ValueError, SyntaxError):
        summary = {}

    if row["type"] == "stages":
        return pd.Series(
            {
                "minutes_deep": summary.get("deep", {}).get("minutes"),
                "minutes_light": summary.get("light", {}).get("minutes"),
                "minutes_rem": summary.get("rem", {}).get("minutes"),
                "minutes_wake": summary.get("wake", {}).get("minutes"),
            }
        )
    return pd.Series(
        {"minutes_deep": None, "minutes_light": None, "minutes_rem": None, "minutes_wake": None}
    )


def _load_sleep() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "sleep.csv")
    df = df[(df["study_interval"] == STUDY_INTERVAL) & (df["mainsleep"] == True)].copy()  # noqa: E712
    stages = df.apply(_parse_sleep_stages, axis=1)
    df = pd.concat([df.reset_index(drop=True), stages], axis=1)
    keep = [
        "id",
        "sleep_start_day_in_study",
        "minutesasleep",
        "minutes_deep",
        "minutes_light",
        "minutes_rem",
        "minutes_wake",
    ]
    df = df[keep].rename(
        columns={
            "sleep_start_day_in_study": "day_in_study",
            "minutesasleep": "sleep_min_total",
            "minutes_deep": "sleep_min_deep",
            "minutes_light": "sleep_min_light",
            "minutes_rem": "sleep_min_rem",
            "minutes_wake": "sleep_min_wake",
        }
    )
    stage_cols = ["sleep_min_deep", "sleep_min_light", "sleep_min_rem", "sleep_min_wake"]
    df[stage_cols] = df[stage_cols].fillna(0)
    # Some days have two mainsleep records (nap + overnight); keep the longer one.
    return df.sort_values("sleep_min_total", ascending=False).drop_duplicates(
        subset=["id", "day_in_study"], keep="first"
    )


def _load_computed_temperature() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "computed_temperature.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = df[
        [
            "id",
            "sleep_start_day_in_study",
            "temperature_samples",
            "nightly_temperature",
            "baseline_relative_sample_sum",
        ]
    ].rename(
        columns={
            "sleep_start_day_in_study": "day_in_study",
            "baseline_relative_sample_sum": "temperature_deviation",
        }
    )
    # Same-day duplicates arise from nap + overnight sessions; keep the better-sampled one.
    df = df.sort_values("temperature_samples", ascending=False).drop_duplicates(
        subset=["id", "day_in_study"], keep="first"
    )
    return df.drop(columns=["temperature_samples"])


def _load_hrv() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "heart_rate_variability_details.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df["rmssd"] = pd.to_numeric(df["rmssd"], errors="coerce")
    df = df.rename(columns={"rmssd": "heart_rate_var"})
    return (
        df.groupby(["id", "day_in_study"])["heart_rate_var"]
        .agg(heart_rate_var_mean="mean", heart_rate_var_min="min", heart_rate_var_max="max")
        .reset_index()
    )


def _compute_first_sleep_day() -> dict[int, int]:
    """Return {id: first sleep_start_day_in_study} as per-subject start filter."""
    df = pd.read_csv(RAW_DATA_DIR / "sleep.csv")
    df = df[(df["study_interval"] == STUDY_INTERVAL) & (df["mainsleep"] == True)]  # noqa: E712
    return df.groupby("id")["sleep_start_day_in_study"].min().to_dict()


def build_interday(first_sleep_days: dict[int, int]) -> pd.DataFrame:
    """Assemble interday CSV from all interday sources."""
    hormones_raw = _load_hormones()
    hormones, _ = filter_hormone_cycles(hormones_raw)

    sleep = _load_sleep()
    temp = _load_computed_temperature()
    hrv = _load_hrv()

    rhr = pd.read_csv(RAW_DATA_DIR / "resting_heart_rate.csv")
    rhr = rhr[rhr["study_interval"] == STUDY_INTERVAL][["id", "day_in_study", "value"]].rename(
        columns={"value": "resting_heart_rate"}
    )

    active_min = pd.read_csv(RAW_DATA_DIR / "active_minutes.csv")
    active_min = active_min[active_min["study_interval"] == STUDY_INTERVAL][
        ["id", "day_in_study", "sedentary", "lightly", "moderately", "very"]
    ].rename(
        columns={
            "sedentary": "active_min_sedentary",
            "lightly": "active_min_light",
            "moderately": "active_min_moderate",
            "very": "active_min_high",
        }
    )

    interday = hormones
    for other in [sleep, temp, hrv, rhr, active_min]:
        interday = interday.merge(other, on=["id", "day_in_study"], how="left")

    before = len(interday)
    min_day = interday["id"].map(first_sleep_days)
    interday = interday[interday["day_in_study"] >= min_day].reset_index(drop=True)
    logger.info(f"Removed {before - len(interday)} interday rows before first sleep day")

    day_counts = interday.groupby("id")["day_in_study"].nunique()
    valid_subjects = day_counts[day_counts >= MIN_DAYS_PER_SUBJECT].index
    before = interday["id"].nunique()
    interday = interday[interday["id"].isin(valid_subjects)].reset_index(drop=True)
    logger.info(
        f"Removed {before - len(valid_subjects)} subjects with < {MIN_DAYS_PER_SUBJECT} days"
    )

    first_cols = ["id", "day_in_study", "is_weekend"]
    rest = [c for c in interday.columns if c not in first_cols + ["study_interval"]]
    return interday[first_cols + rest]


# ---------------------------------------------------------------------------
# Intraday loaders
# ---------------------------------------------------------------------------


def _load_intraday_file(name: str, value_cols: tuple[str, ...]) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"{name}.csv"
    logger.info(f"Loading {name}...")
    df = pd.read_csv(path)
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    cols = ["id", "study_interval", "day_in_study", "is_weekend", "timestamp"] + list(value_cols)
    return _make_datetime(df[cols])


def _load_active_zone_minutes() -> pd.DataFrame:
    logger.info("Loading active_zone_minutes...")
    df = pd.read_csv(RAW_DATA_DIR / "active_zone_minutes.csv")
    df = df[df["study_interval"] == STUDY_INTERVAL].copy()
    df = _make_datetime(df)
    pivoted = df.pivot_table(
        index=["id", "study_interval", "day_in_study", "datetime"],
        columns="heart_zone_id",
        values="total_minutes",
        aggfunc="sum",
    ).reset_index()
    pivoted.columns.name = None
    for zone, col in [
        ("CARDIO", "azm_cardio"),
        ("FAT_BURN", "azm_fat_burn"),
        ("PEAK", "azm_peak"),
    ]:
        if zone in pivoted.columns:
            pivoted = pivoted.rename(columns={zone: col})
        else:
            pivoted[col] = float("nan")
    return pivoted


def _filter_to_valid_days(df: pd.DataFrame, valid_days: set[tuple[int, int]]) -> pd.DataFrame:
    mask = pd.Series(list(zip(df["id"], df["day_in_study"])), index=df.index).isin(valid_days)
    return df[mask]


def build_intraday(
    valid_days: set[tuple[int, int]], first_sleep_days: dict[int, int]
) -> pd.DataFrame:
    """Load, filter, resample, merge, and interpolate all intraday sources."""
    sleep_filtered = {(sid, day) for sid, day in valid_days if day >= first_sleep_days.get(sid, 0)}
    n_pre_sleep = len(valid_days) - len(sleep_filtered)
    if n_pre_sleep:
        logger.info(f"Filtered {n_pre_sleep} subject-days before first sleep day from intraday")
    valid_days = sleep_filtered

    parts: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    for name, value_cols in INTRADAY_FILES.items():
        df = _load_intraday_file(name, value_cols)
        df = _filter_to_valid_days(df, valid_days)
        df = _resample_intraday(df, list(value_cols))
        parts[name] = (df, list(value_cols))

    azm_df = _load_active_zone_minutes()
    azm_df = _filter_to_valid_days(azm_df, valid_days)
    azm_cols = [c for c in azm_df.columns if c.startswith("azm_")]
    azm_df = _resample_intraday(azm_df, azm_cols)
    parts["active_zone_minutes"] = (azm_df, azm_cols)

    # Merge all sources on MERGE_KEYS; heart_rate is the base
    merged, _ = parts["heart_rate"]
    for name, (df, _) in parts.items():
        if name == "heart_rate":
            continue
        val_cols = [c for c in df.columns if c not in MERGE_KEYS and c != "is_weekend"]
        merged = merged.merge(df[MERGE_KEYS + val_cols], on=MERGE_KEYS, how="outer")

    # Filter days with < 18h intraday coverage (bpm as reference signal)
    coverage = merged.groupby(["id", "day_in_study"])["bpm"].count()
    valid_cov = coverage[coverage >= MIN_INTERVALS].reset_index()[["id", "day_in_study"]]
    before = merged[["id", "day_in_study"]].drop_duplicates().shape[0]
    merged = merged.merge(valid_cov, on=["id", "day_in_study"])
    after = merged[["id", "day_in_study"]].drop_duplicates().shape[0]
    logger.info(
        f"Removed {before - after} subject-days for < {MIN_INTRADAY_HOURS}h intraday coverage"
    )

    interp_cols = [c for c in merged.columns if c in ("bpm", "glucose_value")]
    zero_fill_cols = [c for c in merged.columns if c.startswith("azm_")]
    interpolated = []
    for (sid, day), grp in tqdm(
        merged.groupby(["id", "day_in_study"]), desc="Interpolating", leave=False
    ):
        grp = grp.sort_values("datetime").copy()
        grp[interp_cols] = grp[interp_cols].interpolate(method="linear")
        grp[zero_fill_cols] = grp[zero_fill_cols].fillna(0)
        interpolated.append(grp)

    result = pd.concat(interpolated, ignore_index=True)

    input_subjects = {sid for sid, _ in valid_days}
    day_counts = result.groupby("id")["day_in_study"].nunique()
    valid_subjects = day_counts[day_counts >= MIN_DAYS_PER_SUBJECT].index
    result = result[result["id"].isin(valid_subjects)].reset_index(drop=True)
    logger.info(
        f"Removed {len(input_subjects) - len(valid_subjects)} subjects with < {MIN_DAYS_PER_SUBJECT} intraday days (coverage filter)"
    )

    first_cols = ["id", "day_in_study", "is_weekend", "datetime", "time"]
    rest = [c for c in result.columns if c not in first_cols + ["study_interval"]]
    return result[first_cols + rest]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main() -> None:
    logger.info("Starting data pipeline...")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    first_sleep_days = _compute_first_sleep_day()

    logger.info("Processing interday data...")
    hormones_raw = _load_hormones()
    _, valid_days = filter_hormone_cycles(hormones_raw)
    interday = build_interday(first_sleep_days)

    logger.info("Processing intraday data...")
    valid_subjects = set(interday["id"].unique())
    valid_days = {(sid, day) for sid, day in valid_days if sid in valid_subjects}
    intraday = build_intraday(valid_days, first_sleep_days)

    final_subjects = set(intraday["id"].unique())
    if final_subjects != valid_subjects:
        dropped = len(valid_subjects) - len(final_subjects)
        logger.info(f"Dropping {dropped} subjects from interday lost in intraday coverage filter")
        interday = interday[interday["id"].isin(final_subjects)].reset_index(drop=True)

    interday_path = PROCESSED_DATA_DIR / "interday.csv"
    interday.to_csv(interday_path, index=False)
    logger.success(f"Saved interday: {interday_path} ({len(interday):,} rows)")

    intraday_path = PROCESSED_DATA_DIR / "intraday.csv"
    intraday.to_csv(intraday_path, index=False)
    logger.success(f"Saved intraday: {intraday_path} ({len(intraday):,} rows)")


if __name__ == "__main__":
    app()
