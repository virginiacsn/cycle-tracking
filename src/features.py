"""Feature engineering pipeline for cycle tracking."""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

INTERDAY_INPUT = PROCESSED_DATA_DIR / "interday.csv"
INTRADAY_INPUT = PROCESSED_DATA_DIR / "intraday.csv"
INTERDAY_OUTPUT = PROCESSED_DATA_DIR / "features_interday.csv"
INTRADAY_OUTPUT = PROCESSED_DATA_DIR / "features_intraday.csv"


# ---------------------------------------------------------------------------
# Interday and intraday features
# ---------------------------------------------------------------------------


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle_id, cycle_day, and cycle_pct to an interday dataframe.

    cycle_id increments each time the Menstrual phase begins.
    cycle_day is the 1-indexed day within the current cycle.
    cycle_pct is cycle_day / total observed days in the cycle.

    Rows before the first Menstrual phase get cycle_id=0.
    """
    df = df.copy().sort_values(["id", "day_in_study"]).reset_index(drop=True)

    conditions = [
        df["phase"] == "Mentrual",
        df["phase"] == "Fertility",
    ]
    choices = ["Follicular", "Luteal"]
    df["phase_dual"] = np.select(conditions, choices, default=df["phase"])

    for sid, grp in df.groupby("id", sort=True):
        cycle = 0
        c_day = 0
        prev = None
        c_ids: list[int] = []
        c_days: list[int] = []
        for phase in grp["phase"]:
            if phase == "Menstrual" and prev != "Menstrual":
                cycle += 1
                c_day = 1
            else:
                c_day += 1
            c_ids.append(cycle)
            c_days.append(c_day)
            prev = phase
        df.loc[grp.index, "cycle_id"] = c_ids
        df.loc[grp.index, "cycle_day"] = c_days

    df["cycle_id"] = df["cycle_id"].astype(int)
    df["cycle_day"] = df["cycle_day"].astype(int)

    cycle_lengths = df.groupby(["id", "cycle_id"])["cycle_day"].max().rename("cycle_total_days")
    df = df.merge(cycle_lengths, on=["id", "cycle_id"])
    df["cycle_pct"] = ((df["cycle_day"] / df["cycle_total_days"]) * 100).round(1)

    all_phases = {"Menstrual", "Follicular", "Fertility", "Luteal"}
    real = df[df["cycle_id"] > 0]
    cycles_per_subject = real.groupby("id")["cycle_id"].apply(set)

    complete_set: set[tuple] = set()
    for (sid, cid), grp in real.groupby(["id", "cycle_id"]):
        if not all_phases.issubset(set(grp["phase"].dropna())):
            continue
        sid_cycles = cycles_per_subject[sid]
        has_prev = (cid - 1) in sid_cycles
        has_next = (cid + 1) in sid_cycles
        ordered = grp.sort_values("day_in_study")
        if has_prev and has_next:
            complete_set.add((sid, cid))
        elif not has_prev and has_next:
            if (
                ordered["phase"].iloc[0] == "Menstrual"
                and (grp["phase"] == "Menstrual").sum() >= 2
            ):
                complete_set.add((sid, cid))
        elif has_prev and not has_next:
            if ordered["phase"].iloc[-1] == "Luteal" and (grp["phase"] == "Luteal").sum() >= 2:
                complete_set.add((sid, cid))

    complete = pd.DataFrame(list(complete_set), columns=["id", "cycle_id"]).assign(
        is_complete_cycle=True
    )
    df = df.merge(complete, on=["id", "cycle_id"], how="left")
    df["is_complete_cycle"] = df["is_complete_cycle"].fillna(False)
    return df.drop(columns=["cycle_total_days"])


def add_hormone_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["es_to_lh_ratio"] = (df["estrogen"] / df["lh"]).round(2)
    df["estrogen_smooth"] = (
        df.groupby(["id"])["estrogen"]
        .transform(lambda x: x.rolling(window=5, center=True, min_periods=1).mean())
        .round(1)
    )
    df["lh_smooth"] = (
        df.groupby(["id"])["lh"]
        .transform(lambda x: x.rolling(window=5, center=True, min_periods=1).mean())
        .round(1)
    )
    return df


def add_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["active_min_total"] = (
        df["active_min_light"] + df["active_min_moderate"] + df["active_min_high"]
    )
    return df


def add_sleep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["light_to_total_ratio"] = (df["sleep_min_light"] / df["sleep_min_total"]).round(2)
    df["rem_to_total_ratio"] = (df["sleep_min_rem"] / df["sleep_min_total"]).round(2)
    df["deep_to_total_ratio"] = (df["sleep_min_deep"] / df["sleep_min_total"]).round(2)
    return df


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temperature_z_scored"] = (
        df.groupby(["id"])["temperature"].transform(lambda x: (x - x.mean()) / x.std()).round(2)
    )
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
    flow_map = {v: i for i, v in enumerate(flow_scale)}
    df["flow_volume"] = df["flow_volume"].map(flow_map).astype("Int64")

    report_scale = ["Not at all", "Very Low/Little", "Low", "Moderate", "High", "Very High"]
    scale_map = {v: i for i, v in enumerate(report_scale)}
    cols = [
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
    ]

    for col in cols:
        df[col] = df[col].map(scale_map).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    interday_input: Path = INTERDAY_INPUT,
    intraday_input: Path = INTRADAY_INPUT,
    interday_output: Path = INTERDAY_OUTPUT,
    intraday_output: Path = INTRADAY_OUTPUT,
) -> None:
    logger.info("Starting feature engineering...")

    interday = pd.read_csv(interday_input)
    intraday = pd.read_csv(intraday_input, parse_dates=["datetime"])

    logger.info("Adding features to interday...")
    interday = add_cycle_features(interday)
    interday = add_hormone_features(interday)
    interday = add_activity_features(interday)
    interday = add_sleep_features(interday)
    interday = add_temperature_features(interday)
    interday = reports_to_numeric(interday)

    logger.info("Adding features to intraday...")
    intraday = pd.merge(
        intraday,
        interday[["id", "day_in_study", "phase", "cycle_id", "cycle_day", "cycle_pct"]],
        on=["id", "day_in_study"],
    )

    time_bins = [0, 6, 12, 18, 21, 24]
    # pd.cut requires unique labels per bin; "night2" covers 21-24 and is collapsed after
    time_labels = ["night", "morning", "afternoon", "evening", "night2"]
    intraday["time_of_day"] = pd.cut(
        intraday["datetime"].dt.hour, bins=time_bins, labels=time_labels, right=False
    )
    intraday["time_of_day"] = intraday["time_of_day"].replace("night2", "night")

    interday.to_csv(interday_output, index=False)
    logger.success(f"Saved features_interday: {interday_output} ({len(interday):,} rows)")

    intraday.to_csv(intraday_output, index=False)
    logger.success(f"Saved features_intraday: {intraday_output} ({len(intraday):,} rows)")


if __name__ == "__main__":
    app()
