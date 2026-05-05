"""Feature engineering pipeline for cycle tracking."""

from pathlib import Path

from loguru import logger
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
    df["cycle_pct"] = df["cycle_day"] / df["cycle_total_days"]
    return df.drop(columns=["cycle_total_days"])


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
    intraday = pd.read_csv(intraday_input)

    logger.info("Adding cycle features to interday...")
    interday = add_cycle_features(interday)

    interday_output.parent.mkdir(parents=True, exist_ok=True)
    interday.to_csv(interday_output, index=False)
    logger.success(f"Saved features_interday: {interday_output} ({len(interday):,} rows)")

    intraday_output.parent.mkdir(parents=True, exist_ok=True)
    intraday.to_csv(intraday_output, index=False)
    logger.success(f"Saved features_intraday: {intraday_output} ({len(intraday):,} rows)")


if __name__ == "__main__":
    app()
