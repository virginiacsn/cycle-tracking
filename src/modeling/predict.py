"""Inference: predict cycle phase using a trained classification model."""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

DEFAULT_MODEL = MODELS_DIR / "obj1_fitbit_random_forest.pkl"
DEFAULT_FEATURES = PROCESSED_DATA_DIR / "interday_fitbit.csv"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "predictions.csv"


@app.command()
def main(
    features_path: Path = DEFAULT_FEATURES,
    model_path: Path = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Run train.py first.")
        raise typer.Exit(1)

    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)

    df = pd.read_csv(features_path)
    X = df[feature_names].astype(float)

    df["predicted_phase"] = model.predict(X)
    out_cols = ["id", "day_in_study", "phase", "predicted_phase"]
    df[out_cols].to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path} ({len(df):,} rows)")


if __name__ == "__main__":
    app()
