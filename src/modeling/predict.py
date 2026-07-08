"""Inference: predict cycle phase using a trained classification model.

Usage:
    python -m src.modeling.predict fitbit phase random_forest
    python -m src.modeling.predict combined phase xgboost --tune
"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

DATASET_PATHS = {
    "fitbit": PROCESSED_DATA_DIR / "interday_fitbit.csv",
    "selfreports": PROCESSED_DATA_DIR / "interday_selfreports.csv",
    "combined": PROCESSED_DATA_DIR / "interday_combined.csv",
}

TASK_TARGETS = {
    "phase": "phase_label",
}

PHASE_CATEGORIES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "predictions.csv"


@app.command()
def main(
    dataset: str = typer.Argument(help="fitbit | selfreports | combined"),
    task: str = typer.Argument(help="phase"),
    clf_name: str = typer.Argument(help="logreg | random_forest | xgboost | catboost"),
    tune: bool = typer.Option(False, "--tune", help="Use the model trained with --tune."),
    models_dir: Path = typer.Option(MODELS_DIR, help="Directory containing trained model files."),
    output_path: Path = typer.Option(DEFAULT_OUTPUT, help="Where to write predictions."),
) -> None:
    if dataset not in DATASET_PATHS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_PATHS)}")
    if task not in TASK_TARGETS:
        raise typer.BadParameter(f"task must be one of {sorted(TASK_TARGETS)}")

    exp_name = f"{dataset}_{task}" + ("_tune" if tune else "")
    model_path = models_dir / f"{exp_name}_{clf_name}.pkl"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Run train.py first.")
        raise typer.Exit(1)

    features_path = DATASET_PATHS[dataset]
    if not features_path.exists():
        logger.error(f"{features_path.name} not found — run the data pipeline first")
        raise typer.Exit(1)

    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)

    df = pd.read_csv(features_path)
    X = df[feature_names].astype(float)

    target = TASK_TARGETS[task]
    df[f"predicted_{target}"] = model.predict(X)
    df["predicted_phase"] = pd.Categorical.from_codes(
        df[f"predicted_{target}"], categories=PHASE_CATEGORIES
    )

    out_cols = ["id", "day_in_study", "phase", "predicted_phase"]
    df[out_cols].to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path} ({len(df):,} rows)")


if __name__ == "__main__":
    app()
