"""Inference: predict cycle phase using a trained classification model.

Usage:
    python -m src.modeling.predict fitbit logreg
    python -m src.modeling.predict fitbit_hormones_selfreports xgboost
"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from src.config import PKL_DIR, PROCESSED_DATA_DIR
from src.features import DATASET_COMBOS, DATASET_PATHS, load_dataset
from src.modeling.train import TARGET

app = typer.Typer()

PHASE_CATEGORIES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "predictions.csv"


@app.command()
def main(
    dataset: str = typer.Argument(help=f"one of {sorted(DATASET_COMBOS)}"),
    clf_name: str = typer.Argument(help="logreg | xgboost"),
    models_dir: Path = typer.Option(PKL_DIR, help="Directory containing trained model files."),
    output_path: Path = typer.Option(DEFAULT_OUTPUT, help="Where to write predictions."),
) -> None:
    if dataset not in DATASET_COMBOS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_COMBOS)}")

    model_path = models_dir / f"{dataset}_{clf_name}.pkl"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Run train.py first.")
        raise typer.Exit(1)

    missing = [c for c in DATASET_COMBOS[dataset] if not DATASET_PATHS[c].exists()]
    if missing:
        logger.error(f"Missing base dataset file(s) {missing} — run the data pipeline first")
        raise typer.Exit(1)

    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)

    df = load_dataset(dataset)
    X = df[feature_names].astype(float)

    df[f"predicted_{TARGET}"] = model.predict(X)
    df["predicted_phase"] = pd.Categorical.from_codes(
        df[f"predicted_{TARGET}"], categories=PHASE_CATEGORIES
    )

    out_cols = ["id", "day_in_study", "phase", "predicted_phase"]
    df[out_cols].to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path} ({len(df):,} rows)")


if __name__ == "__main__":
    app()
