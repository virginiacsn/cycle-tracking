"""Modeling pipeline: cycle phase classification from fitbit and self-reports."""

import json
from pathlib import Path
import warnings

import joblib
from lightgbm import LGBMClassifier
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.features import SHARED_FEATURES

app = typer.Typer()

RESULTS_DIR = MODELS_DIR / "results"
TARGET_LABEL = "phase_id"
CV_FOLDS = 5
RANDOM_STATE = 42

# Columns that are not features (shared metadata + target)
NON_FEATURE_COLS = set(SHARED_FEATURES)

RQS: dict[str, Path] = {
    "rq1_fitbit": PROCESSED_DATA_DIR / "interday_fitbit.csv",
    "rq2_selfreports": PROCESSED_DATA_DIR / "interday_selfreports.csv",
}


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _make_classifiers() -> dict[str, Pipeline]:
    def imputer():
        return SimpleImputer(strategy="median")

    return {
        "logreg": Pipeline(
            [
                ("imputer", imputer()),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "clf",
                    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        eval_metric="mlogloss",
                        verbosity=0,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "clf",
                    LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_val_classify(
    df: pd.DataFrame,
    features: list[str],
    label: str = TARGET_LABEL,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """GroupKFold CV by subject for all classifiers.

    Returns per-fold metrics DataFrame and final models trained on full data.
    """
    available = [f for f in features if f in df.columns]
    subset = df.dropna(subset=available, how="all").copy()
    X = subset[available].astype(float)
    y = subset[label].values
    groups = subset["id"].values

    n_splits = min(CV_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    classifiers = _make_classifiers()

    X_arr = X.values

    records = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_arr, y, groups)):
        for clf_name, pipe in classifiers.items():
            p = clone(pipe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p.fit(X.iloc[train_idx], y[train_idx])
                y_pred = p.predict(X.iloc[test_idx])
            records.append(
                {
                    "fold": fold,
                    "model": clf_name,
                    "accuracy": round(accuracy_score(y[test_idx], y_pred), 4),
                    "f1_macro": round(
                        f1_score(y[test_idx], y_pred, average="macro", zero_division=0), 4
                    ),
                    "f1_weighted": round(
                        f1_score(y[test_idx], y_pred, average="weighted", zero_division=0),
                        4,
                    ),
                    "n_test": int(len(test_idx)),
                }
            )

    final_models: dict[str, Pipeline] = {}
    for clf_name, pipe in classifiers.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
        final_models[clf_name] = pipe

    return pd.DataFrame(records), final_models


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_summary(rq: str, results: pd.DataFrame) -> None:
    for model_name, grp in results.groupby("model"):
        acc = grp["accuracy"]
        f1 = grp["f1_macro"]
        logger.success(
            f"  [{rq}] {model_name}: "
            f"accuracy={acc.mean():.3f}±{acc.std():.3f}  "
            f"f1_macro={f1.mean():.3f}±{f1.std():.3f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    results_dir: Path = RESULTS_DIR,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}

    for rq_name, features_path in RQS.items():
        logger.info(f"Running {rq_name} from {features_path.name}...")
        df = pd.read_csv(features_path)
        complete = df[df["is_full_cycle"].astype(bool)].copy()
        logger.info(
            f"  {len(complete):,} rows from {complete['id'].nunique()} subjects (full cycles only)"
        )

        features = [c for c in complete.columns if c not in NON_FEATURE_COLS]
        results, final_models = cross_val_classify(complete, features, label=TARGET_LABEL)
        results["rq"] = rq_name
        results.to_csv(results_dir / f"{rq_name}.csv", index=False)
        all_results.append(results)

        for clf_name, model in final_models.items():
            joblib.dump(model, models_dir / f"{rq_name}_{clf_name}.pkl")

        _log_summary(rq_name, results)

        for model_name, grp in results.groupby("model"):
            summary[f"{rq_name}_{model_name}"] = {
                "accuracy": round(grp["accuracy"].mean(), 4),
                "f1_macro": round(grp["f1_macro"].mean(), 4),
            }

    pd.concat(all_results, ignore_index=True).to_csv(results_dir / "all_results.csv", index=False)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.success(f"All results saved to {results_dir}")


if __name__ == "__main__":
    app()
