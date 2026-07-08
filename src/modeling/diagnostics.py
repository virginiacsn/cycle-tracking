"""Fast model diagnostics that avoid full LOSO + Optuna runs.

Usage:
    python -m src.modeling.diagnostics gap
    python -m src.modeling.diagnostics learning-curve fitbit --model random_forest
    python -m src.modeling.diagnostics validation-curve combined --model xgboost
"""

import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
import typer

from src.modeling.classifiers import RANDOM_STATE, make_classifiers
from src.modeling.reporting import log_best_params
from src.modeling.train import DATASET_PATHS, NON_FEATURE_COLS, RESULTS_DIR, TASK_TARGETS
from src.plots import (
    DIAGNOSTICS_FIGURES_DIR,
    MODEL_NAMES,
    plot_learning_curve,
    plot_train_test_gap,
    plot_validation_curve,
)

app = typer.Typer()

# One complexity knob per classifier: param name, values to sweep, log-scale x-axis.
COMPLEXITY_PARAM: dict[str, tuple[str, list[float], bool]] = {
    "logreg": ("clf__C", [0.001, 0.01, 0.1, 1.0, 10.0], True),
    "random_forest": ("clf__max_depth", [3, 5, 8, 12, 20, 22], False),
    "xgboost": ("clf__max_depth", [2, 3, 4, 6, 8], False),
    "catboost": ("clf__depth", [2, 3, 4, 6, 8], False),
}


def _load_dataset(dataset: str) -> tuple[pd.DataFrame, list[str], str]:
    features_path = DATASET_PATHS[dataset]
    if not features_path.exists():
        logger.error(f"{features_path.name} not found — run the data pipeline first")
        raise typer.Exit(1)

    df = pd.read_csv(features_path)
    target = TASK_TARGETS["phase"]
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    subset = df.dropna(subset=features, how="all").copy()
    return subset, features, target


def _train_val_split(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[list, list]:
    subjects = df["id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_frac))
    val_subjects = subjects[:n_val].tolist()
    train_subjects = subjects[n_val:].tolist()
    return train_subjects, val_subjects


def _fit_and_score(pipe, X_train, y_train, X_val, y_val) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
    train_pred = pipe.predict(X_train)
    val_pred = pipe.predict(X_val)
    return {
        "train": {
            "accuracy": accuracy_score(y_train, train_pred),
            "f1_macro": f1_score(y_train, train_pred, average="macro", zero_division=0),
        },
        "validation": {
            "accuracy": accuracy_score(y_val, val_pred),
            "f1_macro": f1_score(y_val, val_pred, average="macro", zero_division=0),
        },
    }


@app.command()
def gap(
    tune: bool = typer.Option(True, "--tune/--no-tune", help="Read tuned or untuned result CSVs."),
) -> None:
    """Plot train-minus-test metric gaps from existing LOSO result CSVs. No training."""
    suffix = "_tune" if tune else ""
    frames = []
    for dataset in DATASET_PATHS:
        path = RESULTS_DIR / f"{dataset}_phase{suffix}.csv"
        if not path.exists():
            logger.warning(f"Skipping {dataset}: {path.name} not found")
            continue
        result = pd.read_csv(path)
        result["dataset"] = dataset
        frames.append(result)

    if not frames:
        logger.error("No result CSVs found — run src.modeling.train first")
        raise typer.Exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined["model"] = combined["model"].map(MODEL_NAMES)
    fig_name = f"eval_gap{suffix}"
    plot_train_test_gap(combined, ["accuracy", "f1_macro", "auroc"], fig_name)
    logger.success(f"Saved {DIAGNOSTICS_FIGURES_DIR / fig_name}.png")


@app.command("best-params")
def best_params(
    tune: bool = typer.Option(True, "--tune/--no-tune", help="Read tuned or untuned result CSVs."),
) -> None:
    """Print tuned hyperparameters per model from existing LOSO result CSVs. No training."""
    suffix = "_tune" if tune else ""
    results_by_dataset = {}
    for dataset in DATASET_PATHS:
        path = RESULTS_DIR / f"{dataset}_phase{suffix}.csv"
        if not path.exists():
            logger.warning(f"Skipping {dataset}: {path.name} not found")
            continue
        results_by_dataset[dataset] = pd.read_csv(path)

    if not results_by_dataset:
        logger.error("No result CSVs found — run src.modeling.train first")
        raise typer.Exit(1)

    log_best_params(results_by_dataset, f"phase{suffix}")


@app.command("learning-curve")
def learning_curve(
    dataset: str = typer.Argument(help="fitbit | selfreports | combined"),
    model: str = typer.Option("random_forest", help="logreg | random_forest | xgboost | catboost"),
    subject_counts: str = typer.Option(
        "5,10,15,20,30", help="Comma-separated training subject counts."
    ),
    n_repeats: int = typer.Option(3, help="Random subject subsets per size."),
    val_frac: float = typer.Option(
        0.2, help="Fraction of subjects held out as a fixed validation set."
    ),
    seed: int = typer.Option(RANDOM_STATE),
) -> None:
    """Train vs validation score as training-subject count grows. Untuned baseline classifier only."""
    if dataset not in DATASET_PATHS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_PATHS)}")
    classifiers = make_classifiers()
    if model not in classifiers:
        raise typer.BadParameter(f"model must be one of {sorted(classifiers)}")

    df, features, target = _load_dataset(dataset)
    train_subjects, val_subjects = _train_val_split(df, val_frac, seed)

    val_mask = df["id"].isin(val_subjects)
    X_val = df.loc[val_mask, features].astype(float)
    y_val = df.loc[val_mask, target].values

    base_pipe = classifiers[model]
    sizes = sorted({min(int(s), len(train_subjects)) for s in subject_counts.split(",")})
    rng = np.random.default_rng(seed)

    records = []
    for size in sizes:
        for rep in range(n_repeats):
            sampled = rng.choice(train_subjects, size=size, replace=False)
            train_mask = df["id"].isin(sampled)
            X_train = df.loc[train_mask, features].astype(float)
            y_train = df.loc[train_mask, target].values

            scores = _fit_and_score(clone(base_pipe), X_train, y_train, X_val, y_val)
            for split, metrics in scores.items():
                records.append({"n_subjects": size, "rep": rep, "split": split, **metrics})

    results = pd.DataFrame(records)
    fig_name = f"learning_curve_{dataset}_{model}"
    plot_learning_curve(results, ["accuracy", "f1_macro"], fig_name)
    logger.success(f"Saved {DIAGNOSTICS_FIGURES_DIR / fig_name}.png")


@app.command("validation-curve")
def validation_curve(
    dataset: str = typer.Argument(help="fitbit | selfreports | combined"),
    model: str = typer.Option("random_forest", help="logreg | random_forest | xgboost | catboost"),
    val_frac: float = typer.Option(
        0.2, help="Fraction of subjects held out as a fixed validation set."
    ),
    seed: int = typer.Option(RANDOM_STATE),
) -> None:
    """Train vs validation score across one hyperparameter, fixed train/validation subject split."""
    if dataset not in DATASET_PATHS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_PATHS)}")
    if model not in COMPLEXITY_PARAM:
        raise typer.BadParameter(f"model must be one of {sorted(COMPLEXITY_PARAM)}")

    df, features, target = _load_dataset(dataset)
    train_subjects, val_subjects = _train_val_split(df, val_frac, seed)

    train_mask = df["id"].isin(train_subjects)
    val_mask = df["id"].isin(val_subjects)
    X_train = df.loc[train_mask, features].astype(float)
    y_train = df.loc[train_mask, target].values
    X_val = df.loc[val_mask, features].astype(float)
    y_val = df.loc[val_mask, target].values

    param_name, param_values, log_x = COMPLEXITY_PARAM[model]
    base_pipe = make_classifiers()[model]

    records = []
    for value in param_values:
        pipe = clone(base_pipe)
        pipe.set_params(**{param_name: value})
        scores = _fit_and_score(pipe, X_train, y_train, X_val, y_val)
        for split, metrics in scores.items():
            records.append({"param_value": value, "split": split, **metrics})

    results = pd.DataFrame(records)
    fig_name = f"validation_curve_{dataset}_{model}_{param_name.split('__')[-1]}"
    plot_validation_curve(results, param_name, ["accuracy", "f1_macro"], fig_name, log_x=log_x)
    logger.success(f"Saved {DIAGNOSTICS_FIGURES_DIR / fig_name}.png")


if __name__ == "__main__":
    app()
