"""Cycle phase classification training.

Usage:
    python -m src.modeling.train main fitbit
    python -m src.modeling.train main fitbit_hormones_selfreports
    python -m src.modeling.train all
"""

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import typer
import xgboost as xgb
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PKL_DIR
from src.features import (
    DATASET_COMBOS,
    SHARED_FEATURES,
    load_dataset,
    missing_dataset_files,
)
from src.modeling.classifiers import (
    RANDOM_STATE,
    fit_classifier,
    make_classifiers,
    tune_classifier,
)
from src.modeling.reporting import log_best_params, log_class_balance, log_summary

app = typer.Typer()

RESULTS_DIR = MODELS_DIR / "results"
NON_FEATURE_COLS = set(SHARED_FEATURES)

TARGET = "phase_label"

BASE_METRICS = ["accuracy", "f1_macro", "recall_macro", "auroc"]


def _feature_importances(pipe: Pipeline, feature_names: list[str]) -> dict | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return dict(zip(feature_names, clf.feature_importances_.round(6).tolist()))
    if hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_).mean(axis=0)
        return dict(zip(feature_names, coef.round(6).tolist()))
    return None


def shap_contribs(pipe: Pipeline, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray | None:
    """Per-sample, per-class SHAP contributions, shape (n_samples, n_classes, n_features).

    Computed exactly, without the `shap` dependency (its `numba` requirement
    doesn't support this project's Python version):
    XGBoost uses exact Tree SHAP via the booster's native `pred_contribs`.
    LogisticRegression uses exact linear SHAP, phi_i = coef_i * (x_i - mean(x_i)).
    """
    clf = pipe.named_steps["clf"]
    X_t = pipe[:-1].transform(X)

    if isinstance(clf, XGBClassifier):
        dmat = xgb.DMatrix(X_t, feature_names=feature_names)
        contribs = np.asarray(clf.get_booster().predict(dmat, pred_contribs=True))
        contribs = contribs[..., :-1]  # drop the bias/base-value column
        if contribs.ndim == 2:  # binary: (n_samples, n_features) -> add a class axis
            contribs = contribs[:, None, :]
        return contribs

    if isinstance(clf, LogisticRegression):
        coef = np.atleast_2d(clf.coef_)  # (n_classes_or_1, n_features)
        centered = X_t - X_t.mean(axis=0)
        return centered[:, None, :] * coef[None, :, :]  # (n_samples, n_classes, n_features)

    return None


def _shap_importances(contribs: np.ndarray | None, feature_names: list[str]) -> dict | None:
    """Mean |SHAP value| per feature, collapsed across samples and classes."""
    if contribs is None:
        return None
    axes = tuple(range(contribs.ndim - 1))
    importance = np.abs(contribs).mean(axis=axes)
    return dict(zip(feature_names, importance.round(6).tolist()))


def subject_split(
    groups: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle subjects and split into train/val/test boolean masks over `groups`.

    A zero fraction means "no split" (0 subjects); a positive fraction always gets
    at least 1 subject, however small, rather than rounding down to none.
    """
    subjects = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, round(len(subjects) * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(len(subjects) * val_frac)) if val_frac > 0 else 0
    test_subjects = subjects[:n_test]
    val_subjects = subjects[n_test : n_test + n_val]
    train_subjects = subjects[n_test + n_val :]
    return (
        np.isin(groups, train_subjects),
        np.isin(groups, val_subjects),
        np.isin(groups, test_subjects),
    )


def _auroc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    if n_classes > 2:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    return float(roc_auc_score(y_true, y_proba[:, 1]))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "auroc": round(_auroc(y_true, y_proba, n_classes), 4),
    }


def single_split(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    val_frac: float = 0.1,
    test_frac: float = 0.3,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Single grouped train/val/test split.

    Subjects are split three ways. Hyperparameters are tuned by scoring trials on
    the validation set; the final model is refit on train only. Test is a pure
    holdout, scored once.
    """
    available = [f for f in features if f in df.columns]
    subset = df.dropna(subset=available, how="all").copy()
    X = subset[available].astype(float)
    y = subset[label].values
    groups = subset["id"].values
    n_classes = len(np.unique(y))

    train_mask, val_mask, test_mask = subject_split(groups, val_frac, test_frac, seed)

    classifiers = make_classifiers()
    classes = np.unique(y)
    records = []
    final_models: dict[str, Pipeline] = {}
    conf_matrices: dict[str, np.ndarray] = {}

    clf_bar = tqdm(classifiers.items(), desc="Models", unit="model")
    for clf_name, base_pipe in clf_bar:
        clf_bar.set_postfix(model=clf_name)
        pipe, best_params = tune_classifier(
            clf_name, base_pipe, X[train_mask], y[train_mask], X[val_mask], y[val_mask]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_classifier(clf_name, pipe, X[train_mask], y[train_mask], X[val_mask], y[val_mask])
            y_train_pred = pipe.predict(X[train_mask])
            y_train_proba = pipe.predict_proba(X[train_mask])
            y_val_pred = pipe.predict(X[val_mask])
            y_val_proba = pipe.predict_proba(X[val_mask])
            y_test_pred = pipe.predict(X[test_mask])
            y_test_proba = pipe.predict_proba(X[test_mask])

        record: dict = {
            "model": clf_name,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            **_metrics(y[test_mask], y_test_pred, y_test_proba, n_classes),
            **{
                f"val_{k}": v
                for k, v in _metrics(y[val_mask], y_val_pred, y_val_proba, n_classes).items()
            },
            **{
                f"train_{k}": v
                for k, v in _metrics(y[train_mask], y_train_pred, y_train_proba, n_classes).items()
            },
            "best_params": json.dumps(best_params),
            "feature_importances": json.dumps(_feature_importances(pipe, available)),
            "shap_importances": json.dumps(
                _shap_importances(shap_contribs(pipe, X[test_mask], available), available)
            ),
        }
        records.append(record)
        final_models[clf_name] = pipe
        conf_matrices[clf_name] = confusion_matrix(y[test_mask], y_test_pred, labels=classes)

    return pd.DataFrame(records), final_models, conf_matrices


def _save_results(
    results: pd.DataFrame,
    final_models: dict[str, Pipeline],
    conf_matrices: dict[str, np.ndarray],
    classes: np.ndarray,
    exp_name: str,
    file_base: str,
    dataset: str,
    n_subjects: int,
    n_rows: int,
    results_dir: Path,
    comment: str,
    extra_run_fields: dict | None = None,
) -> None:
    results = results.copy()
    results["experiment"] = exp_name
    results["confusion_matrix"] = results["model"].map(
        lambda m: json.dumps(conf_matrices[m].tolist())
    )
    results["confusion_matrix_labels"] = json.dumps(classes.tolist())
    results.to_csv(results_dir / f"{file_base}_results.csv", index=False)

    for clf_name, fitted_model in final_models.items():
        joblib.dump(fitted_model, PKL_DIR / f"{file_base}_{clf_name}.pkl")

    log_summary(results, exp_name)
    log_best_params({dataset: results}, exp_name)

    flat_results = {
        f"{exp_name}_{model}_{metric}": round(grp[metric].mean(), 4)
        for model, grp in results.groupby("model")
        for metric in BASE_METRICS
    }
    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exp_name": exp_name,
        "dataset": dataset,
        "n_subjects": n_subjects,
        "n_rows": n_rows,
        "comment": comment,
        **(extra_run_fields or {}),
        **flat_results,
    }
    with open(results_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(run_entry) + "\n")

    logger.success(f"Done — {results_dir}")


RunFn = Callable[[pd.DataFrame, list[str], str], tuple[pd.DataFrame, dict, dict]]


def _run_experiment(
    dataset: str,
    exp_name: str,
    file_base: str,
    results_dir: Path,
    comment: str,
    run_fn: RunFn,
    extra_run_fields: dict | None = None,
) -> None:
    if dataset not in DATASET_COMBOS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_COMBOS)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    missing = missing_dataset_files(dataset)
    if missing:
        logger.error(f"Missing base dataset file(s) {missing} — run the data pipeline first")
        raise typer.Exit(1)

    df = load_dataset(dataset)

    if TARGET not in df.columns:
        logger.error(
            f"Target column '{TARGET}' not in '{dataset}' dataset — add it in features.py"
        )
        raise typer.Exit(1)

    logger.info(f"Running {exp_name}")
    logger.info(f"  {len(df):,} rows  {df['id'].nunique()} subjects")
    log_class_balance(df, TARGET, exp_name)

    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    results, final_models, conf_matrices = run_fn(df, features, TARGET)
    classes = np.unique(df[TARGET].dropna())

    _save_results(
        results,
        final_models,
        conf_matrices,
        classes,
        exp_name,
        file_base,
        dataset,
        int(df["id"].nunique()),
        len(df),
        results_dir,
        comment,
        extra_run_fields=extra_run_fields,
    )


def _train_dataset(
    dataset: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    results_dir: Path,
    comment: str,
) -> None:
    _run_experiment(
        dataset,
        f"{dataset}_phase",
        dataset,
        results_dir,
        comment,
        lambda df, features, target: single_split(
            df, features, target, val_frac=val_frac, test_frac=test_frac, seed=seed
        ),
        extra_run_fields={
            "val_frac": val_frac,
            "test_frac": test_frac,
            "seed": seed,
        },
    )


@app.command()
def main(
    dataset: str = typer.Argument(help=f"one of {sorted(DATASET_COMBOS)}"),
    val_frac: float = typer.Option(0.15, help="Fraction of subjects held out for validation."),
    test_frac: float = typer.Option(0.15, help="Fraction of subjects held out for test."),
    seed: int = typer.Option(RANDOM_STATE, help="Subject-split random seed."),
    results_dir: Path = typer.Option(RESULTS_DIR, help="Where to write CSVs and model files."),
    comment: str = typer.Option("", "--c", help="Free-text note saved to runs.jsonl."),
) -> None:
    _train_dataset(dataset, val_frac, test_frac, seed, results_dir, comment)


@app.command("all")
def train_all(
    val_frac: float = typer.Option(0.15, help="Fraction of subjects held out for validation."),
    test_frac: float = typer.Option(0.15, help="Fraction of subjects held out for test."),
    seed: int = typer.Option(RANDOM_STATE, help="Subject-split random seed."),
    results_dir: Path = typer.Option(RESULTS_DIR, help="Where to write CSVs and model files."),
    comment: str = typer.Option("", "--c", help="Free-text note saved to runs.jsonl."),
) -> None:
    """Train every dataset combination in DATASET_COMBOS."""
    for dataset in tqdm(sorted(DATASET_COMBOS), desc="Datasets", unit="dataset"):
        missing = missing_dataset_files(dataset)
        if missing:
            logger.warning(f"Skipping {dataset}: missing base dataset file(s) {missing}")
            continue
        _train_dataset(dataset, val_frac, test_frac, seed, results_dir, comment)


if __name__ == "__main__":
    app()
