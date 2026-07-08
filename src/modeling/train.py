"""Cycle phase classification training.

Usage:
    python -m src.modeling.train fitbit phase
    python -m src.modeling.train combined phase --tune
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.features import SHARED_FEATURES
from src.modeling.classifiers import make_classifiers, tune_classifier
from src.modeling.reporting import log_best_params, log_class_balance, log_summary
from src.plots import plot_confusion_matrices

app = typer.Typer()

RESULTS_DIR = MODELS_DIR / "results"
NON_FEATURE_COLS = set(SHARED_FEATURES)

DATASET_PATHS = {
    "fitbit": PROCESSED_DATA_DIR / "interday_fitbit.csv",
    "selfreports": PROCESSED_DATA_DIR / "interday_selfreports.csv",
    "combined": PROCESSED_DATA_DIR / "interday_combined.csv",
}

TASK_TARGETS = {
    "phase": "phase_label",
}

BASE_METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auroc"]


def _feature_importances(pipe: Pipeline, feature_names: list[str]) -> dict | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return dict(zip(feature_names, clf.feature_importances_.round(6).tolist()))
    if hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_).mean(axis=0)
        return dict(zip(feature_names, coef.round(6).tolist()))
    return None


def cross_val_loso(
    df: pd.DataFrame,
    features: list[str],
    label: str,
    tune: bool,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """LOSO-CV for all classifiers.

    Returns per-subject metrics DataFrame and final models trained on full data.
    """
    available = [f for f in features if f in df.columns]
    subset = df.dropna(subset=available, how="all").copy()
    X = subset[available].astype(float)
    y = subset[label].values
    groups = subset["id"].values
    n_classes = len(np.unique(y))

    classifiers = make_classifiers()

    loso = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups))
    records = []
    oof_preds: dict[str, tuple[list, list]] = {name: ([], []) for name in classifiers}

    fold_bar = tqdm(loso.split(X, y, groups), total=n_subjects, desc="LOSO", unit="subject")
    for _, (train_idx, test_idx) in enumerate(fold_bar, 1):
        left_out = groups[test_idx[0]]
        fold_bar.set_postfix(subject=left_out)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        groups_train = groups[train_idx]

        clf_bar = tqdm(classifiers.items(), desc=" Models", leave=False, unit="model")
        for clf_name, base_pipe in clf_bar:
            clf_bar.set_postfix(model=clf_name)
            if tune:
                pipe, best_params = tune_classifier(
                    clf_name, base_pipe, X_train, y_train, groups_train
                )
            else:
                pipe, best_params = clone(base_pipe), {}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)
                y_train_pred = pipe.predict(X_train)
                y_train_proba = pipe.predict_proba(X_train)
                y_pred = pipe.predict(X.iloc[test_idx])
                y_proba = pipe.predict_proba(X.iloc[test_idx])

            def _auroc(y_true: np.ndarray, proba: np.ndarray) -> float:
                if n_classes > 2:
                    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
                return float(roc_auc_score(y_true, proba[:, 1]))

            record: dict = {
                "subject": left_out,
                "model": clf_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                # test metrics
                "accuracy": round(accuracy_score(y[test_idx], y_pred), 4),
                "f1_macro": round(
                    f1_score(y[test_idx], y_pred, average="macro", zero_division=0), 4
                ),
                "precision_macro": round(
                    precision_score(y[test_idx], y_pred, average="macro", zero_division=0), 4
                ),
                "recall_macro": round(
                    recall_score(y[test_idx], y_pred, average="macro", zero_division=0), 4
                ),
                "auroc": round(_auroc(y[test_idx], y_proba), 4),
                # train metrics
                "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
                "train_f1_macro": round(
                    f1_score(y_train, y_train_pred, average="macro", zero_division=0), 4
                ),
                "train_precision_macro": round(
                    precision_score(y_train, y_train_pred, average="macro", zero_division=0), 4
                ),
                "train_recall_macro": round(
                    recall_score(y_train, y_train_pred, average="macro", zero_division=0), 4
                ),
                "train_auroc": round(_auroc(y_train, y_train_proba), 4),
                "best_params": json.dumps(best_params),
                "feature_importances": json.dumps(_feature_importances(pipe, available)),
            }

            oof_preds[clf_name][0].extend(y[test_idx].tolist())
            oof_preds[clf_name][1].extend(y_pred.tolist())
            records.append(record)

    classes = np.unique(y)
    conf_matrices = {
        name: confusion_matrix(true, pred, labels=classes)
        for name, (true, pred) in oof_preds.items()
    }

    final_models: dict[str, Pipeline] = {}
    final_bar = tqdm(classifiers.items(), desc="Final models", unit="model")
    for clf_name, base_pipe in final_bar:
        final_bar.set_postfix(model=clf_name)
        if tune:
            pipe, _ = tune_classifier(clf_name, base_pipe, X, y, groups)
        else:
            pipe = clone(base_pipe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
        final_models[clf_name] = pipe

    return pd.DataFrame(records), final_models, conf_matrices


@app.command()
def main(
    dataset: str = typer.Argument(help="fitbit | selfreports | combined"),
    task: str = typer.Argument(help="phase"),
    tune: bool = typer.Option(False, "--t", help="Run Optuna hyperparameter search."),
    results_dir: Path = typer.Option(RESULTS_DIR, help="Where to write CSVs and model files."),
    comment: str = typer.Option("", "--c", help="Free-text note saved to runs.jsonl."),
) -> None:
    if dataset not in DATASET_PATHS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_PATHS)}")
    if task not in TASK_TARGETS:
        raise typer.BadParameter(f"task must be one of {sorted(TASK_TARGETS)}")

    features_path = DATASET_PATHS[dataset]
    target = TASK_TARGETS[task]
    exp_name = f"{dataset}_{task}" + ("_tune" if tune else "")

    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        logger.error(f"{features_path.name} not found — run the data pipeline first")
        raise typer.Exit(1)

    df = pd.read_csv(features_path)

    if target not in df.columns:
        logger.error(
            f"Target column '{target}' not in {features_path.name} — add it in features.py"
        )
        raise typer.Exit(1)

    logger.info(f"Running {exp_name}  tune={tune}")
    logger.info(f"  {len(df):,} rows  {df['id'].nunique()} subjects")
    log_class_balance(df, target, exp_name)

    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    results, final_models, conf_matrices = cross_val_loso(df, features, target, tune)
    results["experiment"] = exp_name

    results.to_csv(results_dir / f"{exp_name}.csv", index=False)

    classes = np.unique(df[target].dropna())
    plot_confusion_matrices(conf_matrices, classes, f"confusion_{exp_name}")

    for clf_name, fitted_model in final_models.items():
        joblib.dump(fitted_model, results_dir.parent / f"{exp_name}_{clf_name}.pkl")

    log_summary(results, exp_name)
    log_best_params({dataset: results}, exp_name)

    metric_cols = BASE_METRICS
    summary = {
        f"{exp_name}_{m}": {
            **{metric: round(grp[metric].mean(), 4) for metric in metric_cols},
            "confusion_matrix": conf_matrices[m].tolist(),
            "confusion_matrix_labels": classes.tolist(),
        }
        for m, grp in results.groupby("model")
    }
    (results_dir / f"{exp_name}_summary.json").write_text(json.dumps(summary, indent=2))

    flat_results = {
        f"{model}_{metric}": val
        for model, metrics in summary.items()
        for metric, val in metrics.items()
        if not metric.startswith("confusion_matrix")
    }
    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exp_name": exp_name,
        "dataset": dataset,
        "task": task,
        "tune": tune,
        "n_subjects": int(df["id"].nunique()),
        "n_rows": len(df),
        "comment": comment,
        **flat_results,
    }
    with open(results_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(run_entry) + "\n")

    logger.success(f"Done — {results_dir}")


if __name__ == "__main__":
    app()
