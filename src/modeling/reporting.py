"""Logging and formatting utilities for training runs."""

import json

from loguru import logger
import pandas as pd


def log_class_balance(df: pd.DataFrame, label: str, name: str) -> None:
    counts = df[label].value_counts().sort_index()
    total = len(df)
    parts = "  ".join(f"class {c}: {n} ({n / total:.1%})" for c, n in counts.items())
    logger.info(f"  [{name}] class balance — {parts}")

    per_subject = df.groupby("id")[label].value_counts().unstack(fill_value=0)
    missing = per_subject.columns[per_subject.min() == 0].tolist()
    if missing:
        logger.warning(
            f"  [{name}] subjects missing classes {missing}: "
            f"{per_subject[per_subject[missing].min(axis=1) == 0].index.tolist()}"
        )

    sizes = df.groupby("id").size()
    logger.info(
        f"  [{name}] rows per subject — min={sizes.min()}  max={sizes.max()}  median={sizes.median():.0f}"
    )


def log_summary(results: pd.DataFrame, name: str) -> None:
    for model_name, grp in results.groupby("model"):
        logger.success(
            f"  [{name}] {model_name}: "
            f"accuracy={grp['accuracy'].mean():.3f}±{grp['accuracy'].std():.3f}  "
            f"f1_macro={grp['f1_macro'].mean():.3f}±{grp['f1_macro'].std():.3f}  "
            f"precision={grp['precision_macro'].mean():.3f}±{grp['precision_macro'].std():.3f}  "
            f"recall={grp['recall_macro'].mean():.3f}±{grp['recall_macro'].std():.3f}  "
            f"auroc={grp['auroc'].mean():.3f}±{grp['auroc'].std():.3f}  "
            f"[train] accuracy={grp['train_accuracy'].mean():.3f} f1={grp['train_f1_macro'].mean():.3f}"
        )


def log_best_params(results_by_dataset: dict[str, pd.DataFrame], name: str) -> None:
    """Log tuned-hyperparameter median/std across LOSO folds, one table per model.

    Each table has params as rows and (dataset, median/std) as columns.
    """
    by_model: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset, results in results_by_dataset.items():
        for model_name, grp in results.groupby("model"):
            params = pd.DataFrame(grp["best_params"].apply(json.loads).tolist())
            if params.empty:
                continue
            params.columns = [c.removeprefix("clf__") for c in params.columns]
            by_model.setdefault(model_name, {})[dataset] = params

    if not by_model:
        logger.info(f"  [{name}] no tuned params to show (run with --tune)")
        return

    for model_name in sorted(by_model):
        table = pd.DataFrame(
            {
                (dataset, stat): getattr(params, stat)().round(4)
                for dataset, params in by_model[model_name].items()
                for stat in ("median", "std")
            }
        ).sort_index(axis=1, level=0)
        logger.info(
            f"  [{name}] {model_name} best params (median/std):\n{table.to_string(na_rep='')}"
        )
