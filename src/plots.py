"""Visualization functions for cycle tracking analysis."""

import json
import re

from loguru import logger
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from src.config import FIGURES_DIR, MODELS_DIR
from src.features import FEATURE_GROUPS

app = typer.Typer()

RESULTS_DIR = MODELS_DIR / "results"
DIAGNOSTICS_FIGURES_DIR = FIGURES_DIR / "diagnostics"
RESULTS_FIGURES_DIR = FIGURES_DIR / "results"

PHASE_DATASETS = ["fitbit", "selfreports", "combined"]

TEST_METRICS = ["accuracy", "f1_macro", "recall_macro", "auroc"]
TRAIN_METRICS = ["train_accuracy", "train_f1_macro", "train_auroc"]
MODEL_NAMES = {"logreg": "LR", "random_forest": "RF", "xgboost": "XGB", "catboost": "CatB"}

_ROLLING_RE = re.compile(r"_(?:rm|rs)_\d+$")

_GROUP_COLORS: dict[str, str] = {
    "Heart": "#e07b54",
    "Sleep": "#5b8db8",
    "Activity": "#6ab187",
    "Body": "#a65dab",
    "Menstrual": "#e03c6e",
    "Stomach": "#f5a623",
    "Symptoms": "#7c7c7c",
    "Position": "#4ecdc4",
    "Other": "#cccccc",
}


def _assign_group(feature: str) -> str:
    base = _ROLLING_RE.sub("", feature)
    for group, members in FEATURE_GROUPS.items():
        if base in members:
            return group
    return "Other"


def plot_eval_metrics(df: pd.DataFrame, metrics: list, fig_name: str):
    """Point plots (mean +/- 95% CI) per metric, datasets color-coded by hue."""
    datasets = PHASE_DATASETS

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4), sharey=True)

    for c, metric in enumerate(metrics):
        ax = axes[c]
        sns.barplot(
            data=df,
            x="model",
            y=metric,
            hue="dataset",
            hue_order=datasets,
            err_kws={"linewidth": 1.5},
            ax=ax,
        )
        ax.set_ylim([0, 1.05])
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if c < len(metrics):
            ax.get_legend().remove()

    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return


def plot_feature_importance(df: pd.DataFrame, fig_name: str, top_n: int = 15):
    """Grid of horizontal bar charts: rows=datasets, cols=models."""
    subset = df[["dataset", "model", "feature_importances"]].copy()
    subset = subset[
        subset["feature_importances"].notna() & (subset["feature_importances"] != "null")
    ]
    subset["feature_importances"] = subset["feature_importances"].apply(json.loads)

    imp_df = pd.concat(
        [
            subset[["dataset", "model"]].reset_index(drop=True),
            pd.DataFrame(subset["feature_importances"].tolist()),
        ],
        axis=1,
    )
    mean_imp = imp_df.groupby(["dataset", "model"]).mean()

    datasets = PHASE_DATASETS
    models = imp_df["model"].unique().tolist()

    fig, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(4 * len(models), 3 * len(datasets)),
    )

    for r, dataset in enumerate(datasets):
        for c, model in enumerate(models):
            ax = axes[r][c]
            if (dataset, model) not in mean_imp.index:
                ax.axis("off")
                continue
            top = mean_imp.loc[(dataset, model)].nlargest(top_n).iloc[::-1]
            ax.barh(top.index, top.values)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel("Mean importance", fontsize=8)
            if r == 0:
                ax.set_title(model, fontsize=10)
            if c == 0:
                ax.set_ylabel(dataset, fontsize=10)

    plt.tight_layout()
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return


def plot_feature_importance_grouped(df: pd.DataFrame, fig_name: str, top_n: int = 20) -> None:
    """Feature importance bars grouped by category, raw and rolling combined in one figure."""
    subset = df[["dataset", "model", "feature_importances"]].copy()
    subset = subset[
        subset["feature_importances"].notna() & (subset["feature_importances"] != "null")
    ]
    subset["feature_importances"] = subset["feature_importances"].apply(json.loads)

    imp_df = pd.concat(
        [
            subset[["dataset", "model"]].reset_index(drop=True),
            pd.DataFrame(subset["feature_importances"].tolist()),
        ],
        axis=1,
    )
    mean_imp = imp_df.groupby(["dataset", "model"]).mean()

    datasets = PHASE_DATASETS
    models = imp_df["model"].unique().tolist()

    fig, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(4 * len(models), 3 * len(datasets)),
        squeeze=False,
    )

    used_groups: set[str] = set()

    for r, dataset in enumerate(datasets):
        for c, model in enumerate(models):
            ax = axes[r][c]
            if (dataset, model) not in mean_imp.index:
                ax.axis("off")
                continue

            importances = mean_imp.loc[(dataset, model)]

            groups = importances.index.map(_assign_group)
            feat_df = pd.DataFrame(
                {"importance": importances.values, "group": groups}, index=importances.index
            )
            feat_df = feat_df.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
            used_groups.update(feat_df["group"].unique())

            is_rolling = feat_df.index.str.contains(_ROLLING_RE.pattern, regex=True)
            colors = feat_df["group"].map(_GROUP_COLORS)
            ax.barh(
                feat_df.index,
                feat_df["importance"],
                color=colors,
                hatch=["///" if r else "" for r in is_rolling],
                edgecolor="white",
                linewidth=0,
            )
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel("Mean importance", fontsize=8)
            if r == 0:
                ax.set_title(model, fontsize=10)
            if c == 0:
                ax.set_ylabel(dataset, fontsize=10)

    legend_handles = [
        mpatches.Patch(color=_GROUP_COLORS[g], label=g) for g in _GROUP_COLORS if g in used_groups
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Rolling")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout()
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    conf_matrices: dict[str, np.ndarray],
    classes: np.ndarray,
    fig_name: str,
) -> None:
    """Heatmap grid of LOSO out-of-fold confusion matrices, one panel per model."""
    models = list(conf_matrices.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        cm = conf_matrices[model_name]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=False,
        )
        ax.set_title(MODEL_NAMES.get(model_name, model_name), fontsize=10)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_train_test_gap(df: pd.DataFrame, metrics: list[str], fig_name: str) -> None:
    """Bar chart of mean (train - test) metric gap per model, dataset side by side."""
    datasets = df["dataset"].unique().tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        gap_col = f"gap_{metric}"
        plot_df = df.assign(**{gap_col: df[f"train_{metric}"] - df[metric]})
        sns.barplot(
            data=plot_df,
            x="model",
            y=gap_col,
            hue="dataset",
            hue_order=datasets,
            err_kws={"linewidth": 1.5},
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{metric} (train - test)", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ax is not axes[-1]:
            ax.get_legend().remove()

    axes[-1].legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    DIAGNOSTICS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIAGNOSTICS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(df: pd.DataFrame, metrics: list[str], fig_name: str) -> None:
    """Train vs validation score across training-subject count, one panel per metric."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.lineplot(
            data=df,
            x="n_subjects",
            y=metric,
            hue="split",
            marker="o",
            errorbar="sd",
            ax=ax,
        )
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("Training subjects")
        ax.set_ylabel("")

    plt.tight_layout()
    DIAGNOSTICS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIAGNOSTICS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curve(
    df: pd.DataFrame,
    param_name: str,
    metrics: list[str],
    fig_name: str,
    log_x: bool = False,
) -> None:
    """Train vs validation score across hyperparameter values, one panel per metric."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.lineplot(
            data=df,
            x="param_value",
            y=metric,
            hue="split",
            marker="o",
            errorbar="sd",
            ax=ax,
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel(param_name.replace("clf__", ""))
        ax.set_ylabel("")

    plt.tight_layout()
    DIAGNOSTICS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIAGNOSTICS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    tune: bool = typer.Option(True, "--tune/--no-tune", help="Read tuned or untuned result CSVs."),
) -> None:
    suffix = "_tune" if tune else ""
    frames = []
    for dataset in PHASE_DATASETS:
        path = RESULTS_DIR / f"{dataset}_phase{suffix}.csv"
        if not path.exists():
            logger.warning(f"Skipping {dataset}: {path.name} not found")
            continue
        df = pd.read_csv(path)
        df["dataset"] = dataset
        df["model"] = df["model"].map(MODEL_NAMES)
        frames.append(df)

    if not frames:
        logger.error("No result CSVs found — run src.modeling.train first")
        raise typer.Exit(1)

    combined = pd.concat(frames, ignore_index=True)
    plot_eval_metrics(combined, TEST_METRICS, f"eval_test{suffix}")
    plot_feature_importance(combined, f"feature_im{suffix}")
    plot_feature_importance_grouped(combined, f"feature_im{suffix}_grouped")


if __name__ == "__main__":
    app()
