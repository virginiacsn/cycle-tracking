"""Visualization functions for cycle tracking analysis."""

import json
import re

import joblib
from loguru import logger
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from src.config import FIGURES_DIR, PKL_DIR
from src.features import (
    DATASET_COMBOS,
    HORMONE_FEATURES,
    SELFREPORT_FEATURES,
    WEARABLE_FEATURES,
    load_dataset,
)
from src.modeling.train import (
    NON_FEATURE_COLS,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    shap_contribs,
    subject_split,
)

app = typer.Typer()

DIAGNOSTICS_FIGURES_DIR = FIGURES_DIR / "diagnostics"
RESULTS_FIGURES_DIR = FIGURES_DIR / "results"

PHASE_DATASETS = [
    "fitbit",
    "hormones",
    "selfreports",
    "fitbit_selfreports",
    "fitbit_hormones",
    "hormones_selfreports",
    "fitbit_hormones_selfreports",
]
assert set(PHASE_DATASETS) == set(DATASET_COMBOS)
COMBINED_DATASET = "fitbit_hormones_selfreports"

TEST_METRICS = ["f1_macro"]
TRAIN_METRICS = ["train_accuracy", "train_f1_macro", "train_auroc"]
MODEL_NAMES = {"logreg": "LR", "xgboost": "XGB"}
METRIC_LABELS = {"f1_macro": "F1 (macro)", "auroc": "AUROC"}

_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_SERIES_MARKERS = ["o", "s", "D", "^"]

_ROLLING_RE = re.compile(r"_(?:rm|rs)_\d+$")

_COMPONENT_LABELS = {"fitbit": "wearables", "selfreports": "self-reports"}


def _dataset_label(dataset: str) -> str:
    return " + ".join(_COMPONENT_LABELS.get(c, c) for c in dataset.split("_"))


def plot_eval_metrics(df: pd.DataFrame, metrics: list, fig_name: str):
    """Slope plot per metric: datasets on the y-axis, metric value on the x-axis,
    one point per model (color + marker) joined by a line within each dataset."""
    datasets = PHASE_DATASETS
    labels = [_dataset_label(d) for d in datasets]
    longest = max(range(len(labels)), key=lambda i: len(labels[i]))
    last_plus = labels[longest].rfind(" + ")
    labels[longest] = labels[longest][:last_plus] + "\n+ " + labels[longest][last_plus + 3 :]
    models = df["model"].unique().tolist()

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.5 * len(metrics), 0.4 * len(datasets) + 1.5),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means = df.groupby(["dataset", "model"])[metric].mean().unstack().reindex(datasets)
        means.index = labels

        for label in labels:
            ax.plot(
                means.loc[label, models],
                [label] * len(models),
                color="#c3c2b7",
                linewidth=1.5,
                zorder=1,
            )

        for i, model in enumerate(models):
            ax.scatter(
                means[model],
                means.index,
                color=_SERIES_COLORS[i % len(_SERIES_COLORS)],
                marker=_SERIES_MARKERS[i % len(_SERIES_MARKERS)],
                s=40,
                label=model,
                zorder=2,
            )

        ax.set_xlim([0, 1.05])
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)

    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return


_SOURCE_FEATURES: dict[str, list[str]] = {
    "Wearables": [*WEARABLE_FEATURES, "lf_hf_ratio"],
    "Hormones": [*HORMONE_FEATURES, "estrogen_lh_ratio"],
    "Self-reports": [*SELFREPORT_FEATURES, "sin", "cos"],
}
_SOURCE_COLORS = {"Wearables": "#2a78d6", "Hormones": "#eb6834", "Self-reports": "#1baf7a"}


def _assign_source(feature: str) -> str:
    base = _ROLLING_RE.sub("", feature)
    for source, members in _SOURCE_FEATURES.items():
        if base in members:
            return source
    return "Other"


def _beeswarm_offsets(
    x: np.ndarray, band: float, seed: int = 0, point_width: float = 0.03
) -> np.ndarray:
    """Vertical jitter for a row of points: bins points with similar x and fans each
    bin out symmetrically, with spread proportional to local point density (capped
    at `band`) so sparse regions stay tight and only dense clusters use the full row."""
    if len(x) == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    n_bins = max(10, len(x) // 8)
    edges = np.linspace(x.min(), x.max(), n_bins + 1) if x.max() > x.min() else np.array([0, 1])
    bin_idx = np.clip(np.digitize(x, edges), 0, len(edges))
    offsets = np.zeros(len(x))
    for b in np.unique(bin_idx):
        members = np.where(bin_idx == b)[0]
        width = min(band, len(members) * point_width)
        spread = np.linspace(-width / 2, width / 2, len(members))
        rng.shuffle(spread)
        offsets[members] = spread
    return offsets


PHASE_LABELS = ["Menstrual", "Follicular", "Fertility", "Luteal"]


def _shap_data(
    dataset: str, model: str, seed: int, top_n: int | None, raw_only: bool
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load the fitted pipeline + held-out test split and compute SHAP contributions.

    Returns (feature_names, contribs, X_t, y_test): contribs has shape
    (n_samples, n_classes, n_features); X_t is the matching (imputed) feature matrix,
    used for beeswarm point color; y_test is each sample's true class. All are
    restricted to `raw_only`/`top_n` (X_t/y_test only in row count, via the test split).
    """
    clf_key = {v: k for k, v in MODEL_NAMES.items()}[model]
    pipe = joblib.load(PKL_DIR / f"{dataset}_{clf_key}.pkl")

    raw_df = load_dataset(dataset)
    features = [c for c in raw_df.columns if c not in NON_FEATURE_COLS]
    subset = raw_df.dropna(subset=features, how="all").copy()
    X = subset[features].astype(float)
    y = subset[TARGET].values.astype(int)
    groups = subset["id"].values

    _, _, test_mask = subject_split(groups, val_frac=0.15, test_frac=0.15, seed=seed)
    X_test, y_test = X[test_mask], y[test_mask]

    contribs = shap_contribs(pipe, X_test, features)
    X_t = np.asarray(pipe[:-1].transform(X_test))

    if raw_only:
        keep_mask = np.array([not _ROLLING_RE.search(f) for f in features])
        features = [f for f, keep in zip(features, keep_mask) if keep]
        contribs = contribs[:, :, keep_mask]
        X_t = X_t[:, keep_mask]

    mean_abs = np.abs(contribs).mean(axis=(0, 1))
    if top_n is not None:
        top_idx = np.argsort(-mean_abs)[:top_n]
        features = [features[i] for i in top_idx]
        contribs = contribs[:, :, top_idx]
        X_t = X_t[:, top_idx]

    return features, contribs, X_t, y_test


def _draw_shap_panel(
    ax: plt.Axes,
    signed: np.ndarray,
    X_t: np.ndarray,
    order: list[int],
    feature_names: list[str],
    sources: list[str],
    source_order: list[str],
    cmap,
    band: float,
    show_labels: bool,
) -> None:
    """Draw one beeswarm panel (features on y, SHAP value on x) onto `ax`."""
    for row, i in enumerate(order):
        vals = X_t[:, i]
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(vals, 0.5)
        y_offsets = _beeswarm_offsets(signed[:, i], band)
        ax.scatter(
            signed[:, i],
            row + y_offsets,
            c=norm,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=8,
            alpha=0.75,
            linewidths=0,
        )

    ax.axvline(0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_yticks(range(len(order)))
    if show_labels:
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=7)
        for tick, i in zip(ax.get_yticklabels(), order):
            tick.set_color(_SOURCE_COLORS.get(sources[i], "#000000"))
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(len(order) - 0.5, -0.5)

    boundary = 0
    for source in source_order:
        n = sum(1 for s in sources if s == source)
        if n == 0:
            continue
        if boundary > 0:
            ax.axhline(boundary - 0.5, color="#e1e0d9", linewidth=0.8, zorder=0)
        boundary += n


def plot_shap_summary_by_class(
    dataset: str = COMBINED_DATASET,
    model: str = "XGB",
    fig_name: str = "shap_summary_by_class",
    seed: int = RANDOM_STATE,
    top_n: int | None = 20,
    raw_only: bool = False,
) -> None:
    """One SHAP beeswarm panel per class label. Unlike `plot_shap_summary` (which picks
    each sample's contribution toward its own true class), every panel here uses *all*
    held-out test samples' contribution toward that one class — the standard multiclass
    SHAP convention, so points within a panel are directly comparable. Panels share the
    same feature set/order (grouped by source: wearables, hormones, self-reports)."""
    features, contribs, X_t, _ = _shap_data(dataset, model, seed, top_n, raw_only)

    mean_abs = np.abs(contribs).mean(axis=(0, 1))
    sources = [_assign_source(f) for f in features]
    source_order = list(_SOURCE_FEATURES)
    order = sorted(
        range(len(features)), key=lambda i: (source_order.index(sources[i]), -mean_abs[i])
    )

    n_classes = contribs.shape[1]
    cmap = plt.get_cmap("coolwarm")
    band = 0.7

    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(3.2 * n_classes + 1.5, 0.28 * len(order) + 1.8),
        sharey=True,
    )

    for c, ax in enumerate(axes):
        _draw_shap_panel(
            ax, contribs[:, c, :], X_t, order, features, sources, source_order, cmap, band, c == 0
        )
        ax.set_xlabel("SHAP value", fontsize=8)
        ax.set_title(PHASE_LABELS[c] if c < len(PHASE_LABELS) else f"class {c}", fontsize=10)

    fig.suptitle(f"{model} — {_dataset_label(dataset)}", fontsize=11, y=1.02)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend_handles = [
        mpatches.Patch(color=color, label=source) for source, color in _SOURCE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_FIGURES_DIR / f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    conf_matrices: dict[str, np.ndarray],
    classes: np.ndarray,
    fig_name: str,
) -> None:
    """Heatmap grid of held-out test confusion matrices, one panel per model."""
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
    """Slope plot per metric: datasets on the y-axis, mean (train - test) gap on the
    x-axis, one point per model (color + marker) joined by a line within each dataset."""
    datasets = PHASE_DATASETS
    labels = [_dataset_label(d) for d in datasets]
    longest = max(range(len(labels)), key=lambda i: len(labels[i]))
    last_plus = labels[longest].rfind(" + ")
    labels[longest] = labels[longest][:last_plus] + "\n+ " + labels[longest][last_plus + 3 :]
    models = df["model"].unique().tolist()

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.5 * len(metrics), 0.4 * len(datasets) + 1.5),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        gap_col = f"gap_{metric}"
        plot_df = df.assign(**{gap_col: df[f"train_{metric}"] - df[metric]})
        means = plot_df.groupby(["dataset", "model"])[gap_col].mean().unstack().reindex(datasets)
        means.index = labels

        for label in labels:
            ax.plot(
                means.loc[label, models],
                [label] * len(models),
                color="#c3c2b7",
                linewidth=1.5,
                zorder=1,
            )

        for i, model in enumerate(models):
            ax.scatter(
                means[model],
                means.index,
                color=_SERIES_COLORS[i % len(_SERIES_COLORS)],
                marker=_SERIES_MARKERS[i % len(_SERIES_MARKERS)],
                s=40,
                label=model,
                zorder=2,
            )

        ax.axvline(0, color="black", linewidth=0.8, zorder=0)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} (train - test)", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)

    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
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


def _load_confusion_matrices(dataset: str) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    path = RESULTS_DIR / f"{dataset}_results.csv"
    if not path.exists():
        logger.warning(f"Skipping {dataset} confusion matrix: {path.name} not found")
        return None

    df = pd.read_csv(path)
    conf_matrices = {
        row["model"]: np.array(json.loads(row["confusion_matrix"])) for _, row in df.iterrows()
    }
    classes = np.array(json.loads(df["confusion_matrix_labels"].iloc[0]))
    return conf_matrices, classes


@app.command()
def main(
    dataset: str = typer.Option(
        COMBINED_DATASET,
        help=f"one of {sorted(DATASET_COMBOS)} — dataset to plot the confusion "
        "matrix and SHAP summaries for.",
    ),
) -> None:
    if dataset not in DATASET_COMBOS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DATASET_COMBOS)}")

    frames = []
    for combo in PHASE_DATASETS:
        path = RESULTS_DIR / f"{combo}_results.csv"
        if not path.exists():
            logger.warning(f"Skipping {combo}: {path.name} not found")
            continue
        df = pd.read_csv(path)
        df["dataset"] = combo
        df["model"] = df["model"].map(MODEL_NAMES)
        frames.append(df)

    if not frames:
        logger.error("No result CSVs found — run src.modeling.train first")
        raise typer.Exit(1)

    combined = pd.concat(frames, ignore_index=True)
    plot_eval_metrics(combined, TEST_METRICS, "eval_test")

    plot_shap_summary_by_class(dataset=dataset, fig_name="shap_summary_by_class")

    loaded = _load_confusion_matrices(dataset)
    if loaded is None:
        raise typer.Exit(1)
    conf_matrices, classes = loaded
    plot_confusion_matrices(conf_matrices, classes, f"{dataset}_confusion")


if __name__ == "__main__":
    app()
