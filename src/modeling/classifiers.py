"""Shared classifier pipelines and Optuna hyperparameter search."""

import warnings

from catboost import CatBoostClassifier
import numpy as np
import optuna as opt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_TRIALS = 15
N_INNER_FOLDS = 3

opt.logging.set_verbosity(opt.logging.WARNING)


def make_classifiers() -> dict[str, Pipeline]:
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
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "clf",
                    XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0),
                ),
            ]
        ),
        "catboost": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "clf",
                    CatBoostClassifier(random_seed=RANDOM_STATE, eval_metric="TotalF1", verbose=0),
                ),
            ]
        ),
    }


def suggest_params(trial: opt.Trial, clf_name: str) -> dict:
    if clf_name == "logreg":
        return {"clf__C": trial.suggest_float("clf__C", 1e-3, 10.0, log=True)}
    if clf_name == "random_forest":
        return {
            "clf__n_estimators": trial.suggest_int("clf__n_estimators", 50, 300),
            "clf__max_depth": trial.suggest_int("clf__max_depth", 3, 20),
            "clf__min_samples_split": trial.suggest_int("clf__min_samples_split", 2, 10),
            "clf__min_samples_leaf": trial.suggest_int("clf__min_samples_leaf", 1, 5),
        }
    if clf_name == "xgboost":
        return {
            "clf__n_estimators": trial.suggest_int("clf__n_estimators", 50, 300),
            "clf__max_depth": trial.suggest_int("clf__max_depth", 2, 8),
            "clf__learning_rate": trial.suggest_float("clf__learning_rate", 0.01, 0.3, log=True),
            "clf__subsample": trial.suggest_float("clf__subsample", 0.6, 1.0),
            "clf__colsample_bytree": trial.suggest_float("clf__colsample_bytree", 0.6, 1.0),
        }
    if clf_name == "catboost":
        return {
            "clf__n_estimators": trial.suggest_int("clf__n_estimators", 50, 300),
            "clf__depth": trial.suggest_int("clf__depth", 3, 8),
            "clf__learning_rate": trial.suggest_float("clf__learning_rate", 0.01, 0.3, log=True),
            "clf__l2_leaf_reg": trial.suggest_float("clf__l2_leaf_reg", 1e-3, 10.0, log=True),
        }
    return {}


def tune_classifier(
    clf_name: str,
    base_pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
) -> tuple[Pipeline, dict]:
    """Optuna search on GroupKFold inner CV; returns best pipeline and params.

    Optimizes macro F1.
    """
    n_inner = min(N_INNER_FOLDS, len(np.unique(groups_train)))
    inner_cv = GroupKFold(n_splits=n_inner)

    def objective(trial: opt.Trial) -> float:
        params = suggest_params(trial, clf_name)
        scores = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train, groups_train):
            p = clone(base_pipe)
            p.set_params(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            y_val = p.predict(X_train.iloc[val_idx])
            score = f1_score(y_train[val_idx], y_val, average="macro", zero_division=0)
            scores.append(score)
        return float(np.mean(scores))

    study = opt.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_pipe = clone(base_pipe)
    best_pipe.set_params(**study.best_params)
    return best_pipe, study.best_params
