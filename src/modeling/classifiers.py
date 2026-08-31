"""Shared classifier pipelines and Optuna hyperparameter search."""

import warnings

import numpy as np
import optuna as opt
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_TRIALS = 50
XGB_N_ESTIMATORS_CAP = 500
XGB_EARLY_STOPPING_ROUNDS = 20

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
        "xgboost": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=XGB_N_ESTIMATORS_CAP,
                        random_state=RANDOM_STATE,
                        eval_metric="mlogloss",
                        verbosity=0,
                    ),
                ),
            ]
        ),
    }


def suggest_params(trial: opt.Trial, clf_name: str) -> dict:
    if clf_name == "logreg":
        return {"clf__C": trial.suggest_float("clf__C", 1e-3, 10.0, log=True)}
    if clf_name == "xgboost":
        return {
            "clf__max_depth": trial.suggest_int("clf__max_depth", 2, 6),
            "clf__learning_rate": trial.suggest_float("clf__learning_rate", 0.01, 0.3, log=True),
            "clf__subsample": trial.suggest_float("clf__subsample", 0.6, 1.0),
            "clf__colsample_bytree": trial.suggest_float("clf__colsample_bytree", 0.6, 1.0),
            "clf__min_child_weight": trial.suggest_int("clf__min_child_weight", 1, 20),
            "clf__reg_alpha": trial.suggest_float("clf__reg_alpha", 1e-3, 10.0, log=True),
            "clf__reg_lambda": trial.suggest_float("clf__reg_lambda", 1e-3, 10.0, log=True),
        }
    return {}


def fit_classifier(
    clf_name: str,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Pipeline:
    """Fit a pipeline in place. For xgboost, stops on validation mlogloss instead of
    training the full n_estimators cap, so tree count adapts to how much a given
    hyperparameter combination can learn before overfitting.
    """
    if clf_name != "xgboost":
        pipe.fit(X_train, y_train)
        return pipe

    pre = pipe[:-1]
    _, clf = pipe.steps[-1]
    X_train_t = pre.fit_transform(X_train, y_train)
    X_val_t = pre.transform(X_val)
    sample_weight = compute_sample_weight("balanced", y_train)
    clf.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
    clf.fit(
        X_train_t, y_train, sample_weight=sample_weight, eval_set=[(X_val_t, y_val)], verbose=False
    )
    return pipe


def tune_classifier(
    clf_name: str,
    base_pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[Pipeline, dict]:
    """Optuna search scored on a held-out validation set; returns best pipeline and params.

    Fits each trial on train and optimizes macro F1 on val.
    """

    def objective(trial: opt.Trial) -> float:
        params = suggest_params(trial, clf_name)
        p = clone(base_pipe)
        p.set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_classifier(clf_name, p, X_train, y_train, X_val, y_val)
        y_pred = p.predict(X_val)
        return f1_score(y_val, y_pred, average="macro", zero_division=0)

    study = opt.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_pipe = clone(base_pipe)
    best_pipe.set_params(**study.best_params)
    return best_pipe, study.best_params
