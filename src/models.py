"""Machine learning models with hyperparameter tuning and cross-validation."""

from typing import Any

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor


def get_baseline_model() -> LinearRegression:
    """Get baseline Linear Regression model.

    Returns:
        Untrained Linear Regression model
    """
    return LinearRegression()


def get_models_dict(random_state: int = 42) -> dict[str, Any]:
    """Get dictionary of models to train.

    Args:
        random_state: Random seed

    Returns:
        Dictionary with model names and instances
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state),
        "ElasticNet": ElasticNet(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            n_estimators=100, random_state=random_state, verbose=0
        ),
    }
    return models


def get_hyperparameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Get hyperparameter search grids for each model.

    Returns:
        Dictionary mapping model names to parameter grids
    """
    grids = {
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "ElasticNet": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63],
            "subsample": [0.6, 0.8, 1.0],
        },
        "CatBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        },
    }
    return grids


def tune_model_hyperparameters(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list[Any]],
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[Any, dict[str, float]]:
    """Tune model hyperparameters using RandomizedSearchCV.

    Args:
        model: Base model instance
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid for search
        n_iter: Number of parameter combinations to try
        cv: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Tuple of (best_model, best_params)
    """
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate model performance.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    return metrics


def train_and_evaluate(
    model: Any, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> dict[str, float]:
    """Train model and evaluate on validation set.

    Args:
        model: ML model instance
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        Dictionary with evaluation metrics
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)
    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> dict[str, float]:
    """Perform cross-validation on a model.

    Args:
        model: ML model instance
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric

    Returns:
        Dictionary with CV mean and std
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "CV_mean": float(-scores.mean()),
        "CV_std": float(scores.std()),
    }


def get_voting_ensemble(random_state: int = 42) -> VotingRegressor:
    """Create a Voting Regressor ensemble.

    Args:
        random_state: Random seed

    Returns:
        VotingRegressor ensemble
    """
    return VotingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=100, random_state=random_state, n_jobs=-1
                ),
            ),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0
                ),
            ),
            (
                "lgbm",
                LGBMRegressor(
                    n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1
                ),
            ),
            (
                "catboost",
                CatBoostRegressor(n_estimators=100, random_state=random_state, verbose=0),
            ),
        ],
        n_jobs=-1,
    )


def get_stacking_ensemble(random_state: int = 42) -> StackingRegressor:
    """Create a Stacking Regressor ensemble.

    Args:
        random_state: Random seed

    Returns:
        StackingRegressor ensemble
    """
    base_estimators = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0
            ),
        ),
        (
            "lgbm",
            LGBMRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1
            ),
        ),
    ]
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(random_state=random_state),
        n_jobs=-1,
    )