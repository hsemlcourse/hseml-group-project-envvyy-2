"""Feature engineering, selection, and dimensionality reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engagement-related features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with new engagement features
    """
    df_new = df.copy()

    # Average viewers per stream
    if "Watch time(Minutes)" in df_new.columns and "Stream time(minutes)" in df_new.columns:
        df_new["avg_viewers"] = df_new["Watch time(Minutes)"] / (df_new["Stream time(minutes)"] + 1)

    # Follower to viewer ratio
    if "Followers" in df_new.columns and "Average viewers" in df_new.columns:
        df_new["follower_viewer_ratio"] = df_new["Followers"] / (df_new["Average viewers"] + 1)

    # Peak to average ratio
    if "Peak viewers" in df_new.columns and "Average viewers" in df_new.columns:
        df_new["peak_avg_ratio"] = df_new["Peak viewers"] / (df_new["Average viewers"] + 1)

    # Efficiency: followers gained per hour streamed
    if "Followers" in df_new.columns and "Stream time(minutes)" in df_new.columns:
        df_new["follower_per_hour"] = df_new["Followers"] / (df_new["Stream time(minutes)"] / 60 + 1)

    # Watch time per stream (average session length)
    if "Watch time(Minutes)" in df_new.columns and "Streams" in df_new.columns:
        df_new["watch_per_stream"] = df_new["Watch time(Minutes)"] / (df_new["Streams"] + 1)

    # Viewers per follower
    if "Average viewers" in df_new.columns and "Followers" in df_new.columns:
        df_new["viewers_per_follower"] = df_new["Average viewers"] / (df_new["Followers"] + 1)

    # Log-transform of skewed features
    skewed_cols = ["Followers", "Average viewers", "Peak viewers", "Watch time(Minutes)"]
    for col in skewed_cols:
        if col in df_new.columns:
            df_new[f"log_{col.replace(' ', '_')}"] = np.log1p(df_new[col])

    return df_new


def encode_categorical_features(
    df: pd.DataFrame, categorical_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode categorical features using Label Encoding.

    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names

    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df_encoded = df.copy()
    encoders = {}

    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Tuple of (scaled X_train, X_val, X_test, scaler)
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Apply PCA for dimensionality reduction.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        variance_threshold: Minimum cumulative variance to retain

    Returns:
        Tuple of (X_train_pca, X_val_pca, X_test_pca, pca_model)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_val_pca, X_test_pca, pca


def select_features_kbest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_features: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SelectKBest, list[str]]:
    """Select top K features using univariate feature selection.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        X_test: Test features
        n_features: Number of features to select

    Returns:
        Tuple of (X_train_selected, X_val_selected, X_test_selected, selector, feature_names)
    """
    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names (if DataFrame was provided with column names)
    selected_indices = selector.get_support(indices=True)
    feature_names = [f"feature_{i}" for i in selected_indices]

    return X_train_selected, X_val_selected, X_test_selected, selector, feature_names


def get_feature_importance(
    model: BaseEstimator, feature_names: list[str]
) -> pd.DataFrame:
    """Extract feature importance from a model.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names

    Returns:
        DataFrame with feature names and importance values, sorted descending
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return importance_df


def build_feature_pipeline(
    n_pca_components: int | float = 0.95,
) -> Pipeline:
    """Build a sklearn pipeline with scaling and PCA.

    Args:
        n_pca_components: Number of components or variance threshold for PCA

    Returns:
        Sklearn Pipeline with StandardScaler and PCA
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca_components, random_state=42)),
        ]
    )
