"""Preprocessing utilities for Twitch data."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "drop"
) -> pd.DataFrame:
    """Handle missing values in the DataFrame.

    Args:
        df: Input DataFrame
        strategy: 'drop' or 'fill'

    Returns:
        DataFrame with missing values handled
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill":
        # Fill numeric with median, categorical with mode
        df_out = df.copy()
        for col in df_out.columns:
            if df_out[col].dtype in ["float64", "int64"]:
                df_out[col] = df_out[col].fillna(df_out[col].median())
            else:
                df_out[col] = df_out[col].fillna(df_out[col].mode()[0])
        return df_out
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates()


def remove_outliers(
    df: pd.DataFrame, columns: list[str], method: str = "iqr"
) -> pd.DataFrame:
    """Remove outliers from specified columns using IQR method.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: Method to use ('iqr')

    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns:
            continue
        if df_clean[col].dtype not in ["float64", "int64"]:
            continue

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Split data into train, validation, and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining data)
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val (from train_val portion)
    # Adjust val_size relative to train_val size
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test