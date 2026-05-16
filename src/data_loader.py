"""Data loading utilities for Twitch dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_twitch_data(filepath: str | Path) -> pd.DataFrame:
    """Load Twitch streamer data from CSV.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(filepath)
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
    }


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset.

    Alias for get_data_info for backward compatibility.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    return get_data_info(df)