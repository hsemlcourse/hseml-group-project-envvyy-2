"""Utility functions for the project."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def save_experiment_results(
    experiment_name: str,
    model_name: str,
    metrics: dict,
    best_params: dict,
    output_path: str | Path,
) -> None:
    """Append experiment results to a CSV file.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model
        metrics: Dictionary of metric values
        best_params: Dictionary of best hyperparameters
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment": experiment_name,
        "model": model_name,
    }
    row.update(metrics)
    if best_params:
        row["best_params"] = str(best_params)

    df_new = pd.DataFrame([row])

    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)


def load_experiment_results(path: str | Path) -> pd.DataFrame:
    """Load experiment results from CSV.

    Args:
        path: Path to results CSV

    Returns:
        DataFrame with results
    """
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_metrics(metrics: dict) -> str:
    """Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metric name to value

    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:,.2f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)