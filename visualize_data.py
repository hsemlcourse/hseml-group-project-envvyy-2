"""Visualization script for Twitch dataset — includes EDA plots and results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent / "src"))

from config import RANDOM_SEED
from data_loader import load_twitch_data
from utils import set_seed

set_seed(RANDOM_SEED)
sns.set_style("whitegrid")


def plot_eda(df: pd.DataFrame, output_path: Path) -> None:
    """Create EDA visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("EDA — Twitch Streamers Dataset", fontsize=16, fontweight="bold")

    # 1. Distribution of Watch time (target)
    if "Watch time(Minutes)" in df.columns:
        ax = axes[0, 0]
        df["Watch time(Minutes)"].hist(bins=40, ax=ax, edgecolor="black", alpha=0.7)
        ax.set_title("Распределение Watch Time (целевая переменная)")
        ax.set_xlabel("Watch Time (Minutes)")
        ax.set_ylabel("Частота")
        ax.grid(True, alpha=0.3)

    # 2. Distribution of Watch time (log scale)
    if "Watch time(Minutes)" in df.columns:
        ax = axes[0, 1]
        np.log1p(df["Watch time(Minutes)"]).hist(bins=40, ax=ax, edgecolor="black", alpha=0.7, color="orange")
        ax.set_title("Распределение log(Watch Time + 1)")
        ax.set_xlabel("log(Watch Time + 1)")
        ax.set_ylabel("Частота")
        ax.grid(True, alpha=0.3)

    # 3. Average viewers vs Followers
    if "Average viewers" in df.columns and "Followers" in df.columns:
        ax = axes[0, 2]
        ax.scatter(df["Followers"], df["Average viewers"], alpha=0.5)
        ax.set_title("Followers vs Average Viewers")
        ax.set_xlabel("Followers")
        ax.set_ylabel("Average Viewers")
        ax.grid(True, alpha=0.3)

    # 4. Top languages
    if "Language" in df.columns:
        ax = axes[1, 0]
        top_langs = df["Language"].value_counts().head(10)
        top_langs.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Топ-10 языков по количеству стримеров")
        ax.set_xlabel("Количество стримеров")
        ax.grid(True, alpha=0.3, axis="x")

    # 5. Correlation heatmap
    ax = axes[1, 1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0,
                    annot_kws={"size": 8})
        ax.set_title("Корреляционная матрица")

    # 6. Stream time vs Watch time
    if "Stream time(minutes)" in df.columns and "Watch time(Minutes)" in df.columns:
        ax = axes[1, 2]
        ax.scatter(df["Stream time(minutes)"], df["Watch time(Minutes)"], alpha=0.5, color="green")
        ax.set_title("Stream Time vs Watch Time")
        ax.set_xlabel("Stream Time (minutes)")
        ax.set_ylabel("Watch Time (Minutes)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[VIZ] EDA графики сохранены: {output_path}")
    plt.close()


def plot_results(results_csv: Path, output_path: Path) -> None:
    """Create results comparison visualization."""
    if not results_csv.exists():
        print(f"[VIZ] Файл результатов не найден: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    if "RMSE" not in df.columns or df.empty:
        print("[VIZ] Нет данных для визуализации результатов")
        return

    df = df.sort_values("RMSE")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Сравнение моделей по метрикам", fontsize=14, fontweight="bold")

    # RMSE
    ax = axes[0]
    colors = ["green" if "baseline" in str(m).lower() else "steelblue" for m in df["model"]]
    bars = ax.barh(df["model"], df["RMSE"], color=colors)
    ax.set_xlabel("RMSE")
    ax.set_title("Root Mean Squared Error (меньше = лучше)")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, df["RMSE"], strict=True):
        ax.text(val + df["RMSE"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8)

    # MAE
    if "MAE" in df.columns:
        ax = axes[1]
        bars = ax.barh(df["model"], df["MAE"], color="orange")
        ax.set_xlabel("MAE")
        ax.set_title("Mean Absolute Error (меньше = лучше)")
        ax.grid(True, alpha=0.3, axis="x")

    # R2
    if "R2" in df.columns:
        ax = axes[2]
        bars = ax.barh(df["model"], df["R2"], color="purple")
        ax.set_xlabel("R2")
        ax.set_title("R² Score (больше = лучше)")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[VIZ] Результаты моделей сохранены: {output_path}")
    plt.close()


def plot_feature_importance(importance_csv: Path, output_path: Path) -> None:
    """Plot feature importance."""
    if not importance_csv.exists():
        print(f"[VIZ] Файл важности признаков не найден: {importance_csv}")
        return

    df = pd.read_csv(importance_csv)
    if df.empty or "feature" not in df.columns or "importance" not in df.columns:
        print("[VIZ] Нет данных для визуализации важности признаков")
        return

    df = df.sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Топ-15 важных признаков (Random Forest)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[VIZ] Важность признаков сохранена: {output_path}")
    plt.close()


def main() -> None:
    """Create all visualizations."""
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ДАННЫХ — CP2")
    print("=" * 80)

    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # EDA plots
    data_path = Path("data/raw/twitchdata-update.csv")
    if data_path.exists():
        df = load_twitch_data(data_path)
        print(f"\n[VIZ] Создание EDA графиков для {len(df)} строк...")
        plot_eda(df, output_dir / "viz_eda.png")
    else:
        print(f"\n[ERROR] Датасет не найден: {data_path}")

    # Results comparison
    results_csv = output_dir / "results.csv"
    if results_csv.exists():
        print("\n[VIZ] Создание графиков результатов...")
        plot_results(results_csv, output_dir / "viz_results.png")
    else:
        print("\n[WARN] Файл results.csv не найден, запустите train_models.py сначала")

    # Feature importance
    importance_csv = output_dir / "feature_importance.csv"
    if importance_csv.exists():
        print("\n[VIZ] Создание графика важности признаков...")
        plot_feature_importance(importance_csv, output_dir / "viz_feature_importance.png")

    print("\n" + "=" * 80)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 80)


if __name__ == "__main__":
    main()
