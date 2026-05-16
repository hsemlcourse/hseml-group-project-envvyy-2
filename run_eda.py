"""Exploratory Data Analysis script for Twitch dataset."""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import RAW_DATA_DIR, RANDOM_SEED
from data_loader import get_data_info, load_twitch_data
from utils import set_seed

# Set seed for reproducibility
set_seed(RANDOM_SEED)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def main():
    """Run EDA on Twitch dataset."""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS - TWITCH STREAMERS")
    print("=" * 80)

    # Load data
    data_path = RAW_DATA_DIR / "twitchdata-update.csv"
    if not data_path.exists():
        print(f"\n❌ Файл не найден: {data_path}")
        print("Пожалуйста, загрузите датасет используя download_data.py")
        return

    print(f"\n📂 Загрузка данных из: {data_path}")
    df = load_twitch_data(data_path)

    # Basic info
    print("\n" + "=" * 80)
    print("1. ОСНОВНАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("=" * 80)
    info = get_data_info(df)
    print(f"\n📊 Размер датасета: {info['n_rows']:,} строк × {info['n_columns']} столбцов")
    print(f"\n📋 Колонки ({info['n_columns']}):")
    for col in info["columns"]:
        print(f"   - {col}")

    # Data types
    print("\n📝 Типы данных:")
    for col, dtype in info["dtypes"].items():
        print(f"   {col}: {dtype}")

    # Missing values
    print("\n" + "=" * 80)
    print("2. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
    print("=" * 80)
    missing = info["missing_values"]
    total_missing = sum(missing.values())
    if total_missing > 0:
        print(f"\n⚠️  Всего пропущенных значений: {total_missing}")
        for col, count in missing.items():
            if count > 0:
                pct = (count / info["n_rows"]) * 100
                print(f"   {col}: {count} ({pct:.2f}%)")
    else:
        print("\n✅ Пропущенных значений нет!")

    # Duplicates
    print("\n" + "=" * 80)
    print("3. ДУБЛИКАТЫ")
    print("=" * 80)
    if info["duplicates"] > 0:
        print(f"\n⚠️  Найдено дубликатов: {info['duplicates']}")
    else:
        print("\n✅ Дубликатов нет!")

    # Statistical summary
    print("\n" + "=" * 80)
    print("4. СТАТИСТИЧЕСКОЕ ОПИСАНИЕ")
    print("=" * 80)
    print("\n" + str(df.describe()))

    # Numeric columns analysis
    print("\n" + "=" * 80)
    print("5. АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print("=" * 80)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nЧисловых признаков: {len(numeric_cols)}")
    for col in numeric_cols:
        print(f"\n📈 {col}:")
        print(f"   Min: {df[col].min():,.2f}")
        print(f"   Max: {df[col].max():,.2f}")
        print(f"   Mean: {df[col].mean():,.2f}")
        print(f"   Median: {df[col].median():,.2f}")
        print(f"   Std: {df[col].std():,.2f}")

    # Categorical columns
    print("\n" + "=" * 80)
    print("6. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ")
    print("=" * 80)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nКатегориальных признаков: {len(categorical_cols)}")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n📊 {col}: {unique_count} уникальных значений")
        if unique_count <= 20:
            print(f"   Топ-5 значений:")
            for val, count in df[col].value_counts().head().items():
                print(f"      {val}: {count}")

    # Correlations
    print("\n" + "=" * 80)
    print("7. КОРРЕЛЯЦИИ")
    print("=" * 80)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        print("\nТоп-10 корреляций (по абсолютному значению):")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr_val in corr_pairs[:10]:
            print(f"   {col1} ↔ {col2}: {corr_val:.3f}")

    # Target variable analysis
    print("\n" + "=" * 80)
    print("8. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("=" * 80)
    print("\n💡 Возможные целевые переменные для предсказания популярности:")
    target_candidates = [
        "Watch time(Minutes)",
        "Average viewers",
        "Followers",
        "Peak viewers",
    ]
    for target in target_candidates:
        if target in df.columns:
            print(f"\n   ✓ {target}")
            print(f"     Range: {df[target].min():,.0f} - {df[target].max():,.0f}")
            print(f"     Mean: {df[target].mean():,.0f}")

    print("\n" + "=" * 80)
    print("✅ EDA ЗАВЕРШЕН!")
    print("=" * 80)
    print("\n📝 Следующие шаги:")
    print("   1. Очистка данных (пропуски, выбросы)")
    print("   2. Feature engineering")
    print("   3. Выбор целевой переменной")
    print("   4. Обучение моделей")


if __name__ == "__main__":
    main()
