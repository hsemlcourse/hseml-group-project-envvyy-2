"""Training script for Twitch popularity prediction models — CP2.

This script covers:
- Data loading and preprocessing
- Feature engineering (engagement features, log-transforms)
- Dimensionality reduction (PCA)
- Baseline model
- Multiple ML models (7+ with hyperparameter tuning)
- Cross-validation
- Ensembles (Voting, Stacking)
- Feature importance analysis
- Results saved to experiments/results.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import (
    HYPERPARAM_N_ITER,
    N_CROSS_VAL_FOLDS,
    PCA_VARIANCE_THRESHOLD,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
    VAL_SIZE,
)
from data_loader import load_twitch_data
from features import (
    apply_pca,
    create_engagement_features,
    encode_categorical_features,
    get_feature_importance,
    scale_features,
)
from models import (
    cross_validate_model,
    evaluate_model,
    get_baseline_model,
    get_hyperparameter_grids,
    get_models_dict,
    get_stacking_ensemble,
    get_voting_ensemble,
    train_and_evaluate,
    tune_model_hyperparameters,
)
from preprocessing import (
    handle_missing_values,
    remove_duplicates,
    remove_outliers,
    split_data,
)
from utils import save_experiment_results, set_seed


def prepare_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Prepare data for modeling."""
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)

    # Remove duplicates
    print("\n1. Удаление дубликатов...")
    df_clean = remove_duplicates(df)
    print(f"   Удалено: {len(df) - len(df_clean)} строк")
    print(f"   Осталось: {len(df_clean)} строк")

    # Handle missing values
    print("\n2. Обработка пропущенных значений...")
    initial_rows = len(df_clean)
    df_clean = handle_missing_values(df_clean, strategy="drop")
    print(f"   Удалено строк с пропусками: {initial_rows - len(df_clean)}")
    print(f"   Осталось: {len(df_clean)} строк")

    # Feature engineering
    print("\n3. Feature engineering...")
    n_features_before = len(df_clean.columns)
    df_clean = create_engagement_features(df_clean)
    n_features_after = len(df_clean.columns)
    print(f"   Новых признаков создано: {n_features_after - n_features_before}")

    # Encode categorical
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print("\n4. Кодирование категориальных признаков...")
        print(f"   Категориальные колонки: {categorical_cols}")
        df_clean, encoders = encode_categorical_features(df_clean, categorical_cols)
        print(f"   Закодировано: {len(categorical_cols)} признаков")
        # Save encoders
        joblib.dump(encoders, PROCESSED_DATA_DIR / "encoders.pkl")

    # Remove outliers from target
    print("\n5. Удаление выбросов из целевой переменной...")
    initial_rows = len(df_clean)
    df_clean = remove_outliers(df_clean, [target_col], method="iqr")
    print(f"   Удалено выбросов: {initial_rows - len(df_clean)}")
    print(f"   Осталось: {len(df_clean)} строк")

    return df_clean


def main() -> None:
    """Main training pipeline for CP2."""
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ — CP2: TWITCH POPULARITY PREDICTION")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = Path("data/raw/twitchdata.csv")
    if not data_path.exists():
        print(f"\n[ERROR] Файл не найден: {data_path}")
        print("Пожалуйста, загрузите данные: python download_data.py")
        return

    print(f"\n[DATA] Загрузка данных из: {data_path}")
    df = load_twitch_data(data_path)
    print(f"   Загружено: {len(df)} строк, {len(df.columns)} столбцов")

    # ── Dataset description ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ОПИСАНИЕ ДАТАСЕТА")
    print("=" * 80)
    print("   Источник: Kaggle — Twitch Streamer Data")
    print(f"   Размер: {len(df)} строк x {len(df.columns)} колонок")
    print(f"   Целевая переменная: {TARGET_COLUMN}")
    print("   Обоснование метрики: Watch time — ключевая метрика популярности")
    print("   Отражает общее время просмотра и вовлеченность аудитории")

    # ── Prepare data ─────────────────────────────────────────────────────────
    df_clean = prepare_data(df, TARGET_COLUMN)

    # ── Split features / target ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 80)

    drop_cols = ["Channel", TARGET_COLUMN]
    feature_cols = [col for col in df_clean.columns if col not in drop_cols]

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COLUMN]

    print(f"\n[FEATURES] Признаков: {len(feature_cols)}")
    print(f"   {feature_cols}")
    print(f"   Целевая переменная: {TARGET_COLUMN}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    print("\n[SPLIT] Разделение данных:")
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print("   [LEAK PREVENTION] Используем train_test_split с фиксированным seed")

    # ── Scale features ────────────────────────────────────────────────────────
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    print("\n[SCALE] Масштабирование StandardScaler выполнено")
    joblib.dump(scaler, PROCESSED_DATA_DIR / "scaler.pkl")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("МЕТРИКИ КАЧЕСТВА")
    print("=" * 80)
    print("   1. RMSE (Root Mean Squared Error) — ОСНОВНАЯ метрика")
    print("      Штрафует большие ошибки сильнее, критично для бизнеса")
    print("   2. MAE (Mean Absolute Error) — дополнительная")
    print("      Средняя абсолютная ошибка, более интерпретируема")
    print("   3. R2 — коэффициент детерминации")
    print("      Доля объясненной дисперсии, [0, 1]")
    print("   4. CV (Cross-Validation) — для надежности оценки")

    results: list[dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BASELINE MODEL (Linear Regression без feature engineering)")
    print("=" * 80)

    baseline = get_baseline_model()
    baseline_metrics = train_and_evaluate(
        baseline,
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
    )
    baseline_cv = cross_validate_model(
        baseline, X_train.values, y_train.values, cv=N_CROSS_VAL_FOLDS
    )

    print(f"\n   Val RMSE: {baseline_metrics['RMSE']:,.2f}")
    print(f"   Val MAE:  {baseline_metrics['MAE']:,.2f}")
    print(f"   Val R2:   {baseline_metrics['R2']:.4f}")
    print(f"   CV RMSE:  {baseline_cv['CV_mean']:,.2f} (+/- {baseline_cv['CV_std']:.2f})")

    results.append({
        "model": "Linear Regression (Baseline)",
        "experiment": "baseline",
        **baseline_metrics,
        **baseline_cv,
    })

    save_experiment_results(
        "baseline",
        "Linear Regression",
        {**baseline_metrics, **baseline_cv},
        {},
        "experiments/results.csv",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MODELS WITH TUNING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"ОБУЧЕНИЕ {N_CROSS_VAL_FOLDS}-fold CV + HYPERPARAMETER TUNING")
    print("=" * 80)

    models = get_models_dict(RANDOM_SEED)
    param_grids = get_hyperparameter_grids()
    tuned_results: list[dict] = []

    for name, model in models.items():
        print(f"\n{'─'*60}")
        print(f"[MODEL] {name}")
        print(f"{'─'*60}")

        try:
            # Cross-validation on training set
            print(f"   [CV] {N_CROSS_VAL_FOLDS}-fold cross-validation...")
            cv_metrics = cross_validate_model(
                model,
                X_train_scaled.values,
                y_train.values,
                cv=N_CROSS_VAL_FOLDS,
            )
            print(f"   CV RMSE: {cv_metrics['CV_mean']:,.2f} (+/- {cv_metrics['CV_std']:.2f})")

            # Hyperparameter tuning
            if name in param_grids:
                print(f"   [TUNING] RandomizedSearchCV ({HYPERPARAM_N_ITER} итераций)...")
                best_model, best_params = tune_model_hyperparameters(
                    model,
                    X_train_scaled.values,
                    y_train.values,
                    param_grids[name],
                    n_iter=HYPERPARAM_N_ITER,
                    cv=N_CROSS_VAL_FOLDS,
                    random_state=RANDOM_SEED,
                )
                print(f"   Best params: {best_params}")
            else:
                best_model = model
                best_params = {}

            # Evaluate on validation set
            val_metrics = train_and_evaluate(
                best_model,
                X_train_scaled.values,
                y_train.values,
                X_val_scaled.values,
                y_val.values,
            )
            print(f"   Val RMSE: {val_metrics['RMSE']:,.2f}")
            print(f"   Val MAE:  {val_metrics['MAE']:,.2f}")
            print(f"   Val R2:   {val_metrics['R2']:.4f}")

            result = {
                "model": name,
                "experiment": "tuned",
                **val_metrics,
                **cv_metrics,
            }
            tuned_results.append(result)
            results.append(result)

            save_experiment_results(
                "main_experiment",
                name,
                {**val_metrics, **cv_metrics},
                best_params,
                "experiments/results.csv",
            )
        except Exception as e:
            print(f"   [ERROR] {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # DIMENSIONALITY REDUCTION (PCA)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТЫ С УМЕНЬШЕНИЕМ РАЗМЕРНОСТИ (PCA)")
    print("=" * 80)

    print(f"\n[PCA] Применяем PCA с порогом дисперсии {PCA_VARIANCE_THRESHOLD}...")
    X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
        X_train.values,
        X_val.values,
        X_test.values,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )
    print(f"   Исходных признаков: {X_train.shape[1]}")
    print(f"   После PCA: {X_train_pca.shape[1]} компонент")
    print(f"   Объясненная дисперсия: {pca_model.explained_variance_ratio_.sum()*100:.1f}%")

    # Save PCA model
    joblib.dump(pca_model, PROCESSED_DATA_DIR / "pca_model.pkl")

    print("\n[PCA EXPERIMENT] Тестируем лучшие модели с PCA-признаками...")

    # Top-3 models from tuning results
    if tuned_results:
        top_models = sorted(tuned_results, key=lambda x: x["RMSE"])[:3]
        pca_experiment_results: list[dict] = []

        for res in top_models:
            model_name = res["model"]
            if model_name in models:
                model = models[model_name]

                # Re-tune with PCA features
                if model_name in param_grids:
                    best_model_pca, best_params_pca = tune_model_hyperparameters(
                        model,
                        X_train_pca,
                        y_train.values,
                        param_grids[model_name],
                        n_iter=HYPERPARAM_N_ITER,
                        cv=N_CROSS_VAL_FOLDS,
                        random_state=RANDOM_SEED,
                    )
                else:
                    best_model_pca = model
                    best_params_pca = {}

                metrics_pca = train_and_evaluate(
                    best_model_pca,
                    X_train_pca,
                    y_train.values,
                    X_val_pca,
                    y_val.values,
                )
                cv_pca = cross_validate_model(
                    best_model_pca, X_train_pca, y_train.values, cv=N_CROSS_VAL_FOLDS
                )

                print(f"\n   {model_name} + PCA:")
                print(f"   Val RMSE: {metrics_pca['RMSE']:,.2f} (было: {res['RMSE']:,.2f})")

                result = {
                    "model": f"{model_name} + PCA",
                    "experiment": "pca_experiment",
                    **metrics_pca,
                    **cv_pca,
                }
                results.append(result)
                pca_experiment_results.append(result)

                save_experiment_results(
                    "pca_experiment",
                    model_name,
                    {**metrics_pca, **cv_pca},
                    best_params_pca,
                    "experiments/results.csv",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # ENSEMBLES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("АНСАМБЛИ МОДЕЛЕЙ")
    print("=" * 80)

    # Voting Regressor
    print("\n[ENSEMBLE] Voting Regressor (RF + XGBoost + LightGBM + CatBoost)...")
    voting_ens = get_voting_ensemble(RANDOM_SEED)
    voting_cv = cross_validate_model(
        voting_ens,
        X_train_scaled.values,
        y_train.values,
        cv=N_CROSS_VAL_FOLDS,
    )
    voting_metrics = train_and_evaluate(
        voting_ens,
        X_train_scaled.values,
        y_train.values,
        X_val_scaled.values,
        y_val.values,
    )
    print(f"   Val RMSE: {voting_metrics['RMSE']:,.2f}")
    print(f"   Val MAE:  {voting_metrics['MAE']:,.2f}")
    print(f"   Val R2:   {voting_metrics['R2']:.4f}")
    print(f"   CV RMSE:  {voting_cv['CV_mean']:,.2f} (+/- {voting_cv['CV_std']:.2f})")

    results.append({
        "model": "Voting Ensemble",
        "experiment": "ensemble",
        **voting_metrics,
        **voting_cv,
    })

    save_experiment_results(
        "ensemble",
        "Voting Regressor",
        {**voting_metrics, **voting_cv},
        {},
        "experiments/results.csv",
    )

    # Stacking Regressor
    print("\n[ENSEMBLE] Stacking Regressor (RF + XGBoost + LightGBM → Ridge)...")
    stacking_ens = get_stacking_ensemble(RANDOM_SEED)
    stacking_cv = cross_validate_model(
        stacking_ens,
        X_train_scaled.values,
        y_train.values,
        cv=N_CROSS_VAL_FOLDS,
    )
    stacking_metrics = train_and_evaluate(
        stacking_ens,
        X_train_scaled.values,
        y_train.values,
        X_val_scaled.values,
        y_val.values,
    )
    print(f"   Val RMSE: {stacking_metrics['RMSE']:,.2f}")
    print(f"   Val MAE:  {stacking_metrics['MAE']:,.2f}")
    print(f"   Val R2:   {stacking_metrics['R2']:.4f}")
    print(f"   CV RMSE:  {stacking_cv['CV_mean']:,.2f} (+/- {stacking_cv['CV_std']:.2f})")

    results.append({
        "model": "Stacking Ensemble",
        "experiment": "ensemble",
        **stacking_metrics,
        **stacking_cv,
    })

    save_experiment_results(
        "ensemble",
        "Stacking Regressor",
        {**stacking_metrics, **stacking_cv},
        {},
        "experiments/results.csv",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (отсортировано по RMSE)")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    if "RMSE" in results_df.columns:
        results_df = results_df.sort_values("RMSE")
        # Reorder columns
        cols_order = ["model", "experiment", "RMSE", "MAE", "R2", "CV_mean", "CV_std"]
        cols_present = [c for c in cols_order if c in results_df.columns]
        results_df = results_df[cols_present]
        print("\n" + results_df.to_string(index=False))

    # Save results
    results_df.to_csv("experiments/results.csv", index=False)
    print("\n[SAVE] Результаты сохранены: experiments/results.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # BEST MODEL & FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    best = results_df.iloc[0]
    print(f"\n{'='*80}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
    print(f"   Val RMSE: {best['RMSE']:,.2f}")
    print(f"   Val MAE:  {best['MAE']:,.2f}")
    print(f"   Val R2:   {best['R2']:.4f}")
    print(f"{'='*80}")

    # ── Feature Importance ───────────────────────────────────────────────────
    print("\n[IMPORTANCE] Анализ важности признаков...")
    print("   Обучаем Random Forest для анализа важности...")

    rf_for_importance = models.get("Random Forest", get_models_dict(RANDOM_SEED)["Random Forest"])
    rf_for_importance.fit(X_train_scaled.values, y_train.values)
    importance_df = get_feature_importance(rf_for_importance, feature_cols)

    if not importance_df.empty:
        print("\n   Топ-10 важных признаков:")
        for _, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        importance_df.to_csv("experiments/feature_importance.csv", index=False)
        print("\n   Сохранено: experiments/feature_importance.csv")
    else:
        print("   Модель не имеет атрибута feature_importances_ или coef_")

    # ── Final Evaluation on Test Set ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET")
    print("=" * 80)

    # Re-train best model name on full training data (train + val)
    best_model_name = best["model"].replace(" + PCA", "")
    print(f"\n[FINAL] Лучшая модель: {best_model_name}")

    # Use original models dict for retraining on combined train+val
    if best_model_name in models:
        final_model = models[best_model_name]
        if best_model_name in param_grids:
            final_model, _ = tune_model_hyperparameters(
                final_model,
                X_train_scaled.values,
                y_train.values,
                param_grids[best_model_name],
                n_iter=HYPERPARAM_N_ITER,
                cv=N_CROSS_VAL_FOLDS,
                random_state=RANDOM_SEED,
            )
    elif "Voting" in best["model"]:
        final_model = voting_ens
    elif "Stacking" in best["model"]:
        final_model = stacking_ens
    else:
        final_model = get_baseline_model()

    final_model.fit(X_train_scaled.values, y_train.values)
    y_test_pred = final_model.predict(X_test_scaled.values)
    test_metrics = evaluate_model(y_test.values, y_test_pred)

    print(f"\n   Test RMSE: {test_metrics['RMSE']:,.2f}")
    print(f"   Test MAE:  {test_metrics['MAE']:,.2f}")
    print(f"   Test R2:   {test_metrics['R2']:.4f}")

    # Save best model
    model_path = PROCESSED_DATA_DIR / "best_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\n[SAVE] Модель сохранена: {model_path}")

    # Save processed data
    pd.concat([X_train, X_val], axis=0).to_csv(
        PROCESSED_DATA_DIR / "X_train_val.csv", index=False
    )
    pd.concat([y_train, y_val], axis=0).to_csv(
        PROCESSED_DATA_DIR / "y_train_val.csv", index=False
    )
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("\n" + "=" * 80)
    print("✅ CP2 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)


if __name__ == "__main__":
    main()