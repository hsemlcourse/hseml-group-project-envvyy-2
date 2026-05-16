"""Run complete pipeline: EDA -> Training -> Visualization."""

import subprocess
import sys


def run_script(script_name: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*80}")
    print(f"Запуск: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {script_name} завершен успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при выполнении {script_name}")
        print(f"Код ошибки: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Файл {script_name} не найден!")
        return False


def main():
    """Run complete ML pipeline."""
    print("="*80)
    print("ПОЛНЫЙ PIPELINE - TWITCH POPULARITY PREDICTION")
    print("="*80)
    print("\nЭтот скрипт выполнит:")
    print("1. EDA (Exploratory Data Analysis)")
    print("2. Обучение моделей")
    print("3. Визуализацию результатов")
    print("\n" + "="*80)
    
    # Check if data exists
    from pathlib import Path
    data_path = Path("data/raw/twitchdata-update.csv")
    if not data_path.exists():
        print("\n❌ ОШИБКА: Датасет не найден!")
        print(f"Пожалуйста, загрузите данные в: {data_path}")
        print("\nИспользуйте:")
        print("  python download_data.py")
        return
    
    # Run pipeline
    scripts = [
        "run_eda.py",
        "train_models.py",
        "visualize_data.py"
    ]
    
    results = {}
    for script in scripts:
        success = run_script(script)
        results[script] = success
        if not success:
            print(f"\n⚠️  Остановка pipeline из-за ошибки в {script}")
            break
    
    # Summary
    print("\n" + "="*80)
    print("ИТОГИ ВЫПОЛНЕНИЯ")
    print("="*80)
    for script, success in results.items():
        status = "✅ Успешно" if success else "❌ Ошибка"
        print(f"{script}: {status}")
    
    if all(results.values()):
        print("\n🎉 Все этапы выполнены успешно!")
        print("\nРезультаты:")
        print("  - experiments/results.csv - таблица экспериментов")
        print("  - experiments/visualizations.png - визуализации")
        print("  - data/processed/best_model.pkl - лучшая модель")
    else:
        print("\n⚠️  Pipeline завершен с ошибками")


if __name__ == "__main__":
    main()
