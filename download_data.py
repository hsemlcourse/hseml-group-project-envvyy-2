"""Script to download Twitch dataset from Kaggle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    from kaggle import KaggleApi
except ImportError:
    print("\n[ERROR] Kaggle API не установлена. Установите: pip install kaggle")
    print("\nАльтернативно скачайте вручную:")
    print("  1. https://www.kaggle.com/datasets/aayushmishra1512/twitchdata")
    print("  2. Download → распакуйте → переименуйте CSV в data/raw/twitchdata-update.csv")
    sys.exit(1)


def download_with_kaggle_api(output_dir: Path) -> bool:
    """Download dataset using Kaggle API."""
    try:
        api = KaggleApi()
        api.authenticate()
        print("[KAGGLE] Аутентификация успешна")
    except Exception as e:
        print(f"\n[ERROR] Ошибка аутентификации Kaggle: {e}")
        print("\nДля настройки Kaggle API:")
        print("  1. Создайте аккаунт на https://www.kaggle.com")
        print("  2. Скачайте kaggle.json из Account → Create New API Token")
        print("  3. Поместите файл в ~/.kaggle/kaggle.json")
        print("     (Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json)")
        print("\nИли скачайте датасет вручную:")
        print("  https://www.kaggle.com/datasets/aayushmishra1512/twitchdata")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("\n[KAGGLE] Загрузка датасета 'aayushmishra1512/twitchdata'...")
        api.dataset_download_files(
            "aayushmishra1512/twitchdata",
            path=str(output_dir),
            unzip=True,
            quiet=False,
        )
        print(f"[KAGGLE] Датасет загружен в: {output_dir}")

        # Find and rename CSV to twitchdata-update.csv (the larger dataset with ~1000 rows)
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            target = output_dir / "twitchdata-update.csv"
            # Remove target if exists
            if target.exists():
                target.unlink()
            # Rename first CSV found
            shutil.move(str(csv_files[0]), str(target))
            print(f"[KAGGLE] CSV переименован в: {target}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Ошибка загрузки: {e}")
        return False


def main() -> None:
    """Download Twitch dataset."""
    print("=" * 80)
    print("ЗАГРУЗКА ДАТАСЕТА TWITCH (1000+ стримеров)")
    print("=" * 80)

    output_dir = Path("data/raw")

    # Check if data already exists
    existing = output_dir / "twitchdata-update.csv"
    if existing.exists():
        print(f"\n[INFO] Датасет уже существует: {existing}")
        print(f"   Размер: {existing.stat().st_size / 1024:.1f} KB")
        response = input("\nПерезаписать? (y/N): ").strip().lower()
        if response != "y":
            print("Отмена.")
            return

    # Try Kaggle API first
    success = download_with_kaggle_api(output_dir)

    if not success:
        # Manual download instructions
        print("\n" + "=" * 80)
        print("РУЧНАЯ ЗАГРУЗКА")
        print("=" * 80)
        print("\n1. Перейдите на Kaggle:")
        print("   https://www.kaggle.com/datasets/aayushmishra1512/twitchdata")
        print("\n2. Нажмите 'Download' и скачайте ZIP архив")
        print("\n3. Распакуйте содержимое в папку:")
        print(f"   {output_dir}")
        print("\n4. Убедитесь что CSV файл называется 'twitchdata-update.csv'")
        print(f"\n5. Проверьте наличие файла: {output_dir / 'twitchdata-update.csv'}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
