from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*arguments: str) -> bool:
    command = [sys.executable, *arguments]
    print("\n>", " ".join(f'"{x}"' if " " in x else x for x in command), "\n")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nОШИБКА: команда завершилась с кодом {result.returncode}")
        return False
    return True


def install() -> bool:
    if not run("-m", "pip", "install", "--upgrade", "pip"):
        return False
    return run("-m", "pip", "install", "-r", str(ROOT / "requirements.txt"))


def prepare() -> bool:
    return run(str(ROOT / "prepare_data.py"))


def train_hierarchical() -> bool:
    return run(str(ROOT / "train_hierarchical.py"))


def train_moe() -> bool:
    return run(str(ROOT / "train_moe.py"))


def train_unet() -> bool:
    return run(str(ROOT / "train_unet.py"))


def train_final() -> bool:
    return run(str(ROOT / "train_final.py"), "--module", "all")


def evaluate() -> bool:
    return run(str(ROOT / "evaluate.py"))


def inference() -> bool:
    return run(str(ROOT / "inference.py"))


def validate() -> bool:
    return run(str(ROOT / "validate_submission.py"))


def cache_predictions() -> bool:
    return run(str(ROOT / "cache_predictions.py"))


def make_variants() -> bool:
    return run(str(ROOT / "make_variants.py"))


def start_service() -> bool:
    return run(str(ROOT / "service" / "run_service.py"))


def quick_submission() -> bool:
    if not (ROOT / "data" / "test" / "sample_submission.csv").exists():
        if not prepare():
            return False
    if not inference():
        return False
    return validate()


def full_retrain() -> bool:
    if not prepare():
        return False
    if not train_final():
        return False
    if not inference():
        return False
    return validate()


def menu() -> None:
    while True:
        print("\n" + "=" * 65)
        print("          COSMOFIRE: ПАНЕЛЬ УПРАВЛЕНИЯ ЧЕМПИОНСКИМ РЕШЕНИЕМ          ")
        print("=" * 65)
        print(" 1. Установить зависимости (requirements.txt)")
        print(" 2. Проверить и распаковать данные (prepare_data)")
        print(" 3. Создать submission (Инференс с постпроцессингом и ансамблем)")
        print(" 4. Обучить финальную иерархическую модель (train_final на 100% данных)")
        print(" 5. Обучить Landcover Mixture of Experts (train_moe: раздельно Пашня/Лес)")
        print(" 6. Обучить Multi-Task U-Net (train_unet: Deep Learning на PyTorch)")
        print(" 7. Оценить модели на честной валидации (evaluate)")
        print(" 8. Закэшировать вероятности теста (cache_predictions)")
        print(" 9. Сгенерировать варианты порогов (make_variants)")
        print("10. Проверить submission.csv строгим валидатором")
        print("11. Запустить гео-сервис (FastAPI + интерактивная веб-карта)")
        print("12. Полный цикл (распаковка -> обучение -> инференс -> проверка)")
        print(" 0. Выход")
        print("-" * 65)

        choice = input("Выберите пункт [0-12]: ").strip()
        if choice == "1":
            install()
        elif choice == "2":
            prepare()
        elif choice == "3":
            quick_submission()
        elif choice == "4":
            train_final()
        elif choice == "5":
            train_moe()
        elif choice == "6":
            train_unet()
        elif choice == "7":
            evaluate()
        elif choice == "8":
            cache_predictions()
        elif choice == "9":
            make_variants()
        elif choice == "10":
            validate()
        elif choice == "11":
            start_service()
        elif choice == "12":
            full_retrain()
        elif choice == "0":
            print("Выход из CosmoFire.")
            break
        else:
            print("Неверный ввод, повторите попытку.")


if __name__ == "__main__":
    menu()
