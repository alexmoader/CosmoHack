from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPECTED = {
    "fire-train-renamed.tar": "9cc2d532312dd91d24beb8c6abcfdddac0cff6ca095501be28d81f751a8f159a",
    "fire-test-renamed.tar": "0431f8ae3af3a266a99074f87ca4a4caf5b6ea2e3645048a905ccac10c51674f",
}

FOLDER_CANDIDATES = [
    "Мониторинг DATA",
    "Monitoring DATA",
    "monitoring_data",
    "data_source",
]


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_source(provided: Path | None) -> Path:
    if provided and provided.exists():
        return provided
    for cand in FOLDER_CANDIDATES:
        p = ROOT / cand
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        f"Не найдена папка с архивами организаторов. Ожидается одна из: {FOLDER_CANDIDATES} в {ROOT}"
    )


def extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        for member in tar.getmembers():
            # Security check for path traversal
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Dangerous path in archive: {member.name}")
        tar.extractall(dest_dir)


def prepare_dataset(source_dir: Path, output_dir: Path) -> None:
    train_dest = output_dir / "train"
    test_dest = output_dir / "test"

    print(f"Используется источник данных: {source_dir}")

    # 1. Train archive
    train_tar = source_dir / "fire-train-renamed.tar"
    if not (train_dest / "af").exists() or not (train_dest / "bs").exists():
        if not train_tar.exists():
            raise FileNotFoundError(f"Файл {train_tar} не найден!")
        print("Проверка SHA-256 train архива...", flush=True)
        h = sha256(train_tar)
        if h != EXPECTED["fire-train-renamed.tar"]:
            print(f"Предупреждение: хэш train ({h}) отличается от эталона ({EXPECTED['fire-train-renamed.tar']})")
        print("Распаковка train данных...", flush=True)
        extract_tar(train_tar, output_dir)
    else:
        print("Данные train уже распакованы.")

    # 2. Test archive
    test_tar = source_dir / "fire-test-renamed.tar"
    if not (test_dest / "af").exists() or not (test_dest / "bs").exists() or not (test_dest / "sample_submission.csv").exists():
        if not test_tar.exists():
            raise FileNotFoundError(f"Файл {test_tar} не найден!")
        print("Проверка SHA-256 test архива...", flush=True)
        h = sha256(test_tar)
        if h != EXPECTED["fire-test-renamed.tar"]:
            print(f"Предупреждение: хэш test ({h}) отличается от эталона ({EXPECTED['fire-test-renamed.tar']})")
        print("Распаковка test данных...", flush=True)
        extract_tar(test_tar, output_dir)
    else:
        print("Данные test уже распакованы.")

    # 3. AOI folder if present
    aoi_source = source_dir / "fire-aoi"
    if aoi_source.exists() and not (output_dir / "fire-aoi").exists():
        shutil.copytree(aoi_source, output_dir / "fire-aoi")
        print("Скопирована папка fire-aoi.")

    print(f"Данные успешно подготовлены в {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack and verify CosmoHackathon datasets")
    parser.add_argument("--source", type=Path, default=None, help="Path to 'Мониторинг DATA' directory")
    parser.add_argument("--output", type=Path, default=ROOT / "data", help="Output directory for train/test")
    args = parser.parse_args()

    src = find_source(args.source)
    prepare_dataset(src, args.output)


if __name__ == "__main__":
    main()
