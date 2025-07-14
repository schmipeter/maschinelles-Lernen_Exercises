from __future__ import annotations
import random
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat

# Zuordnung: Key-Name im .mat → Klassenlabel
MAT_KEYS = {
    "cat": 0,
    "dog": 1,  # deine Dateien
    "catData": 0,
    "dogData": 1,
}  # andere Varianten
CLASSES = {0: "cat", 1: "dog"}


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save(arr: np.ndarray, dst: Path, fmt: str) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(
        dst.with_suffix(f".{fmt}"), format=fmt.upper()
    )


def _read_mat(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Liest .mat und gibt (Bilder, Labels) zurück."""
    mat = loadmat(mat_path)
    images, labels = [], []
    for key, lbl in MAT_KEYS.items():
        if key in mat:
            data = mat[key]
            if data.ndim == 2:  # (pixels, n_img)
                data = data.T  # → (n_img, pixels)
            images.append(data)
            labels.append(np.full(data.shape[0], lbl, dtype=int))
    if not images:
        raise RuntimeError(f"Keine bekannten Keys in {mat_path}")
    return np.vstack(images), np.concatenate(labels)


def extract_and_split(
    mat_path: Path,
    output_dir: Path,
    test_ratio: float = 0.2,
    seed: int = 42,
    image_format: str = "png",
) -> None:
    """Erstellt data/raw/<class>/<train|test>/…"""
    X, y = _read_mat(mat_path)

    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    split = int(len(idx) * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]

    for cls in CLASSES.values():
        _mkdir(output_dir / cls / "train")
        _mkdir(output_dir / cls / "test")

    side = int(np.sqrt(X.shape[1]))  # Bilder sind 1-D-Vektoren

    for i in train_idx:
        cls = CLASSES[int(y[i])]
        _save(
            X[i].reshape(side, side),
            output_dir / cls / "train" / f"{cls}{i}",
            image_format,
        )
    for i in test_idx:
        cls = CLASSES[int(y[i])]
        _save(
            X[i].reshape(side, side),
            output_dir / cls / "test" / f"{cls}{i}",
            image_format,
        )
