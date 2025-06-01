"""
Extrahiert Einzelbilder aus catData_w.mat und dogData_w.mat,
legt sie verlustfrei als PNG ab und erzeugt die Struktur

data/raw/<klasse>/<train|test>/<klassename><laufende_nr>.png
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import io as sio

# ------------------------------------------------------------------
DATA_DIR   = Path("data")
RAW_ROOT   = DATA_DIR / "raw"
TEST_FRAC  = 0.20
RNG_SEED   = 42
CLASSES = {
    "cat": DATA_DIR / "catData_w.mat",
    "dog": DATA_DIR / "dogData_w.mat",
}
# ------------------------------------------------------------------


# ---------- Hilfsfunktionen --------------------------------------


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """bringt das Array sicher ins uint8-Format (0–255)."""
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _load_images(mat_path: Path) -> np.ndarray:
    """
    Liefert das Bild-Array aus einer *_w.mat.

    Reihenfolge:
    1. bevorzugte Keys: 'X', 'images'
    2. bekannte Sonderfälle: 'cat_wave', 'dog_wave'
    3. erster Array-Eintrag ≥ 2 Dimensionen
    """
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    for key in ("X", "images", "cat_wave", "dog_wave"):
        if key in data:
            return data[key]

    for k, v in data.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return v

    raise KeyError(
        f"In {mat_path.name} wurde kein geeigneter Bild-Key gefunden."
    )


def _axis_to_nhw(arr: np.ndarray) -> np.ndarray:
    """bringt das Array in Form (N, H, W)."""
    if arr.ndim == 3 and arr.shape[0] != arr.shape[-1]:
        # (H, W, N) → (N, H, W)
        arr = np.moveaxis(arr, 2, 0)
    elif arr.ndim == 4 and arr.shape[0] in (1, 3):
        # (C, H, W, N) → (N, H, W, C)  (RGB)
        arr = np.moveaxis(arr, 0, -1)
        arr = np.moveaxis(arr, -1, 0)
    return arr


# ---------- Haupt­routine ----------------------------------------


def save_images(label: str, mat_path: Path) -> None:
    print(f"[+] Verarbeite {mat_path.name:<15}  →  Klasse '{label}'")
    imgs = _axis_to_nhw(_load_images(mat_path))

    idxs = list(range(imgs.shape[0]))
    random.shuffle(idxs)

    split_pt = int(len(idxs) * (1 - TEST_FRAC))
    splits = [("train", idxs[:split_pt]), ("test", idxs[split_pt:])]

    for split, subidxs in splits:
        outdir = RAW_ROOT / label / split
        outdir.mkdir(parents=True, exist_ok=True)
        for i, j in enumerate(subidxs):
            img = Image.fromarray(_ensure_uint8(imgs[j]))
            img.save(outdir / f"{label}{i}.png")


# ---------- CLI-Entry-Point --------------------------------------


def main() -> None:
    random.seed(RNG_SEED)
    for lbl, path in CLASSES.items():
        save_images(lbl, path)


if __name__ == "__main__":  # Aufruf:  python -m MECH_M_DUAL_2_MLB_DATA.etl.extract
    main()