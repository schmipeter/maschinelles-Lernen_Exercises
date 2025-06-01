"""
Wandelt catData_w.mat und dogData_w.mat in PNGs um
und legt sie unter data/raw/<klasse>/<train|test>/… ab.
"""

from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy import io as sio

# -------------------------------------------------------------------
DATA_DIR   = Path("data")          # Ordner, in dem die *_w.mat liegen
RAW_ROOT   = DATA_DIR / "raw"      # Zielordner für PNGs
TEST_FRAC  = 0.20                  # 20 % für Test
RNG_SEED   = 42

# Pfade zu den _w.mat + jeweilige Feldnamen der Bild-Arrays
CLASSES = {
    "cat": (DATA_DIR / "catData_w.mat", "cat_wave"),   # ⇦ Feldname anpassen
    "dog": (DATA_DIR / "dogData_w.mat", "dog_wave"),   # ⇦ Feldname anpassen
}
# -------------------------------------------------------------------


def ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Bringt beliebige Bild-Arrays verlustfrei in uint8-Form."""
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def save_images(label: str, mat_path: Path, field: str) -> None:
    print(f"[+] Verarbeite {mat_path.name} ({label})  –  Feld '{field}'")
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if field not in data:
        raise KeyError(f"Feld '{field}' nicht in {mat_path}")

    imgs: np.ndarray = data[field]            # (N,H,W) oder (H,W,N)

    # auf (N,H,W) bringen, falls nötig
    if imgs.ndim == 3 and imgs.shape[0] != imgs.shape[-1]:
        imgs = np.moveaxis(imgs, 2, 0)

    idxs = list(range(imgs.shape[0]))
    random.shuffle(idxs)

    split_pt = int(len(idxs) * (1 - TEST_FRAC))
    splits = [("train", idxs[:split_pt]), ("test", idxs[split_pt:])]

    for split, subidxs in splits:
        outdir = RAW_ROOT / label / split
        outdir.mkdir(parents=True, exist_ok=True)
        for i, j in enumerate(subidxs):
            img = Image.fromarray(ensure_uint8(imgs[j]))
            img.save(outdir / f"{label}{i}.png")


def main() -> None:
    random.seed(RNG_SEED)
    for lbl, (path, field) in CLASSES.items():
        save_images(lbl, path, field)


if __name__ == "__main__":
    main()