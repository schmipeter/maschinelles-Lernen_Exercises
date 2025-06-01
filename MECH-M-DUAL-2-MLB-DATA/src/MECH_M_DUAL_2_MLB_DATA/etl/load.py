"""
Lädt PNGs aus data/raw/<klasse>/<split>/…  und liefert X, y.
"""
from pathlib import Path
import glob
import numpy as np
from PIL import Image
from .transform import transform

def _discover_classes(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def load(root: str | Path = "data/raw",
         *, split: str = "train",
         classes: list[str] | None = None,
         apply_transform: bool = True):
    root = Path(root)
    classes = classes or _discover_classes(root)
    X, y = [], []

    for label, cls in enumerate(classes):
        for fp in glob.glob(str(root/cls/split/"*.png")):
            if apply_transform:
                X.append(transform(fp))           # (H,W) float32
            else:
                X.append(np.asarray(Image.open(fp), dtype=np.uint8))
            y.append(label)

    return np.stack(X), np.asarray(y, dtype=np.int64)