from pathlib import Path
from os import PathLike
from PIL import Image
import numpy as np

def transform(fp: str | PathLike | Path) -> np.ndarray:
    """Graustufen-Resize auf 64×64, normalisiert 0-1."""
    img = Image.open(fp).convert("L").resize((64, 64))
    return np.asarray(img, dtype=np.float32) / 255.0