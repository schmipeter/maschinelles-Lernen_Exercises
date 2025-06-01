"""
Einfache Transform-Funktion: Graustufen + Resize 64×64.
Kann beliebig erweitert werden.
"""
from PIL import Image
import numpy as np

def transform(fp: str | "os.PathLike") -> np.ndarray:
    img = Image.open(fp).convert("L").resize((64, 64))
    return np.asarray(img, dtype=np.float32) / 255.0   # (H,W) 0-1