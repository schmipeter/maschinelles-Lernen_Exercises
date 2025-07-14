from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence
import numpy as np
from PIL import Image
import scipy.io as sio  # ← neu

CLASSES = ("cat", "dog")


# --------------------------------------------------------------------
# Helfer für den Verzeichnis-Modus  (images -> ImageDataset)
# --------------------------------------------------------------------
class ImageDataset:
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image | np.ndarray], np.ndarray] | None = None,
        classes: Sequence[str] = CLASSES,
    ):
        root = Path(root)
        if split not in ("train", "test"):
            raise ValueError("split muss train oder test sein")

        self.files, self.labels = [], []
        for lbl, cls in enumerate(classes):
            folder = root / cls / split
            if not folder.exists():
                raise FileNotFoundError(folder)
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    self.files.append(img_path)
                    self.labels.append(lbl)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img) if self.transform else np.asarray(img)
        return img, self.labels[idx]

    def as_numpy(self):  # nützlich für klassische ML
        imgs, lbls = zip(*[self[i] for i in range(len(self))])
        return np.stack(imgs), np.array(lbls, dtype=np.int64)


# --------------------------------------------------------------------
# load()  – erkennt automatisch, ob eine .mat-Datei oder ein Ordner
# --------------------------------------------------------------------
def load(
    path: str | Path,
    split: str = "train",
    transform: Callable[[Image.Image | np.ndarray], np.ndarray] | None = None,
):
    """
    * Wenn `path` auf eine .mat-Datei zeigt → NumPy-Array zurückgeben
      (kompatibel zu catData_w.mat / dogData_w.mat).

    * Wenn `path` ein Verzeichnis ist → ImageDataset für train / test.
    """
    path = Path(path)

    # ---------- 1) .mat-Datei ----------
    if path.is_file() and path.suffix == ".mat":
        mat = sio.loadmat(path)
        # erster Nicht-Meta-Key wird verwendet
        key = next((k for k in mat if not k.startswith("__")), None)
        if key is None:
            raise RuntimeError(f"Keine Daten in {path}")
        return mat[key]

    # ---------- 2) Verzeichnis ----------
    if path.is_dir():
        return ImageDataset(root=path, split=split, transform=transform)

    raise FileNotFoundError(path)
