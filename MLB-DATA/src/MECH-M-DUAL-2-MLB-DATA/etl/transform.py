from __future__ import annotations
import numpy as np
from PIL import Image, ImageOps


class ImageTransform:
    """
    Einfache deterministische Vorverarbeitung:
      1. quadratischer Center-Crop
      2. Resize auf 'size'
      3. Umwandlung in float32 [0,1]
      4. optionale Kanal-Normalisierung
    """

    def __init__(
        self,
        size: int | tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
    ):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = ImageOps.fit(img, min(img.size), method=Image.Resampling.BILINEAR)
        img = img.resize(self.size, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.mean and self.std:
            arr = (arr - self.mean) / self.std
        return arr
