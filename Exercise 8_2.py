"""
Exercise 8.2  – Max- und Mean-Pooling auf dem Katzenbild
komplett lauffähig unter Python ≥ 3.8, reine NumPy-Implementierung
"""

import io
import requests
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def pool2d(
    img: np.ndarray,
    kernel: tuple[int, int] = (2, 2),
    stride: tuple[int, int] = (2, 2),
    mode: str = "max",
) -> np.ndarray:
    """
    2-D-Pooling (max oder mean) ohne externe Deep-Learning-Bibliotheken.

    Parameters
    ----------
    img : 2-D ndarray
        Graustufenbild.
    kernel : (kh, kw)
        Größe des Pooling-Fensters.
    stride : (sh, sw)
        Schrittweite in Pixeln.
    mode : {"max", "mean"}
        Pooling-Modus.

    Returns
    -------
    pooled : 2-D ndarray
        Gepooltes Bild.

    Raises
    ------
    ValueError
        Bei unbekanntem Modus.
    """
    kh, kw = kernel
    sh, sw = stride
    h, w = img.shape

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    pooled = np.empty((out_h, out_w), dtype=img.dtype)

    for i in range(out_h):
        for j in range(out_w):
            window = img[i * sh : i * sh + kh, j * sw : j * sw + kw]
            if mode == "max":
                pooled[i, j] = np.max(window)
            elif mode == "mean":
                pooled[i, j] = np.mean(window)
            else:
                raise ValueError(f"unknown mode '{mode}', use 'max' or 'mean'")
    return pooled


# --- Bild laden ------------------------------------------------------------
url = (
    "https://github.com/dynamicslab/databook_python/"
    "raw/refs/heads/master/DATA/catData.mat"
)
cat = scipy.io.loadmat(io.BytesIO(requests.get(url).content))["cat"][:, 0]
cat_img = cat.reshape(64, 64).T

# --- Pooling anwenden -------------------------------------------------------
max_pooled = pool2d(cat_img, kernel=(2, 2), stride=(2, 2), mode="max")
mean_pooled = pool2d(cat_img, kernel=(2, 2), stride=(2, 2), mode="mean")

print("Original-Shape :", cat_img.shape)  # (64, 64)
print("Max-Pooling   :", max_pooled.shape)  # (32, 32)
print("Mean-Pooling  :", mean_pooled.shape)  # (32, 32)

# --- Visualisierung ---------------------------------------------------------
plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cat_img, cmap="gray"), plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Max-Pooling 2×2")
plt.imshow(max_pooled, cmap="gray"), plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Mean-Pooling 2×2")
plt.imshow(mean_pooled, cmap="gray"), plt.axis("off")

plt.tight_layout()
plt.show()

# Fazit:

# Max-Pooling bewahrt die stärksten Kanten und Kontraste jeder Region,
# wodurch markante Merkmale erhalten bleiben. Mean-Pooling liefert ein
# geglättetes Bild mit geringerem Rauschen, verliert jedoch feine Details.
