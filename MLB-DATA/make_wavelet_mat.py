import numpy as np, math, pywt, scipy.io as sio
from pathlib import Path


def rescale(x, nb=256):
    x = np.abs(x)
    x = nb * (x - x.min()) / (x.max() - x.min())
    x = 1 + np.fix(x)
    x[x > nb] = nb
    return x


def process(src: str, dst: str, key: str):
    print(f"→ verarbeite {src}")
    mat = sio.loadmat(src)[key].T  # (n_img, pixels)
    l, w = mat.shape  # l = pixels, w = #bilder
    imgs_w = np.zeros((l // 4, w))
    for i in range(w):
        A = mat[:, i].reshape(math.isqrt(l), -1)
        _, (cH, cV, _) = pywt.wavedec2(A, "haar", level=1)
        imgs_w[:, i] = (rescale(cH) + rescale(cV)).ravel()
    sio.savemat(dst, {f"{key}_wave": imgs_w})
    print(f"✓ {dst} geschrieben")


process("catData.mat", "catData_w.mat", "cat")
process("dogData.mat", "dogData_w.mat", "dog")
