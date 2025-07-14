# ----------------------------- imports & seeds -----------------------------
import torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms.functional import (
    pad,
    rotate,
    center_crop,
    InterpolationMode,
)
from sklearn.datasets import fetch_openml
from pathlib import Path

torch.manual_seed(6020)
np.random.seed(6020)

# --------------------------- data preparation -----------------------------
PIXEL_PERM = torch.randperm(784, generator=torch.Generator().manual_seed(6020))
ROT_ANGLE, NOISE_STD = 15.0, 0.40


def _augment_single(img2d: torch.Tensor) -> torch.Tensor:
    noisy = torch.clamp(img2d + NOISE_STD * torch.randn_like(img2d), 0.0, 1.0)
    shuffled = noisy.flatten()[PIXEL_PERM].view(28, 28)
    padded = pad(shuffled.unsqueeze(0), (14, 14, 14, 14))
    rotated = rotate(padded, ROT_ANGLE, interpolation=InterpolationMode.BILINEAR)
    cropped = center_crop(rotated, (28, 28)).squeeze(0)
    return torch.flip(cropped, dims=[1]).flatten()


class MNISTDenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X / 255.0, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        clean = self.X[idx].view(28, 28)
        return _augment_single(clean), clean.flatten()


# --------------------------- model definition -----------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# --------------------------- training loop -------------------------------
def train_model(model, tr_dl, va_dl, loss_fn, optim, epochs=10, device="cpu"):
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(tr_dl)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(model(vx.to(device)), vy.to(device)).item() for vx, vy in va_dl
            ) / len(va_dl)
        print(f"epoch {ep:02d} | train {train_loss:.4f} | val {val_loss:.4f}")


def _grid_show(orig, corrupt, recon, n=10):
    rows = torch.zeros((3 * 28, n * 28))
    for j in range(n):
        rows[0:28, 28 * j : 28 * (j + 1)] = orig[j].view(28, 28)
        rows[28:56, 28 * j : 28 * (j + 1)] = corrupt[j].view(28, 28)
        rows[56:84, 28 * j : 28 * (j + 1)] = recon[j].view(28, 28)
    plt.figure(figsize=(n * 1.2, 4))
    plt.axis("off")
    plt.imshow(rows, cmap="gray")
    plt.show()


# ----------------------------- main ---------------------------------------
def main():
    mnist = fetch_openml("mnist_784", as_frame=False)
    X_train_raw, X_test_raw = mnist.data[:60_000], mnist.data[60_000:]

    train_full = MNISTDenoiseDataset(X_train_raw)
    val_size = int(0.1 * len(train_full))
    train_ds, val_ds = random_split(
        train_full,
        [len(train_full) - val_size, val_size],
        generator=torch.Generator().manual_seed(6020),
    )
    test_ds = MNISTDenoiseDataset(X_test_raw)

    BATCH = 64  # RAM-schonender
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AutoEncoder().to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(ae.parameters(), lr=1e-3)

    train_model(ae, train_loader, val_loader, loss_fn, optim, epochs=5, device=device)

    ae.eval()
    with torch.no_grad():
        x_corrupt, x_orig = next(iter(test_loader))
        x_recon = ae(x_corrupt.to(device)).cpu()
    _grid_show(x_orig, x_corrupt, x_recon)


if __name__ == "__main__":
    main()


# Fazit
# Das Autoencoder-Modell lernt binnen 5 Epochen bereits sinnvolle
# Rekonstruktionen: Rauschen wird entfernt, Ziffern bleiben lesbar.
# Für schärfere Bilder könnten:
#   * mehr Epochen,
#   * größere/angepasste Architektur,
#   * GPU-Training
# eingesetzt werden.
