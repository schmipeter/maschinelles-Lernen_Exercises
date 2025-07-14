# exercise_7_8_dvclive.py
#
# Pytorch-Training mit festem Train/Val-Split
# + DVCLive-Tracking der Kennzahlen UND Checkpoints pro Epoche
#
# Installation (einmalig):
#   python -m pip install dvclive torch numpy scipy requests
#   # falls du DVC-Experimente nutzen möchtest:
#   dvc init
#
# Ausführen:
#   python exercise_7_8_dvclive.py
#   dvc exp show                       # tabellarische Übersicht
#   dvc plots show --target dvclive/plots  # Kurven im Browser
#   Alternativ start docs\index.html  (ohne dvclive)
# -----------------------------------------------------------------------------

import io
import random
from pathlib import Path

import numpy as np
import requests
import scipy.io
import torch
from dvclive import Live
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ─── feste Seeds für Reproduzierbarkeit ─────────────────────────────────────
torch.manual_seed(6020)
np.random.seed(6020)
random.seed(6020)


# ─── Wavelet-Daten laden ────────────────────────────────────────────────────
def load_wavelet(url: str, key: str) -> np.ndarray:
    buf = io.BytesIO(requests.get(url, timeout=30).content)
    return scipy.io.loadmat(buf)[key]


dogs = load_wavelet(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/dogData_w.mat",
    "dog_wave",
)
cats = load_wavelet(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/catData_w.mat",
    "cat_wave",
)

split = 40  # 40 Train-, 40 Test-Bilder je Klasse
X = np.concatenate([dogs[:, :split], cats[:, :split]], axis=1).T / 255.0
y = np.repeat([0, 1], split)

dataset = TensorDataset(
    torch.tensor(X, dtype=torch.float32),
    torch.tensor(y, dtype=torch.long),
)

# ─── fester 90 / 10-Split ───────────────────────────────────────────────────
val_len = int(len(dataset) * 0.1)
train_ds, val_ds = random_split(
    dataset,
    [len(dataset) - val_len, val_len],
    generator=torch.Generator().manual_seed(6020),
)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)


# ─── kleines Netz ───────────────────────────────────────────────────────────
class MyFirstNN(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─── Training + DVCLive-Logging ────────────────────────────────────────────
def train_with_live(model, train_dl, val_dl, epochs: int = 120):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    with Live(dir="dvclive") as live:  # DVCLive-Session
        # Checkpoint-Ordner anlegen, falls er noch nicht existiert
        ckpt_dir = Path(live.dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            # ─ training ───────────────────────────────────────────
            model.train()
            tr_loss = tr_corr = 0
            for xb, yb in train_dl:
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                tr_loss += loss.item()
                tr_corr += (out.argmax(1) == yb).sum().item()

            # ─ validation ────────────────────────────────────────
            model.eval()
            va_loss = va_corr = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    out = model(xb)
                    va_loss += loss_fn(out, yb).item()
                    va_corr += (out.argmax(1) == yb).sum().item()

            # ─ Metriken loggen ───────────────────────────────────
            live.log_metric("train_loss", tr_loss / len(train_dl))
            live.log_metric("val_loss", va_loss / len(val_dl))
            live.log_metric("train_acc", tr_corr / len(train_dl.dataset))
            live.log_metric("val_acc", va_corr / len(val_dl.dataset))

            # ─ Checkpoint sichern und als Artifact registrieren ──
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            live.log_artifact(str(ckpt_path), name=f"ckpt_{epoch:04d}")

            live.next_step()  # neue Epoche


# ─── Hauptprogramm ────────────────────────────────────────────────────────
if __name__ == "__main__":
    net = MyFirstNN(X.shape[1])
    train_with_live(net, train_dl, val_dl)

    # Ergebnis:
#   dvc exp show
#  ──────────────────────────────────────────────────────────────────────────────────────────────
# Experiment                 Created        train_loss   val_loss   train_acc   val_acc   step
# ──────────────────────────────────────────────────────────────────────────────────────────────
# workspace                  -               0.0044861    0.21393           1     0.875    119
# main                       Jun 01, 2025            -          -           -         -      -
# ├── 6d6e625 [soupy-sums]   12:14 AM        0.0044861    0.21393           1     0.875    119
# ├── cba466c [busty-dabs]   Jul 09, 2025      0.73707    0.79354     0.51389     0.375      -
# └── 9f52a4b [fazed-sous]   Jul 09, 2025    0.0044861    0.21393           1     0.875    119
# ──────────────────────────────────────────────────────────────────────────────────────────────


# Fazit

#   Nach ~120 Schritten erreicht das Netz auf dem Trainings-Set
#   eine Accuracy ≈ 1 und einen Loss → 0 → nahezu perfektes Memorieren.
#   Auf dem Validierungs-Set (8 Bilder) pendelt sich die Accuracy bei
#   ≈ 0.87 (7/8 korrekt) und der Loss bei ≈ 0.20 ein.
#    Die Divergenz zwischen sinkendem train_loss und stagnierendem val_loss
#    signalisiert beginnendes Overfitting.
#
#    Modell generalisiert akzeptabel auf das kleine Validierungs-Set,
#    doch die Aussagekraft ist wegen der geringen Datenmenge begrenzt.
#    Für robustere Ergebnisse wären mehr Daten, Data-Augmentation oder
#    Early-Stopping/Regulierung sinnvoll.
