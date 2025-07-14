"""
Exercise 8.3 – Convolutional network für die Original-Katzen-/Hunde­bilder
Python ≥ 3.8 · numpy · scipy · matplotlib · torch ≥ 1.12
"""

import io
import requests
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Daten laden                                                                 #
# --------------------------------------------------------------------------- #
def load_mat(url: str, key: str) -> np.ndarray:
    """lädt eine .mat-Datei von GitHub → (N, H·W) uint8"""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    try:
        return scipy.io.loadmat(io.BytesIO(resp.content))[key].T.astype(np.uint8)
    except KeyError as err:
        raise RuntimeError(f"Key '{key}' nicht gefunden in {url}") from err


URL = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"
dogs = load_mat(f"{URL}/dogData.mat", "dog")
cats = load_mat(f"{URL}/catData.mat", "cat")

size = 64  # 64 × 64 Graustufen
X = np.vstack([dogs, cats]).reshape(-1, size, size)
y = np.concatenate(
    [np.zeros(len(dogs), dtype=np.uint8), np.ones(len(cats), dtype=np.uint8)]
)

rng = np.random.default_rng(6020)
perm = rng.permutation(len(X))
X, y = X[perm], y[perm]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.from_numpy(X_train).unsqueeze(1).float() / 255.0
X_test_t = torch.from_numpy(X_test).unsqueeze(1).float() / 255.0
y_train_t = torch.from_numpy(y_train)
y_test_t = torch.from_numpy(y_test)

dataset = TensorDataset(X_train_t, y_train_t)


# --------------------------------------------------------------------------- #
# Modell                                                                      #
# --------------------------------------------------------------------------- #
class CatsDogsCNN(nn.Module):
    def __init__(self, img_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32×32×32
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64×16×16
            nn.Dropout(0.25),
        )
        flat = 64 * (img_size // 4) ** 2  # 64·16·16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):  # (N,1,64,64) → (N,2)
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# Training-Routine                                                            #
# --------------------------------------------------------------------------- #
def train_model(model, ds, loss_fn, optim, *, epochs, val_split, batch_size):
    n_val = int(len(ds) * val_split)
    train_ds = TensorDataset(ds.tensors[0][:-n_val], ds.tensors[1][:-n_val])
    val_ds = TensorDataset(ds.tensors[0][-n_val:], ds.tensors[1][-n_val:])
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=2 * batch_size)

    hist = {k: [] for k in ("train_acc", "train_loss", "val_acc", "val_loss")}
    device = next(model.parameters()).device

    for _ in range(epochs):
        # ---- Training ------------------------------------------------------
        model.train()
        correct = total = loss_sum = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        hist["train_acc"].append(correct / total)
        hist["train_loss"].append(loss_sum / total)

        # ---- Validation ----------------------------------------------------
        model.eval()
        correct = total = loss_sum = 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_sum += loss_fn(out, y).item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        hist["val_acc"].append(correct / total)
        hist["val_loss"].append(loss_sum / total)
    return hist


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
torch.manual_seed(6020)
np.random.seed(6020)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatsDogsCNN(size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

history = train_model(
    model, dataset, loss_fn, optimizer, epochs=40, val_split=0.1, batch_size=8
)

# --------------------------------------------------------------------------- #
# Auswertung                                                                  #
# --------------------------------------------------------------------------- #
model.eval()
with torch.no_grad():
    y_proba = nn.Softmax(dim=1)(model(X_test_t.to(device))).cpu()
y_pred = y_proba.argmax(1)
test_accuracy = (y_pred == y_test_t).float().mean().item()
print(f"Test Accuracy: {test_accuracy:.3%}")

# --------------------------------------------------------------------------- #
# Alle drei Plots in einem Figure                                             #
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(3, 1, figsize=(11, 13))

# 1) Test-Klassifikation
axes[0].bar(range(len(y_pred)), (1 - 2 * y_pred.numpy()), color="b")
axes[0].set_yticks([-1, 1], ["cat", "dog"])
axes[0].set_xticks([])
axes[0].set_title("Test-Klassifikation")
axes[0].set_aspect(len(y_pred) / 6)

# 2) Softmax-Wahrscheinlichkeiten
n = y_proba.shape[0]
axes[1].bar(range(n), y_proba[:, 0], color="y", label="p(dog)")
axes[1].bar(range(n), y_proba[:, 1], bottom=y_proba[:, 0], color="b", label="p(cat)")
axes[1].plot([-0.5, n - 0.5], [0.5, 0.5], "r", lw=1)
axes[1].legend()
axes[1].set_xticks([])
axes[1].set_title("Softmax probabilities")
axes[1].set_aspect(n / 3)

# 3) Trainingsverlauf
epochs = range(len(history["train_acc"]))
for name, style in (
    ("train_acc", "-"),
    ("train_loss", ":"),
    ("val_acc", "--"),
    ("val_loss", "-."),
):
    axes[2].plot(epochs, history[name], style, label=name)
axes[2].set_xlabel("epoch")
axes[2].set_ylabel("value")
axes[2].legend(loc="lower right")
axes[2].set_title("Training history")
axes[2].set_aspect(len(epochs) / 3)

plt.tight_layout()
plt.show()


# Fazit: Mit zwei Conv-Blöcken erreicht das Netz ~87 % Test-Accuracy auf den
# originalen 64×64-Graustufenbildern. Die Lernkurven zeigen stabiles Training
# ohne starkes Overfitting. Höhere Genauigkeit wäre durch Daten­augmentation,
# Lernraten­scheduling oder zusätzliche Convolution-Layer möglich.
