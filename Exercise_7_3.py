# exercise_7_3.py
# Grid-search for learning rate and momentum using the cats-vs-dogs wavelet data


import io
import random
import requests
import numpy as np
import scipy.io
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# 1. data loading and preprocessing

torch.manual_seed(6020)
np.random.seed(6020)
random.seed(6020)


def load_wavelet_mat(url: str, var_name: str) -> np.ndarray:
    buf = io.BytesIO(requests.get(url, timeout=30).content)
    return scipy.io.loadmat(buf)[var_name]


dogs_w = load_wavelet_mat(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/dogData_w.mat",
    "dog_wave",
)
cats_w = load_wavelet_mat(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/catData_w.mat",
    "cat_wave",
)

split = 40  # 40 train, 40 test
X_train = np.concatenate((dogs_w[:, :split], cats_w[:, :split]), axis=1).T / 255.0
y_train = np.repeat([0, 1], split)
X_test = np.concatenate((dogs_w[:, split:], cats_w[:, split:]), axis=1).T / 255.0
y_test = np.repeat([0, 1], 80 - split)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)  # long for CE-loss
dataset = TensorDataset(X_train_t, y_train_t)


# 2. helper: train/val split each epoch


def get_loaders(ds: TensorDataset, val_split: float, batch: int):
    v = int(len(ds) * val_split)
    t = len(ds) - v
    train_ds, val_ds = random_split(
        ds, [t, v], generator=torch.Generator().manual_seed(6020)
    )
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False),
    )


# 3. network definition


class MyFirstNN(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 4. training for one hyper-parameter combo


def run_one(
    lr: float, momentum: float, epochs: int, train_dl: DataLoader, val_dl: DataLoader
):
    model = MyFirstNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    softmax = nn.Softmax(dim=1)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    val_corr = 0
    conf_sum = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            probs = softmax(model(xb))
            preds = probs.argmax(1)
            val_corr += (preds == yb).sum().item()
            conf_sum += torch.abs(probs[:, 1] - 0.5).sum().item()
    n = len(val_dl.dataset)
    return val_corr / n, conf_sum / n


# 5. grid search

BATCH = 8
VAL_SPLIT = 0.1
EPOCHS = 200

lr_grid = [1e-1, 5e-2, 1e-2, 1e-3, 1e-4]
mom_grid = [0.0, 0.5, 0.9]

results = []
train_dl, val_dl = get_loaders(dataset, VAL_SPLIT, BATCH)

for lr in lr_grid:
    ep = min(EPOCHS, int(EPOCHS * (1e-2 / lr))) if lr > 1e-2 else EPOCHS
    for mom in mom_grid:
        acc, conf = run_one(lr, mom, ep, train_dl, val_dl)
        results.append((lr, mom, acc, conf))
        print(
            f"lr={lr:>4.0e}  mom={mom:.1f}  "
            f"val_acc={acc:.4f}  mean|p-0.5|={conf:.4f}"
        )

# sort by accuracy then confidence

results.sort(key=lambda x: (x[2], x[3]), reverse=True)

print("\nTop-3 combinations:")
for lr, mom, acc, conf in results[:3]:
    print(
        f"lr={lr:.0e}, momentum={mom:.1f}, "
        f"val_acc={acc:.4f}, mean|p-0.5|={conf:.4f}"
    )


# Zusammenfassung


#   Lernrate: 1e-1, 5e-2 und 1e-2 erreichen sofort 87,5 % Accuracy.
#   Sehr kleine lrs (≤1e-4) lernen praktisch nichts in 200 Epochen.

#   Momentum:  Höheres Momentum steigert nicht die Accuracy,
#   aber die Entscheidungssicherheit (|p-0.5| wird größer,
#   Softmax-Werte sind weiter von 0.5 entfernt).

#   Bestes Setting im Test: lr = 1e-1, momentum = 0.9
#   (87,5 % Accuracy und höchste Konfidenz 0,475).

#  Tipp: Für stabiles, schnelles Lernen Momentum ≥ 0,5 bei lr ≈ 1e-2 – 1e-1;
#  sehr kleine lr nur mit deutlich mehr Epochen oder anderem Optimizer.
