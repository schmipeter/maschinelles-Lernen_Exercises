# exercise_7_6.py

# Variante aus Exercise 7.5, diesmal mit Softmax-Layer im Modell.

import io
import random
import requests
import numpy as np
import scipy.io
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(6020)
np.random.seed(6020)
random.seed(6020)


# Daten laden
def load_mat(url: str, key: str):
    buf = io.BytesIO(requests.get(url, timeout=30).content)
    return scipy.io.loadmat(buf)[key]


dogs = load_mat(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/dogData_w.mat",
    "dog_wave",
)
cats = load_mat(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/catData_w.mat",
    "cat_wave",
)

split = 40
X = np.concatenate([dogs[:, :split], cats[:, :split]], axis=1).T / 255.0
y = np.repeat([0, 1], split)

dataset = TensorDataset(
    torch.tensor(X, dtype=torch.float32),
    torch.tensor(y, dtype=torch.long),
)

# fester Train/Val-Split
val_len = int(len(dataset) * 0.1)
train_ds, val_ds = random_split(
    dataset,
    [len(dataset) - val_len, val_len],
    generator=torch.Generator().manual_seed(6020),
)


def get_loaders(bs=8):
    return (
        DataLoader(train_ds, bs, shuffle=True),
        DataLoader(val_ds, bs, shuffle=False),
    )


# Modell mit Softmax
class SoftmaxNN(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


# Training & Evaluation
def train_val_acc(epochs=120, batch=8):
    model = SoftmaxNN(X.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_dl, val_dl = get_loaders(batch)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    correct = 0
    with torch.no_grad():
        model.eval()
        for xb, yb in val_dl:
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(val_ds)


if __name__ == "__main__":
    acc = train_val_acc()
    print(f"Val-Accuracy mit Softmax: {acc:.3f}")

# Ergebnis in meinem Lauf: Val-Accuracy mit Softmax: 0.875

# Fazit: Der zusätzliche Softmax-Layer ändert bei diesem kleinen Netz und Datensatz
# nichts am finalen Ergebnis gegenüber Exercise 7.5.
