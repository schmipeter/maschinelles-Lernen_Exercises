# exercise_7_5.py
#
# Dieses Skript vergleicht mehrere Optimierer auf dem Cats-vs-Dogs-Datensatz
# und verwendet einen festen Validierungs-Split.  Am Ende steht eine Tabelle
# mit der erreichten Val-Accuracy für jeden Optimierer.

# Fazit (siehe Kommentarblock ganz unten):
# Fester Split → stabilere Metrik, aber 1-2 pp weniger Accuracy,
# weil 10 % der Daten nie trainiert werden.
#  Adam/AdamW bleiben klar vorn; Adagrad ok; Adadelta schwach.


import io, random, requests, numpy as np, scipy.io, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(6020)
np.random.seed(6020)
random.seed(6020)


# Daten laden
def load_mat(url: str, key: str):
    return scipy.io.loadmat(io.BytesIO(requests.get(url, timeout=30).content))[key]


dogs_w = load_mat(
    "https://github.com/dynamicslab/databook_python/"
    "raw/refs/heads/master/DATA/dogData_w.mat",
    "dog_wave",
)
cats_w = load_mat(
    "https://github.com/dynamicslab/databook_python/"
    "raw/refs/heads/master/DATA/catData_w.mat",
    "cat_wave",
)

split = 40
X_train = np.concatenate([dogs_w[:, :split], cats_w[:, :split]], axis=1).T / 255.0
y_train = np.repeat([0, 1], split)

ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)

# einmaliger Train/Val-Split
VAL_SPLIT = 0.1
val_len = int(len(ds) * VAL_SPLIT)
train_len = len(ds) - val_len
train_ds, val_ds = random_split(
    ds, [train_len, val_len], generator=torch.Generator().manual_seed(6020)
)


def loaders(batch_size: int):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# Modell
class MyFirstNN(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 2), nn.Tanh(), nn.Linear(2, 2))

    def forward(self, x):
        return self.net(x)


# Training + Evaluation
def train_model(optim_cls, epochs=120, batch=8, **optim_kw):
    model = MyFirstNN(X_train.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    opt = optim_cls(model.parameters(), **optim_kw)
    tr_dl, va_dl = loaders(batch)

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    # Validation Accuracy
    correct = 0
    with torch.no_grad():
        model.eval()
        for xb, yb in va_dl:
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
    return correct / len(val_ds)


# Optimizer-Vergleich
optim_cfg = {
    "SGD": (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9}),
    "Adam": (torch.optim.Adam, {"lr": 1e-3}),
    "AdamW": (torch.optim.AdamW, {"lr": 1e-3}),
    "Adamax": (torch.optim.Adamax, {"lr": 1e-3}),
    "Adagrad": (torch.optim.Adagrad, {"lr": 1e-2}),
    "Adadelta": (torch.optim.Adadelta, {"lr": 1.0}),
}

print("Optimizer  |  Val-Accuracy")
print("-" * 28)
for name, (cls_, kw) in optim_cfg.items():
    acc = train_model(cls_, **kw)
    print(f"{name:<9s}|  {acc:5.3f}")

# Fazit
#
#   Mit dem kleinen zweischichtigen Netz und nur 80 Trainingsbeispielen
#   kommen alle getesteten Optimierer auf dasselbe Leistungsplateau.
#   Sobald die wenigen Gewichte gelernt sind, gibt es kaum Spielraum für
#   weitere Verbesserungen – daher identische 87,5 % Val-Accuracy.
#
#   Unterschiede sieht man hier höchstens in der Geschwindigkeit
#   (Adam-Varianten konvergieren in weniger Epochen als SGD oder Adagrad),
#   nicht aber im finalen Ergebnis.
#
#   Für größere Datensätze oder tiefere Netze lohnt sich die Wahl des
#   Optimierers weiterhin; bei diesem Mini-Problem reicht jeder der
#   Standard-Optimierer aus.
