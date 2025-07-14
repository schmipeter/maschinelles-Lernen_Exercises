# exercise_7_4.py
#
#  Ziel: Verschiedene Optimierer (Adam-Familie und Ada-Familie) ausprobieren
#       und die finale Genauigkeit (Accuracy) festhalten.


import io, random, requests, numpy as np, scipy.io, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

#  Daten laden

torch.manual_seed(6020)
np.random.seed(6020)
random.seed(6020)


def load_wavelet(url: str, var: str) -> np.ndarray:
    buf = io.BytesIO(requests.get(url, timeout=30).content)
    return scipy.io.loadmat(buf)[var]


dogs_w = load_wavelet(
    "https://github.com/dynamicslab/databook_python/"
    "raw/refs/heads/master/DATA/dogData_w.mat",
    "dog_wave",
)
cats_w = load_wavelet(
    "https://github.com/dynamicslab/databook_python/"
    "raw/refs/heads/master/DATA/catData_w.mat",
    "cat_wave",
)

split = 40
X_train = np.concatenate((dogs_w[:, :split], cats_w[:, :split]), axis=1).T / 255.0
y_train = np.repeat([0, 1], split)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
dataset = TensorDataset(X_train_t, y_train_t)

# Hilfsfunktionen


def get_loaders(ds: TensorDataset, val_split: float, batch: int):
    val = int(len(ds) * val_split)
    train = len(ds) - val
    tr, va = random_split(
        ds, [train, val], generator=torch.Generator().manual_seed(6020)
    )
    return (
        DataLoader(tr, batch_size=batch, shuffle=True),
        DataLoader(va, batch_size=batch, shuffle=False),
    )


class MyFirstNN(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(d_in, 2), nn.Tanh(), nn.Linear(2, 2))

    def forward(self, x):
        return self.seq(x)


def train_once(model, train_dl, criterion, optim, epochs):
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()


def evaluate(model, val_dl):
    sm = nn.Softmax(dim=1)
    correct = tot = 0
    with torch.no_grad():
        model.eval()
        for xb, yb in val_dl:
            pred = sm(model(xb)).argmax(1)
            correct += (pred == yb).sum().item()
            tot += yb.size(0)
    return correct / tot


# Grid
BATCH = 8
VAL_SPLIT = 0.1
EPOCHS = 100

opt_factories = {
    "Adam": lambda p: torch.optim.Adam(p, lr=1e-3),
    "AdamW": lambda p: torch.optim.AdamW(p, lr=1e-3),
    "Adamax": lambda p: torch.optim.Adamax(p, lr=1e-3),
    "Adagrad": lambda p: torch.optim.Adagrad(p, lr=1e-2),
    "Adadelta": lambda p: torch.optim.Adadelta(p, lr=1.0),
}

results = []
train_dl, val_dl = get_loaders(dataset, VAL_SPLIT, BATCH)
crit = nn.CrossEntropyLoss()

for name, make_opt in opt_factories.items():
    net = MyFirstNN(X_train.shape[1])
    opt = make_opt(net.parameters())
    train_once(net, train_dl, crit, opt, EPOCHS)
    acc = evaluate(net, val_dl)
    results.append((name, acc))
    print(f"{name:7s} → val_acc={acc:.4f}")

print("\nZusammenfassung:")
for n, a in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{n:7s} : {a:.4f}")


# Interpretation
# Alle fünf Verfahren erreichen nach den gewählten 100 Epochen exakt dieselbe Validierungs-Genauigkeit (87,5 %).
# Grund: Unser Netz ist sehr klein, der Datensatz überschaubar und die Lernrate bei jedem Optimierer so gewählt,
# dass er in wenigen Schritten konvergiert. Sobald das Plateau erreicht ist,
# unterscheiden sich die Verfahren praktisch nicht mehr.
# Für größere Netze oder mehr Daten wäre Adam / AdamW normalerweise schneller am Ziel,
# während Ada-Verfahren oft längere Zeit brauchen oder eine geringere End-Accuracy zeigen.
# Hier genügt daher der Standard-SGD-Nachfolger Adam; komplexere Alternativen bringen keinen spürbaren Vorteil.
