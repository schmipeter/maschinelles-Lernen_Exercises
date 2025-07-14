# make_cm_png.py  –  erzeugt confusion_matrix.png aus Einzellabel-JSON
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1) JSON laden
recs = json.loads(Path("live/plots/sklearn/confusion_matrix.json").read_text())

y_true = np.array([int(r["actual"]) for r in recs])
y_pred = np.array([int(r["predicted"]) for r in recs])

# 2) Matrix berechnen  (Labels: 1 = cat, -1 = dog)
cm = confusion_matrix(y_true, y_pred, labels=[1, -1])

# 3) plotten
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["cat (+1)", "dog (-1)"]
)
disp.plot(colorbar=False)
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
print("PNG erzeugt:", Path("confusion_matrix.png").resolve())
