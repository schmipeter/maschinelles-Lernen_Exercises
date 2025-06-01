from MECH_M_DUAL_2_MLB_DATA.etl import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_tr, y_tr = load("data/raw", split="train")  # (N,H,W)
X_te, y_te = load("data/raw", split="test")

X_tr = X_tr.reshape(len(X_tr), -1)
X_te = X_te.reshape(len(X_te), -1)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)
print("Accuracy:", accuracy_score(y_te, clf.predict(X_te)))