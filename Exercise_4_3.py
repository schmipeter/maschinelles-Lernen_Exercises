"""
exercise_4_3_safe.py
--------------------
Persistieren mit skops.io

1. Original-Ensemble (PCA + LDA/RF/SVC)   →  *.skops
2. PCA-Gewichte nachträglich auf float16  →  deutlich kleinere Datei
3. Eigenes Kernel (benannte Funktion!)    →  speichern & laden

Hinweis:   Lambda-Ausdrücke funktionieren nicht mit skops,
           weil sie beim Laden nicht mehr importierbar sind.
"""

# -------------------------------------------------- Imports
import io, time, pickle, requests, skops.io as sio
from pathlib import Path
import numpy as np, scipy.io
from sklearn.decomposition import PCA
from sklearn.pipeline      import make_pipeline
from sklearn.ensemble      import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm           import SVC
from sklearn.metrics       import accuracy_score

# --------------------------- Hyper-Parameter (nach Belieben)
TRAIN_SIZE = 40        # je Klasse
TEST_SIZE  = 16
PCA_WISH   = 80       # wird auf zulässiges Maximum gekappt
RF_TREES   = 40
SVC_C      = 1.0

DUMP = Path(r"C:\tmp\skops_demo")
DUMP.mkdir(parents=True, exist_ok=True)

# --------------------------- Datensatz laden (Katzen/Hunde)
BASE = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"
load = lambda f: scipy.io.loadmat(io.BytesIO(requests.get(BASE + f).content))
cats = load("/catData_w.mat")["cat_wave"]
dogs = load("/dogData_w.mat")["dog_wave"]

X_train = np.r_[cats[:TRAIN_SIZE],  dogs[:TRAIN_SIZE]]
y_train = np.r_[np.ones(TRAIN_SIZE), -np.ones(TRAIN_SIZE)]
X_test  = np.r_[cats[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE//2],
                dogs[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE//2]]
y_test  = np.r_[np.ones(TEST_SIZE//2), -np.ones(TEST_SIZE//2)]

n_feat   = X_train.shape[1]
n_samp   = X_train.shape[0]
PCA_NCOM = min(PCA_WISH, n_feat, n_samp)   # garantiert gültig

# --------------------------- Benannter Kernel (statt Lambda)
def exp_kernel(x, y):
    """e^(0.01 * |x·yᵀ|) – Beispiel für benutzerdefinierten Kernel."""
    return np.exp(1e-2 * np.abs(x @ y.T))

# --------------------------- Pipeline-Factory
def make_model(svc_kernel="linear", svc_prob=True):
    svc = SVC(kernel=svc_kernel, probability=svc_prob,
              C=SVC_C, shrinking=True, cache_size=200, random_state=42)
    rf  = RandomForestClassifier(n_estimators=RF_TREES,
                                 max_leaf_nodes=2, random_state=42)
    est = [("lda", LDA()), ("rf", rf), ("svc", svc)]
    return make_pipeline(
        PCA(n_components=PCA_NCOM),
        VotingClassifier(estimators=est,
                         voting="hard",
                         flatten_transform=False)
    )

def score(clf, tag):
    tr = accuracy_score(y_train, clf.predict(X_train))
    te = accuracy_score(y_test , clf.predict(X_test ))
    print(f"{tag:<18s} train={tr:.3f}  test={te:.3f}")

def kb(p): return f"{Path(p).stat().st_size//1024} KB"

# =================================================================
# 1) Original-Modell
# =================================================================
orig = make_model()
orig.fit(X_train, y_train)
score(orig, "ORIGINAL")

orig_path = DUMP / "model_orig.skops"
sio.dump(orig, orig_path)
print("orig size:        ", kb(orig_path))

# =================================================================
# 2) PCA → float16
# =================================================================
compressed = pickle.loads(pickle.dumps(orig))        # DeepCopy
pca = compressed[0]
pca.components_ = pca.components_.astype(np.float16, copy=False)
pca.mean_       = pca.mean_.astype(np.float16, copy=False)

score(compressed, "PCA float16")

f16_path = DUMP / "model_float16.skops"
sio.dump(compressed, f16_path)
print("float16 size:     ", kb(f16_path))

trusted = sio.get_untrusted_types(file=f16_path)
score(sio.load(f16_path, trusted=trusted), "float16 loaded")

# =================================================================
# 3) Modell mit eigenem Kernel
# =================================================================
lam = make_model(svc_kernel=exp_kernel, svc_prob=False)

print("\nExp-Kernel wird trainiert …", end="", flush=True)
t0 = time.time()
lam.fit(X_train, y_train)
print(f" fertig nach {time.time()-t0:.1f}s")

score(lam, "Exp kernel")

lam_path = DUMP / "model_exp_kernel.skops"
sio.dump(lam, lam_path)
print("exp size:         ", kb(lam_path))

# beim ersten Laden unbekannte Typen prüfen
unknown = sio.get_untrusted_types(file=lam_path)
# __main__.exp_kernel erscheint hier → bewusst vertrauen
lam_loaded = sio.load(lam_path, trusted=unknown)
score(lam_loaded, "Exp loaded")