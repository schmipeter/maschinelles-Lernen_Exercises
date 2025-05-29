"""
tune_ex_4_2.py
--------------
Mini-Benchmark für verschiedene Persistenz-Techniken (pickle, joblib, cloudpickle)
und SVC-Varianten.  Laufzeit lässt sich über drei Parameter steuern:

    TRAIN_SIZE   – Anzahl Trainings­beispiele pro Klasse
    RF_TREES     – Bäume im RandomForest
    SVC_C        – Komplexitäts­parameter des SVC

Ziel: Gesamt­laufzeit ~1 – 4 s bei aussagekräftigem Test-Score.
"""

# -------------------------------------------------- Imports
import io, time, pickle, joblib, cloudpickle, requests
from pathlib import Path
import numpy as np, scipy.io

from sklearn.decomposition import PCA
from sklearn.pipeline      import make_pipeline
from sklearn.ensemble      import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm           import SVC

# --------------- Stellschrauben (nach Belieben anpassen)
TRAIN_SIZE = 40          # je Klasse   (40–80)
RF_TREES   = 150         # Bäume       (80–200)
SVC_C      = 1.0         # SVC-Parameter (0.3–1.5)

# --------------- Daten laden (nur 2×TRAIN_SIZE+8 Zeilen)
BASE = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"
load = lambda fn: scipy.io.loadmat(io.BytesIO(requests.get(BASE + '/' + fn).content))
cats = load("catData_w.mat")["cat_wave"]
dogs = load("dogData_w.mat")["dog_wave"]

X_train = np.r_[cats[:TRAIN_SIZE],  dogs[:TRAIN_SIZE]]
y_train = np.r_[np.ones(TRAIN_SIZE), -np.ones(TRAIN_SIZE)]
X_test  = np.r_[cats[TRAIN_SIZE:TRAIN_SIZE+8], dogs[TRAIN_SIZE:TRAIN_SIZE+8]]
y_test  = np.r_[np.ones(8), -np.ones(8)]

# --------------- dump-Ordner (OneDrive-frei)
DUMP = Path(r"C:\tmp\ml_fast"); DUMP.mkdir(parents=True, exist_ok=True)

# --------------- Pipeline-Factory
def make_model(kernel="linear", prob=True):
    """erstellt PCA → Voting(LDA, RF, SVC)"""
    svc = SVC(
        kernel=kernel, probability=prob, C=SVC_C,
        shrinking=True, cache_size=200, random_state=6020
    )
    rf  = RandomForestClassifier(
        n_estimators=RF_TREES, max_leaf_nodes=2, random_state=6020
    )
    est = [("lda", LDA()), ("rf", rf), ("svc", svc)]
    return make_pipeline(PCA(n_components=41),
                         VotingClassifier(estimators=est,
                                          voting="hard",
                                          flatten_transform=False))

def report(clf, label):
    print(f"{label:<12s}  train={clf.score(X_train,y_train):.3f}  "
          f"test={clf.score(X_test,y_test):.3f}")

# ==================== Experiment ===========================
t0 = time.time()

# 1) Original-Modell (linear SVC, hard voting)
orig = make_model()
orig.fit(X_train, y_train)
report(orig, "ORIGINAL")
pickle.dump(orig, open(DUMP/"model.pkl", "wb"), protocol=5)

# 2) Soft-Voting-Variante
soft = pickle.loads(pickle.dumps(orig))       # kopieren
soft[1].voting = "soft"
soft[1].named_estimators_["svc"].probability = True
soft.fit(X_train, y_train)
report(soft, "SOFT")
joblib.dump(soft, DUMP/"model.joblib", compress=3)

# 3) SVC mit rbf-Kernel
rbf = make_model(kernel="rbf")
rbf.fit(X_train, y_train)
report(rbf, "SVC rbf")

# 4) Lambda-Kernel (SVC ohne Wahrscheinlich­keiten)
print("\nλ-Kernel …", end="", flush=True)
lambda_kernel = lambda x, y: np.exp(1e-2 * np.abs(x @ y.T))
lam = make_model(kernel=lambda_kernel, prob=False)
t1 = time.time()
lam.fit(X_train, y_train)
print(f" {time.time() - t1:.1f}s")
report(lam, "Lambda")
cloudpickle.dump(lam, open(DUMP/"lambda.cpkl", "wb"))

print(f"\nGesamt: {time.time() - t0:.2f}s  "
      f"(TRAIN_SIZE={TRAIN_SIZE}, RF={RF_TREES}, C={SVC_C})")