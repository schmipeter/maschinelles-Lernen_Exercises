# exercise_4_1_cats_dogs.py

import onnxruntime as ort
ort.set_default_logger_severity(3)     # 0=VERBOSE 1=INFO 2=WARNING 3=ERROR
from pathlib import Path
import io, requests, numpy as np, scipy.io
from sklearn.decomposition import PCA
from sklearn.pipeline     import make_pipeline
from sklearn.ensemble     import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm          import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import accuracy_score
from skl2onnx             import to_onnx

# ----------------------------------------------------
# 1) Daten laden (Katzen- & Hunde-Wavelets)
# ----------------------------------------------------
base = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"
cats_w = scipy.io.loadmat(io.BytesIO(requests.get(f"{base}/catData_w.mat").content))["cat_wave"]
dogs_w = scipy.io.loadmat(io.BytesIO(requests.get(f"{base}/dogData_w.mat").content))["dog_wave"]

X_train = np.concatenate((cats_w[:60],  dogs_w[:60]))
y_train = np.repeat([1, -1], 60)
X_test  = np.concatenate((cats_w[60:80], dogs_w[60:80]))
y_test  = np.repeat([1, -1], 20)

# ----------------------------------------------------
# 2) Hilfsfunktion: Sklearn-Score ➜ ONNX-Round-Trip-Score
# ----------------------------------------------------
def roundtrip(clf, name):
    clf.fit(X_train, y_train)
    sk_train = clf.score(X_train, y_train)
    sk_test  = clf.score(X_test , y_test )

    # --- Export -------------------------------------
    onx = to_onnx(clf, X_train[:1].astype(np.float32))
    Path(f"{name}.onnx").write_bytes(onx.SerializeToString())

    # --- Inference ----------------------------------
    sess = ort.InferenceSession(f"{name}.onnx", providers=["CPUExecutionProvider"])
    xkey = sess.get_inputs()[0].name
    onx_train = accuracy_score(y_train, sess.run(None, {xkey: X_train.astype(np.float32)})[0])
    onx_test  = accuracy_score(y_test , sess.run(None, {xkey: X_test .astype(np.float32)})[0])
    return sk_train, sk_test, onx_train, onx_test

# ----------------------------------------------------
# 3) Ensemble-Modelle (immer flatten_transform=False)
# ----------------------------------------------------
def voting(est, vote, **kw):
    return make_pipeline(PCA(n_components=41),
                         VotingClassifier(estimators=est,
                                          voting=vote,
                                          flatten_transform=False,
                                          **kw))

models = {
    "orig_soft_SVCprob":
        voting([
            ("lda", LDA()),
            ("rf" , RandomForestClassifier(n_estimators=500,
                                           max_leaf_nodes=2,
                                           random_state=6020)),
            ("svc", SVC(kernel="linear",
                        probability=True,
                        random_state=6020)),
        ], vote="soft"),

    "hard_SVC":
        voting([
            ("lda", LDA()),
            ("rf" , RandomForestClassifier(n_estimators=500,
                                           max_leaf_nodes=2,
                                           random_state=6020)),
            ("svc", SVC(kernel="linear",
                        random_state=6020)),
        ], vote="hard"),

    "hard_LinearSVC":
        voting([
            ("lda" , LDA()),
            ("rf"  , RandomForestClassifier(n_estimators=500,
                                            max_leaf_nodes=2,
                                            random_state=6020)),
            ("linsvc", LinearSVC(random_state=6020)),
        ], vote="hard"),

    "soft_RF+LR":
        voting([
            ("lda", LDA()),
            ("rf" , RandomForestClassifier(n_estimators=500,
                                           max_leaf_nodes=2,
                                           random_state=6020)),
            ("lr" , LogisticRegression(max_iter=1000,
                                       random_state=6020)),
        ], vote="soft"),
}

# ----------------------------------------------------
# 4) Experiment durchführen & Ergebnisse ausgeben
# ----------------------------------------------------
print(f"{'MODEL':20s}  sklearn  sklearn  ONNX  ONNX")
print(f"{'':20s}  train    test    train test")

for name, clf in models.items():
    sk_tr, sk_te, onx_tr, onx_te = roundtrip(clf, name)
    print(f"{name:20s}  {sk_tr:.3f}   {sk_te:.3f}   {onx_tr:.3f}  {onx_te:.3f}")