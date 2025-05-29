from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import logging

def get_model():
    logging.debug("Create classifier")
    voting_clf = make_pipeline(
        PCA(n_components=41),
        VotingClassifier(
            estimators=[
                ("lda", LinearDiscriminantAnalysis()),
                ("rf", RandomForestClassifier(
                    n_estimators=500,
                    max_leaf_nodes=2,
                    random_state=6020)),
                ("svc", SVC(
                    kernel="linear",
                    probability=True,
                    random_state=6020)),
            ],
            flatten_transform=False,
        )
    )
    return voting_clf