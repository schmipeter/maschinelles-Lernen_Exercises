from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import logging


def get(components, estimators):
    logging.debug("Create classifier")
    voting_clf = make_pipeline(
        PCA(**components[0].init_args),
        VotingClassifier(
            estimators=[
                ("lda", LinearDiscriminantAnalysis(**estimators[0].init_args)),
                ("rf", RandomForestClassifier(**estimators[1].init_args)),
                ("svc", SVC(**estimators[2].init_args)),
            ],
            **components[1].init_args,
        )
    )
    return voting_clf


def evaluate(clf, data):
    dic = {}
    logging.debug("Score classifier")
    for X, y, prefix in data:
        score = clf.score(X, y)
        dic[prefix + "score"] = score
        logging.info(f"We have a hard voting {prefix}-score of {score}")
    return dic
