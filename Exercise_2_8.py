import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier

# -----------------------------------------------------------------------------
# 1. Generate the Moons Dataset and split into train/test sets
# -----------------------------------------------------------------------------
X, y = make_moons(n_samples=500, noise=0.3, random_state=6020)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print("Moons Dataset:")
print("  Train samples:", X_train.shape[0])
print("  Test samples: ", X_test.shape[0])

# -----------------------------------------------------------------------------
# 2. Voting Classifier with LDA, SVM, and Decision Tree
# -----------------------------------------------------------------------------
# Define base estimators
lda = LinearDiscriminantAnalysis()
svm = SVC(kernel='rbf', probability=True, random_state=42)
tree = DecisionTreeClassifier(random_state=42)

# 2.1 Hard Voting (majority rule)
voting_clf_hard = VotingClassifier(
    estimators=[('lda', lda), ('svm', svm), ('tree', tree)],
    voting='hard'
)
voting_clf_hard.fit(X_train, y_train)

print("\nVoting Classifier (Hard Voting):")
print("  Train Score:", voting_clf_hard.score(X_train, y_train))
print("  Test Score: ", voting_clf_hard.score(X_test, y_test))

# 2.2 Soft Voting (average predicted probabilities)
voting_clf_soft = VotingClassifier(
    estimators=[('lda', lda), ('svm', svm), ('tree', tree)],
    voting='soft'
)
voting_clf_soft.fit(X_train, y_train)

print("\nVoting Classifier (Soft Voting):")
print("  Train Score:", voting_clf_soft.score(X_train, y_train))
print("  Test Score: ", voting_clf_soft.score(X_test, y_test))

# -----------------------------------------------------------------------------
# 3. Bagging Classifier using Decision Trees
# -----------------------------------------------------------------------------
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    max_samples=100,   # each estimator is trained on 100 samples (with replacement)
    bootstrap=True,
    oob_score=True,    # use out-of-bag samples to estimate the performance
    random_state=42
)
bag_clf.fit(X_train, y_train)

print("\nBagging Classifier:")
print("  Out-of-Bag (OOB) Score:", bag_clf.oob_score_)
print("  Train Score:", bag_clf.score(X_train, y_train))
print("  Test Score: ", bag_clf.score(X_test, y_test))

# -----------------------------------------------------------------------------
# 4. Random Forest Classifier
# -----------------------------------------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,  # limit complexity of each tree
    random_state=42
)
rf_clf.fit(X_train, y_train)

print("\nRandom Forest Classifier (max_leaf_nodes=16):")
print("  Train Score:", rf_clf.score(X_train, y_train))
print("  Test Score: ", rf_clf.score(X_test, y_test))

# -----------------------------------------------------------------------------
# 5. Extra Trees Classifier
# -----------------------------------------------------------------------------
et_clf = ExtraTreesClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    random_state=42
)
et_clf.fit(X_train, y_train)

print("\nExtra Trees Classifier (max_leaf_nodes=16):")
print("  Train Score:", et_clf.score(X_train, y_train))
print("  Test Score: ", et_clf.score(X_test, y_test))

# -----------------------------------------------------------------------------
# 6. Checking Feature Importances on the Iris Dataset
# -----------------------------------------------------------------------------
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

rf_iris = RandomForestClassifier(n_estimators=500, random_state=42)
rf_iris.fit(X_iris, y_iris)

print("\nIris Feature Importances (Random Forest):")
for name, importance in zip(iris.feature_names, rf_iris.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# If you wish to see a plot for the moons dataset, you can add:
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm', s=50)
plt.title("Moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()