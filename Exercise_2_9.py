import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Load the Fisher Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # X: features, y: class labels (0,1,2)

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------------------------------------------
# 3. Train an SVM on the Iris dataset
# -----------------------------------------------------------------------------
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_clf.fit(X_train, y_train)

# Evaluate SVM
svm_train_score = svm_clf.score(X_train, y_train)
svm_test_score = svm_clf.score(X_test, y_test)
y_pred_svm = svm_clf.predict(X_test)

print("=== SVM Results ===")
print("Train Accuracy:", svm_train_score)
print("Test Accuracy: ", svm_test_score)

# Create confusion matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\nSVM Confusion Matrix:")
print(cm_svm)

# Optional: Detailed classification report (precision, recall, f1-score)
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

# -----------------------------------------------------------------------------
# 4. Train a Random Forest on the Iris dataset
# -----------------------------------------------------------------------------
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate Random Forest
rf_train_score = rf_clf.score(X_train, y_train)
rf_test_score = rf_clf.score(X_test, y_test)
y_pred_rf = rf_clf.predict(X_test)

print("\n=== Random Forest Results ===")
print("Train Accuracy:", rf_train_score)
print("Test Accuracy: ", rf_test_score)

# Create confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# -----------------------------------------------------------------------------
# 5. Plot the Confusion Matrices
# -----------------------------------------------------------------------------
# SVM confusion matrix
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=iris.target_names)
disp_svm.plot(values_format='d')
plt.title("SVM Confusion Matrix")
plt.show()

# Random Forest confusion matrix
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=iris.target_names)
disp_rf.plot(values_format='d')
plt.title("Random Forest Confusion Matrix")
plt.show()