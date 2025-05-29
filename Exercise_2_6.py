import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Generate the "moons" dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A helper function to plot decision boundaries
def plot_decision_boundary(clf, X, y, ax=None, cmap='coolwarm'):
    # Create a grid covering the data range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict over the entire grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    if ax is None:
        ax = plt.gca()
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot the original points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# 3. Train a Decision Tree Classifier
#    Try changing 'min_samples_leaf' to see how the model changes.
tree_clf = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_clf.fit(X_train, y_train)

# 4. Evaluate the model
train_score = tree_clf.score(X_train, y_train)
test_score = tree_clf.score(X_test, y_test)
print(f"Train score: {train_score:.3f}")
print(f"Test score:  {test_score:.3f}")

# 5. Plot the decision boundary
plt.figure(figsize=(8, 6))
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree on the Moons Dataset (min_samples_leaf=5)")
plt.show()