import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 1. Generate the two moons dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# A helper function to plot the decision boundary
def plot_decision_boundary(clf, X, y, title, ax):
    """
    Plots the decision boundary of an SVM classifier (clf)
    and also plots the training points (X, y) on the same axes (ax).
    """
    # Define a grid for plotting
    x0s = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    x1s = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_grid = np.c_[x0.ravel(), x1.ravel()]
    
    # Get the decision function values (distance from boundary)
    Z = clf.decision_function(X_grid)
    Z = Z.reshape(x0.shape)
    
    # Plot decision boundaries using contour
    ax.contourf(x0, x1, Z, levels=[-1e9, 0, 1e9], alpha=0.2, cmap="coolwarm")
    ax.contour(x0, x1, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], 
               colors="black", alpha=0.7)
    
    # Plot the dataset points
    ax.scatter(X[y==0, 0], X[y==0, 1], c="blue", marker="s", label="Class 0")
    ax.scatter(X[y==1, 0], X[y==1, 1], c="red",  marker="^", label="Class 1")
    
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.legend(loc="upper right")

# 2. Prepare four SVM classifiers with different (gamma, C) values
param_combinations = [
    (0.1, 0.001),
    (5,   0.001),
    (0.1, 1000),
    (5,   1000)
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for (gamma_val, C_val), ax in zip(param_combinations, axes.ravel()):
    # Create and fit the SVM model
    svm_clf = SVC(kernel="rbf", gamma=gamma_val, C=C_val)
    svm_clf.fit(X, y)
    
    # Plot the decision boundary
    title = fr"$\gamma={gamma_val}, C={C_val}$"
    plot_decision_boundary(svm_clf, X, y, title, ax)

plt.tight_layout()
plt.show()