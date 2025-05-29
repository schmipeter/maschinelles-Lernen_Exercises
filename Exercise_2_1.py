import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

# For consistent random numbers
np.random.seed(6020)

# ------------------------------------------------
# 1) TOY EXAMPLE (Two Elliptical Clusters)
# ------------------------------------------------

# Generate first ellipse centered at (0, 0), shape ~ [1, 0.5]
n = 200
X1 = np.random.randn(n, 2) * np.array([1, 0.5])

# Generate second ellipse centered roughly at (1, -2), shape ~ [1, 0.2], rotated by pi/4
X2 = (np.random.randn(n, 2) * np.array([1, 0.2])) + np.array([1, -2])
theta = np.pi / 4
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
X2 = X2 @ rotation

# Combine into one dataset; label them 0 (first ellipse) and 1 (second ellipse)
X_toy = np.vstack((X1, X2))
y_toy = np.concatenate((np.zeros(n), np.ones(n)))

# --- Fit LDA ---
lda_toy = LinearDiscriminantAnalysis()
lda_toy.fit(X_toy, y_toy)

# --- Create a mesh for decision boundary ---
x_min, x_max = X_toy[:, 0].min() - 1, X_toy[:, 0].max() + 1
y_min, y_max = X_toy[:, 1].min() - 1, X_toy[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = lda_toy.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --- Plot toy data + LDA decision boundary ---
plt.figure(figsize=(6,5))
plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")
plt.scatter(X_toy[:, 0], X_toy[:, 1], c=y_toy, cmap="viridis", edgecolor='k')
plt.title("Toy Example: LDA on Two Elliptical Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# ------------------------------------------------
# 2) IRIS DATASET: LDA on Versicolor vs. Virginica
# ------------------------------------------------

# Load Iris data (4D: sepal length, sepal width, petal length, petal width)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# We only want versicolor (class 1) and virginica (class 2)
mask = (y_iris != 0)      # Exclude class 0 (setosa)
X_iris_2class = X_iris[mask]
y_iris_2class = y_iris[mask]

# Re-label so that versicolor = 0, virginica = 1
# Original labels were {1, 2} -> subtract 1
y_iris_2class = y_iris_2class - 1

# Fit LDA on these two classes
lda_iris = LinearDiscriminantAnalysis()
lda_iris.fit(X_iris_2class, y_iris_2class)

# We can check accuracy on the training set
pred_iris = lda_iris.predict(X_iris_2class)
accuracy = np.mean(pred_iris == y_iris_2class)
print(f"LDA accuracy on versicolor vs. virginica: {accuracy:.2f}")

# For visualization, let's just plot the first two features (sepal length & sepal width)
X_plot = X_iris_2class[:, :2]

# Make a mesh for these first two features
x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z_iris = lda_iris.predict(np.c_[xx.ravel(), yy.ravel(), 
                                # We need to provide the other 2 features as well, 
                                # but let's fix them at their mean so we can plot a 2D slice:
                                X_iris_2class[:, 2].mean() * np.ones_like(xx.ravel()),
                                X_iris_2class[:, 3].mean() * np.ones_like(xx.ravel())])
Z_iris = Z_iris.reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, Z_iris, alpha=0.2, cmap="viridis")
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_iris_2class, cmap="viridis", edgecolor='k')
plt.title("Iris: LDA on Versicolor vs. Virginica (First 2 Features)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# Show all figures at once
plt.show()