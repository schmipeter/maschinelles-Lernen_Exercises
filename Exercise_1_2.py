import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# ------------------------------
# 1. Agglomerative Clustering on the Iris Data Set
# ------------------------------

# Load the Iris dataset
iris = load_iris()
X_iris = iris.data      # shape (150, 4)
y_iris = iris.target

## A) 2D version (using the first two features)
X_iris_2d = X_iris[:, :2]  # sepal length and sepal width

# Use Ward linkage (which minimizes variance) for agglomerative clustering
agg_iris_2d = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels_iris_2d = agg_iris_2d.fit_predict(X_iris_2d)

plt.figure(figsize=(5, 4))
plt.scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_iris_2d, cmap="viridis", edgecolor="k")
plt.title("Iris (2D) - Agglomerative Clustering (Ward)")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

## B) 4D version (using all four features)
agg_iris_4d = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels_iris_4d = agg_iris_4d.fit_predict(X_iris)

# Since the data is 4D, project to 2D using PCA for visualization
pca_iris = PCA(n_components=2)
X_iris_4d_pca = pca_iris.fit_transform(X_iris)

plt.figure(figsize=(5, 4))
plt.scatter(X_iris_4d_pca[:, 0], X_iris_4d_pca[:, 1], c=labels_iris_4d, cmap="viridis", edgecolor="k")
plt.title("Iris (4D) - Agglomerative Clustering (Ward) projected to 2D")
plt.xlabel("PC 1")
plt.ylabel("PC 2")


# ------------------------------
# 2. Agglomerative Clustering on the Moons Data Set
# ------------------------------

# Generate the two-moons dataset
X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

# Define the linkage options to explore
linkage_options = ["ward", "complete", "average", "single"]

# Create subplots to display the results for each linkage type
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, linkage in enumerate(linkage_options):
    # Create and fit the model with the current linkage option
    agg_moons = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    labels_moons = agg_moons.fit_predict(X_moons)
    
    # Plot the clustering result
    ax = axes[i]
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap="viridis", edgecolor="k")
    ax.set_title(f"Moons Clustering ({linkage} linkage)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

plt.tight_layout()
plt.show()