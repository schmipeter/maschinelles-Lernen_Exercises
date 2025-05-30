import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA

# ----------------------------------------------
# 1. DBSCAN on a Toy Example: The Two-Moons Dataset
# ----------------------------------------------

# Generate the two-moons dataset
X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

# Define a set of epsilon values and metrics to try.
eps_values = [0.1, 0.3, 0.5]
metrics = ['euclidean', 'manhattan']

# Create subplots for each combination of eps and metric.
fig, axes = plt.subplots(len(metrics), len(eps_values), figsize=(15, 8))
for i, metric in enumerate(metrics):
    for j, eps in enumerate(eps_values):
        # Create and fit DBSCAN for current parameters.
        db = DBSCAN(eps=eps, min_samples=5, metric=metric)
        labels = db.fit_predict(X_moons)
        
        # Plot the clustering result.
        ax = axes[i, j]
        sc = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap="viridis", 
                        edgecolor="k", s=30)
        ax.set_title(f"eps={eps}, metric={metric}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        
plt.suptitle("DBSCAN on Two-Moons: Varying eps and metric", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# ----------------------------------------------
# 2. DBSCAN on the Iris Dataset (Higher Dimensional)
# ----------------------------------------------

# Load the Iris dataset.
iris = load_iris()
X_iris = iris.data  # 4-dimensional features
y_iris = iris.target

# Apply DBSCAN.
# The eps value here is chosen as a starting guess; you may experiment with other values.
db_iris = DBSCAN(eps=0.8, min_samples=5, metric='euclidean')
labels_iris = db_iris.fit_predict(X_iris)

# Count the number of clusters found (ignoring noise, labeled as -1).
n_clusters = len(set(labels_iris)) - (1 if -1 in labels_iris else 0)
n_noise = list(labels_iris).count(-1)
print(f"DBSCAN on Iris dataset: Found {n_clusters} clusters with {n_noise} noise points.")

# For visualization, project the 4D data to 2D using PCA.
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris)

plt.figure(figsize=(6, 5))
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=labels_iris, cmap="viridis", edgecolor="k", s=50)
plt.title("DBSCAN Clustering on Iris (PCA-projected to 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()