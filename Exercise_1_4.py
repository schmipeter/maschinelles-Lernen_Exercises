import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA

# --------------------------------------------------
# Section 1: Toy Example (Synthetic Gaussian Data)
# --------------------------------------------------

# Generate synthetic data: two Gaussian clusters
np.random.seed(42)
n_samples = 200

# Cluster 1: centered at (0,0)
mean1 = [0, 0]
cov1 = [[1, 0.5], [0.5, 1]]
cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# Cluster 2: centered at (5,5)
mean2 = [5, 5]
cov2 = [[1, -0.3], [-0.3, 1]]
cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# Combine data and create ground-truth labels
X_toy = np.vstack((cluster1, cluster2))
y_toy = np.array([0]*n_samples + [1]*n_samples)

# Plot the ground truth of the toy example
plt.figure(figsize=(6, 5))
plt.scatter(X_toy[:, 0], X_toy[:, 1], c=y_toy, cmap='viridis', edgecolor='k')
plt.title("Toy Example: Ground Truth")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Fit GMM with different covariance types
cov_types = ['full', 'tied', 'diag', 'spherical']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, cov_type in enumerate(cov_types):
    gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=42)
    gmm.fit(X_toy)
    labels_toy = gmm.predict(X_toy)
    
    ax = axes[i]
    ax.scatter(X_toy[:, 0], X_toy[:, 1], c=labels_toy, cmap='viridis', edgecolor='k')
    ax.set_title(f"GMM (covariance_type='{cov_type}')")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()

# --------------------------------------------------
# Section 2: Iris Dataset (Higher Dimensional)
# --------------------------------------------------

# Load the Iris dataset
iris = load_iris()
X_iris = iris.data    # 4-dimensional features
y_iris = iris.target

# Apply GMM to split the Iris data into 3 clusters
gmm_iris = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_iris.fit(X_iris)
labels_iris = gmm_iris.predict(X_iris)

# Project the 4D data to 2D using PCA for visualization
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris)

plt.figure(figsize=(6, 5))
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=labels_iris, cmap='viridis', edgecolor='k')
plt.title("Iris Data: GMM Clustering (3 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# --------------------------------------------------
# Section 3: Moons Dataset
# --------------------------------------------------

# Generate the two-moons dataset
X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

# --- Part A: GMM on the Moons Dataset ---
# Fit a Gaussian Mixture Model with 2 components
gmm_moons = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_moons.fit(X_moons)
labels_moons = gmm_moons.predict(X_moons)

plt.figure(figsize=(6, 5))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', edgecolor='k')
plt.title("Moons Data: GMM Clustering (2 Components)")
plt.xlabel("x1")
plt.ylabel("x2")

# --- Part B: Bayesian Gaussian Mixture on the Moons Dataset ---
# BayesianGaussianMixture does not require pre-specifying the exact number of clusters.
# Here we set an upper bound of n_components=10.
bgmm_moons = BayesianGaussianMixture(n_components=10, covariance_type='full', random_state=42)
bgmm_moons.fit(X_moons)
labels_bgmm = bgmm_moons.predict(X_moons)

plt.figure(figsize=(6, 5))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_bgmm, cmap='viridis', edgecolor='k')
plt.title("Moons Data: Bayesian GMM Clustering")
plt.xlabel("x1")
plt.ylabel("x2")

# Show all figures at once
plt.show()