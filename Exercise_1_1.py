# --------------------------------------------------------------------
# 1) Vor *allen* scikit-learn-Imports:  WMIC-Aufruf deaktivieren
# --------------------------------------------------------------------
import os, warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # logische Kerne – anpassen

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

# ------------- Monkey-Patch, damit kein WMIC mehr gestartet wird -----
import importlib

loky_ctx = importlib.import_module("joblib.externals.loky.backend.context")
loky_ctx._count_physical_cores_win32 = lambda: int(os.environ["LOKY_MAX_CPU_COUNT"])
# --------------------------------------------------------------------

# ----- ab hier deine bisherigen Imports / Kommentare / Code ----------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests, scipy.io, io


# A helper function to tile images in a grid for display
def createImage(data, x, y, width=64, height=64):
    """
    data:  2D array of shape (width*height, number_of_images)
    x, y:  how many images to tile vertically (x) and horizontally (y)
    width, height: each image's width and height
    """
    mosaic = np.zeros((height * x, width * y))
    idx = 0
    for i in range(x):
        for j in range(y):
            # Reshape and flip each image
            img = np.flipud(np.reshape(data[:, idx], (height, width)))
            # Place it into the mosaic
            mosaic[i * height : (i + 1) * height, j * width : (j + 1) * width] = img.T
            idx += 1
    return mosaic


# Load the Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# A) 2D version (use first two features for simplicity)
X_2d = X_iris[:, :2]
kmeans_2d = KMeans(n_clusters=3, random_state=42)
labels_2d = kmeans_2d.fit_predict(X_2d)

plt.figure(figsize=(5, 4))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_2d, cmap="viridis", edgecolor="k")
plt.title("Iris (2D) with k=3")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

# B) 4D version (all features)
X_4d = X_iris
kmeans_4d = KMeans(n_clusters=3, random_state=42)
labels_4d = kmeans_4d.fit_predict(X_4d)

pca = PCA(n_components=2)
X_4d_pca = pca.fit_transform(X_4d)

plt.figure(figsize=(5, 4))
plt.scatter(X_4d_pca[:, 0], X_4d_pca[:, 1], c=labels_4d, cmap="viridis", edgecolor="k")
plt.title("Iris (4D) projected to 2D, k=3")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

# Generate the two-moons data
X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)
kmeans_moons = KMeans(n_clusters=2, random_state=42)
labels_moons = kmeans_moons.fit_predict(X_moons)

plt.figure(figsize=(5, 4))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap="viridis", edgecolor="k")
plt.title("Moons dataset with k=2 (k-means)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# Load cat data
response_cats = requests.get(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/catData.mat"
)
cats = scipy.io.loadmat(io.BytesIO(response_cats.content))["cat"]

# Load dog data
response_dogs = requests.get(
    "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/dogData.mat"
)
dogs = scipy.io.loadmat(io.BytesIO(response_dogs.content))["dog"]

# Visualize a few cat images in a 4×4 grid
plt.figure()
plt.imshow(createImage(cats, 4, 4), cmap="gray")
plt.axis("off")
plt.title("Some Sample Cats")

# Visualize a few dog images in a 4×4 grid
plt.figure()
plt.imshow(createImage(dogs, 4, 4), cmap="gray")
plt.axis("off")
plt.title("Some Sample Dogs")

# --- Combine cat and dog data into one array ---
CD = np.concatenate((cats, dogs), axis=1)
CD_centered = CD - CD.mean(axis=1, keepdims=True)

# --- PCA + k-means clustering ---
CD_pca = PCA(n_components=20).fit_transform(CD_centered.T)
labels_cd = KMeans(n_clusters=2, random_state=42).fit_predict(CD_pca)

true_labels = np.r_[np.zeros(cats.shape[1]), np.ones(dogs.shape[1])]
accuracy = (labels_cd == true_labels).mean()
print(f"Number of cat images: {cats.shape[1]}")
print(f"Number of dog images: {dogs.shape[1]}")
print(f"Accuracy (cluster 0 = cat, 1 = dog): {accuracy:.2f}")

# --- 2D PCA plot for cats & dogs ---
CD_2d = PCA(2).fit_transform(CD_centered.T)
plt.figure(figsize=(5, 4))
plt.scatter(CD_2d[:, 0], CD_2d[:, 1], c=labels_cd, cmap="viridis", edgecolor="k")
plt.title("Cats & Dogs in PCA(2D) – KMeans Clusters")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.show()
