import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons

# -------------------------------
# 1) Nonlinear SVM on Circular Data
# -------------------------------
np.random.seed(6020)

# Generate a grid of points from -1 to 1 in each dimension
x = np.linspace(-1, 1, 12)  # 12 points from -1 to 1
y = np.linspace(-1, 1, 12)
XX, YY = np.meshgrid(x, y)

# Create circular labels: class 1 if radius < 0.5, else 0
ZZ = XX**2 + YY**2
labels_circle = np.zeros_like(ZZ, dtype=int)
labels_circle[ZZ < 0.5**2] = 1

# Flatten for fitting
X_circle = np.column_stack((XX.ravel(), YY.ravel()))
y_circle = labels_circle.ravel()

# Build a pipeline: PolynomialFeatures -> StandardScaler -> LinearSVC
# Degree=2 is enough to capture a circular boundary (x^2 + y^2 term).
pipe_circle = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True),
    StandardScaler(),
    LinearSVC(C=1000, random_state=6020, max_iter=10000)
)

# Fit the model
pipe_circle.fit(X_circle, y_circle)

# Create a dense mesh for visualization
xx_vis = np.linspace(-1, 1, 200)
yy_vis = np.linspace(-1, 1, 200)
XX_vis, YY_vis = np.meshgrid(xx_vis, yy_vis)
X_vis = np.column_stack((XX_vis.ravel(), YY_vis.ravel()))

# Predict on the mesh
Z_circle = pipe_circle.predict(X_vis)
Z_circle = Z_circle.reshape(XX_vis.shape)

# Plot the decision boundary
plt.figure(figsize=(6, 5))
plt.contourf(XX_vis, YY_vis, Z_circle, alpha=0.3, cmap="coolwarm")
plt.scatter(X_circle[y_circle == 0, 0], X_circle[y_circle == 0, 1],
            color='blue', label='class 0')
plt.scatter(X_circle[y_circle == 1, 0], X_circle[y_circle == 1, 1],
            color='orange', label='class 1')
plt.title("Nonlinear SVM on Circular Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.legend()

# -------------------------------
# 2) Nonlinear SVM on Two-Moons Data
# -------------------------------
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

pipe_moons = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=True),
    StandardScaler(),
    LinearSVC(C=1000, random_state=6020, max_iter=10000)
)

pipe_moons.fit(X_moons, y_moons)

# Create a mesh covering the range of the moons data
x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
xx_moons = np.linspace(x_min, x_max, 200)
yy_moons = np.linspace(y_min, y_max, 200)
XX_moons, YY_moons = np.meshgrid(xx_moons, yy_moons)

# Predict on the mesh
X_vis_moons = np.column_stack((XX_moons.ravel(), YY_moons.ravel()))
Z_moons = pipe_moons.predict(X_vis_moons)
Z_moons = Z_moons.reshape(XX_moons.shape)

# Plot the decision boundary
plt.figure(figsize=(6, 5))
plt.contourf(XX_moons, YY_moons, Z_moons, alpha=0.3, cmap="coolwarm")
plt.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1],
            color='blue', label='class 0')
plt.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1],
            color='orange', label='class 1')
plt.title("Nonlinear SVM on Two-Moons Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()