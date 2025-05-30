import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Generate the data (same as in Exercise 2.5)
np.random.seed(6020)
m = 100
X = 6 * np.random.rand(m) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m)

# Reshape X for sklearn (needs 2D array)
X = X.reshape(-1, 1)

# We'll create a smooth range of x-values for plotting
x_plot = np.linspace(-3, 3, 200).reshape(-1, 1)

# 2. Define various max_depth settings
max_depth_values = [None, 2, 5, 10]  
# Note: None means the tree can grow until leaves have at least min_samples_leaf samples

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Observations', color='darkorange')

for d in max_depth_values:
    # Create and fit the Decision Tree Regressor
    tree_reg = DecisionTreeRegressor(max_depth=d, min_samples_leaf=10, random_state=42)
    tree_reg.fit(X, y)
    
    # Predict on the smooth range for plotting
    y_plot = tree_reg.predict(x_plot)
    
    label = f"max_depth={d}" if d is not None else "max_depth=None"
    plt.plot(x_plot, y_plot, label=label)

plt.title("Decision Tree Regression (min_samples_leaf=10)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()