import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Seed for reproducibility
np.random.seed(6020)

# 1. Generate the data
m = 100
X = 6 * np.random.rand(m) - 3  # shape: (100,)
y = 0.5 * X**2 + X + 2 + np.random.randn(m)

# Reshape X to a 2D array for sklearn
X = X.reshape(-1, 1)

# 2. Define different SVR models
svr_linear = SVR(kernel='linear', C=100, epsilon=0.1)
svr_poly2  = SVR(kernel='poly',   C=100, degree=2, epsilon=0.1)
svr_poly3  = SVR(kernel='poly',   C=100, degree=3, epsilon=0.1)
svr_rbf    = SVR(kernel='rbf',    C=100, epsilon=0.1)

models = [svr_linear, svr_poly2, svr_poly3, svr_rbf]
labels = ['Linear', 'Poly (degree=2)', 'Poly (degree=3)', 'RBF']

# 3. Fit and predict for each model
x_fit = np.linspace(-3, 3, 200).reshape(-1, 1)  # for plotting smooth curves

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='darkorange', label='Observations')

for model, label in zip(models, labels):
    model.fit(X, y)
    y_fit = model.predict(x_fit)
    plt.plot(x_fit, y_fit, label=label)

plt.title('SVR Regression with Various Kernels')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()