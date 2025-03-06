import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Generate synthetic data: 2 classes in 2D
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Train a linear SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Get the separating hyperplane parameters
# For a linear SVM, the decision function is given by: w0*x0 + w1*x1 + b = 0
w = clf.coef_[0]
b = clf.intercept_[0]
print("Weights:", w)
print("Intercept:", b)

# Calculate the decision boundary line: x1 = (-w0*x0 - b) / w1
# Define a range for x0
x0 = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
x1 = (-w[0] * x0 - b) / w[1]

# Calculate margin lines: distance = 1/||w||
margin = 1 / np.linalg.norm(w)
# Upper margin: w0*x0 + w1*x1 + b = 1 -> x1 = (-w0*x0 - b + 1) / w1
x1_margin_up = (-w[0] * x0 - b + 1) / w[1]
# Lower margin: w0*x0 + w1*x1 + b = -1 -> x1 = (-w0*x0 - b - 1) / w[1]
x1_margin_down = (-w[0] * x0 - b - 1) / w[1]

# Plot the data points and hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', marker='o', label='Class 0')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', marker='s', label='Class 1')

# Plot the decision boundary
plt.plot(x0, x1, 'k-', label='Decision Boundary')
# Plot the margins
plt.plot(x0, x1_margin_up, 'k--', label='Margin')
plt.plot(x0, x1_margin_down, 'k--')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hyperplane and Margins in a 2D Classification Problem')
plt.legend()
plt.grid(True)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# Generate a non-linearly separable dataset using make_moons
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Train an SVM with a non-linear RBF kernel
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)

# Create a grid of points covering the data range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# Compute the decision function for each point on the grid
Z = clf.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(8, 6))
# Fill the contour with color according to the decision function values
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50),
             cmap='RdBu', alpha=0.6)
# Draw the decision boundary where decision function is zero
plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='--', colors='k')
# Scatter plot of the original data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='white', s=50)
plt.title('Non-linear Decision Boundary (RBF SVM)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate a dataset with significant class overlap
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, class_sep=0.5, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)

# Create a grid of points covering the data range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# Compute the decision function for each point on the grid
Z = clf.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50),
             cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='--', colors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='white', s=50)
plt.title('Low Class Separability Due to Significant Overlap')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
