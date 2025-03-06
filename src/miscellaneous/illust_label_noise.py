import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for two classes
# Class 0: "Cats" (e.g., blue)
cluster_A = np.random.randn(50, 2) + np.array([2, 2])
# Class 1: "Dogs" (e.g., red)
cluster_B = np.random.randn(50, 2) + np.array([-2, -2])

# Combine data and create true (clean) labels
data = np.vstack([cluster_A, cluster_B])
true_labels = np.array([0] * 50 + [1] * 50)

# Introduce label noise: randomly flip the labels for 10 data points
noise_indices = np.random.choice(len(true_labels), size=10, replace=False)
noisy_labels = true_labels.copy()
noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]  # Flip 0->1 and 1->0

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Clean Labels
scatter_clean = axes[0].scatter(data[:, 0], data[:, 1], c=true_labels, cmap='bwr', edgecolor='k')
axes[0].set_title('Clean Labels')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True)

# Plot 2: Noisy Labels
scatter_noisy = axes[1].scatter(data[:, 0], data[:, 1], c=noisy_labels, cmap='bwr', edgecolor='k')
# Highlight noisy points with black crosses
axes[1].scatter(data[noise_indices, 0], data[noise_indices, 1],
                c='black', marker='x', s=100, label='Noisy Points')
axes[1].set_title('Noisy Labels')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True)
plt.show()