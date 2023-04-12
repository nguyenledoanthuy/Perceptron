import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Input dataset
X = np.array(
    [
        [0.204, 0.834], [0.222, 0.73], [0.298, 0.822], [
            0.45, 0.842], [0.412, 0.732],
        [0.298, 0.64], [0.588, 0.298], [0.554, 0.398], [
            0.67, 0.466], [0.834, 0.426],
        [0.724, 0.368], [0.79, 0.262], [0.824, 0.338], [
            0.136, 0.26], [0.146, 0.374],
        [0.258, 0.422], [0.292, 0.282], [0.478, 0.568], [
            0.654, 0.776], [0.786, 0.758],
        [0.69, 0.628], [0.736, 0.786], [0.574, 0.742]
    ]
)

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Train the classifier
clf = SVC(kernel='rbf', C=1000)
clf.fit(X, y)

# Visualize the dataset and the separation boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create a grid to evaluate the classifier
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k',
           levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[
           :, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.show()
