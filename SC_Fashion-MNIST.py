import numpy as np
import matplotlib.pyplot as plt
from punct_utils import get_classif_error
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# Load data
X, y = fetch_openml('Fashion-MNIST', return_X_y=True)
X, y = X.values, y.values.astype(int)

classes = [1, 4, 9]
k = len(classes)

mask = np.zeros(y.shape[0], dtype=bool)
for j in classes:
    mask |= (y == j)
Xc, yc = X[mask], y[mask]
Xc -= np.mean(Xc, axis=0) # centering
for i, j in enumerate(np.sort(classes)):
    yc[yc == j] = i
n, p = Xc.shape

# Compute kernel matrix and dominant eigenvectors
K = Xc@Xc.T/p
eigvals, eigvecs = eigsh(K, k=5, which='LA')

# Plot dominant eigenvector
axr = np.arange(n)
for j in range(k):
    cl = (yc == j)
    plt.plot(axr[cl], eigvecs[cl, -1], ls='', marker='.', label=classes[j])
plt.grid(ls=':')
plt.legend()
plt.show()

# Classification
kmeans = KMeans(n_clusters=k, random_state=0).fit(eigvecs)
c_err, per, per_inv = get_classif_error(k, kmeans.labels_, yc)
print(c_err)
