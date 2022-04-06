import numpy as np
import online_utils as utils
import pycle.sketching as sk
import pycle.compressive_learning as cl
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml

np.random.seed(14159)

# Load data
X, y = fetch_openml('Fashion-MNIST', return_X_y=True)
X, y = X.values, y.values.astype(int)

classes = [4, 9]
k = len(classes)

mask = np.zeros(y.shape[0], dtype=bool)
for j in classes:
    mask |= (y == j)
Xc, yc = X[mask], y[mask]
Xc -= np.mean(Xc, axis=0) # centering
for i, j in enumerate(np.sort(classes)):
    yc[yc == j] = i
n, p = Xc.shape

m = 1000*100

Sigma = sk.estimate_Sigma(Xc, 200, c=1, n0=n//50, verbose=1)
Omega = sk.drawFrequencies("FoldedGaussian", p, m, Sigma)
xi = sk.drawDithering(m)
Phi = sk.SimpleFeatureMap("universalQuantization_complex", Omega, xi)

print("Computing sketch... ", end='')
z = sk.computeSketch(Xc, Phi)
print("Done.")

bounds = np.array([Xc.min(axis=0), Xc.max(axis=0)])
Phi_fullprec = sk.SimpleFeatureMap("ComplexExponential", Omega, xi, c_norm=np.pi/2*np.sqrt(2))
solver = cl.CLOMP_CKM(Phi_fullprec, k, bounds, z)
print("Fitting k-means... ", end='')
solver.fit_several_times(10)
print("Done.")
centroids = solver.get_centroids()

dist = cdist(centroids, Xc, metric='euclidean')
est_classes = np.argmin(dist, axis=0)
c_err, _, _ = utils.get_classif_error(k, est_classes, yc)
print("Classification error: {:%}".format(c_err))
