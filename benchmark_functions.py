import numpy as np
import online_utils as utils
import matplotlib.pyplot as plt
import pycle.sketching as sk
import pycle.compressive_learning as cl
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from sklearn.datasets import fetch_openml

def BigGAN_images(folder, classes):
    k = len(classes) # number of classes
    features = [open(folder+'{}_features.csv'.format(cl), 'r') for cl in classes] # load features
    p = len(features[0].readline().rstrip().split(' ')) # get data dimension
    features[0].seek(0)
    # Get data length
    cl_len = np.zeros(k, dtype=int)
    for i, f in enumerate(features):
        cl_len[i] = sum(1 for line in f)
        f.seek(0)
    T = np.sum(cl_len)
    # Define data pipeline
    y = np.repeat(range(k), cl_len)
    np.random.shuffle(y)
    get_data = lambda t: np.array(features[y[t]].readline().rstrip().split(' '), dtype=float)
    # Load dataset
    X = np.zeros((T, p))
    for t in tqdm(range(T)):
        X[t] = get_data(t)
    X -= np.mean(X, axis=0) # centering
    # Close files
    for f in features:
        f.close()
    return k, X, y

def FashionMNIST_images(classes):
    # Load dataset
    X_, y_ = fetch_openml('Fashion-MNIST', return_X_y=True)
    X_, y_ = X_.values, y_.values.astype(int)
    num_classes = {"T-shirt/top": 0, "Trouser": 1, "Pullover": 2, "Dress": 3, "Coat": 4, "Sandal": 5, "Shirt": 6, "Sneaker": 7, "Bag": 8, "Ankle boot": 9}
    k = len(classes)
    cl = [num_classes[c] for c in classes]
    # Select classes
    mask = np.zeros(y_.shape[0], dtype=bool)
    for j in cl:
        mask |= (y_ == j)
    X, y = X_[mask], y_[mask]
    X -= np.mean(X, axis=0) # centering
    for i, j in enumerate(np.sort(cl)):
        y[y == j] = i
    return k, X, y

def offline(k, X, y):
    p = X.shape[1]
    K = X@X.T/p # kernel matrix
    eigvals, eigvecs = eigsh(K, k=1, which='LA') # dominant eigenvalue/eigenvector
    y_est = np.where(eigvecs[:, -1] > 0, 0, 1) # class estimation
    c_err, _, _ = utils.get_classif_error(k, y_est, y) # classification error
    return c_err

def streaming(k, M, X, y, verbose=False):
    T, p = X.shape
    n, L = utils.best_nL(M, p)
    if verbose:
        print("n = {}\nL = {}".format(n, L))
    class_count, (lbda, w, partition_ite, time_ite) = utils.pm1_streaming((lambda t: X[t]), T, n, p, L, k, verbose=verbose)
    y_est = np.argmax(class_count, axis=1) # estimate classes via majority vote
    c_err, per, per_inv = utils.get_classif_error(k, y_est, y)
    delay_c_err = np.mean(per[partition_ite[n-1:]] != np.array([y[t:t+n] for t in range(T-n+1)]), axis=0)[::-1]

    if verbose:
        t = T//2
        xx = np.arange(n)
        for j in range(k):
            mask = (y[t-n+1:t+1] == j)
            plt.plot(xx[mask], w[t][mask], ls='', marker='.', color='C'+str(j))
        plt.title("Top eigenvector")
        plt.grid(ls=':')
        plt.show()

        plt.plot(delay_c_err)
        plt.axhline(y=c_err, ls='--', label="Overall classification error")
        plt.grid(ls=':')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plt.xlabel("Delay $\\Delta t$")
        plt.ylabel("Classification error")
        plt.legend()
        plt.show()

    return c_err

def batch(k, M, X, y, verbose=False):
    T, p = X.shape
    L_ = int(np.round((p/2)*(np.sqrt(1+4*M/(p*p))-1)))
    if verbose:
        print("L' = {}".format(L_))
    y_est = np.zeros_like(y)
    for i in tqdm(range(int(np.ceil(T/L_))), disable=not verbose):
        subX = X[i*L_:(i+1)*L_]
        subK = subX@subX.T/p # kernel matrix
        eigvals, eigvecs = eigsh(subK, k=1, which='LA') # dominant eigenvalue/eigenvector
        y_est[i*L_:(i+1)*L_] = np.where(eigvecs[:, -1] > 0, 0, 1) # class estimation
        _, per, _ = utils.get_classif_error(k, y_est[i*L_:(i+1)*L_], y[i*L_:(i+1)*L_])
        y_est[i*L_:(i+1)*L_] = per[y_est[i*L_:(i+1)*L_]]
    c_err, _, _ = utils.get_classif_error(k, y_est, y)
    return c_err

def sketching(k, M, X, y, n_repetitions=10, verbose=False):
    T, p = X.shape
    X_bounded = (2*X-X.max()-X.min())/(X.max()-X.min()) # bound between -1 and 1
    bounds = np.array([-np.ones(p), np.ones(p)])
    m = M//(p+1) # sketch size

    Sigma = sk.estimate_Sigma(X_bounded, m0=m, n0=T, verbose=verbose) # not realistic -> yields an optimistic error rate
    Omega = sk.drawFrequencies("FoldedGaussian", p, m, Sigma)
    Phi = sk.SimpleFeatureMap("ComplexExponential", Omega)

    if verbose:
        print("Computing sketch...", end='')
    z = sk.computeSketch(X_bounded, Phi)
    if verbose:
        print(" Done.")

    solver = cl.CLOMP_CKM(Phi, k, bounds, z)
    if verbose:
        print("Fitting k-means...", end='')
    solver.fit_several_times(n_repetitions)
    if verbose:
        print(" Done.")
    centroids = solver.get_centroids()

    dist = cdist(centroids, X_bounded, metric='euclidean')
    y_est = np.argmin(dist, axis=0)
    c_err, _, _ = utils.get_classif_error(k, y_est, y)
    return c_err
