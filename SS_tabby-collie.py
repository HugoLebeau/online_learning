import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

collie = np.loadtxt('data/collie_normalized.csv')
tabby = np.loadtxt('data/tabby_normalized.csv')

X = np.concatenate([collie, tabby], axis=0)
n, p = X.shape
y = np.array(['collie']*collie.shape[0]+['tabby']*tabby.shape[0])

idx = np.arange(n)
np.random.shuffle(idx)
X, y = X[idx], y[idx]

K = X@X.T/p
eigvals, eigvecs = eigsh(K, k=1, which='LA')

mu_est = np.sqrt((-(1+(1-eigvals[0])*p/n)+np.sqrt((1+(1-eigvals[0])*p/n)**2-4*p/n))/2)
print("mu_est = {}".format(mu_est))

plt.hist(eigvals, density=True, edgecolor='black', bins='sqrt')
plt.grid(ls=':')
plt.show()

axr = np.arange(n)
for j in np.unique(y):
    cl = (y == j)
    plt.plot(axr[cl], eigvecs[cl, -1], ls='', marker='.')
plt.grid(ls=':')
plt.show()

y_est = np.where(eigvecs[:, -1] > 0, 'tabby', 'collie')
c_err = np.mean(y_est != y)
