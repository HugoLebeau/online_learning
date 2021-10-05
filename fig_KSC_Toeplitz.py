import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.stats as stats

import punct_utils as utils

np.random.seed(3)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 1000
c = 0.2
alpha = 0.1

p = int(c*n)
L = int(alpha*n)
setting = "$n = {} \quad p = {} \quad L = {}$".format(n, p, L)

k = 3
cov_mat = np.kron(np.array([[20, 12, 8], [12, 30, 7], [8, 7, 40]]), np.eye(p))/p
cov_mat = cov_mat[:k*p, :k*p]
pi = np.ones(k)/k
M = stats.multivariate_normal.rvs(cov=cov_mat).reshape((p, k))

J = utils.getJ(n, pi)
true_partition = np.argmax(J, axis=1)
nbMC = 1

eigvals_t, eigvecs_t = utils.simul(nbMC, L, M, J, mask='toeplitz', comp=False)

tau, basis_t = utils.basis(n, L, kind='toeplitz')

axr = np.linspace(eigvals_t.min(), eigvals_t.max(), 1000)
eta0_t = utils.eta0(axr, p, tau)
LSD_t = (1/(1-axr-eta0_t)).imag/np.pi

idx_eigvecs = [-1]
idx_basis = [-1, -2, -3, -4, -5, -6, -7]

basis = basis_t[idx_basis].T
eigvecs = eigvecs_t[0, idx_eigvecs].T
reg = np.zeros((k, len(idx_basis), len(idx_eigvecs)))
curves = np.zeros((k, n, len(idx_eigvecs)))
for j in range(k):
    X_reg_j = basis[true_partition == j]
    reg[j] = linalg.solve((X_reg_j.T)@X_reg_j, (X_reg_j.T)@eigvecs[true_partition == j])
    curves[j] = basis@reg[j]

_, _, per_inv = utils.get_classif_error(k, true_partition, np.argmax(J, axis=1))

plt.hist(eigvals_t.flatten(), bins='auto', edgecolor='black', density=True, zorder=2, label="ESD")
plt.plot(axr, LSD_t, color='red', label="LSD")
plt.grid(ls=':')
plt.ylim(bottom=1e-5, top=1)
plt.yscale('log')
plt.ylabel("Density")
plt.legend()
plt.title(setting)
tikzplotlib.save(fig_folder+"KT_distrib_k"+str(k)+".tex")
plt.show()

x = np.arange(n)
for j in range(k):
    color = 'C'+str(j)
    cl = (true_partition == j)
    plt.plot(x[cl], eigvecs_t[0, -1, cl], alpha=.5, color=color,
             ls='', marker='.', label="Class {}".format(j+1))
    plt.plot(x, curves[per_inv[j], :, 0], color=color)
plt.grid(ls=':')
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"KT_eigvec0_k"+str(k)+".tex")
plt.show()

idx_eigvecs = [-1, -2, -3, -4, -5]
idx_basis = [-1, -2, -3, -4, -5, -6, -7]

partition, (exp_smooth, partition0, reg, curves) = utils.classification(k, eigvecs_t[0, idx_eigvecs], basis_t[idx_basis])
c_err, per, per_inv = utils.get_classif_error(k, partition, true_partition)
print(c_err)

for j in range(k):
    color = 'C'+str(j)
    cl = (true_partition == j)
    plt.plot(x[cl], eigvecs_t[0, -1, cl], ls='', marker='.', label="Class {}".format(j+1), alpha=.5, c=color, zorder=2)
    plt.plot(x[partition0 == per_inv[j]], exp_smooth[0, partition0 == per_inv[j]], ls='--', c=color, zorder=3)
    plt.plot(x, curves[per_inv[j], 0], ls='-', c=color, zorder=4)
plt.grid(ls=':')
plt.legend()
plt.title(setting)
tikzplotlib.save(fig_folder+"KT_est_eigvec0_k"+str(k)+".tex")
plt.show()
