import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import PercentFormatter

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

T = 5000
n = 1000
p = 200
L = 100
setting = "$T = {} \quad n = {} \quad p = {} \quad L = {}$".format(T, n, p, L)

k = 3
cov_mat = 9*np.kron(np.array([[20, 12, 8], [12, 30, 7], [8, 7, 40]]), np.eye(p))/p
cov_mat = cov_mat[:k*p, :k*p]
pi = np.ones(k)/k
M = stats.multivariate_normal.rvs(cov=cov_mat).reshape((p, k))

J = utils.getJ(T, pi)
true_partition = np.argmax(J, axis=1)

P = M@(J.T)
Z = stats.norm.rvs(size=(p, T))
X = P+Z

get_data = lambda t: X[:, t]

smooth_par = 0.15
h_start = 10*k
divided_warmup = True

n_eigvecs = 5
idx_basis = [-1, -2, -3, -4, -5, -6, -7]

basis = utils.basis(n, L, kind='toeplitz')[1][idx_basis]

class_count, details = utils.streaming(get_data, T, p, L, k, n_eigvecs, basis, smooth_par, h_start, divided_warmup)

lbda, w, exp_smooth, partition0, curves, partition_ite, time_ite = details
partition = np.argmax(class_count, axis=1)

c_err, per, per_inv = utils.get_classif_error(k, partition, true_partition)
print("Classification error: {:.2%}".format(c_err))
delay_c_err = np.mean(per[partition_ite[n-1:]] != np.array([true_partition[t:t+n] for t in range(T-n+1)]), axis=0)[::-1]

plt.plot(delay_c_err)
plt.axhline(y=c_err, ls='--', label="Overall classification error")
plt.grid(ls=':')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.xlabel("Delay")
plt.ylabel("Classification error")
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"streaming_c_err.tex")
plt.show()

for i in range(n_eigvecs):
    plt.plot(lbda[i], label=i+1)
plt.grid(ls=':')
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue")
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"streaming_lbda.tex")
plt.show()

t = T//2
i = 0
x = np.arange(max(0, t-n+1), max(n, t+1))
for j in range(k):
    color = 'C'+str(j)
    cl = (true_partition[max(0, t-n+1):t+1] == j)
    cl = np.append(cl, np.zeros(max(0, n-t-1), dtype=bool))
    plt.plot(x[cl], w[t, i, cl], ls='', marker='.', alpha=.3, c=color, label=j+1, zorder=2)
    if (divided_warmup and t <= n-1) or (not divided_warmup and t == n-1):
        mask = ((x <= t) & (partition0 == per_inv[j]))
        plt.plot(x[mask], exp_smooth[i, mask], ls='--', c=color, zorder=3)
    if t >= n-1:
        plt.plot(x, curves[t, per_inv[j], i], c=color, zorder=4)
plt.grid(ls=':')
plt.title("Eigenvector {} | ".format(i+1)+setting)
plt.legend()
tikzplotlib.save(fig_folder+"streaming_eigvec0.tex")
plt.show()
