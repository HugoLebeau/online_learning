import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 2500
L = 750
setting = "$n = {} \quad L = {}$".format(n, L)

_, eigvecs_c = eigsh(utils.gen_mask(n, L, kind='circulant'), k=2, which='LA')
_, eigvecs_t = eigsh(utils.gen_mask(n, L, kind='toeplitz'), k=2, which='LA')

eigvecs_c[:, -2] = np.roll(eigvecs_c[:, -2], -np.argmin(np.abs(eigvecs_c[:, -2])))
for x in [eigvecs_c[:, -1], eigvecs_c[:, -2], eigvecs_t[:, -1], eigvecs_t[:, -2]]:
    if x[0] < 0:
        x *= -1

plt.plot(eigvecs_c[:, -2], label="$C$")
plt.plot(eigvecs_t[:, -2], label="$T$")
ymin, ymax = plt.ylim()
plt.grid(ls=':')
plt.title("Second eigenvector | "+setting)
plt.legend()
tikzplotlib.save(fig_folder+"Toeplitz_circulant_eigvec1.tex")
plt.show()

plt.plot(eigvecs_c[:, -1], label="$C$")
plt.plot(eigvecs_t[:, -1], label="$T$")
plt.ylim(ymin, ymax)
plt.grid(ls=':')
plt.title("First eigenvector | "+setting)
plt.legend()
tikzplotlib.save(fig_folder+"Toeplitz_circulant_eigvec0.tex")
plt.show()
