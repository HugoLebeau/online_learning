import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 50
L = 10
setting = "$n = {} \quad L = {}$".format(n, L)

x = np.linspace(0, 2*np.pi, 500)
nu_x = utils.nu(L, x)

kr = np.arange(n)
psi = utils.nu(L, 2*np.pi*kr/n)

aton = np.argsort(psi) # Ascending order -> natural order
ntoa = np.argsort(aton) # Natural order -> ascending order
tau = linalg.eigh(utils.gen_mask(n, L, kind='toeplitz'), eigvals_only=True)
tau = tau[ntoa]

plt.plot(x, nu_x, color='C0', zorder=3, label="$\\nu_L(x)$")
plt.plot(2*np.pi*kr/n, psi, ls='', marker='.', color='C1', zorder=4, label="$\\psi_k$")
plt.plot(2*np.pi*kr/n, tau, ls='', marker='$\circ$', color='C2', zorder=4, label="$\\tau_k$")
plt.grid(ls=':')
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"nu_psi_tau.tex")
plt.show()
