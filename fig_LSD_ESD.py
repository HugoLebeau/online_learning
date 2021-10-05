import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 2500
alpha = 0.3
L = int(alpha*n)

J = np.zeros((n, 0))

nbMC = 1

tau, basis_t = utils.basis(n, L, kind='toeplitz')
psi, basis_c = utils.basis(n, L, kind='circulant')

for c in [0.5, 0.03]:
    p = int(c*n)
    setting = "$n = {} \quad p = {} \quad L = {}$".format(n, p, L)
    
    M = np.zeros((p, 0))
    
    eigvals_t, eigvecs_t = utils.simul(nbMC, L, M, J, mask='toeplitz', comp=False)
    eigvals_c, eigvecs_c = utils.simul(nbMC, L, M, J, mask='circulant', comp=False)
    
    axr = np.linspace(eigvals_c.min(), eigvals_c.max(), 1000)
    eta0_t = utils.eta0(axr, p, tau)
    eta0_c = utils.eta0(axr, p, psi)
    LSD_t = (1/(1-axr-eta0_t)).imag/np.pi
    LSD_c = (1/(1-axr-eta0_c)).imag/np.pi
    
    plt.hist(eigvals_t.flatten(), bins='auto', color='C0', edgecolor='black', density=True, label="ESD", zorder=2)
    plt.plot(axr, LSD_t, color='C1', label="LSD")
    plt.grid(ls=':')
    plt.ylabel("Density")
    plt.legend()
    plt.title("Toeplitz mask | "+setting)
    tikzplotlib.save(fig_folder+"distrib_Toeplitz_c"+str(c)+".tex")
    plt.show()
    
    plt.hist(eigvals_c.flatten(), bins='auto', color='C0', edgecolor='black', density=True, label="ESD", zorder=2)
    plt.plot(axr, LSD_c, color='C1', label="LSD")
    plt.grid(ls=':')
    plt.ylabel("Density")
    plt.legend()
    plt.title("Circulant mask | "+setting)
    tikzplotlib.save(fig_folder+"distrib_circulant_c"+str(c)+".tex")
    plt.show()
