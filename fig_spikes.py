import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import punct_utils as utils

np.random.seed(2)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 2500
alpha = 0.3
L = int(alpha*n)

k = 2
pi = np.array([0.5, 0.5])
J = utils.getJ(n, pi)

mu_norm = 2

nbMC = 1
n_eigvecs = 2

tau, basis_t = utils.basis(n, L, kind='toeplitz')
psi, basis_c = utils.basis(n, L, kind='circulant')

x = np.arange(n)

for c in [0.5, 0.03]:
    p = int(c*n)
    mu = stats.norm.rvs(size=(p, 1))
    mu = mu_norm*mu/np.linalg.norm(mu)
    M = np.concatenate([+mu, -mu], axis=1)
    setting = "$n = {} \quad p = {} \quad L = {} \quad || \mu || = {}$".format(n, p, L, mu_norm)
    
    eigvals_t, eigvecs_t = utils.simul(nbMC, L, M, J, mask='toeplitz', comp=False)
    eigvals_c, eigvecs_c = utils.simul(nbMC, L, M, J, mask='circulant', comp=False)
    
    spikes_c, spikes_t = utils.get_spikes(n, p, L, mu_norm, tau)
    spikes_idx_c, spikes_c, zeta_c, natural_idx_c = spikes_c
    spikes_idx_t, spikes_t, zeta_t, natural_idx_t = spikes_t
    
    valid_spikes_c = np.abs(psi[spikes_idx_c]) > (2*L-1)/(mu_norm**2+1)
    valid_spikes_t = np.abs(tau[spikes_idx_t]) > (2*L-1)/(mu_norm**2+1)
    
    axr = np.linspace(eigvals_c.min(), eigvals_c.max(), 1000)
    eta0_t = utils.eta0(axr, p, tau)
    eta0_c = utils.eta0(axr, p, psi)
    LSD_t = (1/(1-axr-eta0_t)).imag/np.pi
    LSD_c = (1/(1-axr-eta0_c)).imag/np.pi
    
    plt.hist(eigvals_t.flatten(), bins='auto', color='C0', edgecolor='black', density=True, label="ESD", zorder=2)
    plt.plot(axr, LSD_t, color='C1', label="LSD")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    seen_valid, seen_not_valid = False, False
    for i, spike in enumerate(spikes_t):
        label = "Spike" if i == 0 else None
        label = None
        if valid_spikes_t[i]:
            color = 'C2'
            if not seen_valid:
                seen_valid = True
                label = "Spike I"
        else:
            color = 'C3'
            if not seen_not_valid:
                seen_not_valid = True
                label = "Spike II"
        plt.axvline(x=spike, ls='--', color=color, label=label)
    plt.grid(ls=':')
    plt.ylim(bottom=1e-5, top=1)
    plt.yscale('log')
    plt.ylabel("Density")
    plt.legend()
    plt.title("Toeplitz mask | "+setting)
    tikzplotlib.save(fig_folder+"spikes_Toeplitz_c"+str(c)+".tex")
    plt.show()
    
    plt.hist(eigvals_c.flatten(), bins='auto', color='C0', edgecolor='black', density=True, label="ESD", zorder=2)
    plt.plot(axr, LSD_c, color='C1', label="LSD")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    seen_valid, seen_not_valid = False, False
    for i, spike in enumerate(spikes_c):
        label = "Spike" if i == 0 else None
        label = None
        if valid_spikes_c[i]:
            color = 'C2'
            if not seen_valid:
                seen_valid = True
                label = "Spike I"
        else:
            color = 'C3'
            if not seen_not_valid:
                seen_not_valid = True
                label = "Spike II"
        plt.axvline(x=spike, ls='--', color=color, label=label)
    plt.grid(ls=':')
    plt.ylim(bottom=1e-5, top=1)
    plt.yscale('log')
    plt.ylabel("Density")
    plt.legend()
    plt.title("Circulant mask | "+setting)
    tikzplotlib.save(fig_folder+"spikes_circulant_c"+str(c)+".tex")
    plt.show()
    
    for i in range(n_eigvecs):
        for j in range(k):
            cl = (J[:, j] == 1)
            plt.plot(x[cl], eigvecs_t[0, spikes_idx_t[i], cl], ls='', marker='.', label="Class {}".format(j+1))
        plt.grid(ls=':')
        plt.legend()
        plt.title(setting)
        tikzplotlib.save(fig_folder+"eigvec"+str(natural_idx_t[i])+"_toeplitz_c"+str(c)+".tex")
        plt.show()
        
        for j in range(k):
            cl = (J[:, j] == 1)
            plt.plot(x[cl], eigvecs_c[0, spikes_idx_c[i], cl], ls='', marker='.', label="Class {}".format(j+1))
        plt.grid(ls=':')
        plt.legend()
        plt.title(setting)
        tikzplotlib.save(fig_folder+"eigvec"+str(natural_idx_c[i])+"_circulant_c"+str(c)+".tex")
        plt.show()
