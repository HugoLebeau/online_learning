import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.stats as stats
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 2500
alpha = 0.3
L = int(alpha*n)

k = 2
pi = np.array([0.5, 0.5])

C = utils.gen_mask(n, L, kind='circulant')
T = utils.gen_mask(n, L, kind='toeplitz')

psi, basis_c = utils.basis(n, L, kind='circulant')
tau, basis_t = utils.basis(n, L, kind='toeplitz')

ax_mu2 = np.linspace(0, 7**2, 1000)[1:]

mu2, psi_l = np.meshgrid(ax_mu2, psi)
mu2, tau_l = np.meshgrid(ax_mu2, tau)

for c in [0.5, 0.03]:
    p = int(c*n)
    setting = "$n = {} \quad p = {} \quad L = {}$".format(n, p, L)
    
    zeta_c = (ax_mu2/(ax_mu2+1))*(1-p*np.mean((psi_l/((mu2+1)*psi[-1]-psi_l))**2, axis=0))
    zeta_t = (ax_mu2/(ax_mu2+1))*(1-p*np.mean((tau_l/((mu2+1)*tau[-1]-tau_l))**2, axis=0))
    align_c = np.maximum(zeta_c, 0)
    align_t = np.maximum(zeta_t, 0)
    
    spikes_c, spikes_t = utils.get_spikes(n, p, L, np.sqrt(ax_mu2[-1]), tau)
    spikes_idx_c, spikes_c, _, natural_idx_c = spikes_c
    spikes_idx_t, spikes_t, _, natural_idx_t = spikes_t
    
    n_rep = 10
    mu_r = np.linspace(np.sqrt(ax_mu2[0]), np.sqrt(ax_mu2[-1]), 50)
    c_err_c = np.zeros((n_rep, mu_r.size))
    c_err_t = np.zeros((n_rep, mu_r.size))
    for i in tqdm(range(n_rep)):
        for j, mu_norm in enumerate(mu_r):
            # Set model
            mu = stats.norm.rvs(size=(p, 1))
            mu = mu_norm*mu/linalg.norm(mu)
            M = np.concatenate([+mu, -mu], axis=1)
            J = utils.getJ(n, pi)
            true_partition = np.argmax(J, axis=1)
            
            P = M@(J.T)
            Z = stats.norm.rvs(size=(p, n))
            X = Z+P
            KC = ((X.T)@X)*C/p
            KT = ((X.T)@X)*T/p
            
            # Top empirical eigenvector
            _, w_c = eigsh(KC, k=1, which='LA')
            _, w_t = eigsh(KT, k=1, which='LA')
            
            # Classification
            partition_c = np.where(w_c[:, 0] >= 0, 1, 0)
            partition_t = np.where(w_t[:, 0] >= 0, 1, 0)
            c_err_c[i, j], _, _ = utils.get_classif_error(k, partition_c, true_partition)
            c_err_t[i, j], _, _ = utils.get_classif_error(k, partition_t, true_partition)
    
    c_err_mean_c = c_err_c.mean(axis=0)
    c_err_mean_t = c_err_t.mean(axis=0)
    c_err_std_c = c_err_c.std(axis=0)
    c_err_std_t = c_err_t.std(axis=0)
    
    n_spikes = 4
    
    plt.errorbar(mu_r, c_err_mean_c, yerr=c_err_std_c, ls=':', marker='.',
                 capsize=2, zorder=3, label="Empirical")
    plt.plot(np.sqrt(ax_mu2), stats.norm.sf(np.sqrt(align_c/(1-align_c))), label="$Q(\\sqrt{\\zeta_0/(1-\\zeta_0)})$")
    plt.grid(ls=':')
    plt.xlabel("$\| \mu \|$")
    plt.ylabel("Alignment")
    plt.title("Circulant mask | "+setting)
    plt.legend()
    tikzplotlib.save(fig_folder+"classif_error_circulant_c"+str(c)+".tex")
    plt.show()
    
    plt.errorbar(mu_r, c_err_mean_t, yerr=c_err_std_t, ls=':', marker='.',
                 capsize=2, zorder=3, label="Empirical")
    plt.plot(np.sqrt(ax_mu2), stats.norm.sf(np.sqrt(align_t/(1-align_t))), label="$Q(\\sqrt{\\zeta_0/(1-\\zeta_0)})$")
    plt.grid(ls=':')
    plt.xlabel("$\| \mu \|$")
    plt.ylabel("Alignment")
    plt.title("Toeplitz mask | "+setting)
    plt.legend()
    tikzplotlib.save(fig_folder+"classif_error_toeplitz_c"+str(c)+".tex")
    plt.show()
