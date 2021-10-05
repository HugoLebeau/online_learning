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

C = utils.gen_mask(n, L, kind='circulant')
T = utils.gen_mask(n, L, kind='toeplitz')

kr = np.arange(n)
psi = utils.nu(L, 2*np.pi*kr/n)
aton = np.argsort(psi) # Ascending order -> natural order
ntoa = np.argsort(aton) # Natural order -> ascending order

_, basis_c = utils.basis(n, L, kind='circulant')
tau, basis_t = utils.basis(n, L, kind='toeplitz')

tau = tau[ntoa]

psi_k, psi_l = np.meshgrid(psi, psi)
tau_k, tau_l = np.meshgrid(tau, tau)

ax_mu2 = np.linspace(0, 7**2, 1000)[1:]

for c in [0.5, 0.03]:
    p = int(c*n)
    setting = "$n = {} \quad p = {} \quad L = {}$".format(n, p, L)
    
    zeta_c = np.zeros((ax_mu2.size, n))
    zeta_t = np.zeros((ax_mu2.size, n))
    for i, mu2 in enumerate(tqdm(ax_mu2)):
        zeta_c[i] = (mu2/(mu2+1))*(1-p*np.mean((psi_l/((mu2+1)*psi_k-psi_l))**2, axis=0))
        zeta_t[i] = (mu2/(mu2+1))*(1-p*np.mean((tau_l/((mu2+1)*tau_k-tau_l))**2, axis=0))
    align_c = np.maximum(zeta_c, 0)
    align_t = np.maximum(zeta_t, 0)
    
    spikes_c, spikes_t = utils.get_spikes(n, p, L, np.sqrt(ax_mu2[-1]), tau)
    spikes_idx_c, spikes_c, _, natural_idx_c = spikes_c
    spikes_idx_t, spikes_t, _, natural_idx_t = spikes_t
    
    n_rep = 10
    mu_r = np.linspace(np.sqrt(ax_mu2[0]), np.sqrt(ax_mu2[-1]), 30)
    emp_align_c = np.zeros((n_rep, mu_r.size))
    emp_align_t = np.zeros((n_rep, mu_r.size))
    pi = np.array([1., 0.])
    for i in tqdm(range(n_rep)):
        for j, mu_norm in enumerate(mu_r):
            # Set model
            mu = stats.norm.rvs(size=(p, 1))
            mu = mu_norm*mu/linalg.norm(mu)
            M = np.concatenate([+mu, -mu], axis=1)
            J = utils.getJ(n, pi)
            
            P = M@(J.T)
            Z = stats.norm.rvs(size=(p, n))
            X = Z+P
            KC = ((X.T)@X)*C/p
            KT = ((X.T)@X)*T/p
            
            # Top empirical eigenvector
            _, w_c = eigsh(KC, k=1, which='LA')
            _, w_t = eigsh(KT, k=1, which='LA')
            
            # Alignment
            emp_align_c[i, j] = (basis_c[-1]@w_c[:, 0])**2
            emp_align_t[i, j] = (basis_t[-1]@w_t[:, 0])**2
    
    emp_align_mean_c = emp_align_c.mean(axis=0)
    emp_align_mean_t = emp_align_t.mean(axis=0)
    emp_align_std_c = emp_align_c.std(axis=0)
    emp_align_std_t = emp_align_t.std(axis=0)
    
    n_spikes = 4
    
    plt.errorbar(mu_r, emp_align_mean_c, yerr=emp_align_std_c, ls=':', marker='.',
                 capsize=2, zorder=3, label="Empirical (top eigenvector)")
    for k in natural_idx_c[:n_spikes]:
        plt.plot(np.sqrt(ax_mu2), align_c[:, k], label="$k = {}$".format(k))
    plt.grid(ls=':')
    plt.xlabel("$\| \mu \|$")
    plt.ylabel("Alignment")
    plt.title("Circulant mask | "+setting)
    plt.legend()
    tikzplotlib.save(fig_folder+"alignments_circulant_c"+str(c)+".tex")
    plt.show()
    
    plt.errorbar(mu_r, emp_align_mean_t, yerr=emp_align_std_t, ls=':', marker='.',
                 capsize=2, zorder=3, label="Empirical (top eigenvector)")
    for k in natural_idx_t[:n_spikes]:
        plt.plot(np.sqrt(ax_mu2), align_t[:, k], label="$k = {}$".format(k))
    plt.grid(ls=':')
    plt.xlabel("$\| \mu \|$")
    plt.ylabel("Alignment")
    plt.title("Toeplitz mask | "+setting)
    plt.legend()
    tikzplotlib.save(fig_folder+"alignments_toeplitz_c"+str(c)+".tex")
    plt.show()
