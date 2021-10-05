import tikzplotlib
import numpy as np
import punct_utils as utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg, stats
from scipy.optimize import root_scalar
from scipy.sparse.linalg import eigsh
from matplotlib.colors import Normalize

# %%

fig_folder = "figures/"

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True

# %% Kernel matrices

n = 50
p = int(0.1*n)
L = int(0.1*n)

pi = np.array([0.5, 0.5])
mu = stats.norm.rvs(size=(p, 1))
mu_norm = 2
mu = mu_norm*mu/linalg.norm(mu)
M = np.concatenate([+mu, -mu], axis=1)

J = utils.getJ(n, pi)
P = M@(J.T)
C = utils.genB(n, L, 'circulant')
T = utils.genB(n, L, 'toeplitz')
Z = stats.norm.rvs(size=(p, n))
X = Z+P
K = (X.T)@X
KC = K*C
KT = K*T

# %%

color_bound = np.abs(K).max()
norm = Normalize(-color_bound, color_bound)

# %%

plt.matshow(K, cmap='bwr', norm=norm)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.title("$\\bf K$")
plt.savefig(fig_folder+"K.png", bbox_inches='tight', transparent=True)
plt.show()

plt.matshow(KT, cmap='bwr', norm=norm)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.title("$\\bf K \odot \\bf T$")
plt.savefig(fig_folder+"KoT.png", bbox_inches='tight', transparent=True)
plt.show()

plt.matshow(KC, cmap='bwr', norm=norm)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.title("$\\bf K \odot \\bf C$")
plt.savefig(fig_folder+"KoC.png", bbox_inches='tight', transparent=True)
plt.show()

# %% Phase transition

n = 10000
c = 0.01
p = int(c*n)

epsilons = np.linspace(1e-3, 2e-1, 100)
pht_ss_eps = np.sqrt(c/epsilons)
pht_ss_L = np.empty_like(epsilons)
pht_b = np.empty_like(epsilons)
pht_c = np.empty_like(epsilons)

kr = np.arange(n)
a, b = 1e-5, 20
for i, eps in enumerate(tqdm(epsilons)):
    # L = int(eps*n)
    L = utils.eB2L(n, eps)
    pht_ss_L[i] = np.sqrt(p/L)
    psi = utils.nu(L, 2*np.pi*kr/n)
    func_b = lambda t: t**4+2*t**3+(1-c/eps)*t**2-2*c*t-c
    func_c = lambda t: p*np.mean((psi/((t+1)*psi[0]-psi))**2)-1
    if func_b(a)*func_b(b) < 0:
        res = root_scalar(func_b, method='brentq', bracket=[a, b])
        pht_b[i] = res.root if res.converged else np.nan
    else:
        pht_b[i] = np.nan
    if func_c(a)*func_c(b) < 0:
        res = root_scalar(func_c, method='brentq', bracket=[a, b])
        pht_c[i] = res.root if res.converged else np.nan
    else:
        pht_c[i] = np.nan

# %%

right = np.max(epsilons)
top = min(1, min(np.nanmax(pht_ss_eps), np.nanmax(pht_ss_L), np.nanmax(pht_b), np.nanmax(pht_c)))
plt.plot(epsilons, pht_ss_eps, lw=2, c='C0', label="$\\mathbf{K}_{\\mathrm{ss}, n \\varepsilon}$")
plt.plot(epsilons, pht_ss_L, lw=2, c='C3', label="$\\mathbf{K}_{\\mathrm{ss}, L}$")
plt.plot(epsilons, pht_b, lw=2, c='C1', label="$\\bf K \\odot \\bf B$")
plt.plot(epsilons, pht_c, lw=2, c='C2', label="$\\bf K \\odot \\bf C$")
plt.fill_between(epsilons, pht_c, top, color='C2', alpha=0.15)
plt.axhline(y=np.sqrt(c), ls='--', color='black')
plt.text(right/2, top/2, "Classification possible", color='C2',
         fontsize='xx-large', fontweight='bold', ha='center', va='bottom')
plt.text(right/2, 0, "Classification impossible", color='black',
         fontsize='xx-large', fontweight='bold', ha='center', va='bottom')
plt.grid(ls=':')
plt.xlim(0, right)
plt.ylim(0, top)
plt.xlabel("$\\varepsilon$ ($\\leftarrow$ cheaper calculus)")
plt.ylabel("$\\| \\mu \\|^2$ (easier classification $\\rightarrow$)")
# eB2Ln = lambda eB: 1+0.5/n-np.sqrt(1-eB+0.25/n**2)
# Ln2eB = lambda Ln: 1/n+(Ln-1/n)*(2-Ln)
# secax = plt.gca().secondary_xaxis('top', functions=(eB2Ln, Ln2eB))
# secax.set_xlabel("$L/n$")
plt.legend()
tikzplotlib.save(fig_folder+"phase_transition.tex")
plt.show()

# %% Eigenspectrum

n = 2500
p = int(0.1*n)
L = int(0.1*n)

c = p/n
eB = utils.L2eB(n, L)

setting_b = "$n = {} \\quad p = {} \\quad \\varepsilon_B = {}$".format(n, p, np.round(eB, 3))
setting_t = "$n = {} \\quad p = {} \\quad L = {}$".format(n, p, L)

k = 2
pi = np.array([0.5, 0.5])
mu = stats.norm.rvs(size=(p, 1))
mu_norm = 2
mu = mu_norm*mu/linalg.norm(mu)
M = np.concatenate([+mu, -mu], axis=1)

setting_b += "$\\quad \\| \\mu \\| = {}$".format(mu_norm)
setting_t += "$\\quad \\| \\mu \\| = {}$".format(mu_norm)

J = utils.getJ(n, pi)

eigvals_b, eigvecs_b, _ = utils.simul_bernoulli(1, M, J, 1, eB, 1)
eigvals_t, eigvecs_t = utils.simul(1, L, M, J, mask='toeplitz', comp=False)

axr_b = np.linspace(eigvals_b.min(), eigvals_b.max(), 1000)
LSD_b = utils.get_LSD_bernoulli(axr_b, c, 1, eB, 1)

axr_t = np.linspace(eigvals_t.min(), eigvals_t.max(), 1000)
eta0 = utils.eta0(axr_t, n, p, L)
LSD_c = (1/(1-axr_t-eta0)).imag/np.pi

rho, _ = utils.get_spikes_bernoulli(pi, M, c, 1, eB, 1)
_, (spikes_idx, spikes, zeta, natural_idx) = utils.get_spikes(n, p, L, mu_norm)

basis_t = utils.basis(n, L, kind='toeplitz')

# %%

plt.hist(eigvals_b.flatten(), bins='auto', color='skyblue', edgecolor='black', density=True, zorder=2)
plt.plot(axr_b, LSD_b, color='red', label="LSD")
for spike in rho:
    plt.arrow(spike, 0.075, 0, -0.05, width=0.01, head_width=0.1, head_length=0.01,
              color='green', zorder=3)
plt.grid(ls=':')
plt.ylabel("Density")
plt.legend()
plt.title("$\\bf K \odot \\bf B$ eigenvalues | "+setting_b)
tikzplotlib.save(fig_folder+"KoB_eigvals.tex")
plt.show()

plt.hist(eigvals_t.flatten(), bins='auto', color='skyblue', edgecolor='black', density=True, zorder=2)
plt.plot(axr_t, LSD_c, color='red', label="LSD")
for i in spikes_idx:
    plt.arrow(eigvals_t[0, i], 0.075, 0, -0.05, width=0.01, head_width=0.1, head_length=0.01,
              color='green', zorder=3)
plt.grid(ls=':')
plt.ylabel("Density")
plt.legend()
plt.title("$\\bf K \odot \\bf T$ eigenvalues | "+setting_t)
tikzplotlib.save(fig_folder+"KoT_eigvals.tex")
plt.show()

# %%

x = np.arange(n)

plt.figure(figsize=(4.5, 1.5))
for j in range(k):
    cl = (J[:, j] == 1)
    plt.plot(x[cl], eigvecs_b[0, -1, cl], ls='', marker='.', ms=1)
plt.grid(ls=':')
plt.title("First eigenvector")
plt.savefig(fig_folder+"KoB_eigvec.png", bbox_inches='tight', transparent=True)
plt.show()

# %%

to_plot = spikes_idx[:5]

for i in to_plot:
    plt.figure(figsize=(4.5, 1.5))
    for j in range(k):
        cl = (J[:, j] == 1)
        plt.plot(x[cl], eigvecs_t[0, i, cl], ls='', marker='.', ms=1)
    plt.plot(x, basis_t[i])
    plt.grid(ls=':')
    plt.title("Eigenvector {}".format(n-i))
    plt.savefig(fig_folder+"KoT_eigvec{}.png".format(n-i), bbox_inches='tight', transparent=True)
    plt.show()
    
# %% Classification

mu_norm_est = utils.est_mu(n, p, L, eigvals_t[0, -1])
_, (spikes_idx_est, _, zeta_est, _) = utils.get_spikes(n, p, L, mu_norm_est)
nvecs = 3*len(spikes_idx_est)//4
est = utils.classification(n, zeta_est[:nvecs], eigvecs_t[0, spikes_idx_est[:nvecs]], basis_t[spikes_idx_est[:nvecs]])

plt.figure(figsize=(4.5, 1.5))
for j in range(k):
    cl = (J[:, j] == 1)
    plt.plot(x[cl], est[cl], ls='', marker='.', ms=1)
plt.grid(ls=':')
plt.title("Estimated classes")
plt.savefig(fig_folder+"KoT_classes.png", bbox_inches='tight', transparent=True)
plt.show()

# %% Classification performance

n = 2500
p = int(0.1*n)
L = int(0.1*n)
setting = "$n = {} \quad p = {} \quad L = {}$".format(n, p, L)

k = 2
pi = np.array([0.5, 0.5])
mu = stats.norm.rvs(size=(p, 1))
mu /= np.linalg.norm(mu)
M0 = np.concatenate([+mu, -mu], axis=1)
J = utils.getJ(n, pi)

n_reps = 10

B = utils.genB(n, L, kind='toeplitz')
khi, basis_t = linalg.eigh(B)
basis_t = basis_t.T

mu_norms = np.linspace(0.1, np.sqrt(20), 19)

est = np.zeros((n_reps, 2, len(mu_norms), n))

for i in tqdm(range(n_reps), disable=(n_reps <= 1)):
    for j, mu_norm in enumerate(tqdm(mu_norms, disable=(n_reps > 1))):
        M = mu_norm*M0
        P = M@J.T
        Z = stats.norm.rvs(size=(p, n))
        X = P+Z
        K = (X.T.conj()@X)*B/p
        
        first_spike = eigsh(K, k=1, which='LA')[0][0]
        mu_norm_est = utils.est_mu(n, p, L, first_spike, khi=khi)
        _, (spikes_idx_est, _, zeta_est, _) = utils.get_spikes(n, p, L, mu_norm_est, khi=khi)
        
        i_top = spikes_idx_est[spikes_idx_est >= n//2].min(initial=n)
        i_bottom = spikes_idx_est[spikes_idx_est < n//2].max(initial=-1)+1
        eigvecs_t = np.empty((n, n))
        eigvecs_t[:] = np.nan
        if n-i_top > 0:
            eigvecs_t[i_top:] = eigsh(K, k=n-i_top, which='LA')[1].T
        if i_bottom > 0:
            eigvecs_t[:i_bottom] = eigsh(K, k=i_bottom, which='SA')[1].T
        
        nvecs = int(np.round(0.75*len(spikes_idx_est)))
        est[i, 0, j] = utils.classification(n, zeta_est[:1], eigvecs_t[spikes_idx_est[:1]], basis_t[spikes_idx_est[:1]])
        est[i, 1, j] = utils.classification(n, zeta_est[:nvecs], eigvecs_t[spikes_idx_est[:nvecs]], basis_t[spikes_idx_est[:nvecs]])

cerr = np.mean(np.sign(est) == np.tile(J[:, 0]-J[:, 1], n_reps*2*len(mu_norms)).reshape(est.shape), axis=-1)
cerr = np.minimum(cerr, 1-cerr)
cerr_mean = np.mean(cerr, axis=0)
cerr_std = np.std(cerr, axis=0)

# %%

mu2, khi_k, khi_l = np.meshgrid(mu_norms**2, khi, khi)
zeta_ = np.mean((1-p*(khi_l/((mu2+1)*khi_k-khi_l))**2)*(mu2/(mu2+1)), axis=2)
zeta_ = np.maximum(0, zeta_)
Q = stats.norm.sf(np.sqrt(zeta_/(1-zeta_)))

# %%

plt.errorbar(mu_norms**2, cerr_mean[0], yerr=cerr_std[0], ls=':', marker='.',
             capsize=2, barsabove=True, elinewidth=.8, zorder=3, label="1 spike ({} observations)".format(n_reps))
plt.errorbar(mu_norms**2, cerr_mean[1], yerr=cerr_std[1], ls='--', marker='.',
             capsize=2, barsabove=True, elinewidth=.8, zorder=2, label="¾ visible spikes ({} observations)".format(n_reps))
plt.plot(mu_norms**2, Q[-1, :], zorder=1, label="$\\mathcal{Q}(\\sqrt{\\zeta_0/(1-\\zeta_0)})$")
plt.grid(ls=':')
plt.xlabel("$\\| \\mu \\|^2$ (easier classification $\\rightarrow$)")
plt.ylabel("Classification error")
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"KoT_cerr.tex")
plt.show()
