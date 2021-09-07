import warnings
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optim
import scipy.stats as stats
from itertools import combinations
from time import time
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

khi = -0.21723362821122165741 # minimum of the Dirichlet kernel


# GENERIC FUNCTIONS

def fixed_point(f, x0, delta=1e-6, time_lim=5):
    ''' Fixed point algorithm '''
    xp, xm = f(x0), x0
    t0 = time()
    while np.abs(xp-xm) > delta:
        xp, xm = f(xp), xp
        if time()-t0 > time_lim:
            xp = np.nan
            break
    return xp

def getJ(n, pi):
    ''' Generation of a J matrix '''
    k = len(pi)
    n_ = np.round(n*pi).astype(int)
    n_[0] += n-np.sum(n_)
    J = np.zeros((n, k), dtype=int)
    sum_n = 0
    for i, ni in enumerate(n_):
        J[sum_n:sum_n+ni, i] = 1
        sum_n += ni
    np.random.shuffle(J)
    return J


# BERNOULLI MASK

# Phase transition functions
def F(t, c, eS, eB, b):
    return t**4+(2/eS)*t**3+(1-c/eB)/eS**2*t**2-2*c/eS**3*t-c/eS**4
def G(t, c, eS, eB, b):
    return eS*b+eB*eS*(1+eS*t)/c+eS/(1+eS*t)+eB/(t*(1+eS*t))
def zeta(l, c, eS, eB, b, Gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # avoid division by 0 warning
        return np.where(l > Gamma, F(l, c, eS, eB, b)*eS**3/l/(1+eS*l)**3, 0.)

def genS_bernoulli(n, p, eS):
    ''' Generation of a p x n Bernoulli mask '''
    return stats.bernoulli.rvs(eS, size=(p, n))

def genB_bernoulli(n, eB, b):
    ''' Generation of an n x n Bernoulli mask '''
    B = stats.bernoulli.rvs(eB, size=(n, n))
    for i in range(n):
        B[i, i] = b
        B[i, i+1:] = B[i+1:, i]
    return B

def simul_bernoulli(nbMC, M, J, eS, eB, b, comp=False):
    ''' Two-way Bernoulli puncturing simulation '''
    n, p, k = J.shape[0], M.shape[0], J.shape[1]
    P = M@(J.T)
    n_ = J.sum(axis=0)
    dtype = 'complex' if comp else 'float'
    
    eigvals = np.zeros((nbMC, n))
    eigvecs = np.zeros((nbMC, k, n), dtype=dtype)
    alignments = np.zeros((nbMC, k))
    
    for iMC in tqdm(range(nbMC)):
        Z = stats.norm.rvs(size=(p, n))
        if comp:
            Z += 1j*stats.norm.rvs(size=(p, n))
            Z /= np.sqrt(2)
        X = Z+P
        S, B = genS_bernoulli(n, p, eS), genB_bernoulli(n, eB, b)
        XS = X*S
        K = ((XS.T.conj())@XS)*B/p
    
        eigvals[iMC] = linalg.eigh(K, eigvals_only=True)
        eigvecs[iMC] = eigsh(K, k=k, which='LA')[1].T
        alignments[iMC] = (np.abs(eigvecs[iMC]@J)**2).sum(axis=1)/n_
    
    return eigvals, eigvecs, alignments

def get_LSD_bernoulli(axr, c, eS, eB, b, y=1e-6):
    ''' Limiting spectral distribution for a Bernoulli mask '''
    m_stieltjes = np.zeros(axr.shape, dtype='complex')
    m_stieltjes[-1] = 1j
    density = np.zeros(axr.shape, dtype='float')
    for i, x in enumerate(tqdm(axr)):
        z = x+1j*y
        func = lambda m: 1/(eS*b-z-eB*eS**2*m/c+(eB**3*eS**3*m**2/c**2)/(1+eB*eS*m/c))
        m_stieltjes[i] = fixed_point(func, m_stieltjes[i-1] if not np.isnan(m_stieltjes[i-1]) else 1j)
        density[i] = m_stieltjes[i].imag/np.pi
    return density

def get_spikes_bernoulli(pi, M, c, eS, eB, b, bracket=[0.1, 1000], method='brentq'):
    ''' Position of the spikes and eigenvectors alignments for a Bernoulli mask '''
    L = (np.sqrt(pi).reshape((-1, 1))@np.sqrt(pi).reshape((1, -1)))*(M.T.conj()@M)
    ell = linalg.eigh(L, eigvals_only=True)[::-1]
    root = optim.root_scalar(F, args=(c, eS, eB, b), bracket=bracket, method=method)
    Gamma = root.root # phase transition threshold
    rho = G(ell[ell > Gamma], c, eS, eB, b) # position of the spikes
    alignments = zeta(ell, c, eS, eB, b, Gamma)
    return rho, alignments


# TOEPLITZ / CIRCULANT MASK

def nu(L, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # avoid division by 0 warning
        return np.where(np.isclose(x%(2*np.pi), 0.), 2*L-1, np.divide(np.sin((L-0.5)*x), np.sin(x/2)))

# L <-> eB for a band Toeplitz matrix
def eB2L(n, eB):
    return int(np.round(n+0.5-np.sqrt(n**2*(1-eB)+0.25)))
def L2eB(n, L):
    return 1/n+(L-1)*(2*n-L)/n**2

def gen_mask(n, L, kind='toeplitz'):
    ''' Generation of an n x n Toeplitz/circulant mask '''
    if kind == 'toeplitz':
        c = np.zeros(n) # first column
        for i in range(min(n, L)):
            c[i] = 1.
        B = linalg.toeplitz(c)
        return B
    elif kind == 'circulant':
        c = np.zeros(n) # first column
        for i in range(-min(n, L)+1, min(n, L)):
            c[i] = 1.
        B = linalg.circulant(c)
        return B
    else:
        raise NotImplementedError(kind)

def basis(n, L, kind='toeplitz'):
    ''' Eigenvectors of the Toeplitz mask '''
    B = gen_mask(n, L, kind=kind)
    eigvals, eigvecs = linalg.eigh(B)
    return eigvals, eigvecs.T

def simul(nbMC, L, M, J, mask='toeplitz', comp=False, verbose=True):
    ''' Toeplitz/circulant puncturing simulation '''
    n, p = J.shape[0], M.shape[0]
    P = M@(J.T)
    dtype = 'complex' if comp else 'float'
    
    eigvals = np.zeros((nbMC, n))
    eigvecs = np.zeros((nbMC, n, n), dtype=dtype)
    
    for iMC in tqdm(range(nbMC), disable=not verbose):
        Z = stats.norm.rvs(size=(p, n))
        if comp:
            Z += 1j*stats.norm.rvs(size=(p, n))
            Z /= np.sqrt(2)
        X = Z+P
        B = gen_mask(n, L, kind=mask)
        K = ((X.T.conj())@X)*B/p
    
        eigvals[iMC], eigvecs[iMC] = linalg.eigh(K)
        eigvecs[iMC] = eigvecs[iMC].T
    
    return eigvals, eigvecs

def eta0(axr, n, p, L, psi, y=1e-6, delta=1e-6, verbose=True):
    ''' 1/(1-z-eta0) is the Stieltjes transform for a circulant/Toeplitz mask '''
    eta0 = np.zeros(axr.shape, dtype='complex')
    eta0[-1] = 1j
    for i, x in enumerate(tqdm(axr, disable=not verbose)):
        z = x+1j*y
        func = lambda eta0: np.mean(psi**2/(p*(1-z-eta0)+psi))
        eta0[i] = fixed_point(func, eta0[i-1] if not np.isnan(eta0[i-1]) else 1j, delta=delta)
    return eta0

def get_spikes(n, p, L, mu_norm, tau=None):
    ''' Position, value and alignments of spikes for a circulant and Toeplitz mask '''
    psi = nu(L, 2*np.arange(n)*np.pi/n)
    aton = np.argsort(psi) # Ascending order -> natural order
    ntoa = np.argsort(aton) # Natural order -> ascending order
    if tau is None:
        tau = linalg.eigh(gen_mask(n, L, kind='toeplitz'), eigvals_only=True)
    tau = tau[ntoa]
    psi_k, psi_l = np.meshgrid(psi, psi)
    E = np.isclose(psi_k, psi_l)
    tau_k, tau_l = np.meshgrid(tau, tau)
    spikes_c = (mu_norm**2+1)*psi*(1/p+np.mean(psi_l/((mu_norm**2+1)*psi_k-psi_l), axis=0))
    zeta_c = (1-p*np.mean((psi_l/((mu_norm**2+1)*psi_k-psi_l))**2, axis=0))*((mu_norm**2)/(mu_norm**2+1))
    spikes_t = (mu_norm**2+1)*tau*(1/p+np.mean(tau_l/((mu_norm**2+1)*tau_k-tau_l), axis=0))
    zeta_t = (1-p*np.mean((tau_l/((mu_norm**2+1)*tau_k-tau_l))**2, axis=0))*((mu_norm**2)/(mu_norm**2+1))
    
    # Get indices of spikes grouped by multiplicity
    visible_idx_c, visible_idx_t = [], []
    visible = (zeta_c > 0) | (zeta_t > 0)
    visible_where = np.where(visible)[0]
    for i in visible_where:
        if i != -1:
            group = list(np.where(E[i])[0])
            group_c = [j for j in group if zeta_c[j] > 0]
            group_t = [j for j in group if zeta_t[j] > 0]
            if group_c:
                visible_idx_c.append(group_c)
            if group_t:
                visible_idx_t.append(group_t)
            for j in group:
                visible_where[visible_where == j] = -1
    visible_idx_c = [x for group in visible_idx_c for x in group]
    visible_idx_t = [x for group in visible_idx_t for x in group]
    
    res_c = (ntoa[visible_idx_c], spikes_c[visible_idx_c], zeta_c[visible_idx_c], visible_idx_c)
    res_t = (ntoa[visible_idx_t], spikes_t[visible_idx_t], zeta_t[visible_idx_t], visible_idx_t)
    
    return res_c, res_t

def est_mu(n, p, L, first_spike, psi, mu0=3):
    ''' Estimation of ||mu|| with the first spike for a circulant/Toeplitz mask '''
    func = lambda x: x*(1+p*np.mean(psi/(x-psi)))/p-first_spike
    res = optim.fsolve(func, (mu0**2+1)*psi[-1])
    return np.sqrt(res/psi[-1]-1)

def classification(k, eigvecs, basis, smooth_par=0.15, h_start=None):
    df = basis.shape[0]
    n_eigvecs = eigvecs.shape[0]
    n = eigvecs.shape[1]
    
    if h_start is None:
        h_start = 5*k
    
    idx = np.arange(n)
    partition = -np.ones(n, dtype=int)
    
    # First step: pre-classification with an exponential smoothing
    exp_smooth = np.zeros((n_eigvecs, n))
    
    id_prev = -np.ones(k, dtype=int)
    # Initialisation with Agglomerative Clustering
    partition[:h_start] = AgglomerativeClustering(n_clusters=k).fit(eigvecs[:, :h_start].T).labels_
    # Exponential smoothing of the first points
    for i in range(h_start):
        exp_smooth[:, i] = smooth_par*eigvecs[:, i]+(1-smooth_par)*exp_smooth[:, id_prev[partition[i]]]
        id_prev[partition[i]] = i
    # Pre-classification by minimizing the growth
    for i in range(h_start, n):
        growth = linalg.norm(eigvecs[:, [i]]-exp_smooth[:, id_prev], axis=0)/(idx[i]-idx[id_prev])
        j = np.argmin(growth)
        exp_smooth[:, i] = smooth_par*eigvecs[:, i]+(1-smooth_par)*exp_smooth[:, id_prev[j]]
        partition[i] = j
        id_prev[j] = i
    
    partition0 = partition.copy()
    
    # Second step: projection on the theoretical basis
    reg = np.zeros((k, df, n_eigvecs))
    curves = np.zeros((k, n_eigvecs, n))
    dist = np.zeros((k, n))
    
    convergence = False
    while not convergence:
        for j in range(k):
            X_reg_j = basis[:, partition == j]
            reg[j] = linalg.solve(X_reg_j@(X_reg_j.T), X_reg_j@(eigvecs[:, partition == j].T))
            curves[j] = ((basis.T)@reg[j]).T
            dist[j] = linalg.norm(eigvecs-curves[j], axis=0)
        partition_new = np.argmin(dist, axis=0)
        convergence = np.all(partition == partition_new)
        partition = partition_new
    
    return partition, (curves, reg, exp_smooth, partition0)
