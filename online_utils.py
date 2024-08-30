import warnings
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optim
import scipy.stats as stats
from itertools import combinations, permutations
from time import time
from tqdm import tqdm
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigsh
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
    ''' Generation of a matrix J '''
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

def get_classif_error(k, partition, true_partition):
    ''' Compute classification error '''
    permut = np.array(list(permutations(range(k))))
    c_err_list = [np.mean(pp[partition] != true_partition) for pp in permut]
    per = permut[np.argmin(c_err_list)]
    per_inv = np.argsort(per)
    c_err = np.min(c_err_list)
    return c_err, per, per_inv

# TOEPLITZ / CIRCULANT MASK

def nu(L, x):
    ''' sin( (2 L - 1) x / 2 ) / sin( x / 2 ) '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # avoid division by 0 warning
        return np.where(np.isclose(x%(2*np.pi), 0.), 2*L-1, np.divide(np.sin((L-0.5)*x), np.sin(x/2)))

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
    ''' Eigenvectors of the Toeplitz/circulant mask '''
    B = gen_mask(n, L, kind=kind)
    return linalg.eigh(B)

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
        R = gen_mask(n, L, kind=mask)
        K = ((X.T.conj())@X)*R/p
    
        eigvals[iMC], eigvecs[iMC] = linalg.eigh(K)
    
    return eigvals, eigvecs

def eta0(axr, p, tau, y=1e-6, delta=1e-6, verbose=True):
    ''' 1/(1-z-eta0) is the Stieltjes transform for a circulant mask '''
    eta0 = np.zeros(axr.shape, dtype='complex')
    eta0[-1] = 1j
    for i, x in enumerate(tqdm(axr, disable=not verbose)):
        z = x+1j*y
        func = lambda eta0: np.mean(tau**2/(p*(1-z-eta0)+tau))
        eta0[i] = fixed_point(func, eta0[i-1] if not np.isnan(eta0[i-1]) else 1j, delta=delta)
    return eta0

def get_pht(n, p, L, tau=None, a=1e-5, b=50):
    ''' Phase transition position for the first spike '''
    if tau is None:
        tau = linalg.eigh(gen_mask(n, L, kind='toeplitz'), eigvals_only=True)
    func = lambda t: p*np.mean((tau/((t+1)*tau[0]-tau))**2)-1
    if func(a)*func(b) < 0:
        res = optim.root_scalar(func, method='brentq', bracket=[a, b])
        return res.root if res.converged else np.nan
    else:
        return np.nan

def phi(c, eps, a=1e-5, b=50, infty=10000):
    ''' Asymptotic version of the previous function '''
    range_l = np.arange(1, infty)
    func = lambda t: 1-c*(1/(t*t)+2*np.sum(((t+1)/(np.sinc(eps*range_l))-1)**(-2)))
    if func(a)*func(b) < 0:
        res = optim.root_scalar(func, method='brentq', bracket=[a, b])
        return res.root if res.converged else np.nan
    else:
        return np.nan

def best_nL(M, p):
    ''' Compute best parameters n and L for a given memory (M) and data dimension (p) '''
    # Grid search parameters
    ll_r = np.linspace(0.01, 0.99, 99)
    alpha = 1e-5
    r = M/(p*p)
    func = lambda ll: phi(1/((1-ll)/ll+ll*r/2), 2*ll*r/((1-ll)/ll+ll*r/2)) # phase transition position given lambda
    # Minimize phase transition position on lambda
    grid = np.array([func(ll) for ll in ll_r]) # grid search
    ll_max = (np.sqrt(2*r+1)-1)/r
    ll_argmin = ll_r[np.argmin(grid[ll_r < ll_max])]
    a, b = max(alpha, ll_argmin-0.01), min(1-alpha, ll_argmin+0.01)
    res = optim.minimize_scalar(func, bracket=(a, b), method='brent') # minimization
    if res.success and res.x > 0 and res.x <= ll_max:
        n, L = int(np.round((1-res.x)*p/res.x+(res.x*M/p-1)/2)), int(np.round(res.x*M/p)) # resulting n, L
        return n, L
    else:
        return np.nan, np.nan

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

def est_mu(p, first_spike, tau, mu0=3):
    ''' Estimation of ||mu|| with the first spike for a circulant/Toeplitz mask '''
    func = lambda x: x*(1+p*np.mean(tau/(x-tau)))/p-first_spike
    res = optim.fsolve(func, (mu0**2+1)*tau[-1])
    return np.sqrt(res/tau[-1]-1)


# CLASSIFICATION FUNCTIONS

def init_pre_classif(k, eigvecs, h_start, smooth_par, partition, exp_smooth, id_prev):
    ''' Initialisation of the first step of the classification algorithm using agglomerative clustering '''
    partition[:h_start] = AgglomerativeClustering(n_clusters=k).fit(eigvecs[:h_start]).labels_
    for i in range(h_start): # exponential smoothing of the first points
        memory = ((1-smooth_par)/(i-id_prev))*(1+((1-smooth_par)/smooth_par)*(1-(1-smooth_par)**(i-id_prev-1)))
        exp_smooth[i] = (smooth_par*eigvecs[i]+memory[partition[i]]*exp_smooth[id_prev[partition[i]]])/(smooth_par+memory[partition[i]])
        id_prev[partition[i]] = i

def pre_classif_step(i, eigvecs, smooth_par, partition, exp_smooth, id_prev):
    ''' Iteration of the first step of the classification algorithm '''
    memory = ((1-smooth_par)/(i-id_prev))*(1+((1-smooth_par)/smooth_par)*(1-(1-smooth_par)**(i-id_prev-1)))
    growth = linalg.norm(eigvecs[[i]]-exp_smooth[id_prev], axis=1)/((1+memory/smooth_par)*(i-id_prev))
    j = np.argmin(growth)
    exp_smooth[i] = (smooth_par*eigvecs[i]+memory[j]*exp_smooth[id_prev[j]])/(smooth_par+memory[j])
    partition[i] = j
    id_prev[j] = i

def classif_reg(k, eigvecs, basis, partition, reg, curves, dist):
    ''' Second step of the classification algorithm '''
    convergence = False
    while not convergence:
        for j in range(k):
            X_reg_j = basis[partition == j]
            reg[j] = linalg.solve((X_reg_j.T)@X_reg_j, (X_reg_j.T)@eigvecs[partition == j])
            curves[j] = basis@reg[j]
            dist[j] = linalg.norm(eigvecs-curves[j], axis=1)
        partition_new = np.argmin(dist, axis=0)
        convergence = np.all(partition == partition_new)
        partition = partition_new
    return partition

def classif_reg_with_correction(k, eigvecs, basis, partition, reg, curves, dist):
    ''' Second step of the classification algorithm with a switch correction '''
    partition = classif_reg(k, eigvecs, basis, partition, reg, curves, dist)
    
    scores = [np.sum(np.take_along_axis(dist, partition[None, :], axis=0))]
    settings = [(partition.copy(), reg.copy(), curves.copy())]
    
    comb = list(combinations(range(k), 2))
    for a, b in comb:
        signs = (settings[0][2][a, :, 0]-settings[0][2][b, :, 0] > 0) # check crossing between curves
        if np.any(np.diff(signs)): # if there is a crossing
            partition = settings[0][0]
            # Switch classes
            mask_a, mask_b = (partition == a), (partition == b)
            partition[signs & mask_a] = b
            partition[signs & mask_b] = a
            # New partition
            partition = classif_reg(k, eigvecs, basis, partition, reg, curves, dist)
            scores.append(np.sum(np.take_along_axis(dist, partition[None, :], axis=0)))
            settings.append((partition.copy(), reg.copy(), curves.copy()))
    partition, reg, curves = settings[np.argmin(scores)]
    return partition

def classification(k, eigvecs, basis, smooth_par=0.15, h_start=None, correction=False):
    ''' Classification algorithm '''
    n = eigvecs.shape[0]
    n_eigvecs = eigvecs.shape[1]
    df = basis.shape[1]
    
    if h_start is None:
        h_start = 10*k
    
    partition0 = -np.ones(n, dtype=int)
    
    # First step: pre-classification with an exponential smoothing
    exp_smooth = np.zeros((n, n_eigvecs))
    id_prev = -np.ones(k, dtype=int)
    init_pre_classif(k, eigvecs, h_start, smooth_par, partition0, exp_smooth, id_prev)
    for i in range(h_start, n):
        pre_classif_step(i, eigvecs, smooth_par, partition0, exp_smooth, id_prev)
    
    # Second step: projection on the theoretical basis
    reg = np.zeros((k, df, n_eigvecs)) # regression coefficients of each class
    curves = np.zeros((k, n, n_eigvecs)) # curve associated to each class
    dist = np.zeros((k, n)) # distance of each point to each class
    
    if correction:
        partition = classif_reg_with_correction(k, eigvecs, basis, partition0, reg, curves, dist)
    else:
        partition = classif_reg(k, eigvecs, basis, partition0, reg, curves, dist)
    
    return partition, (exp_smooth, partition0, reg, curves)
    
def streaming(get_data, T, p, L, k, n_eigvecs, basis, smooth_par, h_start, divided_warmup, correction=False, verbose=True):
    ''' Streaming with online classification '''
    rk = np.arange(k)[None, :]
    n, df = basis.shape
    
    # Initialisation
    data = np.zeros((L, p)) # data pipeline
    K_data = np.zeros((n, L)) # sparse kernel matrix
    lbda = np.zeros((T, n_eigvecs)) # top eigenvalues
    w = np.ones((T, n, n_eigvecs)) # top eigenvectors
    
    exp_smooth = np.zeros((n, n_eigvecs)) # exponential smoothing of the first n points
    id_prev = -np.ones(k, dtype=int) # position of the last point seen of each class (used for exponential smoothing only)
    partition0 = -np.ones(n, dtype=int) # pre-classification of the first n points
    partition = -np.ones((T, n), dtype=int) # classification of the last n points
    class_count = np.zeros((T, k), dtype=int) # class_count[t, j] = number of times point x_t is classified in class j
    reg = np.zeros((k, df, n_eigvecs)) # regression coefficients of each class
    curves = np.zeros((T, k, n, n_eigvecs)) # curve associated to each class
    dist = np.zeros((k, n)) # distance of each point to each class
    
    time_ite = np.zeros(T) # duration of each iteration
    
    def make_K(K_data):
        data_u = K_data.T
        offsets_u = np.arange(L)
        K_u = dia_matrix((data_u, offsets_u), shape=(n, n))
        data_l = K_data[:, 1:].T
        offsets_l = np.arange(1, L)
        K_l = dia_matrix((data_l, offsets_l), shape=(n, n)).T
        return K_u+K_l
    
    classif = classif_reg_with_correction if correction else classif_reg
    
    # Streaming
    for t in tqdm(range(T), disable=not verbose):
        tic = time()
        
        # Get a new point in the pipeline
        data = np.roll(data, 1, axis=0)
        data[0] = get_data(t)
        
        # Compute kernel data
        if t < n:
            K_data[t] = data@data[0]/p
        else:
            K_data = np.roll(K_data, -1, axis=0)
            K_data[-1] = data@data[0]/p
        K = make_K(K_data) # make the (n, n) sparse kernel matrix
        
        # Compute top eigenpairs
        if t < n_eigvecs:
            lbda[t, -t-1:], w[t, :, -t-1:] = eigsh(K, k=t+1, v0=w[t-1, :, 0], which='LA')
        else:
            lbda[t], w[t] = eigsh(K, k=n_eigvecs, v0=w[t-1, :, 0], which='LA')
        
        # Sign correction on eigenvectors
        if t >= n:
            w[t] *= np.sign(np.sum(w[t, :-1]*w[t-1, 1:], axis=0))
        elif t > 0:
            w[t] *= np.sign(np.sum(w[t]*w[t-1], axis=0))
        
        # Classification
        if divided_warmup and t < n: # warm-up
            if t == h_start: # initialisation with agglomerative clustering
                init_pre_classif(k, w[t], h_start, smooth_par, partition0, exp_smooth, id_prev)
            elif t > h_start: # pre-classification by minimizing the growth
                pre_classif_step(t, w[t], smooth_par, partition0, exp_smooth, id_prev)
        if t == n-1:
            if not divided_warmup: # do warm-up at once
                init_pre_classif(k, w[t], h_start, smooth_par, partition0, exp_smooth, id_prev)
                for i in range(h_start, n):
                    pre_classif_step(i, w[t], smooth_par, partition0, exp_smooth, id_prev)
            # End of warm-up
            partition[t] = classif(k, w[t], basis, partition0, reg, curves[t], dist)
            class_count[max(0, t-n+1):t+1][rk == partition[t][:, None]] += 1
        elif t >= n:
            partition[t] = classif(k, w[t], basis, np.roll(partition[t-1], -1), reg, curves[t], dist)
            class_count[max(0, t-n+1):t+1][rk == partition[t][:, None]] += 1
        
        toc = time()
        time_ite[t] = toc-tic
    
    return class_count, (lbda[:, ::-1], w[:, :, ::-1], exp_smooth[:, ::-1], partition0, curves[:, :, :, ::-1], partition, time_ite)

def pm1_streaming(get_data, T, n, p, L, k, verbose=True):
    ''' Streaming with online classification (+/-1 setting) '''
    rk = np.arange(k)[None, :]
    
    # Initialisation
    data = np.zeros((L, p)) # data pipeline
    K_data = np.zeros((n, L)) # sparse kernel matrix
    lbda = np.empty(T) # top eigenvalue
    lbda[:] = np.nan
    w = np.ones((T, n, 1)) # top eigenvector
    
    partition = -np.ones((T, n), dtype=int) # classification of the last n points
    class_count = np.zeros((T, k), dtype=int) # class_count[t, j] = number of times point x_t is classified in class j
    
    time_ite = np.zeros(T) # duration of each iteration
    
    def make_K(K_data):
        data_u = K_data.T
        offsets_u = np.arange(L)
        K_u = dia_matrix((data_u, offsets_u), shape=(n, n))
        data_l = K_data[:, 1:].T
        offsets_l = np.arange(1, L)
        K_l = dia_matrix((data_l, offsets_l), shape=(n, n)).T
        return K_u+K_l
    
    # Streaming
    for t in tqdm(range(T), disable=not verbose):
        tic = time()
        
        # Get a new point in the pipeline
        data = np.roll(data, 1, axis=0)
        data[0] = get_data(t)
        
        # Compute kernel data
        if t < n:
            K_data[t] = data@data[0]/p
        else:
            K_data = np.roll(K_data, -1, axis=0)
            K_data[-1] = data@data[0]/p
        K = make_K(K_data) # make the (n, n) sparse kernel matrix
        
        if t >= n:
            lbda[t], w[t] = eigsh(K, k=1, v0=w[t-1], which='LA') # compute top eigenpair
            w[t] *= np.sign(np.sum(w[t, :-1]*w[t-1, 1:])) # sign correction on eigenvectors
            partition[t] = (w[t, :, 0] > 0).astype(int) # classification
            class_count[max(0, t-n+1):t+1][rk == partition[t][:, None]] += 1
        
        toc = time()
        time_ite[t] = toc-tic
    
    return class_count, (lbda, w[:, :, 0], partition, time_ite)
