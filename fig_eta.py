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
p = 30
L = 10
z = 1+1j
setting = "$n = {} \quad p = {} \quad L = {} \quad z = {}$".format(n, p, L, z)

delta = 1e-5

C = utils.gen_mask(n, L, kind='circulant')
T = utils.gen_mask(n, L, kind='toeplitz')

def f(eta, R):
    eta_new = np.zeros_like(eta)
    for r in range(n):
        for s in range(n):
            eta_new[r, s] = np.sum(R[:, r]*(R[s, :]-eta[:, s])/(1-z-np.diag(eta)))/p
    return eta_new

R = C

for R, R_name in zip([C, T], ['C', 'T']):
    etam = np.eye(n, dtype='complex')
    etap = f(etam, R)
    while linalg.norm(etap-etam) > delta:
        etap, etam = f(etap, R), etap
    
    plt.colorbar(plt.matshow(etap.real, cmap='Blues'))
    plt.title("$\\Re(\\bar{\\eta})$ | $\\bf R = "+R_name+"$")
    plt.xticks([])
    plt.yticks([])
    tikzplotlib.save(fig_folder+"eta_real_"+R_name+".tex")
    plt.show()
    
    plt.colorbar(plt.matshow(etap.imag, cmap='Oranges'))
    plt.title("$\\Im(\\bar{\\eta})$ | $\\bf R = "+R_name+"$")
    plt.xticks([])
    plt.yticks([])
    tikzplotlib.save(fig_folder+"eta_imag_"+R_name+".tex")
    plt.show()
