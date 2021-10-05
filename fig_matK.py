import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.stats as stats
from matplotlib.colors import Normalize

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

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
C = utils.gen_mask(n, L, 'circulant')
T = utils.gen_mask(n, L, 'toeplitz')
Z = stats.norm.rvs(size=(p, n))
X = Z+P
K = (X.T)@X
KC = K*C
KT = K*T

color_bound = np.abs(K).max()
norm = Normalize(-color_bound, color_bound)

plt.matshow(K, cmap='bwr', norm=norm)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.title("$\\bf K$")
tikzplotlib.save(fig_folder+"K.tex")
plt.show()

plt.matshow(KT, cmap='bwr', norm=norm)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.title("$\\bf K \odot \\bf T$")
tikzplotlib.save(fig_folder+"KoT.tex")
plt.show()

plt.matshow(KC, cmap='bwr', norm=norm)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.title("$\\bf K \odot \\bf C$")
tikzplotlib.save(fig_folder+"Koc.tex")
plt.show()
