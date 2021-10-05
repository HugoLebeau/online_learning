import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.stats as stats

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

def MP(x, c):
    MP_distrib = np.zeros_like(x)
    id_min = np.argmin(np.abs(x))
    if np.isclose(x[id_min], 0):
        MP_distrib[id_min] = max(0, 1-c)
    Em, Ep = (1-np.sqrt(1/c))**2, (1+np.sqrt(1/c))**2
    mask = ((x >= Em) & (x <= Ep))
    MP_distrib[mask] += c*np.sqrt((x[mask]-Em)*(Ep-x[mask]))/(2*np.pi*x[mask])
    return MP_distrib

n = 1000
c = 2
p = int(c*n)

pi = np.array([0.5, 0.5])
mu = stats.norm.rvs(size=(p, 1))
mu_norm = 2.5
mu = mu_norm*mu/linalg.norm(mu)
M = np.concatenate([+mu, -mu], axis=1)

setting = "$n = {} \\quad p = {} \\quad \\| \\mu \\| = {}$".format(n, p, mu_norm)

J = utils.getJ(n, pi)
P = M@(J.T)
Z = stats.norm.rvs(size=(p, n))
X = P+Z
K = (X.T)@X/p

eigvals, eigvecs = linalg.eigh(K)

x = np.linspace(eigvals.min(), eigvals.max(), 500)
MP_distrib = MP(x, c)

right_edge = (1+np.sqrt(1/c))**2
theta = mu_norm**2/c
spike_pos = (1+theta)*(1+1/(c*theta))

hist = plt.hist(eigvals, bins='auto', color='C0', edgecolor='black', density=True, zorder=3, label="ESD")
plt.plot(x, MP_distrib, color='C1', zorder=4, label="LSD")
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
l_arrow = (ymax-ymin)/10
width = (xmax-xmin)/1000
head_width = (xmax-xmin)/100
plt.arrow(spike_pos, hist[0][-1]+l_arrow+l_arrow/10, 0, -l_arrow, width=width, length_includes_head=True,
          head_width=head_width, head_length=head_width/5, color='C2', zorder=5, label="Spike")
plt.axvline(x=right_edge, ls='--', color='C3', zorder=5, label="Right edge")
plt.grid(ls=':')
plt.ylabel("Density")
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"MP_distrib.tex")
plt.show()

axr = np.arange(n)
for j in range(2):
    cl = (J[:, j] == 1)
    plt.plot(axr[cl], eigvecs[cl, -1], ls='', marker='.', label="Class {}".format(j+1))
plt.grid(ls=':')
plt.legend()
plt.title(setting)
tikzplotlib.save(fig_folder+"MP_eigvec.tex")
plt.show()

ax_mu2 = np.linspace(0, 20, 1000)*np.sqrt(c)
Pe = stats.norm.sf(np.sqrt(np.maximum((ax_mu2**2-c)/(ax_mu2+c*(ax_mu2+1)), 0)))
plt.plot(np.sqrt(ax_mu2), Pe)
plt.axvline(x=np.sqrt(np.sqrt(c)), color='black', ls='--', label="$\| \mu \|^2 = \sqrt{c}$")
plt.grid(ls=':')
plt.xlabel("$\| \mu \|$")
plt.ylabel("$P_e$")
plt.legend()
tikzplotlib.save(fig_folder+"MP_classif_error.tex")
plt.show()
