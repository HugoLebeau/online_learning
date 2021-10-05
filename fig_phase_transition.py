import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import root_scalar

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 10000

c_r = np.arange(1, 100)/100
p_r = (c_r*n).astype(int)

L_r = np.arange(1, n//10)
eps_r = (2*L_r-1)/n
alpha_r = L_r/n

pht_ss_neps = np.sqrt(np.divide(*np.meshgrid(c_r, eps_r))).T
pht_ss_L = np.sqrt(np.divide(*np.meshgrid(c_r, alpha_r))).T
pht_b = np.zeros((p_r.size, L_r.size))
pht_c = np.zeros((p_r.size, L_r.size))

k_nu = 2*np.pi*np.arange(n)/n

a, b = 1e-5, 50

for i, (L, eps) in enumerate(tqdm(zip(L_r.astype(float), eps_r))):
    psi = utils.nu(L, k_nu)
    for j, (c, p) in enumerate(zip(c_r, p_r)):
        func_b = lambda t: t**4+2*t**3+(1-c/eps)*t**2-2*c*t-c
        func_c = lambda t: p*np.mean((psi/((t+1)*psi[0]-psi))**2)-1
        if func_b(a)*func_b(b) < 0:
            res_b = root_scalar(func_b, method='brentq', bracket=[a, b])
            pht_b[j, i] = res_b.root if res_b.converged else np.nan
        else:
            pht_b[j, i] = np.nan
        if func_c(a)*func_c(b) < 0:
            res_c = root_scalar(func_c, method='brentq', bracket=[a, b])
            pht_c[j, i] = res_c.root if res_c.converged else np.nan
        else:
            pht_c[j, i] = np.nan

right = eps_r[-1]
top = 1
plt.plot(eps_r, pht_b[0], color='C0', label="$\\bf R = B$")
plt.plot(eps_r, pht_c[0], color='C2', label="$\\bf R = C$")
plt.plot(eps_r, pht_ss_neps[0], color='C1', label="${\\bf K}_{ss, n\\varepsilon}$")
plt.plot(eps_r, pht_ss_L[0], color='C3', label="${\\bf K}_{ss, L}$")
plt.axhline(y=np.sqrt(c_r[0]), ls='--', color='black', label="$\\| \\mu \\|^2 = \\sqrt{c}$")
plt.fill_between(eps_r, pht_c[0], top, color='C2', alpha=0.15)
plt.grid(ls=':')
plt.xlim(0, right)
plt.ylim(0, top)
plt.text(right/2, 3*top/4, "Classification possible", color='C2',
         fontsize='xx-large', fontweight='bold', ha='center', va='center')
plt.text(right/2, np.sqrt(c_r[0])/2, "Classification impossible", color='black',
         fontsize='xx-large', fontweight='bold', ha='center', va='center')
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\| \\mu \\|^2$")
plt.legend()
plt.title("$c = {}$".format(c_r[0]))
tikzplotlib.save(fig_folder+"phase_transition_c"+str(c_r[0])+".tex")
plt.show()

i_start = np.argmax(eps_r >= 0.02)
mesh = plt.pcolormesh(eps_r[i_start:], c_r, pht_c[:, i_start:], cmap='Greens', shading='auto')
cbar = plt.colorbar(mesh)
cbar.set_label("$\\| \\mu \\|^2$")
plt.xlabel("$\\varepsilon$")
plt.ylabel("$c$")
tikzplotlib.save(fig_folder+"phase_transition.tex")
plt.show()
