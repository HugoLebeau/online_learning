import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import PercentFormatter

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 1000
L = 100

file = 'data/'
classes = ['collie', 'tabby']

features = [open(file+'{}_features.csv'.format(cl), 'r') for cl in classes]

k = len(classes)
p = len(features[0].readline().rstrip().split(' '))
features[0].seek(0)

cl_len = np.zeros(k, dtype=int)
cl_mean = np.zeros((k, p))
for i, f in enumerate(features):
    for line in f:
        cl_len[i] += 1
        cl_mean[i] += np.array(line.rstrip().split(' '), dtype=float)
    f.seek(0)
T = np.sum(cl_len)
mean = np.sum(cl_mean, axis=0)/T

setting = "$T = {} \quad n = {} \quad p = {} \quad L = {}$".format(T, n, p, L)

true_partition = np.repeat(range(k), cl_len)
np.random.shuffle(true_partition)
get_data = lambda t: np.array(features[true_partition[t]].readline().rstrip().split(' '), dtype=float)-mean

smooth_par = 0.15
h_start = 10*k
divided_warmup = True

n_eigvecs = 5
idx_basis = [-1, -2, -3, -4, -5, -6, -7]

basis = utils.basis(n, L, kind='toeplitz')[1][idx_basis]

class_count, details = utils.streaming(get_data, T, p, L, k, n_eigvecs, basis, smooth_par, h_start, divided_warmup)

for f in features:
    f.close()

lbda, w, exp_smooth, partition0, curves, partition_ite, time_ite = details
partition = np.argmax(class_count, axis=1)

c_err, per, per_inv = utils.get_classif_error(k, partition, true_partition)
print("Classification error: {:.2%}".format(c_err))
delay_c_err = np.mean(per[partition_ite[n-1:]] != np.array([true_partition[t:t+n] for t in range(T-n+1)]), axis=0)[::-1]

plt.plot(delay_c_err)
plt.axhline(y=c_err, ls='--', label="Overall classification error")
plt.grid(ls=':')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.xlabel("Delay")
plt.ylabel("Classification error")
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"real_data_c_err.tex")
plt.show()

for i in range(n_eigvecs):
    plt.plot(lbda[i], label=i+1)
plt.grid(ls=':')
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue")
plt.title(setting)
plt.legend()
#tikzplotlib.save(fig_folder+"real_data_lbda.tex")
plt.show()

t = 5000
i = 0
x = np.arange(max(0, t-n+1), max(n, t+1))
for j in range(k):
    color = 'C'+str(j)
    cl = (true_partition[max(0, t-n+1):t+1] == j)
    cl = np.append(cl, np.zeros(max(0, n-t-1), dtype=bool))
    plt.plot(x[cl], w[t, i, cl], ls='', marker='.', alpha=.3, c=color, label=j+1, zorder=2)
    if (divided_warmup and t <= n-1) or (not divided_warmup and t == n-1):
        mask = ((x <= t) & (partition0 == per_inv[j]))
        plt.plot(x[mask], exp_smooth[i, mask], ls='--', c=color, zorder=3)
    if t >= n-1:
        plt.plot(x, curves[t, per_inv[j], i], c=color, zorder=4)
plt.grid(ls=':')
plt.title("Eigenvector {} | ".format(i+1)+setting)
plt.legend()
tikzplotlib.save(fig_folder+"real_data_eigvec0_{}.tex".format(t))
plt.show()


fig, ax = plt.subplots()

for j in range(k):
    color = 'C'+str(j)
    ax.plot([], ls='', marker='.', label=j, c=color)
    ax.plot([], ls='--', c=color)
    ax.plot([], ls='-', c=color)
ax.grid(ls=':')
ax.set_title("Eigenvector {} | ".format(i+1)+setting)
ax.legend()

def animate(t):
    x = np.arange(max(0, t-n+1), max(n, t+1))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim((w[t, i].min(), w[t, i].max()))
    for j in range(k):
        cl = (true_partition[max(0, t-n+1):t+1] == j)
        cl = np.append(cl, np.zeros(max(0, n-t-1), dtype=bool))
        ax.lines[3*j].set_data((x[cl], w[t, i, cl]))
        if (divided_warmup and t <= n-1) or (not divided_warmup and t == n-1):
            mask = ((x <= t) & (partition0 == per_inv[j]))
            ax.lines[3*j+1].set_data((x[mask], exp_smooth[i, mask]))
        if t >= n-1:
            ax.lines[3*j+2].set_data((x, curves[t, per_inv[j], i]))

anim = FuncAnimation(fig, animate, frames=T, interval=30*1000/T, repeat=False)
anim.save("streaming_tabby_collie.mp4")
