import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import punct_utils as utils

np.random.seed(14159)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = True
fig_folder = 'figures/'

n = 1000
L = 200

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

class_count, details = utils.easy_streaming(get_data, T, n, p, L, k)

for f in features:
    f.close()

lbda, w, partition_ite, time_ite = details
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
tikzplotlib.save(fig_folder+"real_data_easy_c_err.tex")
plt.show()

plt.plot(lbda, label=i+1)
plt.grid(ls=':')
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue")
plt.title(setting)
plt.legend()
#tikzplotlib.save(fig_folder+"real_data_easy_lbda.tex")
plt.show()

t = T//2
x = np.arange(max(0, t-n+1), max(n, t+1))
for j in range(k):
    color = 'C'+str(j)
    cl = (true_partition[max(0, t-n+1):t+1] == j)
    cl = np.append(cl, np.zeros(max(0, n-t-1), dtype=bool))
    plt.plot(x[cl], w[t, cl], ls='', marker='.', alpha=.3, c=color, label=j+1, zorder=2)
plt.grid(ls=':')
plt.title(setting)
plt.legend()
tikzplotlib.save(fig_folder+"real_data_easy_eigvec0.tex")
plt.show()
