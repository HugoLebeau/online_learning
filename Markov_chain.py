import numpy as np
import matplotlib.pyplot as plt
infinity = 9999

L = 2
p = np.random.rand(L)
p /= np.sum(p)

Vx = np.append(np.arange(1, L+1), infinity)
Vr = np.arange(1, L+1)

P = np.zeros(((L+1)*L, (L+1)*L))
for i, x in enumerate(Vx):
    for j, r in enumerate(Vr):
        k = i*L+j # index of the actual state
        if r == x:
            P[k, L*L:(L+1)*L] = p
        elif r < x:
            P[k, i*L:(i+1)*L] = p
        else:
            P[k, (i+1)*L:(i+2)*L] = p

t_max = 1000
mu = np.zeros((t_max, (L+1)*L))
mu[0][:L] = p

for t in range(1, t_max):
    mu[t] = mu[t-1]@P

proba_well = np.sum(mu[:, L*L:(L+1)*L], axis=1)

ET = np.sum(1-proba_well)
VT = np.sum(np.diff(proba_well)*(np.arange(1, t_max)**2))-L**2
print(ET)
print(np.sqrt(VT))
print(p)

plt.plot(proba_well)
plt.axvline(x=ET, ls='--', color='red')
plt.grid(ls=':')
plt.show()
