import re
import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm

files = ['epsilon_normalized', 'epsilon_normalized_t']

x, y = [], []
regex = re.compile('[0-9]+:(-?[0-9\.]+)')

for file in files:
    f = open(file, 'r')
    for line in tqdm(f):
        x.append(re.findall(regex, line))
        y.append(line.split(' ', 1)[0])
    f.close()

x = np.array(x, dtype=float)
y = np.array(y, dtype=int)
classes = np.unique(y)

print("X shape:", x.shape) # (500000, 2000)
print("Y shape:", y.shape) # (500000,)

mean_x = np.mean(x, axis=0)
cov_x = ((x-mean_x).T)@(x-mean_x)/x.shape[0]

print("BEFORE NORMALIZATION")
print("X mean inf-norm:", np.linalg.norm(mean_x, ord=np.inf))
print("X mean 2-norm:", np.linalg.norm(mean_x, ord=2))
print("X cov - I_p max element:", np.max(np.abs(cov_x-np.eye(x.shape[1]))))
print("X cov - I_p max singular value:", np.linalg.norm(cov_x-np.eye(x.shape[1]), ord=2))
for cl in classes:
    print("Class {} | X mean 2-norm:".format(cl), np.linalg.norm(np.mean(x[y == cl], axis=0), ord=2))

x0 = linalg.solve(linalg.sqrtm(cov_x), (x-mean_x).T).T
mean_x0 = np.mean(x0, axis=0)
cov_x0 = ((x0-mean_x0).T)@(x0-mean_x0)/x0.shape[0]

print("AFTER NORMALIZATION")
print("X mean inf-norm:", np.linalg.norm(mean_x0, ord=np.inf))
print("X mean 2-norm:", np.linalg.norm(mean_x0, ord=2))
print("X cov - I_p max element:", np.max(np.abs(cov_x0-np.eye(x0.shape[1]))))
print("X cov - I_p max singular value:", np.linalg.norm(cov_x0-np.eye(x0.shape[1]), ord=2))
for cl in classes:
    print("Class {} | X mean 2-norm:".format(cl), np.linalg.norm(np.mean(x0[y == cl], axis=0), ord=2))

np.savetxt('epsilon_data', x0)
np.savetxt('epsilon_classes', y, fmt='%i')
