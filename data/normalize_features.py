import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

classes = ['collie', 'tabby']

features_files = [open('{}_features.csv'.format(cl), 'r') for cl in classes]
features = []
for f in tqdm(features_files):
    features.append(np.loadtxt(f))
    f.close()
all_features = np.concatenate(features, axis=0)
n, p = all_features.shape

mean = np.mean(all_features, axis=0)
std = np.std(all_features, ddof=1, axis=0)
centered_features = (all_features-mean)/std
cov = (centered_features.T)@centered_features/n
# eigval, eigvec = eigsh(cov, k=1, which='LA')

normalized_features = []

print("tr cov / p = {}".format(np.trace(cov)/p))
for cl, f in zip(classes, features):
    nf = (f-mean)/std
    normalized_features.append(nf)
    cov_nf = (nf.T)@nf/nf.shape[0]
    eigval_nf, eigvec_nf = eigsh(cov_nf, k=1, which='LA')
    mu = np.mean(nf, axis=0)
    mu_norm = np.linalg.norm(mu)
    print(cl)
    print("||mu|| = {}".format(mu_norm))
    print("cov max eigenvalue = {}".format(eigval_nf[0]))
    print("< mu/||mu|| , max cov eigvec > = {}".format(mu@eigvec_nf[:, 0]/mu_norm))

output = [open('{}_normalized.csv'.format(cl), 'w') for cl in classes]

for nf, o in zip(tqdm(normalized_features), output):
    np.savetxt(o, nf)
    o.close()
