import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm

file = 'data/'
classes = ['collie', 'tabby']

features_files = [open(file+'{}_features.csv'.format(cl), 'r') for cl in classes]
features = []
for f in tqdm(features_files):
    features.append(np.loadtxt(f))
    f.close()
all_features = np.concatenate(features, axis=0)

mean = np.mean(all_features, axis=0)
cov = (all_features.T)@all_features/all_features.shape[0]

output = [open(file+'{}_normalized.csv'.format(cl), 'w') for cl in classes]

for f, o in tqdm(zip(features, output)):
    np.savetxt(o, linalg.solve(cov, f.T-mean[:, None]).T)
    o.close()
