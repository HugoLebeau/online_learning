import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm

classes = ['collie', 'tabby']

features_files = [open('{}_features.csv'.format(cl), 'r') for cl in classes]
features = []
for f in tqdm(features_files):
    features.append(np.loadtxt(f))
    f.close()
all_features = np.concatenate(features, axis=0)

mean = np.mean(all_features, axis=0)
centered_features = all_features-mean
cov = (centered_features.T)@centered_features/centered_features.shape[0]
sqrt_cov = linalg.sqrtm(cov)

output = [open('{}_normalized.csv'.format(cl), 'w') for cl in classes]

for f, o in tqdm(zip(features, output)):
    np.savetxt(o, linalg.solve(sqrt_cov, (f-mean).T).T)
    o.close()
