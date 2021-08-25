d = J[:, 0]-J[:, 1]
theta = eigvecs_t[0]@((basis_t*d).T)
theta2 = theta**2
quant = np.var(theta, ddof=1)*stats.chi2.ppf(1-1e-3, df=1)
mask = np.any(theta2[spikes_idx_t] > quant, axis=0)

theta_smooth = theta.copy()
theta_smooth[theta2 < quant] = 0
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	theta_smooth = theta_smooth/np.linalg.norm(theta_smooth, axis=1)[:, None]
mix_basis = theta_smooth@basis_t
mix_basis[np.isnan(mix_basis)] = 0

xlabels = n-np.arange(n)[mask]
norm = Normalize(0, theta2.max())

imshow = plt.imshow(theta2[spikes_idx_t][:, mask], aspect='auto', interpolation='none', cmap='Greens', norm=norm)
plt.colorbar(imshow)
plt.xticks(range(len(xlabels)), xlabels, rotation=90)
plt.yticks(range(len(spikes_idx_t)), n-spikes_idx_t)
plt.xlabel("Eigenvector")
plt.ylabel("Spike")
plt.title(setting)
plt.show()
