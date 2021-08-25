# # Classification

# In[17]:


if easy_setting:
    print("Optimal classification error: {:.2%}".format(stats.norm.sf(np.sqrt(zeta_t[0]/(1-zeta_t[0])))))


# #### Separating hyperplane

# In[18]:


if easy_setting:
    nvecs = 9
    first_spike = eigvals_t[0, spikes_idx_t[0]]
    mu_norm_est = utils.est_mu(n, p, L, first_spike, tau)
    _, (spikes_idx_est, _, zeta_est, _) = utils.get_spikes(n, p, L, mu_norm_est, tau=tau)
    U_hat = eigvecs_t[0, spikes_idx_est[:nvecs]]
    w = np.mean(np.sign(U_hat[0])*U_hat, axis=1)
    w /= np.linalg.norm(w)
    classes_hp = 0.5*np.sign((w.T)@U_hat)+0.5


# In[19]:


if easy_setting:
    i = spikes_idx_t[0]
    x = np.arange(n)
    for j in range(k):
        cl = (classes_hp == j)
        plt.plot(x[cl], eigvecs_t[0, i, cl], ls='', marker='.', label=j)
    plt.grid(ls=':')
    plt.title("Eigenvector {} ($k = {}$) | Toeplitz mask".format(n-i, natural_idx_t[0]))
    plt.suptitle(setting)
    plt.legend(title="Predicted")
    plt.show()


# In[20]:


if easy_setting:
    cerr_hp = np.mean(classes_hp == np.sum([j*J[:, j] for j in range(k)], axis=0))
    print("Classification error: {:.2%}".format(min(cerr_hp, 1-cerr_hp), 5))


# In[21]:


if easy_setting:
    w_true = np.mean(mix_basis[spikes_idx_t[:nvecs]], axis=1)
    w_true /= np.linalg.norm(w_true)


# In[22]:


if easy_setting:
    plots = [[0, 1], [0, 2], [1, 2]]
    fig, ax = plt.subplots(len(plots), 2, figsize=(10, len(plots)*4), squeeze=False)
    for i, [a, b] in enumerate(plots):
        for j in range(k):
            cl = (J[:, j] == 1)
            x = eigvecs_t[0, spikes_idx_t[a], cl]
            y = eigvecs_t[0, spikes_idx_t[b], cl]
            ax[i, 0].scatter(x, y, marker='.', edgecolors='none', alpha=.5, label=j)
            x = eigvecs_t[0, spikes_idx_t[a], cl]/basis_t[spikes_idx_t[a], cl]
            y = eigvecs_t[0, spikes_idx_t[b], cl]/basis_t[spikes_idx_t[b], cl]
            ax[i, 1].scatter(x, y, marker='.', edgecolors='none', alpha=.5, label=j)
        xlim = ax[i, 0].get_xlim()
        ax[i, 0].plot([0, w_true[a]], [0, w_true[b]], c='C4', ls='--', label="$w$")
        ax[i, 0].plot([0, w[a]], [0, w[b]], c='C5', ls=':', label="$\\hat{w}$")
        ax[i, 0].set_xlim(xlim)
        for j in range(2):
            ax[i, j].grid(ls=':')
            ax[i, j].set_ylim(ax[i, j].get_xlim())
            ax[i, j].legend()
        ax[i, 0].plot(mix_basis[spikes_idx_t[a]], mix_basis[spikes_idx_t[b]], color='C3')
        ax[i, 0].set_xlabel("$\\hat{{u}}_{{{}}}$".format(natural_idx_t[a]))
        ax[i, 0].set_ylabel("$\\hat{{u}}_{{{}}}$".format(natural_idx_t[b]))
        ax[i, 1].set_xlabel("$\\hat{{u}}_{{{}}} / u_{{{}}}$".format(natural_idx_t[a], natural_idx_t[a]))
        ax[i, 1].set_ylabel("$\\hat{{u}}_{{{}}} / u_{{{}}}$".format(natural_idx_t[b], natural_idx_t[b]))
    fig.suptitle(setting)
    fig.tight_layout()
    plt.show()


# #### First eigenvector

# In[23]:


if easy_setting:
    classes_fe = (eigvecs_t[0, spikes_idx_t[0]] < 0).astype(int)


# In[24]:


if easy_setting:
    i = spikes_idx_t[0]
    x = np.arange(n)
    for j in range(k):
        cl = (classes_fe == j)
        plt.plot(x[cl], eigvecs_t[0, i, cl], ls='', marker='.', label=j)
    plt.grid(ls=':')
    plt.title("Eigenvector {} ($k = {}$) | Toeplitz mask".format(n-i, natural_idx_t[0]))
    plt.suptitle(setting)
    plt.legend(title="Predicted")
    plt.show()


# In[25]:


if easy_setting:
    cerr_fe = np.mean(classes_fe == np.sum([j*J[:, j] for j in range(k)], axis=0))
    print("Classification error: {:.2%}".format(min(cerr_fe, 1-cerr_fe), 5))


# #### Projection on eigenspace

# In[26]:


if easy_setting:
    from scipy.linalg import eigh

    U = basis_t[spikes_idx_est[:nvecs]]

    kurtosis = np.mean(U**4, axis=1)/np.mean(U**2, axis=1)**2
    gamma = 2*kurtosis*zeta_est[:nvecs]*(2-zeta_est[:nvecs])*(1-zeta_est[:nvecs])**2/n
    w = 1/gamma

    Pi_perp = np.eye(n)-(U_hat.T)@U_hat

    d = eigh(Pi_perp*((U.T*w)@U), subset_by_index=(0, 0))[1][:, 0]
    classes_pr = 0.5*np.sign(d)+0.5


# In[27]:


if easy_setting:
    i = spikes_idx_t[0]
    x = np.arange(n)
    for j in range(k):
        cl = (classes_pr == j)
        plt.plot(x[cl], eigvecs_t[0, i, cl], ls='', marker='.', label=j)
    plt.grid(ls=':')
    plt.title("Eigenvector {} ($k = {}$) | Toeplitz mask".format(n-i, natural_idx_t[0]))
    plt.suptitle(setting)
    plt.legend(title="Predicted")
    plt.show()


# In[28]:


if easy_setting:
    cerr_pr = np.mean(classes_pr == np.sum([j*J[:, j] for j in range(k)], axis=0))
    print("Classification error: {:.2%}".format(min(cerr_pr, 1-cerr_pr)))

