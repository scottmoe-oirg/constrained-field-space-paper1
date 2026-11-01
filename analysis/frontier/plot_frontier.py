"""Fig. 2: median SE ratio (a) and median |correlated displacement| (b) vs median achieved leverage,
for the fixed-temporal-covariance (open, dashed) and noise-inclusive (filled, solid) field-space covariances."""
import pandas as pd, numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
files = {'fixed': sys.argv[1] if len(sys.argv) > 1 else 'frontier.csv', 'nested': sys.argv[2] if len(sys.argv) > 2 else 'frontier_nested.csv'}
plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
col = {'P': '#1f77b4', 'N': '#ff7f0e'}; mk = {'P': 'o', 'N': 's'}
for cov, fn in files.items():
    d = pd.read_csv(fn); d['absz'] = d.z_disp.abs()
    g = d.groupby(['particle', 'tau'], sort=False).agg(L=('Lambda', 'median'), se=('se_ratio', 'median'), z=('absz', 'median')).reset_index()
    print(cov); print(g.round(4).to_string())
    for p in 'PN':
        s = g[g.particle == p]; lab = f"{'proton' if p=='P' else 'neutron'}, {'$\\Sigma_{\\rm fix}$' if cov=='fixed' else '$\\Sigma_{\\rm nest}$'}"
        kw = dict(color=col[p], marker=mk[p], ms=4, lw=1.1, ls='--' if cov == 'fixed' else '-', mfc='white' if cov == 'fixed' else col[p], label=lab)
        ax[0].plot(s.L, s.se, **kw); ax[1].plot(s.L, s.z, **kw)
for a in ax: a.set_xscale('log'); a.set_xlabel(r'median achieved leverage $\Lambda=\max(|L_4|/4,|L_6|/20)$'); a.grid(alpha=0.3)
ax[0].set_ylabel(r'median $\sigma_\tau/\sigma_{\rm O4}$'); ax[1].set_ylabel(r'median $|\Delta|/\sigma_\Delta$')
ax[0].set_title('(a) precision', fontsize=9); ax[1].set_title('(b) displacement from O4', fontsize=9); ax[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig('fig_tau_frontier.pdf')
