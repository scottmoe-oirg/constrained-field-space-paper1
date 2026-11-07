"""Fig. 3 (v5): (a) field response of Delta+ and Delta++/2 at the lightest mass; (b) deviation of mu(Delta+)/mu(Delta++) from 1/2
in paired standard errors for the fixed stencils; (c) size of the miss in percent for fixed stencils and per-state BLUE estimators.
Run in a directory containing arr_k0_10_18.npz, dec_scan_nested_summary.csv and dec_blue_nested.csv."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e3e2dc', 'grid.linewidth': 0.5, 'axes.edgecolor': '#8a897f', 'lines.linewidth': 1.0})
B1, B2, B3, B4 = '#2a78d6', '#eb6834', '#1baf7a', '#8a5cd6'; G = '#5f5e57'
a0 = np.load('arr_k0_10_18.npz'); x = np.array([1, 4, 16, 64.])
r = pd.read_csv('dec_scan_nested_summary.csv'); r = r[r.window.isin(['10:18', '12:20'])].sort_values(['window', 'k'])
bl = pd.read_csv('dec_blue_nested.csv').sort_values(['window', 'mpi'])
WIN = [('10:18', '-', True), ('12:20', '--', False)]
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw=dict(wspace=0.42))

a = ax[0]
for lab, v, c in [(r'$\Delta^+$', a0['yA'], B1), (r'$\frac{1}{2}\Delta^{++}$', 0.5 * a0['yB'], B2)]:
    v = 3 * 0.88658 * v; a.plot(x, v, 'o', color=c, ms=3.5, label=lab)
    cf = np.polyfit(x[:3], v[:3], 2); xx = np.linspace(0, 64, 200); a.plot(xx, np.polyval(cf, xx), '-', color=c, alpha=0.8)
    a.plot([0], [np.polyval(cf, 0)], 's', mfc='white', color=c, ms=4.5)
a.set_xlabel(r'$x=f^2$  (field $=fB$)'); a.set_ylabel(r'$3\times0.88658\,y(fB)$  [$\mu_N$]')
a.set_title(r'(a) $m_\pi=338$ MeV, window 10–18', fontsize=8); a.legend(frameon=False, fontsize=7, loc='upper left')
a.text(18, 3 * 0.88658 * 0.775, 'curves: O6 quadratic\nthrough $f=1,2,4$;\nsquares: $B\\to0$', fontsize=6, color=G)

a = ax[1]
for nm, c, mk in [('O2', B2, 'o'), ('O4', B3, '^'), ('O6', B1, 's'), ('O8', B4, 'D')]:
    for w, ls, fill in WIN:
        d = r[r.window == w]; a.plot(d.mpi, d[f'{nm}_sig'], ls=ls, marker=mk, color=c, ms=3, mfc=c if fill else 'white', label=nm if fill else None)
a.axhspan(-2, 2, color='#f1f0ea', zorder=0); a.axhline(0, color='#8a897f', lw=0.6)
a.set_yscale('symlog', linthresh=3); a.set_ylim(-5, 150); a.set_yticks([-3, -2, -1, 0, 1, 2, 3, 10, 30, 100])
a.set_yticklabels(['$-3$', '$-2$', '$-1$', '0', '1', '2', '3', '10', '30', '100'])
a.set_xlabel(r'$m_\pi$ (MeV)'); a.set_ylabel(r'$(\mu_{\Delta^+}/\mu_{\Delta^{++}}-\frac{1}{2})/\sigma$')
a.set_title('(b) deviation in standard errors', fontsize=8)
a.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=4, columnspacing=0.7, handlelength=1.4, handletextpad=0.3)
a.text(0.98, 0.03, 'solid 10–18, open 12–20\nband: $|\\cdot|<2\\sigma$', transform=a.transAxes, ha='right', va='bottom', fontsize=6, color=G)

a = ax[2]
def blue(est, w): d = bl[(bl.window == w) & (bl.est == est)]; return d.mpi, d.dev * 1e-4
series = [('O2 (fixed)', lambda w: (r[r.window == w].mpi, r[r.window == w].O2_dev), B2, 'o'),
          ('O6 (fixed)', lambda w: (r[r.window == w].mpi, r[r.window == w].O6_dev), B1, 's'),
          ('BLUE-O6', lambda w: blue('BLUE-O6', w), '#7a7a72', 'v'),
          ('BLUE-O4', lambda w: blue('BLUE-O4', w), '#c0392b', '^'),
          ('BLUE-1', lambda w: blue('BLUE-1', w), '#7b241c', 'D')]
for nm, f, c, mk in series:
    for w, ls, fill in WIN:
        m, dv = f(w); a.plot(m, 100 * np.abs(np.asarray(dv)) / 0.5, ls=ls, marker=mk, color=c, ms=3, mfc=c if fill else 'white', label=nm if fill else None)
a.set_yscale('log'); a.set_ylim(1e-4, 300)
a.set_xlabel(r'$m_\pi$ (MeV)'); a.set_ylabel(r'$|\mu_{\Delta^+}/\mu_{\Delta^{++}}-\frac{1}{2}|\,/\,\frac{1}{2}$  [%]')
a.set_title('(c) size of the miss', fontsize=8)
a.legend(frameon=False, fontsize=6, loc='lower right', ncol=2, columnspacing=0.7, handlelength=1.6, handletextpad=0.3)

fig.savefig('fig_decuplet_ratio_test_v5.pdf', bbox_inches='tight'); fig.savefig('fig_decuplet_ratio_test_v5.png', dpi=250, bbox_inches='tight')
