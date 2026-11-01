import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e3e2dc','grid.linewidth':0.6,'axes.edgecolor':'#8a897f'})
B1,B2,B3,B4='#2a78d6','#eb6834','#1baf7a','#8a5cd6'
a0=np.load('arr_k0_10_18.npz'); x=np.array([1,4,16,64.]); r=pd.read_csv('dec_scan_nested_summary.csv'); r=r[r.window.isin(['10:18','12:20'])].sort_values(['window','k'])
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9))
a=ax[0]
for lab,v,c in [(r'$\Delta^+$',a0['yA'],B1),(r'$\frac{1}{2}\Delta^{++}$',0.5*a0['yB'],B2)]:
    v=3*0.88658*v; a.plot(x,v,'o',color=c,ms=5,label=lab)
    cf=np.polyfit(x[:3],v[:3],2); xx=np.linspace(0,64,200); a.plot(xx,np.polyval(cf,xx),'-',color=c,lw=1.2,alpha=0.8)
    a.plot([0],[np.polyval(cf,0)],'s',mfc='white',color=c,ms=6)
a.set_xlabel(r'$x=f^2$  (field $=fB$)'); a.set_ylabel(r'$3\times0.88658\,y(fB)$  [$\mu_N$]')
a.set_title(r'(a) $m_\pi=338$ MeV, window 10–18',fontsize=9); a.legend(frameon=False,fontsize=8,loc='upper left')
a.text(30,3*0.88658*0.76,'curve: O6 quadratic through\n$f=1,2,4$; squares: $B\\to0$',fontsize=7,color='#5f5e57')
a=ax[1]
for nm,c,mk in [('O2',B2,'o'),('O4',B3,'^'),('O6',B1,'s'),('O8',B4,'D')]:
    for w,ls,fill in [('10:18','-',True),('12:20','--',False)]:
        x_=r[r.window==w]; a.plot(x_.mpi,x_[f'{nm}_sig'],ls=ls,marker=mk,color=c,ms=4,mfc=c if fill else 'white',label=nm if w=='10:18' else None,lw=1)
a.axhspan(-2,2,color='#f1f0ea',zorder=0); a.axhline(0,color='#8a897f',lw=0.6)
a.set_yscale('symlog',linthresh=3); a.set_ylim(-4,100); a.set_yticks([-5,-3,-2,-1,0,1,2,3,10,30,100])
a.set_yticklabels(['$-5$','$-3$','$-2$','$-1$','0','1','2','3','10','30','100'])
a.set_xlabel(r'$m_\pi$ (MeV)'); a.set_ylabel(r'$(\mu_{\Delta^+}/\mu_{\Delta^{++}}-\frac{1}{2})/\sigma$')
a.set_title('(b) deviation from exact symmetry',fontsize=9); a.legend(frameon=False,fontsize=8,loc='upper left',ncol=4)
a.text(520,-3.6,'solid 10–18, open 12–20; band: $|\\cdot|<2\\sigma$',fontsize=7,color='#5f5e57')
fig.tight_layout(); fig.savefig('fig_ratio_test_nested.pdf'); fig.savefig('fig_ratio_test_nested.png',dpi=160); fig.savefig('fig_decuplet_ratio_test_v3.pdf')
