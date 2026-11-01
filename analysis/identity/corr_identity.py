"""Configuration-level check of the decuplet identity. The code builds Delta++ = F(u,u,u') and
Delta+ = (1/3)[F(u,u,d') + F(d,u,u') + F(u,d,u')], so at first order in the quark charges the B-odd part of every
correlator satisfies  O_{D+}(t) = (2e_u+e_d)/(3e_u) O_{D++}(t) = 1/2 O_{D++}(t), exactly, per configuration.
Higher orders enter at O(B^3)/O(B) ~ B^2. We test r_f = O_{D+}/O_{D++} for f = 1,2,4,8 (O = C(+B)-C(-B), MINUS tag = +B)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np
D=str(DECUPLET)
out=[]
for k in range(6):
    a=np.load(f'{D}/s_DP_k{k}.npz'); b=np.load(f'{D}/s_DPP_k{k}.npz'); com=np.intersect1d(a['cfg'],b['cfg'])
    A=a['C'][np.searchsorted(a['cfg'],com)]; B=b['C'][np.searchsorted(b['cfg'],com)]     # cfg,label8,t,comp
    for fi,f in enumerate([1,2,4,8]):
        OA=A[:,2*fi+1]-A[:,2*fi]; OB=B[:,2*fi+1]-B[:,2*fi]                                   # MINUS - PLUS = C(+B)-C(-B)
        EA=A[:,2*fi+1]+A[:,2*fi]; EB=B[:,2*fi+1]+B[:,2*fi]
        for t in [5,10,14,18]:
            r=OA[:,t,:]/OB[:,t,:]                                                            # per config, per spin comp
            ens=OA[:,t,:].sum(0)/OB[:,t,:].sum(0)
            out.append((k,f,t,np.median(np.abs(r-0.5)),np.abs(r-0.5).max(),ens-0.5, (EA[:,t,:].sum(0)/EB[:,t,:].sum(0))-1))
import pandas as pd
d=pd.DataFrame(out,columns=['k','f','t','median_abs_cfg_dev','max_abs_cfg_dev','ens_dev_comp','even_ratio_minus1_comp'])
pd.set_option('display.width',220); print(d[d.k.isin([0,5])].to_string(index=False))
# scaling with f: median per-config deviation vs f^2
g=d.groupby('f').median_abs_cfg_dev.median(); print(g, '\nratio successive (expect ~4 for B^2 scaling):', (g.values[1:]/g.values[:-1]).round(2))
d.to_csv('corr_identity.csv',index=False)
