"""Regularized covariance optimization on the known-answer controls: BLUE with the field-space covariance shrunk toward
its diagonal, S_lam = (1-lam) S + lam diag(S), and with the Ledoit-Wolf-type data-driven intensity; noise-inclusive replicas.
Targets: Delta+/Delta++ = 1/2 and Delta0/Delta++ = 0. Deviations in units of 1e-4 and in standard errors (fixed weights)."""
import numpy as np, pandas as pd, glob, re, sys
x=np.array([1,4,16,64.]); MPI=[337.7,361.5,383.8,404.9,444.3,693.0]
def cov(Y): n=len(Y); c=Y-Y.mean(0); return (n-1)/n*(c.T@c)
def jk(v): n=len(v); return np.sqrt((n-1)/n*((v-v.mean())**2).sum())
def blue(S,nc):
    A=np.stack([x**p for p in range(nc)],1); e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
def lw_lambda(Y):
    # Ledoit-Wolf-type optimal shrinkage toward the diagonal for the covariance of the mean (jackknife replicas)
    n=len(Y); Z=(Y-Y.mean(0))*np.sqrt(n-1); S=Z.T@Z/n           # per-configuration-scale covariance from jackknife replicas
    D=np.diag(np.diag(S)); off=S-D; num=sum(((np.outer(z,z)-S)**2)[~np.eye(4,dtype=bool)].sum() for z in Z)/n**2
    return float(np.clip(num/((off**2).sum()),0,1))
W6=np.array([64/45,-4/9,1/45,0]); rows=[]
for tag,pat,T in [('Delta+','%s/arr_k%d_%s.npz',0.5),('Delta0','%s/arr_D0_k%d_%s.npz',0.0)]:
    base=sys.argv[1] if tag=='Delta+' else sys.argv[2]
    for k in range(6):
        for w in ['10_18','12_20']:
            d=np.load(pat%(base,k,w)); yA,yB,An,Bn=d['yA'],d['yB'],d['yAn'],d['yBn']; SA,SB=cov(An),cov(Bn)
            lamA,lamB=lw_lambda(An),lw_lambda(Bn)
            for nc,lab in [(2,'BLUE-O4'),(3,'BLUE-O6')]:
                for lam in [0,0.1,0.5,0.9,'LW']:
                    la,lb=(lamA,lamB) if lam=='LW' else (lam,lam)
                    sa=(1-la)*SA+la*np.diag(np.diag(SA)); sb=(1-lb)*SB+lb*np.diag(np.diag(SB))
                    wA,wB=blue(sa,nc),blue(sb,nc); rc=(yA@wA)/(yB@wB); se=jk((An@wA)/(Bn@wB))
                    rows.append(dict(control=tag,mpi=MPI[k],window=w.replace('_',':'),est=lab,lam=lam if lam!='LW' else f'LW({la:.2f})',
                        dev=(rc-T)*1e4,se=se*1e4,sig=(rc-T)/se,eA=jk(An@wA)/jk(An@W6),L6=wA@x**3))
d=pd.DataFrame(rows); d.to_csv('shrinkage.csv',index=False); pd.set_option('display.width',200)
q=d[d.mpi.isin([337.7,693.0])&(d.window=='10:18')]; print(q.round(3).to_string(index=False))
