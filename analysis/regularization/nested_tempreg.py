"""Nested delete-one jackknife for the regularized temporal fits (diagonal = uncorrelated, half = shrunk halfway to the
diagonal). As in analysis/nested/nested_all.py, for each deleted configuration the regularized temporal covariance is
re-estimated from the remaining n-1 configurations and the central fit is redone; compared with the fixed-covariance
jackknife for the moments and the O4/O6 null contrasts. Usage: nested_tempreg.py [ensembles, e.g. P0,P5,N0,N5]"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET
import numpy as np, pandas as pd, poly11_small_nullspace_chisq_v2 as v2
from scipy.stats import chi2
from multiprocessing import Pool
F=np.array([1,2,4,8.]); ETA0=0.001364; CONV=0.88658
W4=np.array([4/3,-1/3,0,0]); W6=np.array([64/45,-4/9,1/45,0])
N4=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]]); N6=np.array([[-64.],[84.],[-21.],[1.]])
def ratio(M): return (M[...,1,0,:]*M[...,0,1,:])/(M[...,0,0,:]*M[...,1,1,:])
def reg(Ct,mode): Dg=np.diag(np.diag(Ct)); return Dg if mode=='diag' else 0.5*Ct+0.5*Dg
def job(a):
    p,k,mode=a; X,_,_=v2.load_particle(str(OCTET),p,k); t=v2.WINDOWS[p]; n=len(X); S=X.sum(0)
    RF=ratio(S/n)[:,t]; loo=(S[None]-X)/(n-1); R=ratio(loo)[:,:,t]
    y=np.empty(4); yj=np.empty((n,4))
    for f in range(4):
        Z=R[:,f]; D=Z-Z.mean(0); Ct=reg((n-1)/n*(D.T@D),mode)
        y[f]=v2.fit_ls(RF[f],Ct,t)/(12*ETA0*F[f])
        for i in range(n): yj[i,f]=v2.fit_ls(Z[i],Ct,t)/(12*ETA0*F[f])
    C=(n-1)/n*((yj-yj.mean(0)).T@(yj-yj.mean(0)))
    yn=np.empty((n,4))
    for i in range(n):
        Si=S-X[i]; RFi=ratio(Si/(n-1))[:,t]
        Ri=ratio((Si[None]-np.delete(X,i,0))/(n-2))[:,:,t]
        for f in range(4):
            Z=Ri[:,f]; D=Z-Z.mean(0); Ct=reg((n-2)/(n-1)*(D.T@D),mode)
            yn[i,f]=v2.fit_ls(RFi[f],Ct,t)/(12*ETA0*F[f])
    Cn=(n-1)/n*((yn-yn.mean(0)).T@(yn-yn.mean(0)))
    T=lambda N,CC: float((N.T@y)@np.linalg.solve(N.T@CC@N,N.T@y))
    r=dict(particle=p,k=k,mode=mode,O6=CONV*W6@y,se_O6_fixed=CONV*np.sqrt(W6@C@W6),se_O6_nested=CONV*np.sqrt(W6@Cn@W6),
           T4_fixed=T(N4,C),T4_nested=T(N4,Cn),T6_fixed=T(N6,C),T6_nested=T(N6,Cn),
           null6_se_ratio=float(np.sqrt((N6.T@Cn@N6)[0,0]/(N6.T@C@N6)[0,0])))
    for s in ['fixed','nested']: r[f'p4_{s}']=chi2.sf(r[f'T4_{s}'],2); r[f'p6_{s}']=chi2.sf(r[f'T6_{s}'],1)
    np.savez(f'nested_tempreg_{mode}_{p}{k}.npz',y=y,C=C,Cn=Cn,yn=yn); print(r,flush=True); return r
if __name__=='__main__':
    ens=(sys.argv[1] if len(sys.argv)>1 else 'P0,P1,P2,P3,P4,P5,N0,N1,N2,N3,N4,N5').split(',')
    jobs=[(e[0],int(e[1]),m) for m in ['diag','half'] for e in ens]
    with Pool(int(sys.argv[2]) if len(sys.argv)>2 else 2) as pool: rows=pool.map(job,jobs,chunksize=1)
    pd.DataFrame(rows).to_csv('nested_tempreg.csv',index=False)
