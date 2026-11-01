"""Nested delete-one jackknife: for each deleted configuration i, the temporal covariance of R is re-estimated from the
delete-one replicas of the remaining n-1 configurations, and the central fit is redone. This includes the noise of the
estimated temporal covariance, which the production jackknife (covariance fixed) omits. Compared with the production
(fixed) covariance for the moments and for the O4/O6 null contrasts, for all 12 ensembles."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import chi2
pass; import poly11_small_nullspace_chisq_v2 as v2
ROOT=str(OCTET)
F=np.array([1,2,4,8.]); ETA0=0.001364; CONV=0.88658
W4=np.array([4/3,-1/3,0,0]); W6=np.array([64/45,-4/9,1/45,0])
N4=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]]); N6=np.array([[-64.],[84.],[-21.],[1.]])
def ratio(M): return (M[...,1,0,:]*M[...,0,1,:])/(M[...,0,0,:]*M[...,1,1,:])
def job(a):
    p,k=a; X,_,_=v2.load_particle(ROOT,p,k); t=v2.WINDOWS[p]; n=len(X)
    y,C,yj=v2.ordinary_analysis(X,t)
    S=X.sum(0); yn=np.empty((n,4))
    for i in range(n):
        Si=S-X[i]; RF=ratio(Si/(n-1))[:,t]
        loo=(Si[None]-np.delete(X,i,0))/(n-2); R=ratio(loo)[:,:,t]     # (n-1, field, t)
        for f in range(4):
            Z=R[:,f]; D=Z-Z.mean(0); Ct=(n-2)/(n-1)*(D.T@D)
            yn[i,f]=v2.fit_ls(RF[f],Ct,t)/(12*ETA0*F[f])
    Cn=(n-1)/n*((yn-yn.mean(0)).T@(yn-yn.mean(0)))
    T=lambda N,CC: float((N.T@y)@np.linalg.solve(N.T@CC@N,N.T@y))
    r=dict(particle=p,k=k,O4=CONV*W4@y,se_O4_fixed=CONV*np.sqrt(W4@C@W4),se_O4_nested=CONV*np.sqrt(W4@Cn@W4),
           se_O6_fixed=CONV*np.sqrt(W6@C@W6),se_O6_nested=CONV*np.sqrt(W6@Cn@W6),
           T4_fixed=T(N4,C),T4_nested=T(N4,Cn),T6_fixed=T(N6,C),T6_nested=T(N6,Cn),
           null6_se_ratio=float(np.sqrt((N6.T@Cn@N6)[0,0]/(N6.T@C@N6)[0,0])))
    r['p4_fixed']=chi2.sf(r['T4_fixed'],2); r['p4_nested']=chi2.sf(r['T4_nested'],2); r['p6_fixed']=chi2.sf(r['T6_fixed'],1); r['p6_nested']=chi2.sf(r['T6_nested'],1)
    np.savez(f'nested_{p}{k}.npz',y=y,C=C,Cn=Cn,yn=yn); print(r,flush=True); return r
if __name__=='__main__':
    todo=[(p,k) for p in 'PN' for k in range(6)]
    with Pool(2) as pool: rows=pool.map(job,todo,chunksize=1)
    d=pd.DataFrame(rows); d.to_csv('nested_vs_fixed.csv',index=False); pd.set_option('display.width',250)
    print(d.round(4).to_string(index=False))
