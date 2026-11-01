"""O6 null statistic with nested (covariance-noise-inclusive) jackknife for the Table II windows."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
pass; import poly11_small_nullspace_chisq_v2 as v2
ROOT=str(OCTET)
F=np.array([1,2,4,8.]); ETA0=0.001364; N6=np.array([-64,84,-21,1.])
def ratio(M): return (M[...,1,0,:]*M[...,0,1,:])/(M[...,0,0,:]*M[...,1,1,:])
_X={}
def job(a):
    p,k,t0,t1=a
    if (p,k) not in _X: _X[(p,k)]=v2.load_particle(ROOT,p,k)[0]
    X=_X[(p,k)]; t=np.arange(t0,t1+1); n=len(X)
    y,C,yj=v2.ordinary_analysis(X,t); S=X.sum(0); yn=np.empty((n,4))
    for i in range(n):
        Si=S-X[i]; RF=ratio(Si/(n-1))[:,t]; R=ratio((Si[None]-np.delete(X,i,0))/(n-2))[:,:,t]
        for f in range(4):
            Z=R[:,f]; D=Z-Z.mean(0); yn[i,f]=v2.fit_ls(RF[f],(n-2)/(n-1)*(D.T@D),t)/(12*ETA0*F[f])
    jk=lambda v: np.sqrt((n-1)/n*((v-v.mean())**2).sum())
    return dict(particle=p,k=k,window=f'{t0}-{t1}',T6_fixed=float((N6@y/jk(yj@N6))**2),T6_nested=float((N6@y/jk(yn@N6))**2))
if __name__=='__main__':
    tasks=[(p,k,t0,20) for p in 'P' for k in (0,5) for t0 in range(10,17)]+[(p,k,t0,26) for p in 'N' for k in (0,5) for t0 in range(14,21)]
    with Pool(2) as pool: rows=pool.map(job,tasks,chunksize=7)
    d=pd.DataFrame(rows); d.to_csv('nested_windows.csv',index=False); print(d.round(2).to_string(index=False))
