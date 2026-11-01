"""Temporal-covariance regularization: repeat the production delete-one jackknife with the temporal covariance of the
double ratio (i) diagonal (uncorrelated fits) and (ii) shrunk halfway toward its diagonal, and recompute the moments,
the O4/O6 null tests and the O4-constrained BLUE (gain and leverage). Nested-jackknife versions are not run."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET
import numpy as np, pandas as pd, poly11_small_nullspace_chisq_v2 as v2
from scipy.stats import chi2
from multiprocessing import Pool
x=np.array([1,4,16,64.]); W4=np.array([4/3,-1/3,0,0]); W6=np.array([64/45,-4/9,1/45,0])
N4=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]]); N6=np.array([[-64.],[84.],[-21.],[1.]])
def blue(S): A=np.stack([x**0,x],1); Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,[1,0])
def analysis(X,times,mode):
    n=len(X); S=X.sum(0); RF=v2.ratio(S/n); loo=(S[None]-X)/(n-1)
    R=(loo[:,:,1,0,:]*loo[:,:,0,1,:])/(loo[:,:,0,0,:]*loo[:,:,1,1,:]); y=np.empty(4); yj=np.empty((n,4))
    for f in range(4):
        Z=R[:,f][:,times]; DD=Z-Z.mean(0); Ct=(n-1)/n*(DD.T@DD); Dg=np.diag(np.diag(Ct))
        Ct=Dg if mode=='diag' else 0.5*Ct+0.5*Dg
        y[f]=v2.fit_ls(RF[f,times],Ct,times)/(12*v2.ETA0*v2.FIELDS[f])
        for i in range(n): yj[i,f]=v2.fit_ls(Z[i],Ct,times)/(12*v2.ETA0*v2.FIELDS[f])
    D=yj-yj.mean(0); return y,(n-1)/n*(D.T@D)
def job(a):
    p,k,mode=a; X,_,_=v2.load_particle(str(OCTET),p,k); y,C=analysis(X,v2.WINDOWS[p],mode)
    T=lambda N: float((N.T@y)@np.linalg.solve(N.T@C@N,N.T@y)); wb=blue(C)
    r=dict(particle=p,k=k,mode=mode,O6=0.88658*W6@y,se_O6=0.88658*np.sqrt(W6@C@W6),T4=T(N4),p4=chi2.sf(T(N4),2),T6=T(N6),p6=chi2.sf(T(N6),1),
           gain=np.sqrt(wb@C@wb/(W4@C@W4)),L6=wb@x**3,z=(W4-wb)@y/np.sqrt((W4-wb)@C@(W4-wb)))
    print(r,flush=True); return r
if __name__=='__main__':
    jobs=[(p,k,m) for m in ['diag','half'] for p in 'PN' for k in range(6)]
    with Pool(2) as pool: rows=pool.map(job,jobs)
    pd.DataFrame(rows).to_csv('tempreg.csv',index=False)
