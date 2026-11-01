"""Two-state temporal fits of the spin ratio, R(t,B)=A e^{-m t}[1+b e^{-D t}], with nested-jackknife field-space covariance.
Variants: gap D with a Gaussian prior (D0, 0.3 D0) refit in every fit ('prior'), or D fixed at D0 ('fixed'); tmin 6 or 8,
tmax 20 (P) / 26 (N). D0 = median over fields of free-gap fits to the central data at tmin=6. Single-state fit in the production
window computed alongside for reference. Usage: ts_nested.py ENS[,ENS..] [nproc]"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import chi2
from multiprocessing import Pool
import pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'lib')); sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from paths import OCTET, WORK; import poly11_small_nullspace_chisq_v2 as v2
from ts_lib import *
CORR=str(OCTET)
W4=np.array([4/3,-1/3,0,0]); W6=np.array([64/45,-4/9,1/45,0])
N4=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]]); N6=np.array([[-64.],[84.],[-21.],[1.]])
PROD={'P':np.arange(12,21),'N':np.arange(17,27)}; TMAX={'P':20,'N':26}
def getX(p,k):
    fn=str(WORK/f'X_{p}{k}.npy')
    if os.path.exists(fn): return np.load(fn)
    X,_,_=v2.load_particle(CORR,p,k); np.save(fn,X); return X
def fitter(kind,D0):
    if kind=='single': return lambda r,W,t,p0: fit1(r,W,t,p0)
    sig=0.3*D0 if kind=='prior' else None
    return lambda r,W,t,p0: fit2(r,W,t,D0,sig,p0)
def run(X,t,kind,D0):
    n=len(X); S=X.sum(0); RF=ratio(S/n)[:,t]; R=ratio((S[None]-X)/(n-1))[:,:,t]; fit=fitter(kind,D0)
    y=np.empty(4); yj=np.empty((n,4)); yn=np.empty((n,4)); c2=np.empty(4); P0=[]
    for f in range(4):
        Z=R[:,f]; D=Z-Z.mean(0); W=whiten((n-1)/n*(D.T@D)); p,c2[f]=fit(RF[f],W,t,None); P0.append(p)
        y[f]=p[1]/(12*ETA0*F[f])
        for i in range(n): yj[i,f]=fit(Z[i],W,t,p)[0][1]/(12*ETA0*F[f])
    for i in range(n):
        Si=S-X[i]; RFi=ratio(Si/(n-1))[:,t]; Ri=ratio((Si[None]-np.delete(X,i,0))/(n-2))[:,:,t]
        for f in range(4):
            Z=Ri[:,f]; D=Z-Z.mean(0); W=whiten((n-2)/(n-1)*(D.T@D)); yn[i,f]=fit(RFi[f],W,t,P0[f])[0][1]/(12*ETA0*F[f])
    J=lambda a:(n-1)/n*((a-a.mean(0)).T@(a-a.mean(0)))
    return y,J(yj),J(yn),c2,P0
def job(a):
    p,k=a; X=getX(p,k); rows=[]
    t6=np.arange(6,TMAX[p]+1); n=len(X); S=X.sum(0); RF=ratio(S/n)[:,t6]; R=ratio((S[None]-X)/(n-1))[:,:,t6]; Ds=[]
    for f in range(4):
        Z=R[:,f]; D=Z-Z.mean(0); Ds.append(fit2(RF[f],whiten((n-1)/n*(D.T@D)),t6,0.3,1.0)[0][3])
    D0=float(np.median(Ds))
    cfg=[('single',PROD[p])]+[(kind,np.arange(tm,TMAX[p]+1)) for tm in (6,8) for kind in ('prior','fixed')]
    for kind,t in cfg:
        y,C,Cn,c2,P=run(X,t,kind,D0)
        T=lambda N,CC: float((N.T@y)@np.linalg.solve(N.T@CC@N,N.T@y))
        r=dict(particle=p,k=k,kind=kind,tmin=int(t[0]),tmax=int(t[-1]),D0=D0,
               D_fit=';'.join(f'{q[3]:.3f}' for q in P) if kind=='prior' else '',
               chi2dof=';'.join(f'{c/(len(t)-(2 if kind=="single" else 3)):.2f}' for c in c2),
               y=';'.join(f'{v:.5f}' for v in y),
               O4=CONV*W4@y,se_O4=CONV*np.sqrt(W4@Cn@W4),O6=CONV*W6@y,se_O6=CONV*np.sqrt(W6@Cn@W6),
               T4_nested=T(N4,Cn),T6_nested=T(N6,Cn),T4_fixed=T(N4,C),T6_fixed=T(N6,C))
        for s in ('nested','fixed'): r[f'p4_{s}']=chi2.sf(r[f'T4_{s}'],2); r[f'p6_{s}']=chi2.sf(r[f'T6_{s}'],1)
        np.savez(f'ts_{p}{k}_{kind}_{t[0]}.npz',y=y,C=C,Cn=Cn); print(r,flush=True); rows.append(r)
    return rows
if __name__=='__main__':
    ens=sys.argv[1].split(','); 
    with Pool(int(sys.argv[2]) if len(sys.argv)>2 else 2) as pool: out=pool.map(job,[(e[0],int(e[1])) for e in ens],chunksize=1)
    df=pd.DataFrame([r for rr in out for r in rr]); fn='ts_nested.csv'
    if os.path.exists(fn): df=pd.concat([pd.read_csv(fn),df]).drop_duplicates(['particle','k','kind','tmin'],keep='last')
    df.to_csv(fn,index=False)
