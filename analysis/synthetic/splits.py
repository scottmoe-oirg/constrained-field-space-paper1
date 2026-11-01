"""Independent train/evaluate splits on the real data (referee point 8). For each of 20 random half/half splits of the
configurations: weights (O4-constrained BLUE, unrestricted) are fitted on the training half from its field-space covariance
(production fixed-temporal-covariance jackknife, and nested), then applied UNCHANGED to the evaluation half, whose error is
computed with the nested jackknife (temporal-covariance noise included). Compared with fixed O4 on the same evaluation half."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
pass; import poly11_small_nullspace_chisq_v2 as v2
ROOT=str(OCTET)
x=np.array([1,4,16,64.]); A=np.c_[np.ones(4),x]; e=np.array([1,0.]); W4=np.array([4/3,-1/3,0,0]); F=np.array([1,2,4,8.]); ETA0=0.001364; CONV=0.88658
def blue(C): Ci=np.linalg.solve(C,A); return Ci@np.linalg.solve(A.T@Ci,e)
def ratio(M): return (M[...,1,0,:]*M[...,0,1,:])/(M[...,0,0,:]*M[...,1,1,:])
def nested(X,t):
    n=len(X); S=X.sum(0); yn=np.empty((n,4))
    for i in range(n):
        Si=S-X[i]; RF=ratio(Si/(n-1))[:,t]; R=ratio((Si[None]-np.delete(X,i,0))/(n-2))[:,:,t]
        for f in range(4):
            Z=R[:,f]; D=Z-Z.mean(0); yn[i,f]=v2.fit_ls(RF[f],(n-2)/(n-1)*(D.T@D),t)/(12*ETA0*F[f])
    return yn
_X={}
def job(a):
    p,s=a
    if p not in _X: _X[p]=v2.load_particle(ROOT,p,0)[0]
    X=_X[p]; t=v2.WINDOWS[p]; n=len(X); rng=np.random.default_rng(1000+s+(500 if p=='N' else 0)); perm=rng.permutation(n); tr,ev=np.sort(perm[:n//2]),np.sort(perm[n//2:])
    ytr,Ctr,_=v2.ordinary_analysis(X[tr],t); yntr=nested(X[tr],t); m=len(yntr); Cntr=(m-1)/m*((yntr-yntr.mean(0)).T@(yntr-yntr.mean(0)))
    yev,Cev,_=v2.ordinary_analysis(X[ev],t); ynev=nested(X[ev],t); k=len(ynev)
    jk=lambda v: np.sqrt((k-1)/k*((v-v.mean())**2).sum())
    out=dict(particle=p,split=s,se_O4_eval=CONV*jk(ynev@W4))
    for lab,C in [('fixedC',Ctr),('nestedC',Cntr)]:
        w=blue(C); out[f'se_BLUE_{lab}_eval']=CONV*jk(ynev@w); out[f'ratio_{lab}']=out[f'se_BLUE_{lab}_eval']/out['se_O4_eval']
        out[f'disp_{lab}_eval_sigma']=((w-W4)@yev)/jk(ynev@(w-W4)); out[f'L6_{lab}']=w@x**3
        wi=blue(Cev); out[f'ratio_{lab}_insample_eval']=CONV*np.sqrt(wi@Cev@wi)/(CONV*np.sqrt(W4@Cev@W4))
    return out
if __name__=='__main__':
    tasks=[(p,s) for p in 'PN' for s in range(20)]
    with Pool(2) as pool: rows=pool.map(job,tasks,chunksize=5)
    d=pd.DataFrame(rows); d.to_csv('splits.csv',index=False)
    print(d.groupby('particle')[[c for c in d.columns if c.startswith('ratio') or c.startswith('disp')]].describe().T.round(3).to_string())
