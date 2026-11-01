"""Learning curve for the unrestricted O4-constrained BLUE (regenerated 2026-10-19; the original summary CSV is lost).
For N in (100,200,400,600,800): 30 random configuration subsets (numpy default_rng, seed 20261008), each passed through
the production analysis (ordinary_analysis: delete-one replicas, temporal covariance fixed within the subset), then the
O4-constrained BLUE weights from the subset's field covariance. Records weights, L4 = w.x^2, L6 = w.x^3, the BLUE/O4
error ratio, and the displacement from O4. Lightest proton and neutron (kappa index 0)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
pass; import poly11_small_nullspace_chisq_v2 as v2
ROOT=str(OCTET)
x=np.array([1,4,16,64.]); A=np.c_[np.ones(4),x]; e=np.array([1,0.]); W4=np.array([4/3,-1/3,0,0]); CONV=0.88658
NS=[100,200,400,600,800]; NSUB=30
_X={}
def init():
    for p in 'PN': _X[p]=v2.load_particle(ROOT,p,0)[0]
def blue(C): Ci=np.linalg.solve(C,A); return Ci@np.linalg.solve(A.T@Ci,e)
def job(a):
    p,N,s,idx=a; y,C,yj=v2.ordinary_analysis(_X[p][idx],v2.WINDOWS[p]); w=blue(C)
    jk=lambda v: np.sqrt((len(v)-1)/len(v)*((v-v.mean())**2).sum())
    return dict(particle=p,N=N,subset=s,w1=w[0],w2=w[1],w3=w[2],w4=w[3],L4=w@x**2,L6=w@x**3,
                O4=CONV*W4@y,BLUE=CONV*w@y,se_O4=CONV*jk(yj@W4),se_BLUE=CONV*jk(yj@w),disp_z=(w-W4)@y/jk(yj@(w-W4)))
if __name__=='__main__':
    rng=np.random.default_rng(20261008); init(); tasks=[]
    for p in 'PN':
        n=len(_X[p])
        for N in NS:
            for s in range(NSUB): tasks.append((p,N,s,np.sort(rng.choice(n,N,replace=False))))
    tasks+= [(p,len(_X[p]),-1,np.arange(len(_X[p]))) for p in 'PN']   # full-sample reference
    with Pool(2,initializer=init) as pool: rows=pool.map(job,tasks,chunksize=2)
    d=pd.DataFrame(rows); d['se_ratio']=d.se_BLUE/d.se_O4; d.to_csv('poly11_small_learning_curve_exact_subsets.csv',index=False)
    g=d[d.subset>=0].groupby(['particle','N'])
    out=[]
    for (p,N),q in g:
        r=dict(particle=p,N=N)
        for c in ['w1','w2','L4','L6','se_ratio','disp_z']:
            r[f'{c}_med']=q[c].median(); r[f'{c}_q10']=q[c].quantile(.1); r[f'{c}_q90']=q[c].quantile(.9)
        out.append(r)
    S=pd.DataFrame(out); S.to_csv('poly11_small_learning_curve_exact_summary.csv',index=False)
    pd.set_option('display.width',250); print(S.round(3).to_string(index=False)); print(d[d.subset<0].round(4).to_string(index=False))
