"""Full-pipeline bootstrap calibration of the O6 null statistic (matches the production statistic exactly):
each replicate resamples configurations with replacement and runs the unchanged production analysis
(ordinary_analysis: temporal covariance estimated from that replicate and held fixed in its delete-one fits).
T_b = (u_b - u_real)^2 / se_b^2 with u = n6.y, n6 = (-64,84,-21,1), se_b from the replicate's own jackknife."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, numpy as np, time, json
from multiprocessing import Pool
pass; import poly11_small_nullspace_chisq_v2 as v2
ROOT=str(OCTET)
n6=np.array([-64,84,-21,1.])
_X=None
def init(p,k):
    global _X,_T; _X,_,_=v2.load_particle(ROOT,p,k); _T=v2.WINDOWS[p]
def one(seed):
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(_X),len(_X))
    y,C,_=v2.ordinary_analysis(_X[idx],_T); return float(n6@y), float(np.sqrt(n6@C@n6)), y.tolist(), np.sqrt(np.diag(C)).tolist(), C.tolist()
if __name__=='__main__':
    p,k,nb=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
    init(p,k); y,C,_=v2.ordinary_analysis(_X,_T); u=float(n6@y); se=float(np.sqrt(n6@C@n6)); T=(u/se)**2
    off=int(sys.argv[4]) if len(sys.argv)>4 else 0
    seeds=[20261019000+1000*(p=='N')+100*k+b for b in range(nb)] if off==0 else [20261020000+100000*(p=='N')+10000*k+off+b for b in range(nb)]
    t0=time.time()
    with Pool(2,initializer=init,initargs=(p,k)) as pool: res=pool.map(one,seeds,chunksize=4)
    ub=np.array([r[0] for r in res]); seb=np.array([r[1] for r in res]); Tb=((ub-u)/seb)**2
    out=dict(particle=p,k=k,nboot=nb,u=u,se=se,T_real=T,q95=float(np.quantile(Tb,.95)),frac_gt_3p84=float((Tb>3.841458820694124).mean()),
             n_ge_Treal=int((Tb>=T).sum()),p_boot=float((Tb>=T).mean()),sd_ub_over_se=float(ub.std(ddof=1)/se),median_seb_over_se=float(np.median(seb)/se),seconds=time.time()-t0)
    tag=f'_{off}' if off else ''
    np.savez(f'fullcal_{p}{k}{tag}.npz',ub=ub,seb=seb,Tb=Tb,yb=np.array([r[2] for r in res]),Cb=np.array([r[4] for r in res]),y=y,C=C); json.dump(out,open(f'fullcal_{p}{k}{tag}.json','w'),indent=1); print(json.dumps(out))
