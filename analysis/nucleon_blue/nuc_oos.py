"""In-sample vs out-of-sample (weights re-derived per replica) errors for covariance-optimised nucleon estimators."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
pass
import poly11_small_nullspace_chisq_v2 as v2
ROOT=str(OCTET)
x=np.array([1,4,16,64.]); CONV=0.88658
def jk(v): N=len(v); return np.sqrt((N-1)/N*((v-v.mean(0))**2).sum(0))
def blue(S,nc):
    A=np.stack([x**p for p in range(nc)],1); e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
def job(a):
    p,k=a; X,_,_=v2.load_particle(ROOT,p,k); y,C,yj=v2.ordinary_analysis(X,v2.WINDOWS[p])
    np.savez(f'nuc_{p}{k}.npz',y=y,C=C,yj=yj); return a
def loo(Z):
    N=len(Z); s1=Z.sum(0); s2=Z.T@Z
    for j in range(N): m=(s1-Z[j])/(N-1); yield (s2-np.outer(Z[j],Z[j]))/(N-1)-np.outer(m,m)
if __name__=='__main__':
    todo=[(p,k) for p in 'PN' for k in range(6)]
    with Pool(3) as pool: pool.map(job,todo)
    w4=np.array([4/3,-1/3,0,0.]); w6=np.array([64/45,-4/9,1/45,0])
    rows=[]
    for p,k in todo:
        d=np.load(f'nuc_{p}{k}.npz'); y,C,yj=d['y'],d['C'],d['yj']; covs=list(loo(yj))
        r=dict(particle=p,k=k,O4=CONV*w4@y,O4_se=CONV*jk(yj@w4),O6_se=CONV*jk(yj@w6))
        for nc,lab in [(1,'BLUE1'),(2,'BLUE4'),(3,'BLUE6')]:
            w=blue(C,nc); fixed=yj@w; re=np.array([yj[j]@blue(S,nc) for j,S in enumerate(covs)])
            r[lab]=CONV*w@y; r[lab+'_se_in']=CONV*jk(fixed); r[lab+'_se_out']=CONV*jk(re)
            ref=w4 if nc<3 else w6
            r[lab+'_z_in']=(ref@y-w@y)/jk(yj@ref-fixed); r[lab+'_z_out']=(ref@y-w@y)/jk(yj@ref-re)
        rows.append(r)
    o=pd.DataFrame(rows); o.to_csv('nuc_oos.csv',index=False)
    o['r1_in']=o.BLUE1_se_in/o.O4_se; o['r1_out']=o.BLUE1_se_out/o.O4_se; o['r4_in']=o.BLUE4_se_in/o.O4_se; o['r4_out']=o.BLUE4_se_out/o.O4_se
    o['r6_in']=o.BLUE6_se_in/o.O6_se; o['r6_out']=o.BLUE6_se_out/o.O6_se
    pd.set_option('display.width',250)
    print(o[['particle','k','r1_in','r1_out','r4_in','r4_out','r6_in','r6_out','BLUE4_z_in','BLUE4_z_out','BLUE6_z_in','BLUE6_z_out']].round(3).to_string(index=False))
    for p in 'PN':
        q=o[o.particle==p]; print(p,'median r1 in/out',q.r1_in.median().round(4),q.r1_out.median().round(4),'r4',q.r4_in.median().round(4),q.r4_out.median().round(4))
