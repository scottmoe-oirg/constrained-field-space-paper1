"""Delta++ (no nucleon partner): shift of O4-constrained BLUE (= correlated linear fit in B^2 to all four fields)
from fixed O6, in paired standard errors, with the Delta++ O4 and O6 null tests; noise-inclusive (nested) replicas."""
import numpy as np, pandas as pd, glob, re, sys
from scipy.stats import chi2
x=np.array([1,4,16,64.]); W4=np.array([4/3,-1/3,0,0]); W6=np.array([64/45,-4/9,1/45,0])
N4=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]]); N6=np.array([-64,84,-21,1.])
MPI=[337.7,361.5,383.8,404.9,444.3,693.0]
def cov(Y): n=len(Y); c=Y-Y.mean(0); return (n-1)/n*(c.T@c)
def jk(v): n=len(v); return np.sqrt((n-1)/n*((v-v.mean())**2).sum())
def blue(S,nc):
    A=np.stack([x**p for p in range(nc)],1); e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
def loo(Z):
    N=len(Z); s1=Z.sum(0); s2=Z.T@Z
    for j in range(N): m=(s1-Z[j])/(N-1); yield (N-1)/N*((s2-np.outer(Z[j],Z[j]))/(N-1)-np.outer(m,m))
rows=[]
for f in sorted(glob.glob(sys.argv[1]+'/arr_k*_10_18.npz')+glob.glob(sys.argv[1]+'/arr_k*_12_20.npz')):
    k,w0,w1=map(int,re.search(r'arr_k(\d)_(\d+)_(\d+)',f).groups()); d=np.load(f); y,Y=d['yB'],d['yBn']; S=cov(Y)
    w=blue(S,2); dlt=w-W6; sh=y@dlt
    re_=np.array([Y[j]@blue(Sj,2) for j,Sj in enumerate(loo(Y))]); se_re=jk(re_-Y@W6)
    T4=float((N4.T@y)@np.linalg.solve(N4.T@S@N4,N4.T@y)); T6=float((N6@y)**2/(N6@S@N6))
    rows.append(dict(window=f'{w0}:{w1}',mpi=MPI[k],O6=y@W6,BLUE_O4=y@w,shift_pct=100*sh/(y@W6),z_fixedw=sh/jk(Y@dlt),z_reopt=sh/se_re,
                     T4=T4,p4=chi2.sf(T4,2),T6=T6,p6=chi2.sf(T6,1),L4=w@x**2,L6=w@x**3))
d=pd.DataFrame(rows).sort_values(['window','mpi']); d.to_csv('dpp_blue_o4.csv',index=False); print(d.round(4).to_string(index=False))
