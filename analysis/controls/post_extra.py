"""Absolute controls: Delta0 (target 0) and Delta- (target -1/2) relative to Delta++, nested replicas.
Fixed stencils O2-O8 and per-state BLUE-1/O4/O6 (noise-inclusive covariance), deviations in 1e-4 and in SE."""
import numpy as np, pandas as pd, glob, re, sys
x=np.array([1,4,16,64.]); W={'O2':np.array([1,0,0,0.]),'O4':np.array([4/3,-1/3,0,0]),'O6':np.array([64/45,-4/9,1/45,0]),'O8':np.linalg.solve(np.vander(x,4,increasing=True).T,[1,0,0,0.])}
MPI=[337.7,361.5,383.8,404.9,444.3,693.0]; T={'D0':0.0,'DM':-0.5}
def jk(v): n=len(v); return np.sqrt((n-1)/n*((v-v.mean())**2).sum())
def cov(Y): n=len(Y); c=Y-Y.mean(0); return (n-1)/n*(c.T@c)
def blue(S,nc):
    A=np.stack([x**p for p in range(nc)],1); e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
rows=[]
for f in sorted(glob.glob('arr_*_k*_*.npz')):
    st,k,w0,w1=re.match(r'arr_(\w+?)_k(\d)_(\d+)_(\d+)',f).groups(); k=int(k); d=np.load(f)
    yA,yB,An,Bn=d['yA'],d['yB'],d['yAn'],d['yBn']
    r=dict(state=st,window=f'{w0}:{w1}',mpi=MPI[k])
    for nm,w in W.items():
        rc=(yA@w)/(yB@w); se=jk((An@w)/(Bn@w)); r[nm]=(rc-T[st])*1e4; r[nm+'_se']=se*1e4; r[nm+'_sig']=(rc-T[st])/se
    for nc,lab in [(1,'BLUE-1'),(2,'BLUE-O4'),(3,'BLUE-O6')]:
        wA,wB=blue(cov(An),nc),blue(cov(Bn),nc); rc=(yA@wA)/(yB@wB); se=jk((An@wA)/(Bn@wB)); r[lab]=(rc-T[st])*1e4; r[lab+'_se']=se*1e4; r[lab+'_sig']=(rc-T[st])/se
    rows.append(r)
d=pd.DataFrame(rows); d.to_csv('dec_extra_summary.csv',index=False); pd.set_option('display.width',300)
for st in d.state.unique():
    print(st); print(d[d.state==st].drop(columns='state').round(2).to_string(index=False))
