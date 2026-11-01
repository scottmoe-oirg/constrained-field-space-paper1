"""Decuplet control with the temporal-covariance noise included: nested delete-one jackknife (covariance of R re-estimated
from the remaining configurations for every deleted configuration), R as defined in the paper. Compares the ratio test and
the Delta null tests with the production (fixed-covariance) jackknife."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np, pandas as pd, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.stats import chi2
D=str(DECUPLET); ETA=0.001364; F=np.array([1,2,4,8.]); C0=0.88658
x=np.array([1,4,16,64.]); W={'O2':np.array([1,0,0,0.]),'O4':np.array([4/3,-1/3,0,0]),'O6':np.array([64/45,-4/9,1/45,0]),
 'O8':np.linalg.solve(np.vander(x,4,increasing=True).T,[1,0,0,0.])}; N6=np.array([-64,84,-21,1.])
WINS=[(t0,18) for t0 in range(8,15)]+[(t0,20) for t0 in range(10,17)]+[(t0,22) for t0 in range(12,17)]
MPI=[337.7,361.5,383.8,404.9,444.3,693.0]
def ratio(G): P=G[...,0::2,:,:]; M=G[...,1::2,:,:]; return M[...,0]*P[...,1]/(P[...,0]*M[...,1])   # R as defined
def fit(R,t,L,p0=None):
    f=lambda p: L@(p[0]*np.exp(-p[1]*t)-R)
    if p0 is None: m0=np.log(abs(R[0]/R[-1]))/(t[-1]-t[0]); p0=[R[0]*np.exp(m0*t[0]),m0]
    return least_squares(f,p0,method='lm',xtol=1e-12,ftol=1e-12).x
def load(k):
    a=np.load(f'{D}/s_DP_k{k}.npz'); b=np.load(f'{D}/s_DPP_k{k}.npz'); com=np.intersect1d(a['cfg'],b['cfg'])
    return {'DP':a['C'][np.searchsorted(a['cfg'],com)],'DPP':b['C'][np.searchsorted(b['cfg'],com)]}
def job(k):
    Dk=load(k); out=[]
    for w0,w1 in WINS:
        t=np.arange(w0,w1+1); res={}
        for st,C in Dk.items():
            n=len(C); S=C.sum(0); RF=ratio(S/n)[:,t]; Rr=ratio((S[None]-C)/(n-1))[:,:,t]
            y=np.zeros(4); yr=np.zeros((n,4)); yn=np.zeros((n,4)); Ls=[]
            for fi in range(4):
                Z=Rr[:,fi]; Cv=(n-1)/n*((Z-Z.mean(0)).T@(Z-Z.mean(0))); L=np.linalg.cholesky(np.linalg.inv(Cv)).T
                p=fit(RF[fi],t,L); y[fi]=C0*p[1]/(12*ETA*F[fi])
                for j in range(n): yr[j,fi]=C0*fit(Z[j],t,L,p)[1]/(12*ETA*F[fi])
                Ls.append(p)
            for i in range(n):
                Si=S-C[i]; RFi=ratio(Si/(n-1))[:,t]; Ri=ratio((Si[None]-np.delete(C,i,0))/(n-2))[:,:,t]
                for fi in range(4):
                    Z=Ri[:,fi]; Cv=(n-2)/(n-1)*((Z-Z.mean(0)).T@(Z-Z.mean(0))); L=np.linalg.cholesky(np.linalg.inv(Cv)).T
                    yn[i,fi]=C0*fit(RFi[fi],t,L,Ls[fi])[1]/(12*ETA*F[fi])
            res[st]=(y,yr,yn)
        (yA,yAr,yAn),(yB,yBr,yBn)=res['DP'],res['DPP']; n=len(yAr)
        jk=lambda v: np.sqrt((n-1)/n*((v-v.mean())**2).sum())
        r=dict(window=f'{w0}:{w1}',k=k,mpi=MPI[k])
        for nm,w in W.items():
            rc=(yA@w)/(yB@w); r[f'{nm}_dev']=rc-0.5; r[f'{nm}_se_fixed']=jk((yAr@w)/(yBr@w)); r[f'{nm}_se_nested']=jk((yAn@w)/(yBn@w))
        for st,(y,yr,yn) in res.items():
            for lab,rep in [('fixed',yr),('nested',yn)]:
                v=jk(rep@N6); r[f'{st}_T6_{lab}']=float((N6@y/v)**2)
        np.savez(f'arr_k{k}_{w0}_{w1}.npz',yA=yA,yAr=yAr,yAn=yAn,yB=yB,yBr=yBr,yBn=yBn)
        out.append(r); print(r['window'],k,flush=True)
    return out
if __name__=='__main__':
    ks=[int(a) for a in sys.argv[1:]] or list(range(6))
    with Pool(2) as pool: rows=sum(pool.map(job,ks),[])
    d=pd.DataFrame(rows)
    for nm in W: d[f'{nm}_sig_fixed']=d[f'{nm}_dev']/d[f'{nm}_se_fixed']; d[f'{nm}_sig_nested']=d[f'{nm}_dev']/d[f'{nm}_se_nested']
    d.to_csv('dec_nested_scan.csv',index=False); pd.set_option('display.width',250)
    print(d[['window','mpi']+[c for c in d.columns if 'sig' in c or 'T6' in c]].round(2).to_string(index=False))
