"""Reproduce the 24 stored field-fit exponents of the historical poly0 notebook (master-files/OCT_PN-mm.nb):
90 configurations, delete-one ratios R_j = G11(+B)G22(-B)/[G11(-B)G22(+B)] (+B = MINUS tag), averages = Mean[R_j],
covariance (n-1)/n sum (R_j - mean)(R_j - mean)^T, correlated fit A exp(-m t) over t = 8..18."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import CORR
import numpy as np, re
from scipy.optimize import least_squares
d=np.load(str(CORR / 'poly0' / '_poly0_P.npz')); C=d['C']; n=len(C)
s=open(str(CORR / 'notebooks' / 'OCT_PN-mm.nb')).read()
pat=re.compile(r'Fitting range =", \{8, 18\}, ",\s+kappa =", (\d+), ",\s+field =", (\d+)\]'); ms=list(pat.finditer(s)); stored={}
for i,m in enumerate(ms):
    seg=s[m.end():m.end()+20000]; r=re.search(r'"m1",\s*"\\\[Rule\]",\s*"?([-0-9.`*^]+)',seg); stored[(int(m.group(1)),int(m.group(2)))]=float(r.group(1).rstrip('`'))
t=np.arange(8,19); tot=C.sum(0); loo=(tot[None]-C)/(n-1)
def fit(r,cov):
    ev,U=np.linalg.eigh(cov); W=(U/np.sqrt(ev)).T
    m0=-np.polyfit(t,np.log(r),1)[0]; A0=np.exp(np.mean(np.log(r)+m0*t))
    return least_squares(lambda p: W@(p[0]*np.exp(-p[1]*t)-r),[A0,m0],method='trf',xtol=1e-14,ftol=1e-14,gtol=1e-14).x[1]
rows=[]
for fi in range(4):
    P,M=loo[:,2*fi],loo[:,2*fi+1]
    for k in range(6):
        Rj=(M[:,k,:,0]*P[:,k,:,1])/(P[:,k,:,0]*M[:,k,:,1]); Rj=Rj[:,t]
        avg=Rj.mean(0); cov=(n-1)/n*((Rj-avg).T@(Rj-avg))
        Pf,Mf=tot[2*fi]/n,tot[2*fi+1]/n; Rfull=((Mf[k,:,0]*Pf[k,:,1])/(Pf[k,:,0]*Mf[k,:,1]))[t]
        m_nb=fit(avg,cov); m_core=fit(Rfull,cov); st=stored[(k+1,fi+1)]
        rows.append((k,fi,st,m_nb,m_core))
a=np.array([[r[2],r[3],r[4]] for r in rows])
for r in rows: print(f'kappa {r[0]+1} field {[1,2,4,8][r[1]]}: stored {r[2]:.12f}  mean-of-jack {r[3]:.12f} (diff {r[3]-r[2]:+.2e})  full-sample ratio {r[4]:.12f} (diff {r[4]-r[2]:+.2e})')
print('N stored',len(stored),'max |diff| notebook convention:',np.abs(a[:,1]-a[:,0]).max(),' full-sample-ratio convention:',np.abs(a[:,2]-a[:,0]).max())
