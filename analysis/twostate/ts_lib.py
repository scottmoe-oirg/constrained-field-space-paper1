import numpy as np
from scipy.optimize import least_squares
ETA0=0.001364; CONV=0.88658; F=np.array([1,2,4,8.])
def ratio(M): return (M[...,1,0,:]*M[...,0,1,:])/(M[...,0,0,:]*M[...,1,1,:])
def whiten(C):
    ev,U=np.linalg.eigh(C); keep=ev>ev.max()*1e-12; return (U[:,keep]/np.sqrt(ev[keep])).T
def fit1(r,W,t,p0=None):
    m0=-np.polyfit(t,np.log(np.abs(r)),1)[0]; A0=np.exp(np.mean(np.log(np.abs(r))+m0*t))
    res=least_squares(lambda p: W@(p[0]*np.exp(-p[1]*t)-r),[A0,m0] if p0 is None else p0,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    return res.x, 2*res.cost
def fit2(r,W,t,D0,Dsig=None,p0=None):
    """R=A e^{-m t}(1+b e^{-D t}); D fixed (Dsig None) or Gaussian prior (D0,Dsig). returns params, chi2(data part)"""
    if p0 is None:
        late=t>=t.max()-8; (A0,m0),_=fit1(r[late],np.eye(late.sum()),t[late])
        b0=np.mean(r[:3]/(A0*np.exp(-m0*t[:3]))-1)/np.mean(np.exp(-D0*t[:3]))
        p0=[A0,m0,b0]+([D0] if Dsig else [])
    def res(p):
        D=p[3] if Dsig else D0
        rr=W@(p[0]*np.exp(-p[1]*t)*(1+p[2]*np.exp(-D*t))-r)
        return np.append(rr,(D-D0)/Dsig) if Dsig else rr
    s=least_squares(res,p0,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=4000)
    rr=s.fun[:-1] if Dsig else s.fun
    return s.x, float(rr@rr)
