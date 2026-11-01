#!/usr/bin/env python3
"""
Recovered exact field-space validation core.
Recovered 2026-10-09 from raw poly11-small archive using historical numerical fingerprints.

IMPORTANT:
- Scientific fitter is scipy.optimize.least_squares with eigen-whitened correlated residuals.
- Do not substitute minimize_scalar/profile fitting: it was explicitly tested and rejected.
- O4 real-data regression fingerprints are hard-coded below.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import numpy as np
from scipy.optimize import least_squares

ETA0=.001364
FIELDS=np.array([1.,2.,4.,8.])
TIMES=np.arange(12,21)
CONV=.88658

EXPECTED_Y=np.array([2.20917302,2.20063138,2.16545036,2.04216874])
EXPECTED_O4=1.9611329055
EXPECTED_O4_SE=.0408682390
EXPECTED_T_O4=37.26943127
EXPECTED_T_O6=4.040752945

def ratio(M):
    return (M[:,1,0,:]*M[:,0,1,:])/(M[:,0,0,:]*M[:,1,1,:])

def fit_ls(r,C):
    ev,U=np.linalg.eigh(C)
    keep=ev>ev.max()*1e-12
    W=(U[:,keep]/np.sqrt(ev[keep])).T
    t=TIMES.astype(float)
    m0=-np.polyfit(t,np.log(np.maximum(np.abs(r),1e-300)),1)[0]
    A0=np.exp(np.mean(np.log(np.maximum(np.abs(r),1e-300))+m0*t))
    def resid(p):
        A,m=p
        return W@(A*np.exp(-m*t)-r)
    res=least_squares(resid,[A0,m0],method="trf",
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=1000)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[1]

def ordinary_analysis(X):
    n=len(X); S=X.sum(0); RF=ratio(S/n)
    loo=(S[None]-X)/(n-1)
    R=(loo[:,:,1,0,:]*loo[:,:,0,1,:])/(loo[:,:,0,0,:]*loo[:,:,1,1,:])
    y=np.empty(4); yj=np.empty((n,4))
    for f in range(4):
        Z=R[:,f][:,TIMES]; DD=Z-Z.mean(0)
        Ct=(n-1)/n*(DD.T@DD)
        y[f]=fit_ls(RF[f,TIMES],Ct)/(12*ETA0*FIELDS[f])
        for i in range(n):
            yj[i,f]=fit_ls(Z[i],Ct)/(12*ETA0*FIELDS[f])
    C=(n-1)/n*((yj-yj.mean(0)).T@(yj-yj.mean(0)))
    return y,C,yj

def nested_covariance(X):
    n=len(X); S=X.sum(0); ni=n-1
    yj=np.empty((n,4)); mask=np.ones(n,dtype=bool)
    for i in range(n):
        Si=S-X[i]; RF=ratio(Si/ni)[:,TIMES]
        mask[:]=True; mask[i]=False
        loo=(Si[None]-X[mask])/(ni-1)
        R=(loo[:,:,1,0,:]*loo[:,:,0,1,:])/(loo[:,:,0,0,:]*loo[:,:,1,1,:])
        for f in range(4):
            Z=R[:,f][:,TIMES]; DD=Z-Z.mean(0)
            Ct=(ni-1)/ni*(DD.T@DD)
            yj[i,f]=fit_ls(RF[f],Ct)/(12*ETA0*FIELDS[f])
    C=(n-1)/n*((yj-yj.mean(0)).T@(yj-yj.mean(0)))
    return C,yj

Z4_NULL=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]])
Z6_NULL=np.array([[-64.],[84.],[-21.],[1.]])

def null_T(y,C,A):
    # Deterministic algebraic null bases for the fixed field hierarchy m=(1,2,4,8).
    # Avoid platform-dependent QR choices in an ill-conditioned projected covariance.
    if A.shape==(4,2):
        Z=Z4_NULL
    elif A.shape==(4,3):
        Z=Z6_NULL
    else:
        raise ValueError("Only fixed O4 (4x2) and O6 (4x3) designs are supported.")
    z=Z.T@y; V=Z.T@C@Z
    return float(z@np.linalg.solve(V,z))

def regression_check(X):
    y,C,_=ordinary_analysis(X)
    w4=np.array([4/3,-1/3,0,0.])
    o4=CONV*(w4@y)
    se=CONV*np.sqrt(w4@C@w4)
    A4=np.column_stack([np.ones(4),FIELDS**2])
    A6=np.column_stack([np.ones(4),FIELDS**2,FIELDS**4])
    t4=null_T(y,C,A4); t6=null_T(y,C,A6)
    checks=[
        ("y",np.max(np.abs(y-EXPECTED_Y)),2e-7),
        ("O4",abs(o4-EXPECTED_O4),2e-7),
        ("O4_SE",abs(se-EXPECTED_O4_SE),2e-7),
        ("T_O4",abs(t4-EXPECTED_T_O4),2e-5),
        ("T_O6",abs(t6-EXPECTED_T_O6),2e-5)]
    for name,err,tol in checks:
        if err>tol: raise RuntimeError(f"Regression failure {name}: error={err} tol={tol}")
    return dict(y=y,O4=o4,O4_SE=se,T_O4=t4,T_O6=t6)
