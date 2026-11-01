"""Synthetic field-space study with KNOWN population covariance and KNOWN response coefficients (referee points 5, 9, 10).
Population: the real lightest-proton and lightest-neutron field-space covariances (nested jackknife, i.e. covariance of the
mean, condition ~1e5) and responses y(x) = y0 + c1 x + c2 x^2 + c3 x^3 with (y0,c1,c2,c3) from the exact cubic through
the real y. Scenarios scale the omitted cubic coefficient: c3 = 0 (O6 model exact), c3 as in data, 3x data.
Each mock: n configurations z_i ~ N(y_true, n*Sigma); estimators use ybar and the sample covariance of the mean.
Estimators: fixed O2/O4/O6/O8; BLUE-1/O4/O6 with the TRUE Sigma and with the ESTIMATED Sigma; correlated (GLS) polynomial
fits of degree 1 and 2 with the estimated Sigma (algebraically equal to BLUE-O4/O6 with that Sigma - checked).
Reports: displacement from the true y0 (mean over mocks) in units of the true SD, actual SD, mean reported (in-sample)
SE, coverage of +-1 SE, and the mean leverage L4, L6 of the estimated weights vs the population-optimal weights."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np, pandas as pd, json
x=np.array([1,4,16,64.]); V=np.vander(x,4,increasing=True)
W={'O2':np.array([1,0,0,0.]),'O4':np.array([4/3,-1/3,0,0]),'O6':np.array([64/45,-4/9,1/45,0]),'O8':np.linalg.solve(V.T,[1,0,0,0.])}
def blue(S,nc):
    A=V[:,:nc]; e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
def gls_intercept(y,S,deg):
    A=V[:,:deg+1]; Si=np.linalg.inv(S); beta=np.linalg.solve(A.T@Si@A,A.T@Si@y); return beta[0]
rng=np.random.default_rng(20261020); rows=[]; lev=[]
for p in ['P0','N0']:
    d=np.load(str(DERIVED / f'nested_{p}.npz')); Sig=d['Cn']*0.88658**2; yreal=d['y']*0.88658
    coef=np.linalg.solve(V,yreal)            # exact cubic through the real responses
    for scen,s3 in [('c3=0',0.0),('c3=data',1.0),('c3=3xdata',3.0)]:
        c=coef.copy(); c[3]*=s3; ytrue=V@c; y0=c[0]
        for n in [100,200,400,992]:
            M=2000 if n<992 else 1500; L=np.linalg.cholesky(n*Sig); est={}
            pop={nc:blue(Sig,nc) for nc in (1,2,3)}
            for m in range(M):
                Z=ytrue+rng.standard_normal((n,4))@L.T; yb=Z.mean(0); S=np.cov(Z.T)/n
                for nm,w in W.items(): est.setdefault(nm,[]).append((w@yb,np.sqrt(w@S@w)))
                for nc,lab in [(1,'BLUE-1'),(2,'BLUE-O4'),(3,'BLUE-O6')]:
                    wt=pop[nc]; we=blue(S,nc)
                    est.setdefault(lab+' (true Sigma)',[]).append((wt@yb,np.sqrt(wt@Sig@wt)))
                    est.setdefault(lab+' (estimated Sigma)',[]).append((we@yb,np.sqrt(we@S@we)))
                    if m<300: lev.append(dict(particle=p,scenario=scen,n=n,est=lab,L4_est=we@x**2,L6_est=we@x**3,L4_pop=pop[nc]@x**2,L6_pop=pop[nc]@x**3))
                for deg in (1,2):
                    est.setdefault(f'GLS poly deg {deg} (estimated Sigma)',[]).append((gls_intercept(yb,S,deg),np.nan))
            sd_O6=np.std([v for v,_ in est['O6']])
            for nm,v in est.items():
                v=np.array(v); val,se=v[:,0],v[:,1]; sd=val.std(ddof=1)
                rows.append(dict(particle=p,scenario=scen,n=n,estimator=nm,true_y0=y0,mean_disp=val.mean()-y0,disp_over_sd=(val.mean()-y0)/sd,
                    sd=sd,sd_over_O6=sd/sd_O6,mean_reported_se=np.nanmean(se),se_over_sd=np.nanmean(se)/sd,
                    coverage_1se=np.nanmean(np.abs(val-y0)<=se) if not np.all(np.isnan(se)) else np.nan,
                    rmse_over_O6=np.sqrt(((val-y0)**2).mean())/np.sqrt(((np.array([a for a,_ in est['O6']])-y0)**2).mean())))
            print(p,scen,n,'done',flush=True)
d=pd.DataFrame(rows); d.to_csv('synthetic_results.csv',index=False); pd.DataFrame(lev).to_csv('synthetic_leverage.csv',index=False)
