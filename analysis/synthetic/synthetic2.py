"""Draft-17 additions to the synthetic study.
(1) Pre-test: power of the weight-free O4 (2 dof) and O6 (1 dof) null tests vs the displacement of fixed and
    covariance-optimized estimators, as the omitted response is scaled (lambda multiplies c2 and c3, or c3 only).
(2) Population optimum: precision gain sigma(BLUE)/sigma(O4) and leverage of the population-optimal weights,
    for the nested (noise-inclusive) covariance, the full-pipeline bootstrap covariance, and its resampling spread.
(3) Field design: the same population (exact cubic 'common-curve' model K = V^-1 Sigma V^-T) evaluated at
    equally spaced fields, to see how leverage and gain depend on the geometric design x = (1,4,16,64)."""
import numpy as np, pandas as pd, sys
from scipy.stats import chi2
D=sys.argv[1]; FC=sys.argv[2]
def V_(x,m=4): return np.vander(x,m,increasing=True)
def blue(S,x,nc):
    A=V_(x,nc); e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
def stencil(x,nc): A=V_(x[:nc],nc); w=np.zeros(len(x)); w[:nc]=np.linalg.solve(A.T,np.eye(nc)[0]); return w
x=np.array([1,4,16,64.]); W4,W6=stencil(x,2),stencil(x,3)
N4=np.array([[4.,20.],[-5.,-21.],[1.,0.],[0.,1.]]); N6=np.array([-64,84,-21,1.])
rng=np.random.default_rng(20261030)
# (1) pre-test power
rows=[]
for p in ['P0','N0']:
    d=np.load(f'{D}/nested_{p}.npz'); Sig=d['Cn']*0.88658**2; c=np.linalg.solve(V_(x),d['y']*0.88658); n=992; L=np.linalg.cholesky(n*Sig)
    pop4,pop6=blue(Sig,x,2),blue(Sig,x,3)
    for mode in ['c2,c3','c3']:
        for lam in [0,0.25,0.5,1,2,3]:
            cc=c.copy()
            if mode=='c2,c3': cc[2]*=lam; cc[3]*=lam
            else: cc[3]*=lam
            yt=V_(x)@cc; y0=cc[0]; M=1000; rej4=rej6=0; est={k:[] for k in ['O4','O6','BLUE-O4','BLUE-O6']}
            for m in range(M):
                Z=yt+rng.standard_normal((n,4))@L.T; yb=Z.mean(0); S=np.cov(Z.T)/n
                T4=(N4.T@yb)@np.linalg.solve(N4.T@S@N4,N4.T@yb); T6=(N6@yb)**2/(N6@S@N6)
                rej4+=T4>chi2.isf(.05,2); rej6+=T6>chi2.isf(.05,1)
                est['O4'].append(W4@yb); est['O6'].append(W6@yb); est['BLUE-O4'].append(blue(S,x,2)@yb); est['BLUE-O6'].append(blue(S,x,3)@yb)
            r=dict(particle=p,scaled=mode,lam=lam,power_O4=rej4/M,power_O6=rej6/M,mc_err_disp=1/np.sqrt(M))
            for k,v in est.items(): v=np.array(v); r[f'disp_{k}']=(v.mean()-y0)/v.std(ddof=1)
            rows.append(r); print(r,flush=True)
pd.DataFrame(rows).to_csv('pretest_power.csv',index=False)
# (2) population optimum
def gain(S,nc=2):
    w=blue(S,x,nc); return np.sqrt(w@S@w/(W4@S@W4)) if nc==2 else np.sqrt(w@S@w/(W6@S@W6)), w@x**2, w@x**3
prow=[]
for p,tag in [('P0',['P0','P0_1']),('P5',['P5']),('N0',['N0']),('N5',['N5'])]:
    Cn=np.load(f'{D}/nested_{p}.npz')['Cn']; yb=np.concatenate([np.load(f'{FC}/fullcal_{t}.npz')['yb'] for t in tag]); Cb=np.cov(yb.T)
    for lab,S in [('nested',Cn),('bootstrap',Cb)]:
        g,l4,l6=gain(S); prow.append(dict(ens=p,cov=lab,gain_O4=g,L4=l4,L6=l6,gain_O6=gain(S,3)[0]))
    gs=[];l6s=[]
    for b in range(400):
        S=np.cov(yb[rng.integers(0,len(yb),len(yb))].T); g,_,l6=gain(S); gs.append(g); l6s.append(l6)
    prow.append(dict(ens=p,cov='bootstrap resampled 10-90%',gain_O4=f'{np.quantile(gs,.1):.3f}-{np.quantile(gs,.9):.3f}',L6=f'{np.quantile(l6s,.1):.3g} to {np.quantile(l6s,.9):.3g}'))
pd.DataFrame(prow).to_csv('population_optimum.csv',index=False); print(pd.DataFrame(prow).to_string())
# (3) field design
drows=[]
designs={'geometric (1,2,4,8)':np.array([1,2,4,8.]),'equal (2,4,6,8)':np.array([2,4,6,8.]),'equal (1,2,3,4)':np.array([1,2,3,4.])}
for p in ['P0','N0']:
    Cn=np.load(f'{D}/nested_{p}.npz')['Cn']; Vi=np.linalg.inv(V_(x)); K=Vi@Cn@Vi.T
    for nm,f in designs.items():
        xx=f**2; S=V_(xx)@K@V_(xx).T; w4,w6=stencil(xx,2),stencil(xx,3); wb=blue(S,xx,2)
        s=lambda w: np.sqrt(w@S@w); s2=s(stencil(xx,1))
        drows.append(dict(particle=p,design=nm,O4_over_O2=s(w4)/s2,O6_over_O2=s(w6)/s2,BLUE4_over_O4=s(wb)/s(w4),
            L4_O4=w4@xx**2,L6_O4=w4@xx**3,L4_BLUE=wb@xx**2,L6_BLUE=wb@xx**3,rel_L4=abs(wb@xx**2)/abs(w4@xx**2),rel_L6=abs(wb@xx**3)/abs(w4@xx**3)))
pd.DataFrame(drows).to_csv('field_design.csv',index=False); print(pd.DataFrame(drows).round(4).to_string())
