#!/usr/bin/env python3
"""
Fit-window robustness of the manuscript's magnetic-moment hierarchy, using the
manuscript's own pipeline (recovered_fieldspace_core: double ratio, fixed
temporal covariance across delete-one replicas, nonlinear correlated fits).

X layout expected by the core: X[config, field(1,2,4,8), sign(0=PLUS,1=MINUS), spin(0=s1,1=s2), t].
First validates the core's hard-coded regression fingerprints (lightest proton, 12:20),
then scans tmin at fixed tmax for both nucleons and reports O4/O6 with
paired jackknife differences from the production window.

usage: window_scan_moments.py <P|N> <reduced OCT dir> <prod_tmin> <prod_tmax> <tmin list>
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import sys, re, glob, json
import numpy as np
pass
import recovered_fieldspace_core as core
part,rawdir,pt0,pt1,tmins=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),[int(x) for x in sys.argv[5].split(',')]
files=sorted(glob.glob(f'{rawdir}/cfly_2448_*.OCT_{part}'),key=lambda p:int(re.search(r'_(\d+)\.OCT',p).group(1)))
lab=re.compile(rf'OCT_{part}_ETA(\d\d)_(PLUS|MINUS)_(\d\d)$')
FI={'01':0,'02':1,'04':2,'08':3}
X=np.empty((len(files),6,4,2,2,48))
for i,f in enumerate(files):
    for line in open(f):
        L,t,s1,s2=line.split(); m=lab.match(L)
        if not m: continue
        X[i,int(m.group(3)),FI[m.group(1)],0 if m.group(2)=='PLUS' else 1,:,int(t)]=(float(s1),float(s2))
print(part,'configs',len(files),flush=True)
C=core.CONV; W4=np.array([4/3,-1/3,0,0]); W6=np.array([64/45,-4/9,1/45,0])
if part=='P':
    core.TIMES=np.arange(12,21)
    y,Cv,_=core.ordinary_analysis(X[:,0]); o4=C*(W4@y); se=C*np.sqrt(W4@Cv@W4)
    t4=core.null_T(y,Cv,np.column_stack([np.ones(4),core.FIELDS**2])); t6=core.null_T(y,Cv,np.column_stack([np.ones(4),core.FIELDS**2,core.FIELDS**4]))
    print('regression vs core fingerprints: max|dy|=%.1e  dO4=%.1e  dSE=%.1e  T_O4 %.4f vs %.4f  T_O6 %.4f vs %.4f'%(
        np.max(np.abs(y-core.EXPECTED_Y)),abs(o4-core.EXPECTED_O4),abs(se-core.EXPECTED_O4_SE),t4,core.EXPECTED_T_O4,t6,core.EXPECTED_T_O6),flush=True)
res={}
for t0 in sorted(set(tmins+[pt0])):
    core.TIMES=np.arange(t0,pt1+1)
    for k in range(6):
        y,Cv,yj=core.ordinary_analysis(X[:,k]); res[(t0,k)]=(y,yj)
    print('done tmin',t0,flush=True)
out=[]
for k in range(6):
    yp,yjp=res[(pt0,k)]
    for t0 in sorted(set(tmins+[pt0])):
        y,yj=res[(t0,k)]; n=len(yj)
        row=dict(kappa_index=k,tmin=t0,tmax=pt1)
        for name,w in (('O4',W4),('O6',W6)):
            v=C*(yj@w); vp=C*(yjp@w); d=v-vp
            row[name]=float(C*(y@w)); row[name+'_se']=float(np.sqrt((n-1)/n*np.sum((v-v.mean())**2)))
            row[name+'_minus_prod']=float(C*(y@w)-C*(yp@w)); dse=np.sqrt((n-1)/n*np.sum((d-d.mean())**2))
            row[name+'_diff_z']=float(row[name+'_minus_prod']/dse) if dse>0 else 0.0
        T6=core.null_T(y,(n-1)/n*((yj-yj.mean(0)).T@(yj-yj.mean(0))),np.column_stack([np.ones(4),core.FIELDS**2,core.FIELDS**4]))
        row['T_O6']=T6; out.append(row)
json.dump(out,open(f'window_scan_{part}.json','w'),indent=1)
for r in out:
    print(f"k{r['kappa_index']} {r['tmin']}:{r['tmax']}  O6={r['O6']:.4f}({r['O6_se']:.4f})  diff from prod={r['O6_minus_prod']:+.4f} ({r['O6_diff_z']:+.2f} SE)  T_O6={r['T_O6']:.2f}")
