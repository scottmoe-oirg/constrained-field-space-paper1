import numpy as np, pandas as pd, glob, re
x=np.array([1,4,16,64.]); W={'O2':np.array([1,0,0,0.]),'O4':np.array([4/3,-1/3,0,0]),'O6':np.array([64/45,-4/9,1/45,0]),'O8':np.linalg.solve(np.vander(x,4,increasing=True).T,[1,0,0,0.])}
MPI=[337.7,361.5,383.8,404.9,444.3,693.0]
def jk(v): n=len(v); return np.sqrt((n-1)/n*((v-v.mean(0))**2).sum(0))
def cov(Y): n=len(Y); c=Y-Y.mean(0); return (n-1)/n*(c.T@c)
def blue(S,nc):
    A=np.stack([x**p for p in range(nc)],1); e=np.zeros(nc); e[0]=1; Si=np.linalg.inv(S); return Si@A@np.linalg.solve(A.T@Si@A,e)
def loo(Z):
    N=len(Z); s1=Z.sum(0); s2=Z.T@Z
    for j in range(N): m=(s1-Z[j])/(N-1); yield (N-1)/N*((s2-np.outer(Z[j],Z[j]))/(N-1)-np.outer(m,m))
rows=[]; brows=[]
for f in sorted(glob.glob('arr_k*_*.npz')):
    k,w0,w1=map(int,re.match(r'arr_k(\d)_(\d+)_(\d+)',f).groups()); d=np.load(f)
    yA,yB,An,Bn,Ar,Br=d['yA'],d['yB'],d['yAn'],d['yBn'],d['yAr'],d['yBr']
    r=dict(window=f'{w0}:{w1}',tmin=w0,tmax=w1,k=k,mpi=MPI[k])
    for nm,w in W.items():
        rc=(yA@w)/(yB@w); r[nm+'_dev']=rc-.5; r[nm+'_se_fixed']=jk((Ar@w)/(Br@w)); r[nm+'_se']=jk((An@w)/(Bn@w)); r[nm+'_sig']=(rc-.5)/r[nm+'_se']
    for nm,(Y,y) in {'DP':(An,yA),'DPP':(Bn,yB)}.items():
        n6=np.array([-64,84,-21,1.]); r[nm+'_T6']=float((n6@y/jk(Y@n6))**2)
    for nc,lab in [(1,'BLUE-1'),(2,'BLUE-O4'),(3,'BLUE-O6')]:
        SA,SB=cov(An),cov(Bn); wA,wB=blue(SA,nc),blue(SB,nc); rc=(yA@wA)/(yB@wB)
        r[lab+'_dev']=rc-.5; se_in=jk((An@wA)/(Bn@wB)); r[lab+'_sig_in']=(rc-.5)/se_in
        if (w0,w1) in [(10,18),(12,20)]:
            ra=np.array([An[j]@blue(S,nc) for j,S in enumerate(loo(An))]); rb=np.array([Bn[j]@blue(S,nc) for j,S in enumerate(loo(Bn))])
            se_out=jk(ra/rb); o6A,o6B=jk(An@W['O6']),jk(Bn@W['O6'])
            brows.append(dict(window=f'{w0}:{w1}',mpi=MPI[k],est=lab,dev=(rc-.5)*1e4,pct=abs(rc-.5)/.5*100,se_in=se_in*1e4,sig_in=(rc-.5)/se_in,se_out=se_out*1e4,sig_out=(rc-.5)/se_out,
                eA_in=jk(An@wA)/o6A,eB_in=jk(Bn@wB)/o6B,eA_out=jk(ra)/o6A,eB_out=jk(rb)/o6B,wA=np.round(wA,2).tolist(),condA=np.linalg.cond(SA)))
    rows.append(r)
d=pd.DataFrame(rows); d.to_csv('dec_scan_nested_summary.csv',index=False); b=pd.DataFrame(brows); b.to_csv('dec_blue_nested.csv',index=False)
pd.set_option('display.width',250)
P=d[d.window.isin(['10:18','12:20'])].sort_values(['window','k'])
print(P[['window','mpi']+[f'{n}_{c}' for n in W for c in ('dev','se','sig')]].assign(**{f'{n}_dev':lambda q,n=n: q[f'{n}_dev']*1e4 for n in W},**{f'{n}_se':lambda q,n=n: q[f'{n}_se']*1e4 for n in W}).round(2).to_string(index=False))
L,H=d[d.k<5],d[d.k==5]
for n in list(W)+['BLUE-O6']:
    c=n+'_sig' if n in W else n+'_sig_in'
    print(n,'scan sig light %.1f..%.1f heavy %.1f..%.1f'%(L[c].min(),L[c].max(),H[c].min(),H[c].max()))
print('BLUE-O6 pct over scan',(d['BLUE-O6_dev'].abs()/.5*100).min(),(d['BLUE-O6_dev'].abs()/.5*100).max())
print('O6 neg count',(d.O6_dev<0).sum(),len(d))
print('DP T6 range',d.DP_T6.min(),d.DP_T6.max(),'DPP',d.DPP_T6.min(),d.DPP_T6.max(), 'prod windows DP',P.DP_T6.min(),P.DP_T6.max(),'DPP',P.DPP_T6.min(),P.DPP_T6.max())
print(b[b.mpi.isin([337.7,444.3,693.0])].drop(columns=['wA']).round(3).to_string(index=False))
for w in ['10:18','12:20']:
    for e in ['BLUE-1','BLUE-O4','BLUE-O6']:
        q=b[(b.window==w)&(b.est==e)]; print(w,e,'pct %.1f-%.1f sig_out %.1f..%.1f eA_in %.2f-%.2f eA_out %.2f-%.2f eB_out %.2f-%.2f'%(q.pct.min(),q.pct.max(),q.sig_out.min(),q.sig_out.max(),q.eA_in.min(),q.eA_in.max(),q.eA_out.min(),q.eA_out.max(),q.eB_out.min(),q.eB_out.max()))
print(b[(b.window=='10:18')&(b.mpi==337.7)][['est','wA','condA']].to_string())
