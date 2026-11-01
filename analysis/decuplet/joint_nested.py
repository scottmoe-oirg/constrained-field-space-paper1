import numpy as np, pandas as pd
x=np.array([1,4,16,64.]); W6=np.array([64/45,-4/9,1/45,0]); MPI=[337.7,361.5,383.8,404.9,444.3,693.0]
def jk(v): n=len(v); return np.sqrt((n-1)/n*((v-v.mean(0))**2).sum(0))
def joint(S8,nc,r0):
    A=np.stack([x**p for p in range(nc)],1); e=np.zeros(nc); e[0]=1
    D=np.diag(np.r_[np.ones(4),-r0*np.ones(4)]); Q=D@S8@D; Cm=np.zeros((8,2*nc)); Cm[:4,:nc]=A; Cm[4:,nc:]=A
    K=np.block([[2*Q,Cm],[Cm.T,np.zeros((2*nc,2*nc))]]); s=np.linalg.solve(K,np.r_[np.zeros(8),e,e]); return s[:4],s[4:8]
def loo(Z):
    N=len(Z); s1=Z.sum(0); s2=Z.T@Z
    for j in range(N): m=(s1-Z[j])/(N-1); yield (s2-np.outer(Z[j],Z[j]))/(N-1)-np.outer(m,m)
rows=[]
for w in ['10_18','12_20']:
    for k in range(6):
        d=np.load(f'arr_k{k}_{w}.npz'); yA,yB,An,Bn=d['yA'],d['yB'],d['yAn'],d['yBn']; Z=np.c_[An,Bn]; S8=np.cov(Z.T,bias=True)
        r6=(yA@W6)/(yB@W6); se6=jk((An@W6)/(Bn@W6))
        for nc,lab in [(1,'joint-BLUE-1'),(2,'joint-BLUE-O4'),(3,'joint-BLUE-O6')]:
            u,v=joint(S8,nc,r6); r=(yA@u)/(yB@v); se_in=jk((An@u)/(Bn@v))
            rr=np.array([(An[j]@a)/(Bn[j]@b) for j,C in enumerate(loo(Z)) for a,b in [joint(C,nc,r6)]]); se_out=jk(rr)
            rows.append(dict(window=w,mpi=MPI[k],est=lab,dev=(r-.5)*1e4,sig_in=(r-.5)/se_in,sig_out=(r-.5)/se_out,in_over_O6=se_in/se6,out_over_O6=se_out/se6))
o=pd.DataFrame(rows); o.to_csv('joint_nested.csv',index=False)
for (w,e),q in o.groupby(['window','est']): print(w,e,'dev %.2f..%.2f e-4  sig_out %.1f..%.1f  in/O6 %.2f-%.2f  out/O6 %.2f-%.2f'%(q.dev.min(),q.dev.max(),q.sig_out.min(),q.sig_out.max(),q.in_over_O6.min(),q.in_over_O6.max(),q.out_over_O6.min(),q.out_over_O6.max()))
