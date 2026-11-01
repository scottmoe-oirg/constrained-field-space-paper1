import numpy as np
ETA=0.001364; FIELDS=np.array([1,2,4,8]); X=FIELDS**2
K=-0.013151195576411883
W={'O2':np.array([1,0,0,0.]),'O4':np.array([4/3,-1/3,0,0]),'O6':np.array([64/45,-4/9,1/45,0]),
   'O8':np.linalg.solve(np.vander(X,4,increasing=True).T,np.array([1,0,0,0.]))}
NULL6=np.array([-64,84,-21,1.])
MPI={0.1567:337.7,0.1566:361.5,0.1565:383.8,0.1564:404.9,0.1562:444.3,0.1546:693.0}
def jk_means(A,bs=8):
    """A: cfg x ... ; returns full mean and J leave-one-bin-out means."""
    n=(A.shape[0]//bs)*bs; A=A[:n]; J=n//bs
    tot=A.sum(0); bins=A.reshape(J,bs,*A.shape[1:]).sum(1)
    return tot/n,(tot[None]-bins)/(n-bs)
def jk_se(reps):
    J=reps.shape[0]; return np.sqrt((J-1)/J*((reps-reps.mean(0))**2).sum(0))
