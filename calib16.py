import esjet, numpy as np, sys
from multiprocessing import Pool
PHI=0.0673
base=dict(V=16e3,h=0.12,Q=1.11e-10,mu=1.75e-3,G=5.2e3,alpha=0.041,rho=1040.0,a0=3e-6,L0=1e-4,n_dep=60,n_skip=10,tmax=0.3,cfl=0.02,dt_max=2e-6)
def job(a):
    rq,s=a; return rq,s,esjet.simulate(rho_q=rq,seed=s,**base)['d_jet']
if __name__=='__main__':
    grid=[float(x) for x in sys.argv[1:]]
    with Pool(2) as p: res=p.map(job,[(rq,s) for rq in grid for s in range(5)])
    for rq in grid:
        d=np.array([r[2] for r in res if r[0]==rq])*np.sqrt(PHI)*1e9
        print(rq,'d_f %.1f +- %.1f nm'%(d.mean(),d.std()),flush=True)
