import esjet, numpy as np, time
from multiprocessing import Pool
PHI=0.0673
base=dict(V=15e3,h=0.12,Q=1.11e-10,mu=1.75e-3,G=5.2e3,alpha=0.041,rho=1040.0,a0=3e-6,rho_q=125.0,n_skip=10,tmax=0.3)
cases=[('A cfl0.05 dt1e-5 L0=100um',dict(cfl=0.05,dt_max=1e-5,L0=1e-4,n_dep=60)),
       ('B cfl0.02 dt2e-6 L0=100um',dict(cfl=0.02,dt_max=2e-6,L0=1e-4,n_dep=60)),
       ('C cfl0.01 dt1e-6 L0=100um',dict(cfl=0.01,dt_max=1e-6,L0=1e-4,n_dep=60)),
       ('D cfl0.02 dt2e-6 L0=50um',dict(cfl=0.02,dt_max=2e-6,L0=5e-5,n_dep=120,n_skip=20))]
def job(a):
    name,kw,s=a; kk=dict(base); kk.update(kw); t0=time.time(); o=esjet.simulate(seed=s,**kk); return name,s,o['d_jet'],time.time()-t0
if __name__=='__main__':
    with Pool(2) as p: res=p.map(job,[(n,k,s) for n,k in cases for s in range(5)])
    for n,_ in cases:
        d=np.array([r[2] for r in res if r[0]==n]); c=np.mean([r[3] for r in res if r[0]==n])
        print(n,'d_f %.1f +- %.1f nm  cpu/run %.1fs'%(d.mean()*np.sqrt(PHI)*1e9,d.std()*np.sqrt(PHI)*1e9,c),flush=True)
