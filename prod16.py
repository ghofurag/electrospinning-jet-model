import esjet, numpy as np, json, pickle
from multiprocessing import Pool
OUT='/home/claude/v3/runs/'
B=dict(V=16e3,h=0.12,Q=1.11e-10,mu=1.75e-3,G=5.2e3,alpha=0.041,rho=1040.0,a0=3e-6,L0=1e-4,rho_q=130.0,
       n_dep=60,n_skip=10,tmax=0.3,cfl=0.02,dt_max=2e-6)
cases=[]
for V in [10e3,12e3,14e3,16e3,18e3,20e3]:
    cases.append(('V',V,dict(V=V))); cases.append(('V_IV',V,dict(V=V,charge_scale_with_V=16e3)))
for mu in [1.75e-3,2.23e-3,5e-3,15.6e-3,37.5e-3,50e-3,0.15,0.5,1.5,5.0]: cases.append(('mu',mu,dict(mu=mu)))
for h in [0.07,0.10,0.12,0.14,0.17,0.20]: cases.append(('h',h,dict(h=h)))
for V in [10e3,16e3,20e3]: cases.append(('V_L50',V,dict(V=V,L0=5e-5,n_dep=120,n_skip=20)))
for mu in [1.75e-3,15.6e-3,50e-3]: cases.append(('mu_L50',mu,dict(mu=mu,L0=5e-5,n_dep=120,n_skip=20)))
for h in [0.07,0.12,0.20]: cases.append(('h_L50',h,dict(h=h,L0=5e-5,n_dep=120,n_skip=20)))
cases.append(('convA',0,dict(cfl=0.05,dt_max=1e-5))); cases.append(('convC',0,dict(cfl=0.01,dt_max=1e-6)))
def job(a):
    grp,val,kw,s=a; k=dict(B); k.update(kw); o=esjet.simulate(seed=s,**k)
    cur=o['current']*(k['V']/16e3 if kw.get('charge_scale_with_V') else 1)
    return dict(grp=grp,val=val,seed=s,d_jet=float(o['d_jet']),d_sd=float(o['d_jet_sd']),t_first=float(o['t_first']),n_dep=o['n_dep'],cur=float(cur))
if __name__=='__main__':
    jobs=[(g,v,kw,s) for g,v,kw in cases for s in range(5)]
    with Pool(2) as p: res=p.map(job,jobs)
    json.dump(res,open(OUT+'prod.json','w'),indent=0)
    rec={}
    for name,kw in [('base',{}),('V10',dict(V=10e3)),('V20',dict(V=20e3)),('h7',dict(h=0.07)),('h20',dict(h=0.20))]:
        k=dict(B); k.update(kw); rec[name]=esjet.simulate(seed=0,record=True,**k)
    pickle.dump(rec,open(OUT+'rec.pkl','wb'))
    pool=[]
    for s in range(5):
        o=esjet.simulate(seed=s,record=True,**B); pool+=list(o['dep_all'][10:])
    np.save(OUT+'base_dep_nm.npy',np.array(pool)*np.sqrt(0.0673)*1e9)
    print('done',len(res))
