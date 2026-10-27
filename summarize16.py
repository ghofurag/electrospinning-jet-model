import json, numpy as np
PHI=0.0673; s=np.sqrt(PHI)
R=json.load(open('/home/claude/v3/runs/prod.json'))
groups={}
for r in R: groups.setdefault(r['grp'],{}).setdefault(r['val'],[]).append(r)
out={}
for g,dd in groups.items():
    xs=sorted(dd); rows=[]
    for x in xs:
        d=np.array([r['d_jet'] for r in dd[x]]); w=np.array([r['d_sd'] for r in dd[x]]); tf=np.array([r['t_first'] for r in dd[x]])
        rows.append((x,d.mean()*s*1e9,d.std()*s*1e9,w.mean()*s*1e9,tf.mean()*1e3,dd[x][0]['cur']*1e9))
    X=np.array([r[0] for r in rows]); Y=np.array([r[1] for r in rows])
    p=np.polyfit(np.log(X),np.log(Y),1); yp=np.polyval(p,np.log(X)); r2=1-((np.log(Y)-yp)**2).sum()/((np.log(Y)-np.log(Y).mean())**2).sum()
    # bootstrap exponent over seeds
    print('==',g,'exponent %.3f R2 %.3f change %.1f%%'%(p[0],r2,100*(Y[-1]-Y[0])/Y[0]))
    for r in rows: print('   x=%g  d_f=%.1f +- %.1f (seed)  within-run sd=%.1f  t_first=%.2f ms  I=%.1f nA'%r)
    out[g]=dict(rows=rows,exp=p[0],r2=r2)
json.dump(out,open('/home/claude/v3/runs/summary.json','w'))
