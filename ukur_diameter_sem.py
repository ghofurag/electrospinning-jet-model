import numpy as np
from PIL import Image
from skimage import filters, morphology, measure
from scipy import ndimage as ndi
im=np.array(Image.open('sem.jpg').convert('L')).astype(float)[:685,:]
nm_px=200/21.5
sm=filters.gaussian(im,0.8,preserve_range=True)
th=filters.threshold_local(filters.gaussian(im,1,preserve_range=True),151,offset=-5)
fib=morphology.remove_small_objects(filters.gaussian(im,1,preserve_range=True)>th,max_size=200)
sk=morphology.skeletonize(fib)
nb=ndi.convolve(sk.astype(int),np.ones((3,3)),mode='constant')-1
juncd=morphology.dilation(sk&(nb>=3),morphology.disk(12))
pts=sk&~juncd; pts[:40,:]=pts[-40:,:]=False; pts[:,:40]=pts[:,-40:]=False
lab=measure.label(pts,connectivity=2)
res=[]
for reg in measure.regionprops(lab):
    if reg.area<15: continue
    c=reg.coords.astype(float); mu=c.mean(0); u,s,vt=np.linalg.svd(c-mu); t=vt[0]; nrm=np.array([-t[1],t[0]])
    i0=np.argmin(((c-mu)**2).sum(1)); p=c[i0]
    ss=np.arange(-40,40.25,0.25)
    prof=np.mean([ndi.map_coordinates(sm,[p[0]+ss*nrm[0]+k*t[0],p[1]+ss*nrm[1]+k*t[1]],order=1) for k in (-2,-1,0,1,2)],0)
    ci=len(ss)//2; peak=prof[ci-8:ci+9].max(); 
    # local minima on each side within 40 px
    left=prof[:ci]; right=prof[ci:]
    bl=left.min(); br=right.min(); base=max(bl,br)
    half=base+0.5*(peak-base)
    if peak-base<15: continue
    # crossings
    li=np.where(prof[:ci]<half)[0]; ri=np.where(prof[ci:]<half)[0]
    if len(li)==0 or len(ri)==0: continue
    w=(ci+ri[0])-li[-1]
    res.append(w*0.25*nm_px)
res=np.array(res)
print('FWHM n=%d mean=%.0f sd=%.0f median=%.0f p10=%.0f p90=%.0f'%(len(res),res.mean(),res.std(),np.median(res),*np.percentile(res,[10,90])))
np.save('fwhm.npy',res)
