import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset
import readParams as rdp

geofil = 'ocean_geometry.nc'
fil = 'output__002?_01.nc'

D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f = rdp.getgeom(geofil)
(xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)

fh = mfdset(fil)
e = fh.variables['e'][:,:,:,:]
h = fh.variables['h'][:,:,:,:]
fh.close()

e1 = np.swapaxes(e,0,1).reshape((e.shape[1],-1))

z = np.linspace(-np.max(D),-1,num=50)
rho = np.zeros((z.size,e1.shape[1]))
#rho[:] = np.NaN

for i in range(z.size):
    t1 = e1[0:-1,:] - z[i]
    t2 = e1[1:,:] - z[i]
    tt = t1*t2
    indices = np.nonzero(tt<=0)
    rho[i,indices[1]]=zl[indices[0]]

rho = rho.reshape((z.size,e.shape[0],e.shape[2],e.shape[3])).swapaxes(0,1)
T = (1031-rho)/0.2

plt.figure()
plt.contourf(yh,z,np.mean(T[:,:,:,-1],axis=0))
plt.colorbar()
plt.xlabel('lat (degrees)')
plt.ylabel('depth (m)')
#plt.show()
plt.savefig('T_eb.eps', dpi=300, facecolor='w', edgecolor='w', format='eps', transparent=False, bbox_inches='tight')

xstart = -25
xend = 0
ystart = 38
yend = 42
zstart = 1
zend = 10

nx = len(xh) 
ny = len(yh)
nz = len(zl)
xs = [i for i in range(nx) if xh[i] >= xstart and xh[i] <= xend][1]
xe = [i for i in range(nx) if xh[i] >= xstart and xh[i] <= xend][-1]
ys = [i for i in range(ny) if yh[i] >= ystart and yh[i] <= yend][1]
ye = [i for i in range(ny) if yh[i] >= ystart and yh[i] <= yend][-1]
zs = zstart
ze = zend

sy = T[:,:,:,:]
sy1 = sy.reshape((T.shape[0],-1))
sy2 = (sy1 - sy1.mean(axis=0,keepdims=True))
sycovm = np.cov(sy2)
sycorm = np.corrcoef(sy2)
U, s, V = np.linalg.svd(sycovm)
pc = np.dot(V,sy2) + sy1.mean(axis=0,keepdims=True)

n = time.size
f = np.fft.rfftfreq(n)   # cycles/5days
per = 1/f/6              # period in months

plt.figure()
for i in range(3):
    plt.subplot(3,2,2*i+1)
    C = np.fft.rfft(U[:,i])
#    im = plt.plot(time-time[0],U[:,i])
    im = plt.plot(per,np.square(np.absolute(C)))
    plt.ylabel('Eigenvector')
    if i==2:
        plt.xlabel('Period (months)')

    plt.subplot(3,2,2*i+2)
    P = pc[i,:].reshape((sy.shape[1],sy.shape[2],-1))
    print(P.shape,xh.shape,z.shape)
    im = plt.contourf(xh,z,np.mean(P,axis=1))
    plt.title('R^2 = '+str(int(s[i]/np.sum(s)*100)))
    plt.colorbar()
    if i==2:
        plt.xlabel('x (m)')
    plt.ylabel('z (m)')

#plt.show()
plt.savefig('eof_ymean.eps', dpi=300, facecolor='w', edgecolor='w', format='eps', transparent=False, bbox_inches='tight')

#c = np.mean(np.sum(np.sqrt(np.diff(zi[np.newaxis,:,np.newaxis,np.newaxis],axis=1)*h*9.8/1031),axis=1),axis=0)/np.pi
#l = c/f/1000
#plt.figure()
#plt.subplot(1,2,1)
#plt.contourf(xh,yh,c)
#plt.colorbar()
#plt.subplot(1,2,2)
#plt.contourf(xh,yh,l)
#plt.colorbar()
#plt.show()

#plt.savefig('ssh_hm.eps', dpi=300, facecolor='w', edgecolor='w', format='eps', transparent=False, bbox_inches='tight')
