import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt

def plotvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,twa=True,savfil=None):

    D = rdp1.getgeom(geofil)[0]
    
    time = rdp1.getdims(fil)[3]
    
    (t,zl1,yh1,xq1),u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,xhxq='xq')
    (t,zl1,yq1,xh1),v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,yhyq='yq')
    (t,zi1,yh1,xh1),e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze)

    um = np.ma.apply_over_axes(np.mean, u, meanax)
    vm = np.ma.apply_over_axes(np.mean, v, meanax)
    em = np.ma.apply_over_axes(np.mean, e, meanax)
    
    plt.figure()
    
    ax = plt.subplot(3,2,1)
    z = np.linspace(-np.max(D),-1,num=50)
    umatz = getvaratz(um,z,em)
    umax = np.amax(np.absolute(umatz))
    Vctr = np.linspace(-umax,umax,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(xq1, z, umatz[0,:,0,:],Vctr, cmap=plt.cm.RdBu_r)
    #im2 = ax.contour(xh1,zi1,np.mean(em[0,:,:,:],axis=1),colors='k')
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
   # ax.set_xticks([-70, -50, -30, -10])
    
    ax = plt.subplot(3,2,2)
    vmatz = getvaratz(vm,z,em)
    vmax = np.max(np.absolute(vmatz))
    Vctr = np.linspace(-vmax,vmax,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(xh1, z, vmatz[0,:,0,:],Vctr, cmap=plt.cm.RdBu_r)
    #im2 = ax.contour(xh1,zi1,np.mean(em[0,:,:,:],axis=1),colors='k')
    cbar = plt.colorbar(im, ticks=Vcbar)
    ax.set_yticklabels([])
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
   # ax.set_xticks([-70, -50, -30, -10])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

