import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt
from mom_plot import m6plot

def extractvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        twa=True):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    D = rdp1.getgeom(geofil)[0]
    
    time = rdp1.getdims(fil)[3]
    
    dimu,u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,xhxq='xq')
    dimv,v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,yhyq='yq')
    dime,e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze)

    um = np.ma.apply_over_axes(np.mean, u, meanax)
    vm = np.ma.apply_over_axes(np.mean, v, meanax)
    em = np.ma.apply_over_axes(np.mean, e, meanax)

    if twa:
        dimhu,frhatu=rdp1.getvar('frhatu',fil,wlon=xstart,elon=xend,
                slat=ystart,nlat=yend, zs=zs,ze=ze)
        dimhv,frhatv=rdp1.getvar('frhatv',fil,wlon=xstart,elon=xend,
                slat=ystart,nlat=yend, zs=zs,ze=ze)
        h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
        h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
        hm_u = np.ma.apply_over_axes(np.mean, h_u, meanax)
        hm_v = np.ma.apply_over_axes(np.mean, h_v, meanax)
        um /= hm_u
        vm /= hm_v

    Xu = dimu[keepax[1]]
    Xv = dimv[keepax[1]]
    Yu = dimu[keepax[0]]
    Yv = dimv[keepax[0]]
    if 1 in keepax:
        z = np.linspace(-np.max(D),-1,num=50)
        um = getvaratz(um,z,em)
        vm = getvaratz(vm,z,em)
        Yu = z
        Yv = z

    Pu = um.squeeze()
    Pv = vm.squeeze()
    datau = (Xu,Yu,Pu)
    datav = (Xv,Yv,Pv)
    return datau, datav


def plotvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        twa=True,savfil=None):
    datau,datav = extractvel(geofil,fil,xstart,xend,
            ystart,yend,zs,ze,meanax,twa)
    plt.figure()
    ax = plt.subplot(3,2,1)
    im = m6plot(datau,ax)
    
    ax = plt.subplot(3,2,2)
    im = m6plot(datav,ax)
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
