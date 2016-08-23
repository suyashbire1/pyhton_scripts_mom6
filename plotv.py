import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt
from mom_plot import m6plot

def extractvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        twa=True,loop=True):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    D = rdp1.getgeom(geofil)[0]
    
    time = rdp1.getdims(fil)[3]
    nt = time.size

    if loop:
        print('Reading data using loop...')
        dimu,um = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,xhxq='xq',ts=0,te=1)
        dimv,vm = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,yhyq='yq',ts=0,te=1)
        dime,em = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,ts=0,te=1)
        um /= nt
        vm /= nt
        em /= nt
        if twa:
            dimhu,frhatu=rdp1.getvar('frhatu',fil,wlon=xstart,elon=xend,
                    slat=ystart,nlat=yend, zs=zs,ze=ze,ts=0,te=1)
            dimhv,frhatv=rdp1.getvar('frhatv',fil,wlon=xstart,elon=xend,
                    slat=ystart,nlat=yend, zs=zs,ze=ze,ts=0,te=1)
            hm_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            hm_v = frhatv*D[np.newaxis,np.newaxis,:,:]
            hm_u /= nt
            hm_v /= nt

        for i in range(1,nt):
            u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]
            v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,yhyq='yq',ts=i,te=i+1)[1]
            e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,ts=i,te=i+1)[1]
            um += u/nt
            vm += v/nt
            em += e/nt
            if twa:
                frhatu=rdp1.getvar('frhatu',fil,wlon=xstart,elon=xend,
                        slat=ystart,nlat=yend, zs=zs,ze=ze,ts=i,te=i+1)[1]
                frhatv=rdp1.getvar('frhatv',fil,wlon=xstart,elon=xend,
                        slat=ystart,nlat=yend, zs=zs,ze=ze,ts=i,te=i+1)[1]
                h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
                h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
                hm_u += h_u/nt
                hm_v += h_v/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100)))
            sys.stdout.flush()
            
        if twa:
            um /= hm_u
            vm /= hm_v
        
    else:
        print('Reading data...')
        dimu,u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,xhxq='xq')
        dimv,v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,yhyq='yq')
        dime,e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze)
        print('Done!')

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
