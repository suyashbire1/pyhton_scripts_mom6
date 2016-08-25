import sys
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

    #D = rdp1.getgeom(geofil)[0]
    D = rdp1.getgeom(geofil,wlon=xstart,elon=xend,slat=ystart,nlat=yend)[0]
    
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
        if twa:
            print('Using twa..')
            dimhu,frhatu=rdp1.getvar('frhatu',fil,wlon=xstart,elon=xend,
                    slat=ystart,nlat=yend,xhxq='xq',zs=zs,ze=ze,ts=0,te=1)
            dimhv,frhatv=rdp1.getvar('frhatv',fil,wlon=xstart,elon=xend,
                    slat=ystart,nlat=yend,yhyq='yq', zs=zs,ze=ze,ts=0,te=1)
            hm_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            hm_v = frhatv*D[np.newaxis,np.newaxis,:,:]
            hm_u = np.ma.masked_array(hm_u,mask=(hm_u<=1e0).astype(int))
            hm_v = np.ma.masked_array(hm_v,mask=(hm_v<=1e0).astype(int))
            uhm = um*hm_u/nt
            vhm = vm*hm_v/nt
            hm_u /= nt
            hm_v /= nt
        else:
            um /= nt
            vm /= nt

        em /= nt

        for i in range(1,nt):
            u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]
            v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,yhyq='yq',ts=i,te=i+1)[1]
            e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,ts=i,te=i+1)[1]
            if twa:
                frhatu=rdp1.getvar('frhatu',fil,wlon=xstart,elon=xend,
                        slat=ystart,nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]
                frhatv=rdp1.getvar('frhatv',fil,wlon=xstart,elon=xend,
                        slat=ystart,nlat=yend, zs=zs,ze=ze,yhyq='yq',ts=i,te=i+1)[1]
                h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
                h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
                h_u = np.ma.masked_array(h_u,mask=(h_u<=1e0).astype(int))
                h_v = np.ma.masked_array(h_v,mask=(h_v<=1e0).astype(int))
                uh = u*h_u
                vh = v*h_v
                uhm += uh/nt
                vhm += vh/nt
                hm_u += h_u/nt
                hm_v += h_v/nt
            else:
                um += u/nt
                vm += v/nt
                
            em += e/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()
            

        if twa:
            hm_u = np.ma.apply_over_axes(np.mean, hm_u, meanax[0])
            hm_v = np.ma.apply_over_axes(np.mean, hm_v, meanax[0])
            uhm = np.ma.apply_over_axes(np.mean, uhm, meanax[0])
            vhm = np.ma.apply_over_axes(np.mean, vhm, meanax[0])
            um = uhm/hm_u
            vm = vhm/hm_v

        um = np.ma.apply_over_axes(np.mean, um, meanax)
        vm = np.ma.apply_over_axes(np.mean, vm, meanax)
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        
    else:
        print('Reading data...')
        dimu,u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,xhxq='xq')
        dimv,v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,yhyq='yq')
        dime,e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze)
        print('Done!')


        if twa:
            print('Using twa..')
            dimhu,frhatu=rdp1.getvar('frhatu',fil,wlon=xstart,elon=xend,
                    slat=ystart,nlat=yend, zs=zs,ze=ze)
            dimhv,frhatv=rdp1.getvar('frhatv',fil,wlon=xstart,elon=xend,
                    slat=ystart,nlat=yend, zs=zs,ze=ze)
            h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
            h_u = np.ma.masked_array(h_u,mask=(h_u<=1e0).astype(int))
            h_v = np.ma.masked_array(h_v,mask=(h_v<=1e0).astype(int))
            uh = u*h_u
            vh = v*h_v
            um = np.ma.apply_over_axes(np.mean, uh, meanax[0])
            vm = np.ma.apply_over_axes(np.mean, vh, meanax[0])
            hm_u = np.ma.apply_over_axes(np.mean, h_u, meanax[0])
            hm_v = np.ma.apply_over_axes(np.mean, h_v, meanax[0])
            um /= hm_u
            vm /= hm_v

        um = np.ma.apply_over_axes(np.mean, u, meanax)
        vm = np.ma.apply_over_axes(np.mean, v, meanax)
        em = np.ma.apply_over_axes(np.mean, e, meanax)

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
    print(np.max(Pu),np.max(Pv))
    datau = (Xu,Yu,Pu)
    datav = (Xv,Yv,Pv)
    return datau, datav


def plotvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        twa=True,savfil=None,loop=True):
    datau,datav = extractvel(geofil,fil,xstart,xend,
            ystart,yend,zs,ze,meanax,twa,loop)
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
