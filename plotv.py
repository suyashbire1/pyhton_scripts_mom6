import sys
import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt
from mom_plot import m6plot
from netCDF4 import MFDataset as mfdset
import time

def extractvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        twa=True,loop=True):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fh = mfdset(fil)
    (uxs,uxe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
    (xs,xe),(vys,vye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
    D = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0]
    nt = dimu[0].size
    t0 = time.time()
    
    if loop:
        print('Reading data using loop...')
        u = fh.variables['u'][0:1,zs:ze,ys:ye,uxs:uxe]
        v = fh.variables['v'][0:1,zs:ze,vys:vye,xs:xe]
        e = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]

        if twa:
            print('Using twa..')
            frhatu = fh.variables['frhatu'][0:1,zs:ze,ys:ye,uxs:uxe]
            frhatv = fh.variables['frhatv'][0:1,zs:ze,vys:vye,xs:xe]
            hm_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            hm_v = frhatv*D[np.newaxis,np.newaxis,:,:]
            hm_u = np.ma.masked_array(hm_u,mask=(hm_u<=1e-3).astype(int))
            hm_v = np.ma.masked_array(hm_v,mask=(hm_v<=1e-3).astype(int))
            uhm = u*hm_u.filled(0)
            vhm = v*hm_v.filled(0)
            swashu = np.ma.array(np.ones(hm_u.shape),mask=np.ma.getmaskarray(hm_u))
            swashv = np.ma.array(np.ones(hm_v.shape),mask=np.ma.getmaskarray(hm_v))
        else:
            um = u/nt
            vm = v/nt

        em = e/nt

        for i in range(1,nt):
            u = fh.variables['u'][i:i+1,zs:ze,ys:ye,uxs:uxe]
            v = fh.variables['v'][i:i+1,zs:ze,vys:vye,xs:xe]
            e = fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]
            if twa:
                frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,uxs:uxe]
                frhatv = fh.variables['frhatv'][i:i+1,zs:ze,vys:vye,xs:xe]
                h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
                h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
                h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
                h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
                uhm += u*h_u.filled(0)
                vhm += v*h_v.filled(0)
                hm_u += h_u.filled(0)
                hm_v += h_v.filled(0)
                swashu += np.ma.array(np.ones(hm_u.shape),mask=np.ma.getmaskarray(hm_u))
                swashv += np.ma.array(np.ones(hm_v.shape),mask=np.ma.getmaskarray(hm_v))
            else:
                um += u/nt
                vm += v/nt
                
            em += e/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()
            
        fh.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        if twa:
            hm_u = np.ma.apply_over_axes(np.mean, hm_u, meanax[0])
            hm_v = np.ma.apply_over_axes(np.mean, hm_v, meanax[0])
            uhm = np.ma.apply_over_axes(np.mean, uhm, meanax[0])
            vhm = np.ma.apply_over_axes(np.mean, vhm, meanax[0])
            um = uhm/hm_u
            vm = vhm/hm_v

        um = np.ma.apply_over_axes(np.mean, um, meanax)
        vm = np.ma.apply_over_axes(np.mean, vm, meanax)
        swashu = np.ma.apply_over_axes(np.mean, swashu, meanax)
        swashv = np.ma.apply_over_axes(np.mean, swashv, meanax)
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
        elm = np.ma.apply_over_axes(np.mean, elm, meanax)
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        
    Xu = dimu[keepax[1]]
    Xv = dimv[keepax[1]]
    Yu = dimu[keepax[0]]
    Yv = dimv[keepax[0]]
    if 1 in keepax:
        Yu = elm.squeeze()
        Xu = np.meshgrid(Xu,dimu[1])[0]
        Yv = elm.squeeze()
        Xv = np.meshgrid(Xv,dimv[1])[0]

    Pu = um.squeeze()
    Pv = vm.squeeze()
    swashu = swashu.squeeze().filled(0)
    swashv = swashv.squeeze().filled(0)
    datau = (Xu,Yu,Pu,swashu)
    datav = (Xv,Yv,Pv,swashv)
    return datau, datav


def plotvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        twa=True,savfil=None,loop=True):
    datau,datav = extractvel(geofil,fil,xstart,xend,
            ystart,yend,zs,ze,meanax,twa,loop)
    plt.figure()
    ax = plt.subplot(3,2,1)
    im = m6plot(datau,ax,xlab='x from EB (Deg)',ylab='z (m)')
    im2 = plt.contour(datau[0],datau[1],datau[3],1,colors='k')
    
    ax = plt.subplot(3,2,2)
    im = m6plot(datav,ax,xlab='x from EB (Deg)',ylab='z (m)')
    im2 = plt.contour(datav[0],datav[1],datav[3],1,colors='k')
    ax.set_yticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
