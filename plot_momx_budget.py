import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time as tim

def extract_momx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      loop=True,alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        D, (ah,aq) = rdp1.getgeom(geofil,wlon=xstart,
                elon=xend,slat=ystart,nlat=yend)[0:2]
        time = rdp1.getdims(fil)[3]
        nt = time.size
        fh = mfdset(fil)
        t0 = tim.time()

        if loop:
            print('Reading data using loop...')
            dimu,dudtm = rdp1.getvar('dudt',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,xhxq='xq',ts=0,te=1)
            dudtm /= nt
            caum = rdp1.getvar('CAu',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,xhxq='xq',ts=0,te=1)[1]/nt
            pfum = rdp1.getvar('PFu',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,xhxq='xq',ts=0,te=1)[1]/nt
            dudtviscm = rdp1.getvar('du_dt_visc',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                    nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=0,te=1)[1]/nt
            diffum = rdp1.getvar('diffu',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                    nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=0,te=1)[1]/nt
            dudtdiam = rdp1.getvar('dudt_dia',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                    nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=0,te=1)[1]/nt
            dime,em = rdp1.getvar('e',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,ts=0,te=1,zlzi='zi')
            em /=nt
            print(caum.shape,em.shape)

            for i in range(1,nt):
                dudtm += rdp1.getvar('dudt',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                        nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]/nt
                caum += rdp1.getvar('CAu',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]/nt
                pfum += rdp1.getvar('PFu',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]/nt
                dudtviscm += rdp1.getvar('du_dt_visc',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                        nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]/nt
                diffum += rdp1.getvar('diffu',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                        nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]/nt
                dudtdiam += rdp1.getvar('dudt_dia',fh,fil,wlon=xstart,elon=xend,slat=ystart,
                        nlat=yend, zs=zs,ze=ze,xhxq='xq',ts=i,te=i+1)[1]/nt
                em += rdp1.getvar('e',fh,fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,ts=i,te=i+1,zlzi='zi')[1]/nt

                sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
                sys.stdout.flush()
            
        fh.close()
        print('Time taken for data reading: {}s'.format(tim.time()-t0))

        terms = np.ma.concatenate(( dudtm[:,:,:,:,np.newaxis],
                                    caum[:,:,:,:,np.newaxis],
                                    dudtdiam[:,:,:,:,np.newaxis],
                                    pfum[:,:,:,:,np.newaxis],
                                    dudtviscm[:,:,:,:,np.newaxis],
                                    diffum[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.mean, terms, meanax)
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        elm = np.ma.apply_over_axes(np.mean, elm, meanax)

        X = dimu[keepax[1]]
        Y = dimu[keepax[0]]
        if 1 in keepax:
            Y = elm.squeeze()
            X = np.meshgrid(X,dimu[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('momx_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('momx_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_momx(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,loop=True,alreadysaved=False):
    X,Y,P = extract_momx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            loop,alreadysaved)
    cmax = np.nanmax(np.absolute(P))
    plt.figure()
    ax = plt.subplot(3,2,1)
    im = m6plot((X,Y,P[:,:,0]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,2)
    im = m6plot((X,Y,P[:,:,1]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,3)
    im = m6plot((X,Y,P[:,:,2]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,4)
    im = m6plot((X,Y,P[:,:,3]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,5)
    im = m6plot((X,Y,P[:,:,4]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,6)
    im = m6plot((X,Y,P[:,:,5]),ax,Zmax=cmax)

    im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
