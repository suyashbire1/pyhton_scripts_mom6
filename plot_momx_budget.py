import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time

def extract_momx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        D, (ah,aq) = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0:2]
        nt = dimu[0].size
        t0 = time.time()

        print('Reading data using loop...')
        dudtm = fh.variables['dudt'][0:1,zs:ze,ys:ye,xs:xe]/nt
        caum = fh.variables['CAu'][0:1,zs:ze,ys:ye,xs:xe]/nt
        pfum = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]/nt
        dudtviscm = fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]/nt
        diffum = fh.variables['diffu'][0:1,zs:ze,ys:ye,xs:xe]/nt
        dudtdiam = fh.variables['dudt_dia'][0:1,zs:ze,ys:ye,xs:xe]/nt
        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
            print(caum.shape,em.shape)

        for i in range(1,nt):
            dudtm += fh.variables['dudt'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            caum += fh.variables['CAu'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            pfum += fh.variables['PFu'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            dudtviscm += fh.variables['du_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            diffum += fh.variables['diffu'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            dudtdiam += fh.variables['dudt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()
            
        fh.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        terms = np.ma.concatenate(( dudtm[:,:,:,:,np.newaxis],
                                    caum[:,:,:,:,np.newaxis],
                                    dudtdiam[:,:,:,:,np.newaxis],
                                    pfum[:,:,:,:,np.newaxis],
                                    dudtviscm[:,:,:,:,np.newaxis],
                                    diffum[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.mean, terms, meanax)

        X = dimu[keepax[1]]
        Y = dimu[keepax[0]]
        if 1 in keepax:
            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
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
        savfil=None,alreadysaved=False):
    X,Y,P = extract_momx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
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

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()