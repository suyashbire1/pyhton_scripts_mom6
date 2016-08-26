import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time as tim

def extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      loop=True,alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        D, (ah,aq) = rdp1.getgeom(geofil,wlon=xstart,
                elon=xend,slat=ystart,nlat=yend)[0:2]
        time = rdp1.getdims(fil)[3]
        (uxs,uxe),_,dimu = rdp1.getlatlonindx(fil,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        _,(vys,vye),dimv = rdp1.getlatlonindx(fil,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
        (xs,xe),(ys,ye),dime = rdp1.getlatlonindx(fil,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,zlzi='zi')
        _,_,dimh = rdp1.getlatlonindx(fil,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze)
        nt = dimu[0].size
        t0 = tim.time()
        fh = mfdset(fil)

        if loop:
            print('Reading data using loop...')
            uh = fh.variables['uh'][0:1,zs:ze,ys:ye,uxs:uxe]
            vh = fh.variables['vh'][0:1,zs:ze,vys:vye,xs:xe]
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
            dhdtm = fh.variables['dhdt'][0:1,zs:ze,ys:ye,xs:xe]/nt
            wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
            print(uh.shape,vh.shape,wd.shape,dhdtm.shape,em.shape)

            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhxm = uhx/nt

            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhym = vhy/nt

            wdm = np.diff(wd,axis=1)/nt

            for i in range(1,nt):
                uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,uxs:uxe]
                vh = fh.variables['vh'][i:i+1,zs:ze,vys:vye,xs:xe]
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
                dhdtm += fh.variables['dhdt'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
                wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]

                uh = np.ma.filled(uh.astype(float), 0)
                uhx = np.diff(uh,axis = 3)/ah
                uhxm += uhx/nt

                vh = np.ma.filled(vh.astype(float), 0)
                vhy = np.diff(vh,axis = 2)/ah
                vhym += vhy/nt

                wdm += np.diff(wd,axis=1)/nt

                sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
                sys.stdout.flush()
                
        fh.close()
        print('Total reading time: {}s'.format(tim.time()-t0))

        terms = np.ma.concatenate(( dhdtm[:,:,:,:,np.newaxis],
                                    uhxm[:,:,:,:,np.newaxis],
                                    vhym[:,:,:,:,np.newaxis],
                                    wdm[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.mean, terms, meanax)
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        elm = np.ma.apply_over_axes(np.mean, elm, meanax)

        X = dimh[keepax[1]]
        Y = dimh[keepax[0]]
        if 1 in keepax:
            Y = elm.squeeze()
            X = np.meshgrid(X,dimh[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('cb_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('cb_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_cb(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,loop=True,alreadysaved=False):
    X,Y,P = extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
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
    im = m6plot((X,Y,np.sum(P,axis=2)),ax,Zmax=cmax)
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
