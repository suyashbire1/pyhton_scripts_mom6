import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time

def extract_momy_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
        D, (ah,aq) = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0:2]
        nt = dimv[0].size
        t0 = time.time()

        print('Reading data using loop...')
        dvdtm = fh.variables['dvdt'][0:1,zs:ze,ys:ye,xs:xe]/nt
        cavm = fh.variables['CAv'][0:1,zs:ze,ys:ye,xs:xe]/nt
        pfvm = fh.variables['PFv'][0:1,zs:ze,ys:ye,xs:xe]/nt
        dvdtviscm = fh.variables['dv_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]/nt
        diffvm = fh.variables['diffv'][0:1,zs:ze,ys:ye,xs:xe]/nt
        dvdtdiam = fh.variables['dvdt_dia'][0:1,zs:ze,ys:ye,xs:xe]/nt
        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
            print(cavm.shape,em.shape)

        for i in range(1,nt):
            dvdtm += fh.variables['dvdt'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            cavm += fh.variables['CAv'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            pfvm += fh.variables['PFv'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            dvdtviscm += fh.variables['dv_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            diffvm += fh.variables['diffv'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            dvdtdiam += fh.variables['dvdt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()
            
        fh.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        terms = np.ma.concatenate(( dvdtm[:,:,:,:,np.newaxis],
                                    cavm[:,:,:,:,np.newaxis],
                                    dvdtdiam[:,:,:,:,np.newaxis],
                                    pfvm[:,:,:,:,np.newaxis],
                                    dvdtviscm[:,:,:,:,np.newaxis],
                                    diffvm[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.mean, terms, meanax)

        X = dimv[keepax[1]]
        Y = dimv[keepax[0]]
        if 1 in keepax:
            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimv[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('momy_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('momy_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_momy(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,alreadysaved=False):
    X,Y,P = extract_momy_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(3,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 3:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()
