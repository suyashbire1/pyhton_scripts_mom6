import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_momx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fhgeo = dset(geofil)
        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
        fhgeo.close()
        nt = dimu[0].size
        t0 = time.time()

        print('Reading data using loop...')
        #dudtm = fh.variables['dudt'][0:1,zs:ze,ys:ye,xs:xe]/nt
        caum = fh.variables['CAu'][0:1,zs:ze,ys:ye,xs:xe]/nt
        pfum = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]/nt
        dudtviscm = fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]/nt
        diffum = fh.variables['diffu'][0:1,zs:ze,ys:ye,xs:xe]/nt
        dudtdiam = fh.variables['dudt_dia'][0:1,zs:ze,ys:ye,xs:xe]/nt
        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
            print(caum.shape,em.shape)

        for i in range(1,nt):
            #dudtm += fh.variables['dudt'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
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

        terms = np.ma.concatenate(( caum[:,:,:,:,np.newaxis],
                                    dudtdiam[:,:,:,:,np.newaxis],
                                    pfum[:,:,:,:,np.newaxis],
                                    diffum[:,:,:,:,np.newaxis],
                                    dudtviscm[:,:,:,:,np.newaxis]),axis=4)
        terms = terms.filled(0)

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
    fig = plt.figure(figsize=(12, 9))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [ r'$\overline{(f+\zeta)v-(KE)_{\tilde{x}}}$', 
            r'$-\overline{\varpi u_{\tilde{b}}}$',
            r'$-\overline{m_{\tilde{x}}}$', 
            r'$\overline{X^H}$', 
            r'$\overline{X^V}$']

    for i in range(P.shape[-1]):
        ax = plt.subplot(5,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,vmax=cmax,vmin=-cmax,
                txt=lab[i], ylim=(-2500,0),cmap='RdBu_r')
        if i % 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('z (m)')

        if i > 2:
            xdegtokm(ax,0.5*(ystart+yend))

        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax,cmap='RdBu_r')
        plt.show()
