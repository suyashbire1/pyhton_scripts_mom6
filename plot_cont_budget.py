import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      loop=True,alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimh = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze)
        fhgeo = dset(geofil)
        D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
        fhgeo.close()
        nt = dimh[0].size
        t0 = time.time()

        if loop:
            print('Reading data using loop...')
            uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs-1:xe]
            vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye,xs:xe]
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
            #dhdtm = fh.variables['dhdt'][0:1,zs:ze,ys:ye,xs:xe]/nt
            wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]

            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhxm = uhx/nt

            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhym = vhy/nt

            wdm = np.diff(wd,axis=1)/nt

            for i in range(1,nt):
                uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
                vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
                #dhdtm += fh.variables['dhdt'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
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
        print('Total reading time: {}s'.format(time.time()-t0))

        terms = np.ma.concatenate(( uhxm[:,:,:,:,np.newaxis],
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
    fig = plt.figure(figsize=(12, 9))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [ r'$(\overline{uh})_{\tilde{x}}$', 
            r'$(\overline{vh})_{\tilde{y}}$', 
            r'$(\overline{\varpi h})_{\tilde{b}}$'] 

    for i in range(P.shape[-1]):
        ax = plt.subplot(5,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,vmax=cmax,vmin=-cmax,
                txt=lab[i], ylim=(-2500,0),cmap='RdBu_r')
        if i % 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('z (m)')

        if i > 0:
            xdegtokm(ax,0.5*(ystart+yend))

        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax,cmap='RdBu_r')
        plt.show()
