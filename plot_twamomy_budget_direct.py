import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time

def extract_twamomy_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
#        _,(vys,vye),_ = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
#                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
#        (xs,xe),(ys,ye),_ = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
#                slat=ystart, nlat=yend,zs=zs,ze=ze)
        D, (ah,aq) = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0:2]
        nt_const = dimv[0].size

        e = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
        hmfu = fh.variables['twa_hmfu'][0:,zs:ze,ys:ye,xs:xe]
        hpfv = fh.variables['twa_hpfv'][0:,zs:ze,ys:ye,xs:xe]
        huvxpt = fh.variables['twa_huvxpt'][0:,zs:ze,ys:ye,xs:xe]
        hvvymt = fh.variables['twa_hvvymt'][0:,zs:ze,ys:ye,xs:xe]
        hvwb = fh.variables['twa_hvwb'][0:,zs:ze,ys:ye,xs:xe]
        hdvdtvisc = fh.variables['twa_hdvdtvisc'][0:,zs:ze,ys:ye,xs:xe]
        hdiffv = fh.variables['twa_hdiffv'][0:,zs:ze,ys:ye,xs:xe]
            
        fh.close()

        terms = np.ma.concatenate(( hmfu[:,:,:,:,np.newaxis],
                                    hpfv[:,:,:,:,np.newaxis],
                                    hvwb[:,:,:,:,np.newaxis],
                                    huvxpt[:,:,:,:,np.newaxis],
                                    hvvymt[:,:,:,:,np.newaxis],
                                    hdvdtvisc[:,:,:,:,np.newaxis],
                                    hdiffv[:,:,:,:,np.newaxis]),
                                    axis=4)

        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)

        X = dimv[keepax[1]]
        Y = dimv[keepax[0]]
        if 1 in keepax:
            el = 0.5*(e[:,0:-1,:,:]+e[:,1:,:,:])
            e = np.ma.apply_over_axes(np.mean, e, meanax)
            el = np.ma.apply_over_axes(np.mean, el, meanax)
            Y = el.squeeze()
            X = np.meshgrid(X,dimv[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('twamomy_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twamomy_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_twamomy(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil1=None,savfil2=None,alreadysaved=False):
    X,Y,P = extract_twamomy_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(4,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 3:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil1:
        plt.savefig(savfil1+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()

    P1 = np.concatenate((np.sum(P[:,:,0:2],axis=2,keepdims=True),
        P[:,:,[2]],np.sum(P[:,:,3:5],axis=2,keepdims=True),P[:,:,5:]),axis=2)
    cmax = np.nanmax(np.absolute(P1))
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)']
    for i in range(P1.shape[-1]):
        ax = plt.subplot(3,2,i+1)
        im = m6plot((X,Y,P1[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 3:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil2:
        plt.savefig(savfil2+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P1,axis=2)),Zmax=cmax)
        plt.show()
