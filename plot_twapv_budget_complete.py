import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from plot_twamomx_budget_complete import extract_twamomx_terms
from plot_twamomy_budget_complete import extract_twamomy_terms

def extract_twapv_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fhgeo = dset(geofil)
        fh = mfdset(fil)
        zi = rdp1.getdims(fh)[2][0]
        dbl = -np.diff(zi)*9.8/1031
        (xs,xe),(ys,ye),dimq = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yq')
        dxbu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][4]
        dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][5]
        nt_const = dimq[0].size
        fhgeo.close()
        
        xmom = extract_twamomx_terms(geofil,vgeofil,fil,xs,xe,ys,ye+1,zs,ze,(0,),
                alreadysaved=False,xyasindices=True)[2]
        ymom = extract_twamomy_terms(geofil,vgeofil,fil,xs,xe,ys,ye,zs,ze,(0,),
                alreadysaved=False,xyasindices=True)[2]

        xmom = xmom[np.newaxis,:,:,:,:]
        ymom = ymom[np.newaxis,:,:,:,:]
        ymom = np.concatenate((ymom,-ymom[:,:,:,-1:]),axis=3)

        pv = -np.diff(xmom,axis=2)/dybu[:,:,np.newaxis] + np.diff(ymom,axis=3)/dxbu[:,:,np.newaxis]
        pv = np.ma.apply_over_axes(np.nanmean, pv, meanax)

        X = dimq[keepax[1]]
        Y = dimq[keepax[0]]
        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt_const
            for i in range(1,nt_const):
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt_const
                sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
                sys.stdout.flush()

            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimq[1])[0]

        P = pv.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('twapv_complete_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twapv_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_twapv(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxscalefactor = 1, savfil=None,alreadysaved=False):
    X,Y,P = extract_twapv_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))*cmaxscalefactor
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(5,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 7:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()
