import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_northward_transport(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
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
        (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
        dxcv = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][2]
        nt_const = dimv[0].size
        fhgeo.close()
        vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
        vh = vh.filled(0)
        vhplus = np.where(vh>0,vh,0)
        vhminus = np.where(vh<0,vh,0)
        terms = np.ma.concatenate(( vhplus[:,:,:,:,np.newaxis],
                                    vhminus[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.mean, terms, meanax)
        termsm = termsm.squeeze()
        X = dimv[keepax[1]]
        Y = dimv[keepax[0]]

    return X,Y, termsm

def plot_nt(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxscalefactor = 1, savfil=None,alreadysaved=False):
    X,Y,P = extract_northward_transport(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))*cmaxscalefactor
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(1,2,i+1)
        if i = 1:
            im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        else:
            im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i],cbar=False)

        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('y (deg)')

        if i > -1:
            plt.xlabel('x (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()
