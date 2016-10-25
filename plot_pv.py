import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from getvaratz import getvaratz


def extract_twapv(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,
        zs,ze,meanax,savfil=None, cmaxscalefactor=None,
        plotatz=False,Z=None):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fhgeo = dset(geofil)
    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    zi = rdp1.getdims(fh)[2][0]
    dbl = np.diff(zi)*9.8/1031
    (xs,xe),(ys,ye),dimq = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yq')
    dxbu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][4]
    dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][5]
    dycu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][1]
    dxcv = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][2]
    f = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[-1]
    nt_const = dimq[0].size
    fhgeo.close()
    
    em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
    elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

    uh = fh.variables['uh'][0:,zs:ze,ys:ye+1,xs:xe]

    h_cu = fh2.variables['h_Cu'][0:,zs:ze,ys:ye+1,xs:xe]
    utwa = uh/h_cu/dycu
    h_cu = np.where(h_cu>1e-3,h_cu,np.nan)

    vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]

    h_cv = fh2.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]
    vtwa = vh/h_cv/dxcv
    vtwa = np.concatenate((vtwa,-vtwa[:,:,:,-1:]),axis=3)
    h_cv = np.concatenate((h_cv,-h_cv[:,:,:,-1:]),axis=3)
    h_cv = np.where(h_cv>1e-3,h_cv,np.nan)

    fh2.close()
    fh.close()

    hq = 0.25*(h_cu[:,:,:-1,:] + h_cv[:,:,:,:-1] + 
            h_cu[:,:,1:,:] + h_cv[:,:,:,1:])

    pv = f -np.diff(utwa,axis=2)/dybu + np.diff(vtwa,axis=3)/dxbu
    pv = pv/(hq/dbl[:,np.newaxis,np.newaxis])

    X = dimq[keepax[1]]
    Y = dimq[keepax[0]]
    if 1 in keepax:
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        elm = np.ma.apply_over_axes(np.mean, elm, meanax)
        Y = elm.squeeze()
        X = np.meshgrid(X,dimq[1])[0]
    
    if plotatz:
        pv = getvaratz(pv,Z,em)

    pv = np.ma.apply_over_axes(np.nanmean, pv, meanax)
    pv = pv.squeeze()
    cmax = np.nanmax(np.absolute(pv))*cmaxscalefactor
    im = m6plot((X,Y,pv), xlabel=r'x ($^{\circ}$ E)',ylabel=r'y ($^{\circ}$ N)',
            vmin=6e-10,vmax=cmax,aspect='equal',bvnorm=True)

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
