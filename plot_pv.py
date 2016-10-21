import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from getvaratz import getvaratz


def extract_twapv(geofil,vgeofil,fil,xstart,xend,ystart,yend,
        zs,ze,meanax,savfil=None,fil2=None, cmaxscalefactor=None,
        plotatz=False,Z=None):

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
    dycu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][1]
    dxcv = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][2]
    nt_const = dimq[0].size
    fhgeo.close()
    
    try:
        em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
    except:
        fh2 = mfdset(fil2)
        em = fh2.variables['e'][0:,zs:ze,ys:ye,xs:xe]
        fh2.close()

    elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

    try:
        uh = fh.variables['uh'][0:,zs:ze,ys:ye+1,xs:xe]
    except:
        fh2 = mfdset(fil2)
        uh = fh2.variables['uh'][0:,zs:ze,ys:ye+1,xs:xe]
        fh2.close()

    h_cu = fh.variables['h_Cu'][0:,zs:ze,ys:ye+1,xs:xe]
    utwa = uh/h_cu/dycu
    h_cu = np.where(h_cu>1e-3,h_cu,np.nan)

    try:
        vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
    except:
        fh2 = mfdset(fil2)
        vh = fh2.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
        fh2.close()

    h_cv = fh.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]
    vtwa = vh/h_cv/dxcv
    vtwa = np.concatenate((vtwa,-vtwa[:,:,:,-1:]),axis=3)
    h_cv = np.concatenate((h_cv,-h_cv[:,:,:,-1:]),axis=3)
    h_cv = np.where(h_cv>1e-3,h_cv,np.nan)

    hq = 0.25*(h_cu[:,:,:-1,:] + h_cv[:,:,:,:-1] + 
            h_cu[:,:,1:,:] + h_cv[:,:,:,1:])

    pv = -np.diff(utwa,axis=2)/dybu + np.diff(vtwa,axis=3)/dxbu
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
    print(pv.shape,X.shape,Y.shape)
    im = m6plot((X,Y,pv), Zmax=cmax)

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
