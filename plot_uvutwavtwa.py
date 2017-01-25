import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from getvaratz import getvaratz


def extract_uv(geofil,fil,fil2,fil3,xstart,xend,ystart,yend,
        zs,ze,savfil=None, utwamaxscalefactor=1, vtwamaxscalefactor=1):

    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    zi = rdp1.getdims(fh)[2][0]

    fhgeo = dset(geofil)
    

    (xs,xe),(ys,ye),dimutwa = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yh')
    em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
    elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
    elm = np.ma.apply_over_axes(np.mean, elm, (0,2))
    dycu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][1]

    uh = fh.variables['uh'][0:,zs:ze,ys:ye,xs:xe]
    h_cu = fh2.variables['h_Cu'][0:,zs:ze,ys:ye,xs:xe]
    utwa = uh/h_cu/dycu

    fig = plt.figure()
    ax = fig.add_subplot(221)
    X = dimutwa[3]
    X = np.meshgrid(X,dimutwa[1])[0]
    Y = elm.squeeze()
    utwa = np.ma.apply_over_axes(np.nanmean, utwa, (0,2))
    utwa = utwa.squeeze()
    cmax = np.nanmax(np.absolute(utwa))*utwamaxscalefactor
    im = m6plot((X,Y,utwa),ax=ax, ylabel='z (m)',txt=r'$\hat{u}$',
            vmin=-cmax,vmax=cmax,cmap='RdBu_r',ylim=(-3000,0))
    ax.set_xticklabels([])


    (xs,xe),(ys,ye),dimvtwa = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xh',yhyq='yq')
    em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
    elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
    elm = np.ma.apply_over_axes(np.mean, elm, (0,2))
    dxcv = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][2]

    vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
    h_cv = fh2.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]
    vtwa = vh/h_cv/dxcv

    ax = fig.add_subplot(222)
    X = dimvtwa[3]
    X = np.meshgrid(X,dimvtwa[1])[0]
    Y = elm.squeeze()
    vtwa = np.ma.apply_over_axes(np.nanmean, vtwa, (0,2))
    vtwa = vtwa.squeeze()
    cmax = np.nanmax(np.absolute(vtwa))*vtwamaxscalefactor
    im = m6plot((X,Y,vtwa),ax=ax,txt=r'$\hat{v}$',
            vmin=-cmax,vmax=cmax,cmap='RdBu_r',ylim=(-3000,0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fh2.close()
    fh.close()
    fhgeo.close()

    fh3 = mfdset(fil3)
    (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh3,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yh',zlzi='zlremap')
    u = fh3.variables['u'][0:,zs:ze,ys:ye,xs:xe]
    ax = fig.add_subplot(223)
    X = dimu[3]
    Y = dimu[1]
    u = np.ma.apply_over_axes(np.nanmean, u, (0,2))
    u = u.squeeze()
    cmax = np.nanmax(np.absolute(u))
    im = m6plot((X,-Y,u),ax=ax,txt='u', ylabel='z (m)',vmin=-cmax,vmax=cmax,cmap='RdBu_r')
    xdegtokm(ax,0.5*(ystart+yend))

    (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh3,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xh',yhyq='yq',zlzi='zlremap')
    v = fh3.variables['v'][0:,zs:ze,ys:ye,xs:xe]
    ax = fig.add_subplot(224)
    X = dimv[3]
    Y = dimv[1]
    v = np.ma.apply_over_axes(np.nanmean, v, (0,2))
    v = v.squeeze()
    cmax = np.nanmax(np.absolute(v))
    im = m6plot((X,-Y,v),ax=ax,txt='v', vmin=-cmax,vmax=cmax,cmap='RdBu_r')
    xdegtokm(ax,0.5*(ystart+yend))
    ax.set_yticklabels([])
    fh3.close()


    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
