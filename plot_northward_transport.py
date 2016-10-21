import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_northward_transport(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,fil2=None):

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
    f = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[-1]
    nt_const = dimv[0].size
    fhgeo.close()
    vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
    try:
        h = fh.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]
    except:
        fh2 = mfdset(fil2)
        h = fh2.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]
        fh2.close()
    fh.close()
    vh = vh.filled(0)
    vhplus = np.where(vh>0,vh,0)
    vhminus = np.where(vh<0,vh,0)
    lr = np.sum(np.sqrt(-h*dbl[:,np.newaxis,np.newaxis])/np.pi/f/1e3,axis=1,keepdims=True)
    lr = np.mean(lr,axis=0).squeeze()
    terms = np.ma.concatenate(( vhplus[:,:,:,:,np.newaxis],
                                vhminus[:,:,:,:,np.newaxis]),axis=4)

    termsm = np.ma.apply_over_axes(np.mean, terms, meanax)
    termsm = termsm.squeeze()
    X = dimv[keepax[1]]
    Y = dimv[keepax[0]]

    return X,Y, termsm, lr

def plot_nt(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxscalefactor = 1, savfil=None, fil2=None):
    X,Y,P,lr = extract_northward_transport(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,fil2=fil2)
    cmax = np.nanmax(np.absolute(P))*cmaxscalefactor
    plt.figure()
    ti = r'$\int_{-D}^{0} (vdx) dz$ $(m^3s^{-1})$' 
    for i in range(P.shape[-1]):
        ax = plt.subplot(1,4,i+2)
        if i == 1:
            im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,txt=ti)
        else:
            im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,txt=ti,cbar=False)
        ax.set_yticklabels([])
        plt.xlabel(r'$x (^{\circ}$)')

    ax = plt.subplot(1,4,4)
    ax.plot(np.nanmean(P[:,:,0],axis=1)/1e6,Y,lw=2,label='North',color='r')
    ax.plot(np.nanmean(-P[:,:,1],axis=1)/1e6,Y,lw=2,label='South',color='b')
    ax.set_xlabel('EB transport (Sv)')
    ax.set_yticklabels([])
    ax.text(0.1,0.05,r'$\int_{D}^{0} \int_{EB}^{} v dx dz$ (Sv)',transform=ax.transAxes)
    ax.grid(True)
    plt.legend(loc='best')

    ax = plt.subplot(1,4,1)
    ax.plot(np.nanmean(lr,axis=1),Y,lw=2,color='k')
    ax.set_xlabel(r'$L_R$ (km)')
    plt.ylabel(r'$y (^{\circ}$)')
    ax.grid(True)
#    plt.tight_layout()
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
