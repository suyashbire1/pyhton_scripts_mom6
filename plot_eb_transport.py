import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from getvaratz import getvaratz

def plot_eb_transport(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,savfil=None):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fh = mfdset(fil)
    (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
    fhgeo = dset(geofil)
    D = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0]
    fhgeo.close()
    nt_const = dimv[0].size
    vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
    e = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
    fh.close()
    vh = vh.filled(0)

    X = dimv[keepax[1]]
    Y = dimv[keepax[0]]
    if 1 in keepax:
        z = np.linspace(-np.nanmax([2000]),-1,num=50)
        Y = z 
    P = getvaratz(vh,z,e)
    P = np.ma.apply_over_axes(np.mean, P, meanax)
    P = P.squeeze()
    #P = np.ma.apply_over_axes(np.mean, vh, meanax).squeeze()
    im = m6plot((X,Y,P),titl='Transport near EB', ylab='z (m)', xlab='y (Deg)')
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,P),titl='Transport near EB', ylab='z (m)', xlab='y (Deg)')
        plt.show()
