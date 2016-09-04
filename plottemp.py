import sys
import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt
from mom_plot import m6plot
from netCDF4 import MFDataset as mfdset
import time

def extractT(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,ts=0,te=None,
        z=None,drhodt=-0.2,rho0=1031.0,savfil=None,plotit=True):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fh = mfdset(fil)
    (xs,xe),(ys,ye),dimh = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze)
    D = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0]
    nt = dimh[0].size
    t0 = time.time()

    zl = rdp1.getdims(fh)[2][1]
    e = fh.variables['e'][ts:te,zs:ze,ys:ye,xs:xe]
    if z == None:
        z = np.linspace(-np.nanmax(D),-1,num=50)
    T = getTatz(zl,z,e)
    T = (T - rho0)/drhodt
    T = np.ma.apply_over_axes(np.nanmean, T, meanax)

    X = dimh[keepax[1]]
    Y = dimh[keepax[0]]
    if 1 in keepax:
        Y = z 

    P = T.squeeze()
    data = (X,Y,P)

    if plotit:
        Pmax = np.nanmax(P)
        Pmin = np.nanmin(P)
        im = m6plot(data,Zmax=Pmax,Zmin=Pmin,cmap=plt.cm.Reds,
                xlab='y (deg)',ylab='z (m)')
        if savfil:
            plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                        format='eps', transparent=False, bbox_inches='tight')
        else:
            plt.show()
    else:
        return data
