import sys
import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt
from mom_plot1 import m6plot
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extractT(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,ts=0,te=None,
        z=None,drhodt=-0.2,rho0=1031.0,savfil=None,plotit=True,loop=True):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fh = mfdset(fil)
    (xs,xe),(ys,ye),dimh = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze)
    fhgeo = dset(geofil)
    D = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0]
    fhgeo.close()
    nt = dimh[0].size
    t0 = time.time()

    zl = rdp1.getdims(fh)[2][1]
    if loop:
        print('Reading data in loop...')
        e = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
        for i in range(nt):
            e += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()
        print('Time taken for data reading: {}s'.format(time.time()-t0))
    else:
        e = fh.variables['e'][ts:te,zs:ze,ys:ye,xs:xe]

    X = dimh[keepax[1]]
    Y = dimh[keepax[0]]
    if 1 in keepax:
        Y = z 
        if z == None:
            z = np.linspace(-np.nanmax(D),-1,num=50)
            Y = z
    T = getTatz(zl,z,e)
    T = (T - rho0)/drhodt
    T = np.ma.apply_over_axes(np.nanmean, T, meanax)

    P = T.squeeze()
    data = (X,Y,P)

    if plotit:
        Pmax = np.nanmax(P)
        Pmin = np.nanmin(P)
        im = m6plot(data,vmax=Pmax,vmin=Pmin,title=r'T at 40N ($^{\circ}$C)',
                xlabel=r'x ($^{\circ}$)',ylabel='z (m)',bvnorm=True,blevs=15)
        if savfil:
            plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                        format='eps', transparent=False, bbox_inches='tight')
        else:
            plt.show()
    else:
        return data
