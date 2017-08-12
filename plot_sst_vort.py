import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot,xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc, getTatzc, getTatzc2
from pym6 import Domain, Variable, Plotter
import importlib
importlib.reload(Domain)
importlib.reload(Variable)
importlib.reload(Plotter)
gv = Variable.GridVariable

def plot_sst_vort(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,ls,le,
      fil3=None, z=np.linspace(-10,-9,4),htol=1e-3,whichterms=None):

    domain = Domain.Domain(geofil,vgeofil,
            xstart,xend,ystart,yend,ls=ls,le=le,ts=450,te=451)

    with  mfdset(fil2) as fh2:
        rho = gv('e',domain,'hi',fh2).read_array(tmean=False).toz(z,rho=True)
        T = (-rho + 1031)*(1/0.2)
        T.name =''
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
        T.plot('nanmean',(0,1),contour=False,cbar=True,plot_kwargs=dict(cmap='RdYlBu_r',vmax=20,vmin=10),ax=ax[0])
        e = gv('e',domain,'hi',fh2,plot_loc='qi').xsm().ysm().read_array(tmean=False,
                extend_kwargs=dict(method='mirror')).move_to('ui').move_to('qi')
        uy = gv('u',domain,'ul',fh2,plot_loc='ql').yep().read_array(tmean=False,
                extend_kwargs=dict(method='vorticity')).ddx(2)
        vx = gv('v',domain,'vl',fh2,plot_loc='ql').xep().read_array(tmean=False,
                extend_kwargs=dict(method='vorticity')).ddx(3)
        vort = (vx - uy)
        vort.name = ''
        vort.plot('nanmean',(0,1),perc=98,contour=False,cbar=True,plot_kwargs=dict(cmap='RdBu_r'),ax=ax[1])
        ax[1].set_ylabel('')

        vortz = vort.toz([-1],e=e)
        vortz.name = ''
        vortz.plot('nanmean',(0,1),perc=98,contour=False,cbar=True,plot_kwargs=dict(cmap='RdBu_r'),ax=ax[2])
#        vmax = np.nanpercentile(np.fabs(vortz.values),98)
#        xx,yy = np.meshgrid(vortz.dom.lonq[vortz._plot_slice[3,0]:vortz._plot_slice[3,1]],
#                vortz.dom.latq[vortz._plot_slice[2,0]:vortz._plot_slice[2,1]])
#        print(xx.shape, yy.shape, vortz.values[0,0,:,:].shape)
#        im = ax[2].pcolormesh(xx,yy,
#                vortz.values[0,0,:,:],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
#        print(e.values.shape)
#        im = ax[2].pcolormesh(e.dom.lonq[e._plot_slice[3,0]:e._plot_slice[3,1]],
#                e.dom.latq[e._plot_slice[2,0]:e._plot_slice[2,1]],
#                e.values[0,0,:,:],cmap='RdBu_r')
#        cbar = fig.colorbar(im,ax=ax[2])
#        cbar.formatter.set_powerlimits((-2,2))
#        cbar.update_ticks()
#        ax[2].set_ylabel('')

    domain = Domain.Domain(geofil,vgeofil,
            xstart,xend,ystart,yend,ls=0,le=1)
    with mfdset(fil) as fh:
        e = gv('e',domain,'hi',fh).read_array()
        cs = ax[0].contour(e.dom.lonh[e._plot_slice[3,0]:e._plot_slice[3,1]],e.dom.lath[e._plot_slice[2,0]:e._plot_slice[2,1]],e.values.squeeze(),
                levels=[-0.4,-0.3,-0.15,0,0.15,0.3,0.4],colors='k')
        cs.clabel(inline=True)
    return fig
