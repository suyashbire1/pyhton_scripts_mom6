import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def extract_velocities(geofil,vgeofil,fil,fil2,fil3,xstart,xend,ystart,yend,ls,le,
       z=np.linspace(-3000,0,100),htol=1e-3,whichterms=None):

    domain = Domain.Domain(geofil,vgeofil,
        xstart,xend,ystart,yend,ls=ls,le=le,ts=0,te=None) 

    fig,ax = plt.subplots(3,3,sharex=True,sharey=True,figsize=(8,8))
    ax = ax.ravel()

    with mfdset(fil) as fh, mfdset(fil2) as fh2: 

        h = -gv('e',domain,'hi',fh2,units='m').read_array().o1diff(1).values
        h[h<htol] = np.nan
        sigma = gv('e',domain,'hi',fh2).read_array().ddx(1).values
        
        e = gv('e',domain,'hi',fh2).xsm().xep().read_array(
                extend_kwargs={'method':'mirror'}).move_to('ul')
        ur = gv('uh',domain,'ul',fh2,fh,plot_loc='hl',divisor='h_Cu',
                name=r'$\hat{u}$').xsm().read_array(divide_by_dy=True,filled=0)
        u = gv('u',domain,'ul',fh2,fh,plot_loc='hl',
                name=r'$\bar{u}$').xsm().read_array(filled=0)
        ub = ur - u
        ub.name = r'$\frac{\overline{u^{\prime}h^{\prime}}}{\bar{h}}$'
        ax[2],im = ur.plot('nanmean',(0,2),zcoord=True,z=z,e=e,
                plot_kwargs=dict(cmap='RdBu_r'),clevs=[-0.01,0.01],
                contour=True,ax=ax[2],perc=99)
        vmin,vmax = im.get_clim()
        u.plot('nanmean',(0,2),zcoord=True,z=z,e=e,clevs=[-0.01,0.01],
                plot_kwargs=dict(cmap='RdBu_r',vmin=vmin,vmax=vmax),
                cbar=False,contour=True,ax=ax[0],perc=99)
        ub.plot('nanmean',(0,2),zcoord=True,z=z,e=e,clevs=[-0.01,0.01],
                plot_kwargs=dict(cmap='RdBu_r',vmin=vmin,vmax=vmax),
                cbar=False,contour=True,ax=ax[1],perc=99)
        cbar = fig.colorbar(im,ax=ax[:3].tolist())
        cbar.formatter.set_powerlimits((-2,2))
        cbar.update_ticks()

        ex = gv('e',domain,'hi',fh2).xsm().xep().read_array(extend_kwargs={'method':'mirror'}).ddx(3).move_to('ul')
        uex = ur*ex
        uex = uex.move_to('hl')#.values

        e = gv('e',domain,'hi',fh2).ysm().yep().read_array(
                extend_kwargs={'method':'mirror'}).move_to('vl')
        vr = gv('vh',domain,'vl',fh2,fh,plot_loc='hl',divisor='h_Cv',
                name=r'$\hat{v}$').ysm().read_array(divide_by_dx=True,filled=0)
        v = gv('v',domain,'vl',fh2,fh,plot_loc='hl',
                name=r'$\bar{v}$').ysm().read_array(filled=0)
        vb = vr - v
        vb.name = r'$\frac{\overline{v^{\prime}h^{\prime}}}{\bar{h}}$'
        ax[5],im=vr.plot('nanmean',(0,2),zcoord=True,z=z,e=e,
                plot_kwargs=dict(cmap='RdBu_r'),clevs=[-0.12,0.12],
                contour=True,ax=ax[5],perc=99)
        vmin,vmax = im.get_clim()
        v.plot('nanmean',(0,2),zcoord=True,z=z,e=e,clevs=[-0.12,0.12],
                plot_kwargs=dict(cmap='RdBu_r',vmin=vmin,vmax=vmax),
                contour=True,ax=ax[3],perc=99)
        vb.plot('nanmean',(0,2),zcoord=True,z=z,e=e,clevs=[-0.12,0.12],
                plot_kwargs=dict(cmap='RdBu_r'),
                contour=True,ax=ax[4],perc=99)
        cbar = fig.colorbar(im,ax=ax[3:6].tolist())
        cbar.formatter.set_powerlimits((-2,2))
        cbar.update_ticks()

        ey = gv('e',domain,'hi',fh2).ysm().yep().read_array().ddx(2).move_to('vl')
        vey = vr*ey
        vey = vey.move_to('hl')#.values

        e = gv('e',domain,'hi',fh2).read_array()#.values
        wd = gv('wd',domain,'hi',fh2,name=r'$\hat{\varpi}\bar{\zeta}_{\tilde{b}}$').read_array().move_to('hl')#.values
        conv = uex + vey
        conv.name = r'$\hat{u}\bar{\zeta}_{\tilde{x}}+\hat{v}\bar{\zeta}_{\tilde{y}}$'
        whash = wd + conv
        whash.name = r'$w^{\#}$'
        ax[8],im=whash.plot('nanmean',(0,2),zcoord=True,z=z,e=e,
                plot_kwargs=dict(cmap='RdBu_r'),clevs=[-0.0001,0.0001],
                contour=True,fmt='%.0e',ax=ax[8],perc=99)
        vmin,vmax = im.get_clim()
        wd.plot('nanmean',(0,2),zcoord=True,z=z,e=e,clevs=[-0.0001,0.0001],
                plot_kwargs=dict(cmap='RdBu_r',vmin=vmin,vmax=vmax),
                contour=True,fmt='%.0e',ax=ax[7],perc=99)
        conv.plot('nanmean',(0,2),zcoord=True,z=z,e=e,clevs=[-0.0001,0.0001],
                plot_kwargs=dict(cmap='RdBu_r',vmin=vmin,vmax=vmax),
                contour=True,fmt='%.0e',ax=ax[6],perc=99,xtokm=True)
        cbar = fig.colorbar(im,ax=ax[6:9].tolist())
        cbar.formatter.set_powerlimits((-2,2))
        cbar.update_ticks()
    for i,axc in enumerate(ax):
        axc.set_ylabel('z (m)') if i % 3 == 0 else axc.set_ylabel('')
        axc.set_xlabel('x from EB (km)') if i > 5 else axc.set_xlabel('')
    return fig


#    fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3))
#
#    with mfdset(fil3) as fh3:
#
#        domain = Domain.Domain(geofil,vgeofil,
#            xstart,xend,ystart,yend,ls=ls,le=le,ts=0,te=None) 
#
#        e = gv('e',domain,'hi',fh3,plot_loc='hl').xsm().xep().read_array(
#                extend_kwargs={'method':'mirror'},tmean=False).move_to('ui')
#        u = gv('u',domain,'ul',fh3,plot_loc='hl').xsm().read_array(
#                tmean=False,filled=0).toz(z,e=e)
#        ux = u.ddx(3)
#
#        e = gv('e',domain,'hi',fh3,plot_loc='hl').ysm().yep().read_array(
#                extend_kwargs={'method':'mirror'},tmean=False).move_to('vi')
#        v = gv('v',domain,'vl',fh3,plot_loc='hl').ysm().read_array(
#                tmean=False,filled=0).toz(z,e=e)
#        vy = v.ddx(2)
#
#        wz = np.zeros_like(ux.values)
#        wz -= ux.values*ux.dz 
#        wz -= vy.values*vy.dz
#        #w = np.zeros_like(wz) + np.cumsum(np.nan_to_num(wz),axis=1)
#        w = np.cumsum(wz,axis=1)
#        
#        x = ux.dom.lonh[ux.plot_slice[3,0]:ux.plot_slice[3,1]]
#        dx = np.diff(x)[0]
#        extent = [x[0]-dx/2,x[-1]+dx/2,z[0]-ux.dz/2,z[-1]+ux.dz/2]
#        wplot = np.nanmean(w,axis=(0,2))
#        vmax = np.nanpercentile(np.fabs(wplot[:,:-3]),99.0)
#        #wplot[(wplot>vmax) & (wplot<-vmax)] = 0
#        #vmax = np.nanpercentile(np.fabs(wplot),99.0)
#        im = ax[2].imshow(wplot[:,:-3],origin='lower',cmap='RdBu_r',
#                interpolation='none',extent=extent,aspect='auto',vmax=vmax,vmin=-vmax)
#        cbar = fig.colorbar(im)
#        cbar.formatter.set_powerlimits((-2,2))
#        cbar.update_ticks()
#        CS = ax[2].contour(x[3:],z,wplot[:,:-3],np.linspace(-5e-4,-1e-4,2),colors='k')
#        CS.clabel(inline=1,fmt="%.0e")
#        ax[2].set_xlabel('x from EB (km)')
#        ax[2].text(0.05,0.2,r'$\bar{w}^z$',transform=ax[2].transAxes,fontsize=15)
#
#        e = gv('e',domain,'hi',fh3,plot_loc='ul').xep().read_array(
#                extend_kwargs={'method':'mirror'},tmean=False).move_to('ul')
#        u = gv('u',domain,'ul',fh3,math=r'$\bar{u}^z$').read_array(
#                tmean=False,filled=0).toz(z,e=e).plot(
#                'nanmean',(0,2),plot_kwargs=dict(cmap='RdBu_r'),
#                cbar=True,ax=ax[0], contour=True,clevs=np.array([-0.005,0.01]),
#                fmt='%1.2f',xtokm=True,annotate='math')
#        e = gv('e',domain,'hi',fh3,plot_loc='vl').yep().read_array(
#                extend_kwargs={'method':'mirror'},tmean=False).move_to('vl')
#        v = gv('v',domain,'vl',fh3,math=r'$\bar{v}^z$').read_array(
#                tmean=False,filled=0).toz(z,e=e).plot(
#                'nanmean',(0,2),plot_kwargs=dict(cmap='RdBu_r'),
#                cbar=True,ax=ax[1],contour=True,clevs=np.array([-0.2,0.2]),
#                fmt='%1.1f',xtokm=True,annotate='math')
#        ax[1].set_ylabel('')
#
#    return fig


