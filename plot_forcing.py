import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.constants as sc
from netCDF4 import MFDataset as mfdset
from pym6 import Variable,Domain
import importlib
importlib.reload(Variable)
importlib.reload(Domain)
gv = Variable.GridVariable
from mom_plot1 import xdegtokm

def plot_forcing(fil,fil2=None):
    y = np.linspace(10,60,100)
    sflat = 20
    nflat = 40

    T = np.zeros(y.shape)
    Tnorth = 10
    Tsouth = 20
    T[y>=sflat] = (Tsouth - Tnorth)/(sflat - nflat)*(y[y>=sflat] - nflat) + Tnorth
    T[y<sflat] = Tsouth
    T[y>nflat] = Tnorth

    rho0 = 1031
    drhodt = -0.2
    rho = rho0 + drhodt*T

    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[1, 4, 1],
                           height_ratios=[1, 4])
    fig = plt.figure(figsize=(9,4))
    #ax = fig.add_subplot(121)
    ax = plt.subplot(gs[1])
    ax.plot(y,T,'k',lw=1)
#    ax.set_ylabel('T ($^{\circ}$C)')
    ax.set_xlabel('y ($^{\circ}$N)')
    ax.set_ylim(8,22)
    ax.grid(True)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax2 = ax.twinx()
    ax2.set_ylabel(r"""$\rho (kgm^{-3})$""")
    ax2.set_ylim(ax.get_ylim())
    yticks = ax.get_yticks()
    ax2.set_yticks(yticks[::2])
    yticks = rho0 + drhodt*yticks[::2]
    ax2.set_yticklabels(['{}'.format(i) for i in yticks])

    z = np.linspace(0,3000)
    dabyss = 1500
    Tabyss = 5
    Tz = (Tabyss-Tsouth)/dabyss*z + Tsouth
    Tz[z>dabyss] = Tabyss

    #ax = fig.add_subplot(122)
    ax = plt.subplot(gs[3])
    ax.plot(Tz,z,'k',lw=1)
    ax.set_xlabel('T ($^{\circ}$C)')
    ax.set_ylabel('z (m)')
    ax.set_xlim(4,21)
    ax.grid(True)
    ax.invert_yaxis()
#    ax.yaxis.tick_right()
#    ax.yaxis.set_label_position("right")

#    ax2 = ax.twiny()
#    ax2.set_xlabel(r"""$\rho (kgm^{-3})$""")
#    ax2.set_xlim(ax.get_xlim())
#    xticks = ax.get_xticks()
#    ax2.set_xticks(xticks[::2])
#    xticks = rho0 + drhodt*xticks[::2]
#    ax2.set_xticklabels(['{}'.format(i) for i in xticks])

    with mfdset(fil) as fh, mfdset(fil2) as fh2:
        geofil = 'ocean_geometry.nc'
        vgeofil = 'Vertical_coordinate.nc'
        domain = Domain.Domain(geofil,vgeofil,-0.5,0,10,60,ls=0,le=None) 
        e = gv('e',domain,'hi',fh).read_array()
        ax = plt.subplot(gs[4])
        ax.plot(domain.lath[e.plot_slice[2,0]:e.plot_slice[2,1]],np.mean(e.values,axis=(0,3)).T[:,::2],'k',lw=1)
        ax.plot(domain.lath[e.plot_slice[2,0]:e.plot_slice[2,1]],np.mean(e.values,axis=(0,3)).T[:,-1],'k',lw=1)
        ax.set_yticklabels('')
        ax.set_xlabel('y ($^{\circ}$N)')
        ax.axvline(x=38.5,color='k')
        ax.grid()
#        swash = gv('islayerdeep',domain,'ql',fh2,
#                plot_loc='hl').xsm().ysm().read_array(extend_kwargs=dict(method='mirror'),
#                        tmean=False).move_to('ul').move_to('hl')
#        swash0 = fh2.variables['islayerdeep'][-1,0,0,320]
#        swash = (-swash + swash0)*(100/swash0)
#        z=np.linspace(-3000,0)
#        swash = swash.toz(z,e)
#        ax.contour(swash.dom.lath[e.plot_slice[2,0]:e.plot_slice[2,1]],z,np.nanmean(swash.values,axis=(0,3)),levels=[1],colors='r')
        

        domain = Domain.Domain(geofil,vgeofil,-0.5,0,38,39,ls=0,le=None) 
        ax = plt.subplot(gs[5])
        e = gv('e',domain,'hi',fh).read_array()
        ax.plot(domain.lonh[e.plot_slice[3,0]:e.plot_slice[3,1]],np.mean(e.values,axis=(0,2)).T[:,::2],'k',lw=1)
        ax.plot(domain.lonh[e.plot_slice[3,0]:e.plot_slice[3,1]],np.mean(e.values,axis=(0,2)).T[:,-1],'k',lw=1)
        ax.set_yticklabels('')
        ax.set_xlabel('x ($^{\circ}$)')

#        swash = gv('islayerdeep',domain,'ql',fh2,
#                plot_loc='hl').xsm().ysm().read_array(
#                        tmean=False).move_to('ul').move_to('hl')
#        swash0 = fh2.variables['islayerdeep'][-1,0,0,320]
#        swash = (-swash + swash0)*(100/swash0)
#        z=np.linspace(-3000,0)
#        swash = swash.toz(z,e)
#        
#        ax.contour(swash.dom.lonh[swash.plot_slice[3,0]:swash.plot_slice[3,1]],
#                z,np.nanmean(swash.values,axis=(0,2)),levels=[1],colors='r')
        xdegtokm(ax,(38+39)/2)
        ax.grid()
        #swash.plot('nanmean',(0,2),zcoord=True,e=e,z=np.linspace(-3000,0),cbar=True,ax=ax)
    return fig

    #plt.savefig('meridionalTprof.eps', dpi=300, facecolor='w', edgecolor='w', 
    #        format='eps', transparent=False, bbox_inches='tight')


