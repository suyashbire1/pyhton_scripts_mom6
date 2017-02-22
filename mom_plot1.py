import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import scipy.constants as sc

def m6plot(data,ax=None,**plotkwargs):

    if not ax:
        f = plt.figure(figsize=(8,8/sc.golden))
        ax = f.add_subplot(111)
    
    cmap = plotkwargs.get('cmap','viridis')
    cbar = plotkwargs.get('cbar',True)
    xlabel = plotkwargs.get('xlabel','')
    ylabel = plotkwargs.get('ylabel','')
    txt = plotkwargs.get('txt','')
    title = plotkwargs.get('title','')
    aspect = plotkwargs.get('aspect','auto')
    vmin = plotkwargs.get('vmin',None)
    vmax = plotkwargs.get('vmax',None)
    inverty = plotkwargs.get('inverty',False)
    bvnorm = plotkwargs.get('bvnorm',False)
    blevs = plotkwargs.get('blevs',25)
    xticks = plotkwargs.get('xticks',None)
    yticks = plotkwargs.get('yticks',None)
    xlim = plotkwargs.get('xlim',None)
    ylim = plotkwargs.get('ylim',None)
    savfil = plotkwargs.get('savfil',None)

    X,Y,Z = data[0:3]

    if bvnorm:
        bounds = np.linspace(vmin, vmax, blevs)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        im = ax.pcolormesh(X,Y,Z,norm=norm,cmap=cmap,shading='flat')
    else:
        im = ax.pcolormesh(X,Y,Z,vmax=vmax,vmin=vmin,cmap=cmap,shading='flat')

    if cbar:
        cbar = plt.colorbar(im, use_gridspec=True)
        cbar.formatter.set_powerlimits((-3, 4))
        cbar.update_ticks()
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_aspect(aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    tx = ax.text(0.05,0.7,txt,transform=ax.transAxes)
    tx.set_fontsize(12)
    if inverty:
        ax.invert_yaxis()

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                format='eps', transparent=False, bbox_inches='tight')
    else:
        return im

def xdegtokm(ax,y):
    xt = ax.get_xticks()
    R = 6400
    xtinkm = R*np.cos(y*np.pi/180)*xt*np.pi/180
    ax.set_xticklabels(['{:.0f}'.format(i) for i in xtinkm])
    ax.set_xlabel('x from EB (km)')
    return

