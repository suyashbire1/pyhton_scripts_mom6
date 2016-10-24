import sys
import scipy.constants
import matplotlib.pyplot as plt
from netCDF4 import MFDataset as mfdset, Dataset as dset
import numpy as np
from matplotlib import animation, rc
import readParams_moreoptions as rdp1
from mom_plot import m6plot
from datetime import date
import matplotlib.colors as colors

def plottanimation(fil,var,savfil=None,fig=None,wlon=-25,elon=0, 
        slat=10,nlat=60,zs=0,ze=1,fps=5,bitrate=1000,**plotkwargs):
    
    fh = mfdset(fil)
    ani = animate(fh,var,fig=fig,wlon=wlon,elon=elon, 
            slat=slat,nlat=nlat,zs=zs,ze=ze,**plotkwargs)

    if savfil:
        mywriter = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
        ani.save(savfil+'.mp4', writer=mywriter)
    else:
        plt.show()
    fh.close()

def animate(fh,var,fig=None,wlon=-25,elon=0,
            slat=10,nlat=60,zs=0,ze=1,**plotkwargs):
    """Sets up the initial figure window and 
    returns and animation object."""
    if fig is None:
        fig, ax = plt.subplots()
        
    (xs,xe),(ys,ye),dimh = rdp1.getlatlonindx(fh,wlon=wlon,elon=elon,
                 slat=slat, nlat=nlat,zs=zs,ze=ze,zlzi='zlremap')
    pdims = ()
    for i in range(1,4):
        if dimh[i].size != 1:
            pdims += (dimh[i],)
    y,x = pdims
    nt = dimh[0].size
    cmap = plotkwargs.get('cmap','viridis')
    xlabel = plotkwargs.get('xlabel','')
    ylabel = plotkwargs.get('ylabel','')
    title = plotkwargs.get('title','')
    aspect = plotkwargs.get('aspect','auto')
    vmin = plotkwargs.get('vmin',5)
    vmax = plotkwargs.get('vmax',20)
    inverty = plotkwargs.get('inverty',False)
    bvnorm = plotkwargs.get('bvnorm',False)

    pvar = np.squeeze(fh.variables[var][0,zs:ze,ys:ye,xs:xe])
    if var == 'Rho_cv':
        pvar = 1/0.2*(1031-pvar)

    if bvnorm:
        bounds = np.linspace(vmin, vmax, 25)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        im = ax.pcolormesh(x,y,pvar,norm=norm,cmap=cmap,shading='flat')
    else:
        im = ax.pcolormesh(x,y,pvar,vmax=vmax,vmin=vmin,cmap=cmap,shading='flat')

    plt.colorbar(im)
    tim = date.fromordinal(int(fh.variables['Time'][0])).isoformat()
    txt = ax.text(0.05,0.025,tim,transform=ax.transAxes)
    ax.set_aspect(aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if inverty:
        ax.invert_yaxis()


    def animator(i):
        """Updates the figure for each iteration"""
        pvar = np.squeeze(fh.variables[var][i,zs:ze,ys:ye,xs:xe])
        if var == 'Rho_cv':
            pvar = 1/0.2*(1031-pvar)
        pvar = pvar[:-1,:-1]
        im.set_array(pvar.ravel())
        im = ax.contourf(x,y,pvar,norm=norm,cmap=cmap,shading='flat')
        tim = date.fromordinal(int(fh.variables['Time'][i])).isoformat()
        txt.set_text(tim)
        sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
        sys.stdout.flush()
        return im,txt

    anim = animation.FuncAnimation(fig, animator, 
                               frames=nt, interval=100, blit=False)
    return anim

def plotzetaanimation(geofil,fil,savfil=None,fig=None,wlon=-25,elon=0, 
        slat=10,nlat=60,zs=0,ze=1,fps=5,bitrate=1000,**plotkwargs):
    
    fhgeo = dset(geofil)
    fh = mfdset(fil)
    ani = animatezeta(fhgeo,fh,fig=fig,wlon=wlon,elon=elon, 
            slat=slat,nlat=nlat,zs=zs,ze=ze,**plotkwargs)

    if savfil:
        mywriter = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
        ani.save(savfil+'.mp4', writer=mywriter)
    else:
        plt.show()
    fh.close()
    fhgeo.close()

def animatezeta(fhgeo,fh,fig=None,wlon=-25,elon=0,
            slat=10,nlat=60,zs=0,ze=1,**plotkwargs):
    """Sets up the initial figure window and 
    returns and animation object."""
    if fig is None:
        fig, ax = plt.subplots()
        
    (xs,xe),(ys,ye),dimq = rdp1.getlatlonindx(fh,wlon=wlon,elon=elon,
                 slat=slat,nlat=nlat,zs=zs,ze=ze,
                 zlzi='zlremap',xhxq='xq',yhyq='yq')
    dxbu,dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][4:6]
    pdims = ()
    for i in range(1,4):
        if dimq[i].size != 1:
            pdims += (dimq[i],)
    y,x = pdims
    nt = dimq[0].size
    cmap = plotkwargs.get('cmap','viridis')
    xlabel = plotkwargs.get('xlabel','')
    ylabel = plotkwargs.get('ylabel','')
    title = plotkwargs.get('title','')
    aspect = plotkwargs.get('aspect','auto')
    vmin = plotkwargs.get('vmin',None)
    vmax = plotkwargs.get('vmax',None)
    inverty = plotkwargs.get('inverty',False)
    bvnorm = plotkwargs.get('bvnorm',False)

    u = fh.variables['u'][:1,zs:ze,ys:ye,xs:xe]
    u = np.concatenate((u,-u[:,:,-1:,:]),axis=2)
    v = fh.variables['v'][:1,zs:ze,ys:ye,xs:xe]
    v = np.concatenate((v,-v[:,:,:,-1:]),axis=3)
    zeta = np.diff(v,axis=3)/dxbu - np.diff(u,axis=2)/dybu
    pvar = zeta.squeeze()

    if vmin != None and vmax != None:
        if bvnorm:
            bounds = np.linspace(vmin, vmax, 25)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.pcolormesh(x,y,pvar,norm=norm,cmap=cmap,shading='flat')
        else:
            im = ax.pcolormesh(x,y,pvar,vmax=vmax,vmin=vmin,cmap=cmap,shading='flat')
    else:
        im = ax.pcolormesh(x,y,pvar,cmap=cmap,shading='flat')

    plt.colorbar(im)
    tim = date.fromordinal(int(fh.variables['Time'][0])).isoformat()
    txt = ax.text(0.05,0.025,tim,transform=ax.transAxes)
    ax.set_aspect(aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if inverty:
        ax.invert_yaxis()


    def animator(i):
        """Updates the figure for each iteration"""
        u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
        u = np.concatenate((u,-u[:,:,-1:,:]),axis=2)
        v = fh.variables['v'][i:i+1,zs:ze,ys:ye,xs:xe]
        v = np.concatenate((v,-v[:,:,:,-1:]),axis=3)
        zeta = np.diff(v,axis=3)/dxbu - np.diff(u,axis=2)/dybu
        pvar = zeta.squeeze()
        pvar = pvar[:-1,:-1]
        im.set_array(pvar.ravel())
        tim = date.fromordinal(int(fh.variables['Time'][i])).isoformat()
        txt.set_text(tim)
        sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
        sys.stdout.flush()
        return im,txt

    anim = animation.FuncAnimation(fig, animator, 
                               frames=nt, interval=100, blit=False)
    return anim
