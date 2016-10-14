import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import MFDataset as mfdset
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
import sys
import readParams_moreoptions as rdp1


#def plottanimation(fil,levels=None,yinc=True,savfil=None,**iselargs):
#    ds = xr.open_dataset(fil)
#
#    T = 5+1/0.2*(1031-ds.Rho_cv)
#    ani = animate(T.isel(**iselargs),levels=levels,yincrease=yinc)
#
#    if savfil:
#        ani.save(savfil+'.mp4', writer="ffmpeg", fps=30, 
#                extra_args=['-vcodec', 'libx264'])
#    else:
#        plt.show()


#def animate(var,fig=None,**kwargs):
#    """Sets up the initial figure window and 
#    returns and animation object."""
#    if fig is None:
#        fig, ax = plt.subplots()
#
#    var.isel(Time=0).plot.contourf(**kwargs)
#    
#
#    def animator(i):
#        """Updates the figure for each iteration"""
#        fig.clear()
#        var.isel(Time=i+1).plot.contourf(**kwargs)
#        sys.stdout.write('\r'+str(int((i+1)/(var.Time.size-1)*100))+'% done...')
#        sys.stdout.flush()
#        return fig,
#
#    anim = animation.FuncAnimation(fig, animator,frames=var.Time.size-1, 
#            interval=30, blit=True)
#    return anim


def plottanimation(fil,var,savfil=None,fig=None,wlon=-25,elon=0, 
        slat=10,nlat=60,zs=0,ze=1,**plotkwargs):
    
    fh = mfdset(fil)
    ani = animate(fh,var,fig=fig,wlon=wlon,elon=elon, 
            slat=slat,nlat=nlat,zs=zs,ze=ze,**plotkwargs)

    if savfil:
        ani.save(savfil+'.mp4', writer="ffmpeg", fps=30, 
                extra_args=['-vcodec', 'libx264'])
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
    pvar = np.squeeze(fh.variables[var][0,zs:ze,ys:ye,xs:xe])
    pvar = 5+1/0.2*(1031-pvar)
    plt.contourf(x,y,pvar,**plotkwargs)
    plt.colorbar()

    def animator(i):
        """Updates the figure for each iteration"""
        fig.clear()
        pvar = np.squeeze(fh.variables[var][i,zs:ze,ys:ye,xs:xe])
        plt.contourf(x,y,pvar,**plotkwargs)
        plt.colorbar()
        return fig,

    anim = animation.FuncAnimation(fig, animator, 
                               frames=nt, interval=40, blit=True)
    return anim
