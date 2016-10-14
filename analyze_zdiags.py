import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
import sys


def plottanimation(fil,levels=None,yinc=True,savfil=None,**iselargs):
    ds = xr.open_dataset(fil)

    T = 5+1/0.2*(1031-ds.Rho_cv)
    ani = animate(T.isel(**iselargs),levels=levels,yincrease=yinc)

    if savfil:
        ani.save(savfil+'.mp4', writer="ffmpeg", fps=30, 
                extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()


def animate(var,fig=None,**kwargs):
    """Sets up the initial figure window and 
    returns and animation object."""
    if fig is None:
        fig, ax = plt.subplots()

    var.isel(Time=0).plot.contourf(**kwargs)
    

    def animator(i):
        """Updates the figure for each iteration"""
        fig.clear()
        var.isel(Time=i+1).plot.contourf(**kwargs)
        sys.stdout.write('\r'+str(int((i+1)/(var.Time.size-1)*100))+'% done...')
        sys.stdout.flush()
        return fig,

    anim = animation.FuncAnimation(fig, animator,frames=var.Time.size-1, 
            interval=30, blit=True)
    return anim


