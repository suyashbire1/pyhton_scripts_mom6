def m6plot(data,ax=None,xticks=None,yticks=None,
        xlab=None,ylab=None,savfil=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if not ax:
        ax = plt.gca()
    
    X,Y,Z = data
    Zmax = np.amax(np.absolute(Z))
    print(Zmax)
    Zctr = np.linspace(-Zmax,Zmax,num=12,endpoint=True)
    Zcbar = (Zctr[1:] + Zctr[:-1])/2
    im = ax.contourf(X, Y, Z, Zctr, cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(im, ticks=Zcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xlab:
        plt=xlabel(xlab)
    if yticks:
        plt.ylabel(ylab)

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                format='eps', transparent=False, bbox_inches='tight')
    else:
        return im

