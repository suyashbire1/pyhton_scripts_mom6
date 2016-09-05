def m6plot(data,ax=None,xticks=None,yticks=None,
        xlab=None,ylab=None,savfil=None,
        Zmax=None,Zmin=None,titl=None,cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if not ax:
        f = plt.figure()
        ax = f.gca()
    
    X,Y,Z = data[0:3]
    if Zmax != None and Zmin != None:
        Zctr = np.linspace(Zmin,Zmax,num=26,endpoint=True)
    elif Zmax != None:
        Zctr = np.linspace(-Zmax,Zmax,num=12,endpoint=True)
    else:
        Zmax = np.nanmax(np.absolute(Z))
        Zctr = np.linspace(-Zmax,Zmax,num=12,endpoint=True)

    Zcbar = (Zctr[1:] + Zctr[:-1])/2
    if cmap:
        im = ax.contourf(X, Y, Z, Zctr,cmap=cmap)
    else:
        im = ax.contourf(X, Y, Z, Zctr, cmap=plt.cm.RdBu_r)
        
    cbar = plt.colorbar(im, ticks=Zcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if titl:
        ax.set_title(titl)

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                format='eps', transparent=False, bbox_inches='tight')
    else:
        return im
