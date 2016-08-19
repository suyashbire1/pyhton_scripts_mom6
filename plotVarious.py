import os
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset
import readParams as rdp
from manipulateDomain import *
import matplotlib.animation as animation

def plotoceanstats(fil,savfil=None):
    (layer,interface,time), (en,ape,ke), (maxcfltrans,maxcfllin), ntrunc = rdp.getoceanstats(fil)
    time /= 365
    ax =plt.subplot(4,1,1)
    im1 = plt.plot(time,maxcfltrans)
    ax.set_ylabel('Max CFL')
    plt.tick_params(axis='x',labelbottom='off')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    ax = plt.subplot(4,1,2)
    im1 = plt.plot(time,np.mean(ape,axis=1))
    ax.set_ylabel('APE (J)')
    plt.tick_params(axis='x',labelbottom='off')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    ax = plt.subplot(4,1,3)
    im1 = plt.plot(time,np.mean(ke,axis=1))
    ax.set_ylabel('KE (J)')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    plt.tick_params(axis='x',labelbottom='off')
    ax = plt.subplot(4,1,4)
    im1 = plt.plot(time,ntrunc)
    plt.xlabel('Time (years)')
    ax.set_ylabel('Ntruncations')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

def plotvel(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,twa=True,savfil=None):

    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f = rdp.getgeom(geofil)
    
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)
    
    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)
    
    hm = np.zeros((nz,ny,nx))
    hm_cu = np.zeros((nz,ny,nx))
    hm_cv = np.zeros((nz,ny,nx))
    em = np.zeros((nzp1,ny,nx))
    elm = np.zeros((nz,ny,nx))
    um = np.zeros((nz,ny,nx))
    utwa = np.zeros((nz,ny,nx))
    vm = np.zeros((nz,ny,nx))
    vtwa = np.zeros((nz,ny,nx))

    
    for i in range(nt):
        (u,v,h,e,el) = rdp.getuvhe(fil,i)
        em += e/nt
        elm += el/nt
        if twa:
            (dhdt,uh,vh,wd,uhgm,vhgm) = rdp.getcontterms(fil,i)
            (frhatu,frhatv) = rdp.gethatuv(fil,i)
            h_cu = frhatu*D
            h_cu = np.ma.masked_array(h_cu,mask=(h_cu<=1e0).astype(int))
            h_cv = frhatv*D
            h_cv = np.ma.masked_array(h_cv,mask=(h_cv<=1e0).astype(int))
            utwa += uh/dycu
            vtwa += vh/dxcv
            hm_cu += h_cu
            hm_cv += h_cv
        else:
            um += u/nt
            vm += v/nt

        #print((i+1)/nt*100)
        sys.stdout.write('\r'+str(int((i+1)/nt*100)))
        sys.stdout.flush()
    
    if twa:
        um = utwa/hm_cu
        vm = vtwa/hm_cv
    
    plt.figure()
    
    ax = plt.subplot(3,2,1)
    (X1,Y1,epl,eplmn,eplmx),(xs,xe),(ys,ye) = getisopyc(em,xstart,xend,
                                                  ystart,yend,zs,ze,xh,yh,zi,meanax)
    print(np.shape(epl))
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(um[:,:,:],elm,xstart,xend,
                                                  ystart,yend,zs,ze,xq,yh,zl,meanax)
    Vctr = np.linspace(Pmn,Pmx,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    im2 = ax.contour(X1,Y1,epl,epl.shape[0],colors='k')
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
   # ax.set_xticks([-70, -50, -30, -10])
    
    ax = plt.subplot(3,2,2)
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(vm[:,:,:],elm,xstart,xend,
                                                  ystart,yend,zs,ze,xh,yq,zl,meanax)
    Vctr = np.linspace(Pmn,Pmx,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    im2 = ax.contour(X1,Y1,epl,epl.shape[0],colors='k')
    cbar = plt.colorbar(im, ticks=Vcbar)
    ax.set_yticklabels([])
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
   # ax.set_xticks([-70, -50, -30, -10])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()


def plotsshanim(geofil,fil,desfps,savfil=None):
       
    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f = rdp.getgeom(geofil)
    
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)
    
    print(len(time))

    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)
 
    fh = mfdset(fil)
    e1 = fh.variables['e'][:,0,:,:]
    emax = np.ceil(np.amax(np.abs(e1)))
    e = fh.variables['e'][0,:,:,:] 
    fh.close()
     
    xlim = [-25,0]     # actual values of x in degrees
    ylim = [10,50]    # actual values of y in degrees
    zlim = [0,1]     # indices of zl or zi
  
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(e,e,xlim[0],xlim[-1],
                                                  ylim[0],ylim[-1],zlim[0],zlim[-1],xq,yh,zi,0)
#    X,Y,P,Pmn,Pmx = plotrange(e,xlim[0],xlim[-1],
#                                ylim[0],ylim[-1],
#                                zlim[0],zlim[-1],xh,yh,zi,e,0)
    fig = plt.figure()
    ax = plt.axes()
    Vctr = np.linspace(-emax,emax,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(im, ticks=Vcbar)
    ani = animation.FuncAnimation(fig,update_contour_plot,frames=range(nt),
            fargs=(fil,ax,fig,xlim,ylim,zlim,xh,yh,zi,0,emax))
    if savfil:
        ani.save(savfil+'.mp4', writer="ffmpeg", fps=desfps, 
                extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
    #plt.close(fig)
    plt.close('all')

def update_contour_plot(i,fil,ax,fig,xlim,ylim,zlim,x,y,z,meanax,emax):
    fh = mfdset(fil)
    var = fh.variables['e'][i,:,:,:] 
    fh.close()
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(var,var,xlim[0],xlim[-1],
                                                  ylim[0],ylim[-1],zlim[0],zlim[-1],x,y,z,0)
#    X,Y,P,Pmn,Pmx = plotrange(var,xlim[0],xlim[-1],
#                                  ylim[0],ylim[-1],
#                                  zlim[0],zlim[-1],x,y,z,var,meanax)
    ax.cla()
    Vctr = np.linspace(-emax,emax,num=12,endpoint=True)
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    plt.title(str(i))
    #print(i)
    sys.stdout.write('\r'+str(i))
    sys.stdout.flush()
    return im,
