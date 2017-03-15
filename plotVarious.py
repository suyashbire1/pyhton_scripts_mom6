import os
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset
import readParams as rdp
import readParams_moreoptions as rdp1
from manipulateDomain import *
import matplotlib.animation as animation
from getvaratz import *
import xarray as xr

def plotoceanstats(fil,savfil=None):
    ds = xr.open_mfdataset(fil)
    ds['Time'] /= 365

    fig,ax = plt.subplots(4,1,sharex=True,figsize=(9,9))
    ds.max_CFL_trans.plot(ax=ax[0])
    ds.APE.sum('Interface').plot(ax=ax[1])
    ds.KE.sum('Layer').plot(ax=ax[2])
    ds.Ntrunc.plot(ax=ax[3])
    for axc in ax:
        axc.grid()
        axc.get_yaxis().set_label_coords(-0.1,0.5)
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

    time = ds.Time.values
    timedt = time[1:]
    dt = np.diff(time)[:,np.newaxis]
    dt[dt==0] = np.nan
    ape = ds.APE.values
    ape[ape==0] = np.nan
    ke = ds.KE.values
    ke[ke==0] = np.nan
    mass_lay = ds.Mass_lay.values
    mass_lay[mass_lay==0] = np.nan
    dape = np.diff(ape,axis=0)/ape[:-1,:]/dt
    dke = np.diff(ke,axis=0)/ke[:-1,:]/dt
    dm = np.diff(mass_lay,axis=0)/mass_lay[:-1,:]/dt
    layer = ds.Layer.values
    for i in range(layer.size):
        fig,ax = plt.subplots(4,1,sharex=True)
        ax[0].plot(timedt,dape[:,i])
        ax[0].set_ylabel(r'$\frac{1}{APE}\frac{d APE}{dt} (day^{-1})$')
        ax[1].plot(timedt,dke[:,i])
        ax[1].set_ylabel(r'$\frac{1}{KE}\frac{d KE}{dt} (day^{-1})$')
        ax[2].plot(timedt,dm[:,i])
        ax[2].set_ylabel(r'$\frac{1}{M}\frac{d M}{dt} (day^{-1})$')
        ax[3].plot(time,mass_lay[:,i]/layer[i])
        ax[3].set_ylabel(r'Layer volume (m$^3$)')
        ax[3].set_xlabel('Time (years)')
        for axc in ax.ravel():
            axc.get_yaxis().set_label_coords(-0.1,0.5)
            axc.grid()
        if savfil:
            plt.savefig(savfil+'_{}_rates.png'.format(i),facecolor='w', edgecolor='w', 
                    format='png', transparent=False, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def plotoceanstats1(fil,savfil=None):
    ((layer,interface,time), (en,ape,ke), (maxcfltrans,maxcfllin), ntrunc,
    (mass_lay, mass_chg, mass_anom)) = rdp.getoceanstats(fil)
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
    timedt = time[1:]
    dt = np.diff(time)[:,np.newaxis]
    dt[dt==0] = np.nan
    ape[ape==0] = np.nan
    ke[ke==0] = np.nan
    mass_lay[mass_lay==0] = np.nan
    dape = np.diff(ape,axis=0)/ape[:-1,:]/dt
    dke = np.diff(ke,axis=0)/ke[:-1,:]/dt
    dm = np.diff(mass_lay,axis=0)/mass_lay[:-1,:]/dt
    for i in range(layer.size):
        fig,ax = plt.subplots(4,1,sharex=True)
        ax[0].plot(timedt,dape[:,i])
        ax[0].set_ylabel(r'$\frac{1}{APE}\frac{d APE}{dt} (day^{-1})$')
        ax[1].plot(timedt,dke[:,i])
        ax[1].set_ylabel(r'$\frac{1}{KE}\frac{d KE}{dt} (day^{-1})$')
        ax[2].plot(timedt,dm[:,i])
        ax[2].set_ylabel(r'$\frac{1}{M}\frac{d M}{dt} (day^{-1})$')
        ax[3].plot(time,mass_lay[:,i]/layer[i])
        ax[3].set_ylabel(r'Layer volume (m$^3$)')
        ax[3].set_xlabel('Time (years)')
        for axc in ax.ravel():
            axc.get_yaxis().set_label_coords(-0.1,0.5)
            axc.grid()
        if savfil:
            plt.savefig(savfil+'_{}_rates.png'.format(i),facecolor='w', edgecolor='w', 
                    format='png', transparent=False, bbox_inches='tight')
            plt.close(fig)
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

    (t,zl1,yh1,xq1),um = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,ts=0,te=1,xhxq='xq')
    (t,zl1,yq1,xh1),vm = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,ts=0,te=1,yhyq='yq')
    (t,zi1,yh1,xh1),em = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
            zs=zs,ze=ze,ts=0,te=1)
    um /=nt
    vm /=nt
    em /=nt
    elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
    
    for i in range(1,nt):
#        (u,v,h,e,el) = rdp.getuvhe(fil,i)
        (t,zl1,yh1,xq1),u = rdp1.getvar('u',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,ts=i,te=i+1,xhxq='xq')
        (t,zl1,yq1,xh1),v = rdp1.getvar('v',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,ts=i,te=i+1,yhyq='yq')
        (t,zi1,yh1,xh1),e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                zs=zs,ze=ze,ts=i,te=i+1)
        el = 0.5*(e[:,0:-1,:,:]+e[:,1:,:,:])
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
    z = np.linspace(-np.max(D),-1,num=50)
    umatz = getvaratz(um,z,em)
    umax = np.max(np.absolute(np.mean(umatz[0,:,:,:],axis=1)))
    Vctr = np.linspace(-umax,umax,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(xq1, z, np.mean(umatz[0,:,:,:],axis=1),Vctr, cmap=plt.cm.RdBu_r)
    #im2 = ax.contour(xh1,zi1,np.mean(em[0,:,:,:],axis=1),colors='k')
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
   # ax.set_xticks([-70, -50, -30, -10])
    
    ax = plt.subplot(3,2,2)
    vmatz = getvaratz(vm,z,em)
    vmax = np.max(np.absolute(np.mean(vmatz[0,:,:,:],axis=1)))
    Vctr = np.linspace(-vmax,vmax,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(xh1, z, np.mean(vmatz[0,:,:,:],axis=1),Vctr, cmap=plt.cm.RdBu_r)
    #im2 = ax.contour(xh1,zi1,np.mean(em[0,:,:,:],axis=1),colors='k')
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
