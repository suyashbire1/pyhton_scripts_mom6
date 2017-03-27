import readParams_moreoptions as rdp1
from netCDF4 import MFDataset as mfdset, Dataset as dset
import plot_twapv_budget_complete as pv
import plot_twamomy_budget_complete_direct_newest as py
import numpy as np
import pyximport
pyximport.install()
from getvaratzc import getvaratzc, getTatzc
import matplotlib.pyplot as plt
import importlib
importlib.reload(pv)
importlib.reload(py)
from mom_plot1 import m6plot, xdegtokm

def linfit(x,y):
    idx = np.isfinite(y) & np.isfinite(x)
    fit = np.polyfit(x[idx],y[idx],1)
    fit_fn = np.poly1d(fit)
    yfit = fit_fn(x)
    return fit, yfit

def get_heddy_coeffs(geofil,vgeofil,fil,fil2,
                     xstart,xend,ystart,yend,zs=0,ze=None,
                     z=np.linspace(-3000,0,100),perc=5):

    fhgeo = dset(geofil)
    fhvgeo = dset(vgeofil)
    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    db = fhvgeo.variables['g'][:]
    fhvgeo.close()
    dbi = np.append(db,0)
    zi = fh.variables['zi'][:]
    zl = fh.variables['zl'][:]
    dbl = np.diff(zi)*9.8/1031

    sl, dimv = rdp1.getslice(fh,xstart,xend,ystart,yend,yhyq='yq')
    slmx = np.s_[:,:,sl[2],(sl[3].start-1):sl[3].stop]
    slpx = np.s_[:,:,sl[2],(sl[3].start+1):sl[3].stop]
    slmxpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),(sl[3].start-1):sl[3].stop]
    slpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),sl[3]]
    sl2d = sl[2:]
    slmx2d = slmx[2:]
    slpx2d = slpx[2:]

    dxbu = fhgeo.variables['dxBu'][slmx2d]
    dxcv = fhgeo.variables['dxCv'][sl2d]

    v,_ = pv.getvtwa(fhgeo, fh, fh2, slmx)
    vx = np.diff(v,axis=3)/dxbu
    vxx = np.diff(vx,axis=3)/dxcv
    e = fh2.variables['e'][slpy]
    eatv = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])
    vxx = getvaratzc(vxx.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    e = fh2.variables['e'][slmxpy]
    eatvmx = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])
    vaz = getvaratzc(v.astype(np.float32),
                     z.astype(np.float32),
                     eatvmx.astype(np.float32))
    vazx = np.diff(vaz,axis=3)/dxbu
    vazxx = np.diff(vazx,axis=3)/dxcv


    x,y,P,_,_ = py.extract_twamomy_terms(geofil,vgeofil,fil,fil2,
                                         xstart,xend,ystart,yend,
                                         zs,ze,(0,))
    uv = P[:,:,:,:,5]

    vvalid = vaz[:,:,:,1:-1]
    idxv = np.isfinite(uv) & np.isfinite(vvalid)
    fitv = np.polyfit(vvalid[idxv],uv[idxv],1)
    fitv_fn = np.poly1d(fitv)

    idxvperc = (np.isfinite(uv) 
            & np.isfinite(vvalid) 
            & (uv > np.nanpercentile(uv,perc)) 
            & (uv < np.nanpercentile(uv,100-perc))
            & (vvalid > np.nanpercentile(vvalid,perc)) 
            & (vvalid < np.nanpercentile(vvalid,100-perc)))
    fitvperc = np.polyfit(vvalid[idxvperc],uv[idxvperc],1)
    fitvperc_fn = np.poly1d(fitv)

    idxvxx = np.isfinite(uv) & np.isfinite(vxx)
    fitvxx = np.polyfit(vxx[idxvxx],uv[idxvxx],1)
    fitvxx_fn = np.poly1d(fitv)

    idxvxxperc = (np.isfinite(uv) 
            & np.isfinite(vxx) 
            & (uv > np.nanpercentile(uv,perc)) 
            & (uv < np.nanpercentile(uv,100-perc))
            & (vxx > np.nanpercentile(vxx,perc)) 
            & (vxx < np.nanpercentile(vxx,100-perc)))
    fitvxxperc = np.polyfit(vxx[idxvxxperc],uv[idxvxxperc],1)
    fitvxxperc_fn = np.poly1d(fitv)

    vazm = np.apply_over_axes(np.nanmean,vvalid,(0,2))
    vxxm = np.apply_over_axes(np.nanmean,vxx,(0,2))
    vazxxm = np.apply_over_axes(np.nanmean,vazxx,(0,2))
    uvm = np.apply_over_axes(np.nanmean,uv,(0,2))

    fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(12,3))
    cmax = np.fabs(np.percentile(vxxm,(1,99))).max()
    im = ax[0].pcolormesh(dimv[3],z,vxxm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[0])
    cmax = np.fabs(np.percentile(vazxxm,(1,99))).max()
    im = ax[1].pcolormesh(dimv[3],z,vazxxm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[1])
    cmax = np.fabs(np.percentile(vvalid,(0.5,99.5))).max()
    im = ax[2].pcolormesh(dimv[3],z,vazm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[2])
    cmax = np.fabs(np.percentile(uvm,(0.5,99.5))).max()
    im = ax[3].pcolormesh(dimv[3],z,uvm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    cb = fig.colorbar(im,ax=ax[3])
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks() 
    for axc in ax:
        xdegtokm(axc,0.5*(ystart+yend))
        axc.set_xlabel('x (km)')
        axc.grid()
    ax[0].set_ylabel('z (m)')

    fig2,ax = plt.subplots(1,2,sharey=True)
    ax[0].plot(vvalid[idxv],uv[idxv],'.')
    ax[0].plot(vvalid[idxv],fitv_fn(vvalid[idxv]),'k-')
    ax[0].set_xlabel('v (m/s)')
    ax[0].set_ylabel('duvdx (m2/s2)')
    ax[0].grid()
    ax[0].text(0.05,0.95,'{:.2e}*v + {:.2e}'.format(fitv[0],fitv[1]),transform=ax[0].transAxes)
    ax[1].plot(vxx[idxvxx],uv[idxvxx],'.')
    ax[1].plot(vxx[idxvxx],fitvxx_fn(vxx[idxvxx]),'k-')
    ax[1].set_xlabel('vxx (1/ms)')
    ax[1].grid()
    ax[1].text(0.05,0.95,'{:5.2f}*vxx + {:.2e}'.format(fitvxx[0],fitvxx[1]),transform=ax[1].transAxes)

    fig3,ax = plt.subplots(1,2,sharey=True)
    ax[0].plot(vvalid[idxvperc],uv[idxvperc],'.')
    ax[0].plot(vvalid[idxvperc],fitvperc_fn(vvalid[idxvperc]),'k-')
    ax[0].set_xlabel('v (m/s)')
    ax[0].set_ylabel('duvdx (m2/s2)')
    ax[0].grid()
    ax[0].text(0.05,0.95,'{:.2e}*v + {:.2e}'.format(fitvperc[0],fitvperc[1]),transform=ax[0].transAxes)
    ax[1].plot(vxx[idxvxxperc],uv[idxvxxperc],'.')
    ax[1].plot(vxx[idxvxxperc],fitvxxperc_fn(vxx[idxvxxperc]),'k-')
    ax[1].set_xlabel('vxx (1/ms)')
    ax[1].grid()
    ax[1].text(0.05,0.95,'{:5.2f}*vxx + {:.2e}'.format(fitvxxperc[0],fitvxxperc[1]),transform=ax[1].transAxes)
    return fig, fig2, fig3

def get_deddy_coeffs(geofil,vgeofil,fil,fil2,
                     xstart,xend,ystart,yend,zs=0,ze=None,
                     z=np.linspace(-1500,0,100),perc=5):

    fhgeo = dset(geofil)
    fhvgeo = dset(vgeofil)
    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    db = fhvgeo.variables['g'][:]
    fhvgeo.close()
    dbi = np.append(db,0)
    zi = fh.variables['zi'][:]
    zl = fh.variables['zl'][:]
    dbl = np.diff(zi)*9.8/1031

    sl, dimv = rdp1.getslice(fh,xstart,xend,ystart,yend,yhyq='yq')
    sl2d = sl[2:]
    slpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),sl[3]]

    e = fh2.variables['e'][slpy]
    eatv = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])
    v,h = pv.getvtwa(fhgeo, fh, fh2, sl)
    v = v[:,:,:,:-1]
    h = h[:,:,:,:-1]
    sig = h/dbl[:,np.newaxis,np.newaxis]

    vb = -np.diff(v,axis=1)/dbi[1:-1,np.newaxis,np.newaxis]
    vb = np.concatenate((vb[:,:1,:,:],vb,vb[:,-1:,:,:]),axis=1)
    vbb = -np.diff(vb,axis=1)/dbl[:,np.newaxis,np.newaxis]
    vbb = getvaratzc(vbb.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    vi = np.concatenate((v[:,:1,:,:],v,-v[:,-1:,:,:]),axis=1)
    vi = 0.5*(vi[:,:-1,:,:] + vi[:,1:,:,:])
    vib = -np.diff(vi,axis=1)/dbl[:,np.newaxis,np.newaxis]
    svib = sig*vib
    svibb = -np.diff(svib,axis=1)/dbi[1:-1,np.newaxis,np.newaxis]
    svibb = np.concatenate((svibb[:,:1,:,:],svibb,svibb[:,-1:,:,:]),axis=1)
    svibb = 0.5*(svibb[:,:-1,:,:] + svibb[:,1:,:,:])
    svibb = getvaratzc(svibb.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    sv = sig*v
    svb = -np.diff(sv,axis=1)/dbi[1:-1,np.newaxis,np.newaxis]
    svb = np.concatenate((svb[:,:1,:,:],svb,svb[:,-1:,:,:]),axis=1)
    svbb = -np.diff(svb,axis=1)/dbl[:,np.newaxis,np.newaxis]
    svbb = getvaratzc(svbb.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    x,y,P,_,_ = py.extract_twamomy_terms(geofil,vgeofil,fil,fil2,
                                         xstart,xend,ystart,yend,
                                         zs,ze,(0,),z=z)
    fdrag = P[:,:,:,:,7]

    idx = np.isfinite(fdrag) & np.isfinite(vbb)
    fit = np.polyfit(vbb[idx],fdrag[idx],1)
    fit_fn = np.poly1d(fit)

    idxsvibb = np.isfinite(fdrag) & np.isfinite(svibb)
    fitsvibb = np.polyfit(svibb[idxsvibb],fdrag[idxsvibb],1)
    fitsvibb_fn = np.poly1d(fitsvibb)

    idxsvbb = np.isfinite(fdrag) & np.isfinite(svbb)
    fitsvbb = np.polyfit(vbb[idxsvbb],fdrag[idxsvbb],1)
    fitsvbb_fn = np.poly1d(fitsvbb)

    vbbm = np.apply_over_axes(np.nanmean,vbb,(0,2))
    svibbm = np.apply_over_axes(np.nanmean,svibb,(0,2))
    svbbm = np.apply_over_axes(np.nanmean,svbb,(0,2))
    fdragm = np.apply_over_axes(np.nanmean,fdrag,(0,2))

    fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(12,3))
    cmax = np.fabs(np.percentile(vbbm,(1,99))).max()
    im = ax[0].pcolormesh(dimv[3],z,vbbm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[0])

    cmax = np.fabs(np.percentile(svibbm,(1,99))).max()
    im = ax[1].pcolormesh(dimv[3],z,svibbm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[1])

    cmax = np.fabs(np.percentile(svbbm,(1,99))).max()
    im = ax[2].pcolormesh(dimv[3],z,svbbm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[2])

    cmax = np.fabs(np.percentile(fdragm,(1,99))).max()
    im = ax[3].pcolormesh(dimv[3],z,fdragm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[3])

    fig2,ax = plt.subplots(1,3,sharey=True)
    ax[0].plot(vbb.ravel(),fdrag.ravel(),'.')
    ax[0].set_xlabel(r'$v_{bb}$')
    ax[1].plot(svibb.ravel(),fdrag.ravel(),'.')
    ax[1].set_xlabel(r'$(\sigma v_b)_b$')
    ax[2].plot(svbb.ravel(),fdrag.ravel(),'.')
    ax[2].set_xlabel(r'$(\sigma v)_{bb}$')

    ax[0].set_ylabel('Form Drag')
    for axc in ax:
        axc.grid()

    return fig, fig2

def get_veddy_coeffs(geofil,vgeofil,fil,fil2,
                     xstart,xend,ystart,yend,zs=0,ze=None,
                     z=np.linspace(-1500,0,100),perc=5):

    fhgeo = dset(geofil)
    fhvgeo = dset(vgeofil)
    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    db = fhvgeo.variables['g'][:]
    fhvgeo.close()
    dbi = np.append(db,0)
    zi = fh.variables['zi'][:]
    zl = fh.variables['zl'][:]
    dbl = np.diff(zi)*9.8/1031
    dz = np.diff(z)[:1]

    sl, dimv = rdp1.getslice(fh,xstart,xend,ystart,yend,yhyq='yq')
    sl2d = sl[2:]
    slpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),sl[3]]

    e = fh2.variables['e'][slpy]
    eatv = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])
    v,h = pv.getvtwa(fhgeo, fh, fh2, sl)
    v = v[:,:,:,:-1]
    h = h[:,:,:,:-1]
    sig = h/dbl[:,np.newaxis,np.newaxis]
    sig = getvaratzc(sig.astype(np.float32),
                   z.astype(np.float32),
                   eatv.astype(np.float32))

    v = getvaratzc(v.astype(np.float32),
                   z.astype(np.float32),
                   eatv.astype(np.float32))
    vz = -np.diff(v,axis=1)/dz
    vzz = -np.diff(vz,axis=1)/dz

    svz = 0.5*(sig[:,:-1,:,:]+sig[:,1:,:,:])*vz
    svzz = -np.diff(svz,axis=1)/dz

    sv = sig*v
    svz1 = -np.diff(sv,axis=1)/dz
    svzz1 = -np.diff(svz1,axis=1)/dz

    x,y,P,_,_ = py.extract_twamomy_terms(geofil,vgeofil,fil,fil2,
                                         xstart,xend,ystart,yend,
                                         zs,ze,(0,),z=z)
    fdrag = P[:,:,:,:,7]

    fit, vzzfit = linfit(vzz,fdrag[:,1:-1,:,:])
    fitsvzz, svzzfit = linfit(svzz,fdrag[:,1:-1,:,:])
    fitsvzz1, svzz1fit = linfit(svzz1,fdrag[:,1:-1,:,:])

    vzzm = np.apply_over_axes(np.nanmean,vzz,(0,2))
    svzzm = np.apply_over_axes(np.nanmean,svzz,(0,2))
    svzz1m = np.apply_over_axes(np.nanmean,svzz1,(0,2))
    fdragm = np.apply_over_axes(np.nanmean,fdrag,(0,2))

    fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(12,3))
    cmax = np.fabs(np.percentile(vzzm,(1,99))).max()
    im = ax[0].pcolormesh(dimv[3],z[1:-1],vzzm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[0])

    cmax = np.fabs(np.percentile(svzzm,(1,99))).max()
    im = ax[1].pcolormesh(dimv[3],z[1:-1],svzzm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[1])

    cmax = np.fabs(np.percentile(svzz1m,(1,99))).max()
    im = ax[2].pcolormesh(dimv[3],z[1:-1],svzz1m.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[2])

    cmax = np.fabs(np.percentile(fdragm,(1,99))).max()
    im = ax[3].pcolormesh(dimv[3],z,fdragm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[3])

    fig2,ax = plt.subplots(1,3,sharey=True, figsize=(12,3))
    ax[0].plot(vzz.ravel(),fdrag[:,1:-1].ravel(),'.')
    ax[0].plot(vzz.ravel(),vzzfit.ravel(),'k-')
    ax[0].text(0.05,0.95,'{:.2e}*x + {:.2e}'.format(fit[0],fit[1]),transform=ax[0].transAxes)
    ax[0].set_xlabel(r'$v_{zz}$')
    ax[1].plot(svzz.ravel(),fdrag[:,1:-1].ravel(),'.')
    ax[1].plot(svzz.ravel(),svzzfit.ravel(),'k-')
    ax[1].set_xlabel(r'$(\sigma v_z)_z$')
    ax[1].text(0.05,0.95,r'{:.2e}*x + {:.2e}'.format(fitsvzz[0],fitsvzz[1]),transform=ax[1].transAxes)
    ax[2].plot(svzz1.ravel(),fdrag[:,1:-1].ravel(),'.')
    ax[2].plot(svzz1.ravel(),svzz1fit.ravel(),'k-')
    ax[2].set_xlabel(r'$(\sigma v)_{zz}$')
    ax[2].text(0.05,0.95,r'{:.2e}*x + {:.2e}'.format(fitsvzz1[0],fitsvzz1[1]),transform=ax[2].transAxes)

    for axc in ax:
        axc.grid()

    return fig, fig2
