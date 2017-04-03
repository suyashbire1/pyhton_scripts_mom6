import readParams_moreoptions as rdp1
from netCDF4 import MFDataset as mfdset, Dataset as dset
import plot_twapv_budget_complete as pv
import plot_twamomy_budget_complete_direct_newest as py
import numpy as np
import pyximport
pyximport.install()
from getvaratzc import getvaratzc, getTatzc
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
importlib.reload(pv)
importlib.reload(py)
from mom_plot1 import m6plot, xdegtokm
import scipy.integrate as sint

def linfit(self,x,y,percx=0,percy=0):
    idx = (np.isfinite(y) 
           & np.isfinite(x) 
           & (y > np.nanpercentile(y,percy)) 
           & (y < np.nanpercentile(y,100-percy))
           & (x > np.nanpercentile(x,percx)) 
           & (x < np.nanpercentile(x,100-percx)))
    fit = np.polyfit(x[idx],y[idx],1)
    fit_fn = np.poly1d(fit)
    yfit = fit_fn(x[idx])
    self.plot(x[idx].ravel(),y[idx].ravel(),'.')
    self.plot(x[idx].ravel(),yfit.ravel(),'k-')
    self.text(0.05,0.95,
            'y = {:.2e}*x + {:.2e}'.format(fit[0],fit[1]),
            transform=self.transAxes)

mpl.axes.Axes.linfit = linfit

def get_heddy_coeffs(geofil,vgeofil,fil,fil2,
                     xstart,xend,ystart,yend,zs=0,ze=None,
                     z=np.linspace(-3000,0,100),percx=0,percy=0):

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
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]

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
    e = (fh2.variables['e'][slpy]*dt).sum(axis=0,keepdims=True)/dt.sum(axis=0,keepdims=True)
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
    ax[0].linfit(vvalid,uv,percx=percx,percy=percy)
    ax[0].set_xlabel('v (m/s)')
    ax[0].set_ylabel('RS div (m/s2)')
    ax[0].grid()

    ax[1].linfit(vxx,uv,percx=percx,percy=percy)
    ax[1].set_xlabel('vxx (1/ms)')
    ax[1].grid()

    return fig, fig2

def get_heddy_coeffs_fromflx(geofil,vgeofil,fil,fil2,
                     xstart,xend,ystart,yend,zs=0,ze=None,
                     z=np.linspace(-3000,0,100),perc=5,htol=1e-3):

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
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]

    sl, dimv = rdp1.getslice(fh,xstart,xend,ystart,yend,yhyq='yq')
    slmx = np.s_[:,:,sl[2],(sl[3].start-1):sl[3].stop]
    slpx = np.s_[:,:,sl[2],(sl[3].start+1):sl[3].stop]
    slmxpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),(sl[3].start-1):sl[3].stop]
    slpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),sl[3]]
    slmpy = np.s_[:,:,(sl[2].start-1):(sl[2].stop+1),sl[3]]
    sl2d = sl[2:]
    slmx2d = slmx[2:]
    slpx2d = slpx[2:]
    slpy2d = slpy[2:]

    dxbu = fhgeo.variables['dxBu'][slmx2d]
    dybu = fhgeo.variables['dyBu'][slmx2d]
    dxcv = fhgeo.variables['dxCv'][sl2d]
    dycv = fhgeo.variables['dyCv'][sl2d]
    dxt = fhgeo.variables['dxT'][slpy2d]
    dyt = fhgeo.variables['dyT'][slpy2d]

    v,_ = pv.getvtwa(fhgeo, fh, fh2, slmx)
    vx = np.diff(v,axis=3)/dxbu
    vx = vx[:,:,:,1:]
    e = (fh2.variables['e'][slpy]*dt).sum(axis=0,keepdims=True)/dt.sum(axis=0,keepdims=True)
    eatv = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])

    e = fh2.variables['e'][slmxpy]
    eatvmx = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])

    vh = (fh.variables['vh_masked'][sl]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    h_cv = (fh.variables['h_Cv'][sl]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    h_cv[h_cv < htol] = np.nan
    sig = h_cv/dbl[:,np.newaxis,np.newaxis]
    h_vm = h_cv
    vtwa = vh/h_cv/dxcv
    vhforxdiff = (fh.variables['vh_masked'][slmx]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    h_cvforxdiff = (fh.variables['h_Cv'][slmx]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    h_cvforxdiff[h_cvforxdiff < htol] = np.nan
    vtwaforxdiff = vhforxdiff/h_cvforxdiff
    vtwaforxdiff = np.concatenate((vtwaforxdiff,vtwaforxdiff[:,:,:,-1:]),axis=3)
    vtwax = np.diff(vtwaforxdiff,axis=3)/dxbu/dybu
    vtwax = 0.5*(vtwax[:,:,:,:-1] + vtwax[:,:,:,1:])
    uh = (fh.variables['uh_masked'][slmxpy]*dt).filled(0).sum(axis=0,keepdims=True)/np.sum(dt)
    hum = 0.25*(uh[:,:,:-1,:-1] + uh[:,:,:-1,1:] + uh[:,:,1:,:-1] +
            uh[:,:,1:,1:])/dycv
    huvxphvvym = (fh.variables['twa_huvxpt'][sl]*dt +
            fh.variables['twa_hvvymt'][sl]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    hvv = (fh.variables['hvv_Cv'][slmpy]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    hvvym = np.diff(hvv,axis=2)/dxt/dyt
    hvvym = 0.5*(hvvym[:,:,:-1,:] + hvvym[:,:,1:,:])
    huvxm = -(huvxphvvym + hvvym)
    advx = hum*vtwax/h_vm
    humx = np.diff(np.nan_to_num(uh),axis=3)/dxt/dyt
    humx = 0.5*(humx[:,:,:-1,:] + humx[:,:,1:,:])
    xdivep1 = -huvxm/h_vm
    xdivep2 = advx
    xdivep3 = vtwa*humx/h_vm 
    xdivep = (xdivep1 + xdivep2 + xdivep3)

    xdivep *= sig 
    uv = sint.cumtrapz(xdivep[:,:,:,::-1],dx=-dxbu[:,1:-1], axis=3,
            initial=0)[:,:,:,::-1]/sig
    
    uvm = np.apply_over_axes(np.nanmean,uv,(0,2))
    vxm = np.apply_over_axes(np.nanmean,vx,(0,2))
    uvm = getvaratzc(uvm.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))
    vxm = getvaratzc(vxm.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12,3))
    cmax = np.fabs(np.percentile(uvm,(1,99))).max()
    im = ax[0].pcolormesh(dimv[3],z,uvm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    cmax = np.fabs(np.percentile(vxm,(1,99))).max()
    im = ax[1].pcolormesh(dimv[3],z,vxm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    for axc in ax:
        xdegtokm(axc,0.5*(ystart+yend))
        axc.set_xlabel('x (km)')
        axc.grid()
    ax[0].set_ylabel('z (m)')

    fig2,ax = plt.subplots(1,1,sharey=True)
    ax.linfit(vx,uv)
    ax.set_xlabel('v_x (m/s)')
    ax.set_ylabel('RS (m2/s2)')
    ax.grid()

    return fig, fig2


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
    f = fhgeo.variables['f'][sl2d]

    e = fh2.variables['e'][slpy]
    eatv = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])
    v,h = pv.getvtwa(fhgeo, fh, fh2, sl)
    v = v[:,:,:,:-1]
    h = h[:,:,:,:-1]
    sig = h/dbl[:,np.newaxis,np.newaxis]
    sigaz = getvaratzc(sig.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

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

    fig2,ax = plt.subplots(1,3,sharey=True,figsize=(12,3))
    ax[0].linfit(vbb,sigaz*fdrag/f**2)
    ax[0].set_xlabel(r'$v_{bb}$')
    ax[1].linfit(svibb,sigaz*fdrag/f**2)
    ax[1].set_xlabel(r'$(\sigma v_b)_b$')
    ax[2].linfit(svbb,sigaz*fdrag/f**2)
    ax[2].set_xlabel(r'$(\sigma v)_{bb}$')

    ax[0].set_ylabel('Form Drag')
    for axc in ax:
        axc.grid()

    return fig, fig2

def get_deddy_coeffs_fromflx(geofil,vgeofil,fil,fil2,
                     xstart,xend,ystart,yend,zs=0,ze=None,
                     zlim=(-1500,0),percx=0,percy=0,
                     nsqdep=False,fil3=None,swashperc=1):

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
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]
    z = np.linspace(zlim[0],zlim[1],100)

    sl, dimv = rdp1.getslice(fh,xstart,xend,ystart,yend,yhyq='yq')
    sl2d = sl[2:]
    dycv = fhgeo.variables['dyCv'][sl2d]
    slpy = np.s_[:,:,sl[2].start:(sl[2].stop+1),sl[3]]
    f = fhgeo.variables['f'][sl2d]


    e = (fh2.variables['e'][slpy]*dt).sum(axis=0,keepdims=True)/dt.sum(axis=0)
    eatv = 0.5*(e[:,:,:-1,:] + e[:,:,1:,:])
    v,h = pv.getvtwa(fhgeo, fh, fh2, sl)
    v = v[:,:,:,:-1]
    h = h[:,:,:,:-1]
    sig = h/dbl[:,np.newaxis,np.newaxis]
    hi = 0.5*(h[:,:-1]+h[:,1:])
    sigi = hi/dbi[1:-1,np.newaxis,np.newaxis]

    vb = -np.diff(v,axis=1)/dbi[1:-1,np.newaxis,np.newaxis]
    #vb = np.concatenate((vb[:,:1,:,:],vb,vb[:,-1:,:,:]),axis=1)

    vi = np.concatenate((v[:,:1,:,:],v,-v[:,-1:,:,:]),axis=1)
    vi = 0.5*(vi[:,:-1,:,:] + vi[:,1:,:,:])
    vib = -np.diff(vi,axis=1)/dbl[:,np.newaxis,np.newaxis]
    if nsqdep:
        fsqvb = f**2*vb*sigi
    else:
        fsqvb = f**2*vb

    fsqvbaz = getvaratzc(fsqvb.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    esq = (fh.variables['esq'][slpy]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    elmforydiff = 0.5*(e[:,0:-1,:,:]+e[:,1:,:,:])
    edlsqm = (esq - elmforydiff**2)
    edlsqmy = np.diff(edlsqm,axis=2)/dycv

    hpfv = (fh.variables['twa_hpfv'][sl]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    pfvm = (fh2.variables['PFv'][sl]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    edpfvdmb = -(-hpfv + h*pfvm - 0.5*edlsqmy*dbl[:,np.newaxis,np.newaxis])

    fh.close()
    fh2.close()
    fhgeo.close()

    zdpfvd = sint.cumtrapz(edpfvdmb,dx=-dbl[0], axis=1)
#    zdpfvd = np.concatenate((np.zeros(zdpfvd[:,:1,:,:].shape),
#                             zdpfvd,
#                             np.zeros(zdpfvd[:,:1,:,:].shape)),axis=1)
#    zdpfvd = 0.5*(zdpfvd[:,:-1,:,:]+zdpfvd[:,1:,:,:])
    zdpfvdaz = getvaratzc(zdpfvd.astype(np.float32),
                     z.astype(np.float32),
                     eatv.astype(np.float32))

    fsqvbazm = np.apply_over_axes(np.nanmean,fsqvbaz,(0,2))
    zdpfvdazm = np.apply_over_axes(np.nanmean,zdpfvdaz,(0,2))

    fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12,3))
    cmax = np.fabs(np.percentile(zdpfvdazm,(1,99))).max()
    im = ax[0].pcolormesh(dimv[3],z,zdpfvdazm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[0])

    cmax = np.fabs(np.percentile(fsqvbazm,(1,99))).max()
    im = ax[1].pcolormesh(dimv[3],z,fsqvbazm.squeeze(),
                          vmax=cmax, vmin=-cmax, cmap='RdBu_r')
    fig.colorbar(im,ax=ax[1])

    if fil3:
        fh3 = mfdset(fil3)
        slmxtn = np.s_[-1:,sl[1],sl[2],(sl[3].start-1):sl[3].stop]
        islayerdeep0 = fh3.variables['islayerdeep'][-1:,0,0,0]
        islayerdeep = fh3.variables['islayerdeep'][slmxtn]
        if np.ma.is_masked(islayerdeep):
            islayerdeep = islayerdeep.filled(np.nan)
            islayerdeep[:,:,:,-1:] = islayerdeep[:,:,:,-2:-1]

        swash = (islayerdeep0 - islayerdeep)/islayerdeep0*100
        swash = 0.5*(swash[:,:,:,:-1] + swash[:,:,:,1:])
        fh3.close()
        swash = getvaratzc(swash.astype(np.float32),
                             z.astype(np.float32),
                             eatv.astype(np.float32))
        swash = np.apply_over_axes(np.nanmean,swash,(0,2))
        em = np.apply_over_axes(np.nanmean,e,(0,2))
        xx,zz = np.meshgrid(dimv[3],zi)
        for axc in ax:
            axc.contour(dimv[3],z,swash.squeeze(),np.array([swashperc]),
                    colors='k')
            axc.contour(xx,em.squeeze(),zz,ls='k--')
            axc.set_ylim(zlim[0],zlim[1])

    fig2,ax = plt.subplots(1,1,sharey=True,figsize=(12,3))
    ax.linfit(fsqvbaz,zdpfvdaz,percx=percx,percy=percy)
    ax.set_xlabel(r'$f^2 v_b$')
    ax.set_ylabel(r'$\zeta^{\prime} m_y^{\prime}$')
    return fig,fig2
