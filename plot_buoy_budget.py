import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot,xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc, getTatzc, getTatzc2
from pym6 import Domain, Variable, Plotter
import importlib
importlib.reload(Domain)
importlib.reload(Variable)
importlib.reload(Plotter)
gv = Variable.GridVariable

def extract_buoy_terms_pym6(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,ls,le,
      fil3=None, z=np.linspace(-3000,0,100),htol=1e-3,whichterms=None):

    domain = Domain.Domain(geofil,vgeofil,xstart,xend,ystart,yend,ls=ls,le=le) 

    with mfdset(fil) as fh, mfdset(fil2) as fh2:

        h = -gv('e',domain,'hi',fh2,units='m').read_array().o1diff(1).values
        h[h<htol] = np.nan
        sigma = gv('e',domain,'hi',fh2).read_array().ddx(1).values
        
        ex = gv('e',domain,'hi',fh2).xsm().xep().read_array(extend_kwargs={'method':'mirror'}).ddx(3).move_to('ul')
        u = gv('uh',domain,'ul',fh2,fh,plot_loc='hl',divisor='h_Cu').xsm().read_array(divide_by_dy=True)
        uex = u*ex
        uex = uex.move_to('hl')#.values
        ubx = -uex*(1/sigma)

        ey = gv('e',domain,'hi',fh2).ysm().yep().read_array().ddx(2).move_to('vl')
        v = gv('vh',domain,'vl',fh2,fh,plot_loc='hl',divisor='h_Cv').ysm().read_array(divide_by_dx=True)
        vey = v*ey
        vey = vey.move_to('hl')#.values
        vby = -vey*(1/sigma)

        wd = gv('wd',domain,'hi',fh2).read_array().move_to('hl')#.values
        whash = wd + uex + vey
        wbz = whash*(1/sigma)

        hwb = -gv('wd',domain,'hi',fh2).read_array().o1diff(1)*domain.db
        wtwa = hwb*(1/h)

        budgetlist =  [-ubx,-vby,-wbz,-wtwa]
        name = [r'$-\hat{u}b^{\#}_x$',r'$-\hat{v}b^{\#}_y$',r'$-w^{\#}b^{\#}_z$',r'$\hat{\varpi}$']
        if whichterms:
            budgetlist = budgetlist[whichterms]
            name = name[whichterms]
        for i,var in enumerate(budgetlist):
            var.units = r'$ms^{-3}$'
            var.name = name[i]
        z = np.linspace(-2000,0)
        e = gv('e',domain,'hi',fh2).read_array()
        plot_kwargs = dict(cmap='RdBu_r')
        plotter_kwargs = dict(zcoord=True,z=z,e=e,isop_mean=True)

        fig = Plotter.budget_plot(budgetlist,(0,2),plot_kwargs=plot_kwargs,
                plotter_kwargs=plotter_kwargs,perc=90,individual_cbars=False)
        return fig

def getvaravg(fh,varstr,sl):
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]
    var = (fh.variables[varstr][sl]*dt)
    var = np.apply_over_axes(np.sum,var,0)/np.sum(dt)
    return var

def getutwa(fhgeo,fh,fh2,sl,htol=1e-3):
    uh = getvaravg(fh2,'uh',sl)
    h_cu = getvaravg(fh,'h_Cu',sl)
    h_cu = np.ma.masked_array(h_cu,mask=(h_cu<htol))
    dycu = fhgeo.variables['dyCu'][sl[2:]]
    utwa = uh/h_cu/dycu
    return utwa.filled(np.nan)

def getvtwa(fhgeo,fh,fh2,sl,htol=1e-3):
    vh = getvaravg(fh2,'vh',sl)
    h_cv = getvaravg(fh,'h_Cv',sl)
    h_cv = np.ma.masked_array(h_cv,mask=(h_cv<htol))
    dxcv = fhgeo.variables['dxCv'][sl[2:]]
    vtwa = vh/dxcv/h_cv
    return vtwa.filled(np.nan)

def getwhash(fhgeo,vgeofil,fh,fh2,sl,htol=1e-3):
    fhvgeo = dset(vgeofil)
    db = -fhvgeo.variables['g'][:]
    dbi = np.append(db,0)
    fhvgeo.close()
    zi = rdp1.getdims(fh)[2][0]
    dbl = np.diff(zi)*9.8/1031

    zs,ze = sl[1].start, sl[1].stop
    ys,ye = sl[2].start, sl[2].stop
    xs,xe = sl[3].start, sl[3].stop
    slmx = np.s_[:,zs:ze,ys:ye,xs-1:xe]
    slmy = np.s_[:,zs:ze,ys-1:ye,xs:xe]
    slmpy = np.s_[:,zs:ze,ys-1:ye+1,xs:xe]
    
    emforxdiff = getvaravg(fh2,'e',slmx)
    elmforxdiff = 0.5*(emforxdiff[:,0:-1,:,:]+emforxdiff[:,1:,:,:])
    elmforxdiff = np.concatenate((elmforxdiff,elmforxdiff[:,:,:,-1:]),axis=3)
    dxcuforxdiff = fhgeo.variables['dxCu'][slmx[2:]]
    ex = np.diff(elmforxdiff,axis=3)/dxcuforxdiff

    uh = getvaravg(fh2,'uh',slmx)
    h_cu = getvaravg(fh,'h_Cu',slmx)
    h_cu = np.ma.masked_array(h_cu,mask=(h_cu<htol))
    dycu = fhgeo.variables['dyCu'][slmx[2:]]
    uzx = (uh/h_cu/dycu).filled(0)*ex
    uzx = 0.5*(uzx[:,:,:,1:]+uzx[:,:,:,:-1])

    emforydiff = getvaravg(fh2,'e',slmpy)
    elmforydiff = 0.5*(emforydiff[:,0:-1,:,:]+emforydiff[:,1:,:,:])
    dycv = fhgeo.variables['dyCv'][slmy[2:]]
    ey = np.diff(elmforydiff,axis=2)/dycv

    vh = getvaravg(fh2,'vh',slmy)
    h_cv = getvaravg(fh,'h_Cv',slmy)
    h_cv = np.ma.masked_array(h_cv,mask=(h_cv<htol))
    dxcv = fhgeo.variables['dxCv'][slmy[2:]]
    vtwa = vh/dxcv/h_cv
    vzy = vtwa*ey
    vzy = 0.5*(vzy[:,:,1:,:]+vzy[:,:,:-1,:])

    wd = getvaravg(fh2,'wd',sl)
    hw = wd*dbi[:,np.newaxis,np.newaxis]
    hw = 0.5*(hw[:,1:,:,:]+hw[:,:-1,:,:])
    wzb = hw/dbl[:,np.newaxis,np.newaxis]
    whash = uzx + vzy + wzb
    return whash.filled(np.nan)

def extract_buoy_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,
      fil3=None, z=np.linspace(-3000,0,100),htol=1e-3):


    fhgeo = dset(geofil)
    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    dz = np.diff(z)[0]

    (xs,xe), (ys,ye), dimh = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                                            slat=ystart,nlat=yend, 
                                            zs=zs,ze=ze,ts=0,te=None)
    sl = np.s_[:,zs:ze,ys:ye,xs:xe]
    sl2d = sl[2:]
    slmx = np.s_[:,zs:ze,ys:ye,xs-1:xe]
    slmx2d = slmx[2:]
    slmpy = np.s_[:,zs:ze,ys-1:ye+1,xs:xe] 
    slmpy2d = slmpy[2:]
    slmy = np.s_[:,zs:ze,ys-1:ye,xs:xe] 
    slmy2d = slmy[2:]

    dxcu = fhgeo.variables['dxCu'][slmx2d]
    dycv = fhgeo.variables['dyCv'][slmy2d]

    e = getvaravg(fh2,'e',sl)
    h = -np.diff(e,axis=1)
    h = np.ma.masked_array(h,mask=(h<htol)).filled(np.nan)
    zi = fh.variables['zi'][:]
    bi = 9.8*(1-zi/1031)
    dbl = np.diff(zi)*9.8/1031
    bz = -(np.diff(bi)[:,np.newaxis,np.newaxis]/h)
    bz = getvaratzc(bz.astype(np.float32),
                 z.astype(np.float32),
                 e.astype(np.float32))
    whash = getwhash(fhgeo,vgeofil,fh,fh2,sl,htol=htol)
    whash = getvaratzc(whash.astype(np.float32),
                       z.astype(np.float32),
                       e.astype(np.float32))
    wbz = whash*bz

    efxd = getvaravg(fh2,'e',slmx)
    bfxd = getTatzc2(bi.astype(np.float32),
                    z.astype(np.float32),
                    efxd.astype(np.float32))
    bfxd = np.concatenate((bfxd,bfxd[:,:,:,-1:]),axis=3)
    bx = np.diff(bfxd,axis=3)/dxcu
    u = getutwa(fhgeo,fh,fh2,slmx,htol=htol)
    e_cu = np.concatenate((efxd,efxd[:,:,:,-1:]),axis=3)
    e_cu = 0.5*(e_cu[:,:,:,:-1]+e_cu[:,:,:,1:])
    u = getvaratzc(u.astype(np.float32),
                   z.astype(np.float32),
                   e_cu.astype(np.float32))
    ubx = u*bx
    ubx = 0.5*(ubx[:,:,:,:-1]+ubx[:,:,:,1:])

    efyd = getvaravg(fh2,'e',slmpy)
    bfyd = getTatzc2(bi.astype(np.float32),
                    z.astype(np.float32),
                    efyd.astype(np.float32))
    by = np.diff(bfyd,axis=2)/dycv
    e_cv = 0.5*(efyd[:,:,:-1,:]+efyd[:,:,1:,:])
    v = getvtwa(fhgeo,fh,fh2,slmy,htol=htol)
    v = getvaratzc(v.astype(np.float32),
                   z.astype(np.float32),
                   e_cv.astype(np.float32))
    vby = v*by
    vby = 0.5*(vby[:,:,:-1,:]+vby[:,:,1:,:])

    hwb = getvaravg(fh2,'wd',sl)
    hwb = -np.diff(hwb,axis=1)
    hwm = hwb*dbl[:,np.newaxis,np.newaxis]
    wtwa = hwm/h
    wtwa = getvaratzc(wtwa.astype(np.float32),
                   z.astype(np.float32),
                   e.astype(np.float32))

    terms = np.ma.concatenate((ubx[:,:,:,:,np.newaxis],
                            vby[:,:,:,:,np.newaxis],
                            wbz[:,:,:,:,np.newaxis],
                           -wtwa[:,:,:,:,np.newaxis]),axis=4)
    return dimh, terms
    
def plot_buoy_budget(geofil,vgeofil,fil,fil2,xstart,xend,
        ystart,yend,zs,ze,meanax,minperc = 0,savfil=None,
        z=np.linspace(-3000,0,100),htol=1e-3,fil3=None):

    dims,P = extract_buoy_terms(geofil,vgeofil,fil,fil2,xstart,xend,
            ystart,yend,zs,ze,z=z,htol=htol,fil3=fil3)

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fig,ax = plt.subplots(int(np.ceil(P.shape[4]/2)),2,sharex=True,sharey=True,
                          figsize=(6,4))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [r'$\widehat{u}b^{\#}_{x}$',
           r'$\widehat{v}b^{\#}_{y}$',
           r'$w^{\#}b^{\#}_{z}$',
           r'$\widehat{\varpi}$']

    cmax = np.nanpercentile(P.ravel(),[minperc,100-minperc])
    cmax = np.max(np.fabs(cmax))
    P = np.ma.masked_array(P,mask=np.isnan(P))
    P = np.apply_over_axes(np.nanmean,P,meanax).squeeze()
    X = dims[keepax[1]]
    Y = dims[keepax[0]]
    if 1 in keepax:
        Y = z
    for i in range(ax.size):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,P[:,:,i]),axc,vmax=cmax,vmin=-cmax,ptype='imshow',
                    txt=lab[i], cmap='RdBu_r',cbar=False)
                                                
        if i % 2 == 0:
            axc.set_ylabel('z (m)')
        if i >= np.size(ax)-2:
            xdegtokm(axc,0.5*(ystart+yend))

    fig.tight_layout()
    cb = fig.colorbar(im, ax=ax.ravel().tolist())
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

    im = m6plot((X,Y,np.sum(P,axis=2)),vmax=cmax,vmin=-cmax,ptype='imshow',
            cmap='RdBu_r',ylim=(-2500,0))
    if savfil:
        plt.savefig(savfil+'res.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
