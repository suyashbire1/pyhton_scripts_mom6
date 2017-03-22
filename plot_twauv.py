import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import pyximport
pyximport.install()
from getvaratzc import getvaratzc

def getvaravg(fh,varstr,sl):
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]
    var = (fh.variables[varstr][sl]*dt)
    var = np.apply_over_axes(np.sum,var,0)/np.sum(dt)
    return var

def getuv(geofil,vgeofil,fil,fil2,xstart,xend,
        ystart,yend,zs,ze,meanax,xyasindices = False):
    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fhvgeo = dset(vgeofil)
    db = -fhvgeo.variables['g'][:]
    dbi = np.append(db,0)
    fhvgeo.close()
    fhgeo = dset(geofil)
    fh = mfdset(fil)
    fh2 = mfdset(fil2)
    zi = rdp1.getdims(fh)[2][0]
    dbl = np.diff(zi)*9.8/1031

    slh,dimh = rdp1.getslice(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze)
    ys,ye = slh[2].start, slh[2].stop
    xs,xe = slh[3].start, slh[3].stop
    slhmx = np.s_[:,zs:ze,ys:ye,xs-1:xe]
    slhmy = np.s_[:,zs:ze,ys-1:ye,xs:xe]
    slhmpy = np.s_[:,zs:ze,ys-1:ye+1,xs:xe]
    slu,dimu = rdp1.getslice(fh,xstart,xend,ystart,yend,
            zs=zs,ze=ze,ts=0,te=None,xhxq='xq',yhyq='yh',zlzi='zl')
    slv,dimv = rdp1.getslice(fh,xstart,xend,ystart,yend,
            zs=zs,ze=ze,ts=0,te=None,xhxq='xh',yhyq='yq',zlzi='zl')
    ys,ye = slv[2].start, slv[2].stop
    xs,xe = slv[3].start, slv[3].stop
    slvpy = np.s_[:,zs:ze,ys:ye+1,xs:xe]

    uh = getvaravg(fh2,'uh',slu)
    h_cu = getvaravg(fh,'h_Cu',slu)
    h_cu = np.ma.masked_array(h_cu,mask=(h_cu<1e-3))
    dycu = fhgeo.variables['dyCu'][slu[2:]]
    utwa = uh/h_cu/dycu

    vh = getvaravg(fh2,'vh',slv)
    h_cv = getvaravg(fh,'h_Cv',slv)
    h_cv = np.ma.masked_array(h_cv,mask=(h_cv<1e-3))
    dxcv = fhgeo.variables['dxCv'][slv[2:]]
    vtwa = vh/dxcv/h_cv

    emforxdiff = getvaravg(fh2,'e',slhmx)
    elmforxdiff = 0.5*(emforxdiff[:,0:-1,:,:]+emforxdiff[:,1:,:,:])
    elmforxdiff = np.concatenate((elmforxdiff,elmforxdiff[:,:,:,-1:]),axis=3)
    dxcuforxdiff = fhgeo.variables['dxCu'][slhmx[2:]]
    ex = np.diff(elmforxdiff,axis=3)/dxcuforxdiff

    uh = getvaravg(fh2,'uh',slhmx)
    h_cu = getvaravg(fh,'h_Cu',slhmx)
    h_cu = np.ma.masked_array(h_cu,mask=(h_cu<1e-3))
    dycu = fhgeo.variables['dyCu'][slhmx[2:]]
    uzx = (uh/h_cu/dycu).filled(0)*ex
    uzx = 0.5*(uzx[:,:,:,1:]+uzx[:,:,:,:-1])

    emforydiff = getvaravg(fh2,'e',slhmpy)
    elmforydiff = 0.5*(emforydiff[:,0:-1,:,:]+emforydiff[:,1:,:,:])
    dycv = fhgeo.variables['dyCv'][slhmy[2:]]
    ey = np.diff(elmforydiff,axis=2)/dycv

    vh = getvaravg(fh2,'vh',slhmy)
    h_cv = getvaravg(fh,'h_Cv',slhmy)
    h_cv = np.ma.masked_array(h_cv,mask=(h_cv<1e-3))
    dxcv = fhgeo.variables['dxCv'][slhmy[2:]]
    vtwa = vh/dxcv/h_cv
    vzy = vtwa*ey
    vtwa = 0.5*(vtwa[:,:,1:,:]+vtwa[:,:,:-1,:])
    vzy = 0.5*(vzy[:,:,1:,:]+vzy[:,:,:-1,:])

    wd = getvaravg(fh2,'wd',slh)
    hw = wd*dbi[:,np.newaxis,np.newaxis]
    hw = 0.5*(hw[:,1:,:,:]+hw[:,:-1,:,:])
    wzb = -hw/dbl[:,np.newaxis,np.newaxis]
    whash = uzx + vzy + wzb

    terms = [utwa,vtwa,whash]
    slices = [slu,slvpy,slh]
    X = [dimu[keepax[1]],dimv[keepax[1]],dimh[keepax[1]]]
    Y = [dimu[keepax[0]],dimv[keepax[0]],dimh[keepax[0]]]
    termsm = []
    for item in terms:
        try:
            item = item.filled(np.nan)
        except AttributeError:
            item = item
        termsm.append(np.ma.apply_over_axes(np.nanmean, item, meanax))

    if 1 in keepax:
        Y = []
        for i in range(len(terms)):
            z = np.linspace(-3000,0,100)
            em = fh2.variables['e'][slices[i]]
            if i == 0:
                em = np.concatenate((em,em[:,:,:,-1:]),axis=3)
                em = 0.5*(em[:,:,:,:-1]+em[:,:,:,1:])
            elif i == 1:
                em = 0.5*(em[:,:,:-1,:]+em[:,:,1:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            termsm[i] = getvaratzc(termsm[i].astype(np.float32),
                                   z.astype(np.float32),
                                   em.astype(np.float32))
            Y.append(z)

    fh2.close()
    fh.close()
    fhgeo.close()

    P = []
    for i, item in enumerate(termsm):
        P.append(item)
        X[i] = np.ma.filled(X[i].astype(float), np.nan)
        Y[i] = np.ma.filled(Y[i].astype(float), np.nan)

    return X,Y,P

def plot_uv(geofil,vgeofil,fil,fil2,xstart,xend,
        ystart,yend,zs,ze,meanax,minperc = [1,1,0],xyasindices = False):

    X,Y,P = getuv(geofil,vgeofil,fil,fil2,xstart,xend,
            ystart,yend,zs,ze,meanax,xyasindices = False)


    fig,ax = plt.subplots(len(P),1,sharex=True,sharey=True,
                          figsize=(6/(0.5*(1+np.sqrt(5))),6))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [r'$\widehat{u}$',r'$\widehat{v}$',r'$w^{\#}$']

    for i in range(len(P)):
        P[i] = np.ma.masked_array(P[i],mask=np.isnan(P[i]))
        P[i] = P[i].squeeze()
        cmax = np.nanpercentile(P[i],[minperc[i],100-minperc[i]])
        cmax = np.max(np.fabs(cmax))
        axc = ax.ravel()[i]
        im = m6plot((X[i],Y[i],P[i]),axc,vmax=cmax,vmin=-cmax,ptype='imshow',
                    txt=lab[i], ylim=(-2500,0),cmap='RdBu_r',cbar=False)
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
                                                
        axc.set_ylabel('z (m)')
        if i == np.size(ax)-1:
            xdegtokm(axc,0.5*(ystart+yend))
    return fig
