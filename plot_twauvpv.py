import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset


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

    if xyasindices:
        (xs,xe),(ys,ye) = (xstart,xend),(ystart,yend)
        _,_,dimu = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                zs=zs,ze=ze,ts=0,te=None,xhxq='xq',yhyq='yh',zlzi='zl')
        _,_,dimv = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                zs=zs,ze=ze,ts=0,te=None,xhxq='xh',yhyq='yq',zlzi='zl')
        _,_,dimh = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                zs=zs,ze=ze,ts=0,te=None,xhxq='xh',yhyq='yh',zlzi='zl')
    else:
        (xs,xe),(ys,ye),dimh = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze)
        _,_,dimu = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                zs=zs,ze=ze,ts=0,te=None,xhxq='xq',yhyq='yh',zlzi='zl')
        _,_,dimv = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                zs=zs,ze=ze,ts=0,te=None,xhxq='xh',yhyq='yq',zlzi='zl')


    D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
    Dforgetutwaforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[0]
    Dforgetutwaforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[0]
    Dforgethvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[0]
    dxt,dyt = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][6:8]
    dxcuforxdiff,dycuforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][0:2]
    dycuforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[2][1:2]
    dycvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[2][3:4]
    dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][5]
    dyt1 = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][7]
    nt_const = dimh[0].size


    emforxdiff = fh.variables['e'][0:,zs:ze,ys:ye,xs-1:xe]
    elmforxdiff = 0.5*(emforxdiff[:,0:-1,:,:]+emforxdiff[:,1:,:,:])
    elmforxdiff = np.concatenate((elmforxdiff,elmforxdiff[:,:,:,-1:]),axis=3)
    ex = np.diff(elmforxdiff,axis=3)/dxcuforxdiff

    uh = fh2.variables['uh'][0:,zs:ze,ys:ye,xs-1:xe]
    h_cu = fh.variables['h_Cu'][0:,zs:ze,ys:ye,xs-1:xe]
    utwa = uh/h_cu/dycuforxdiff
    utwa = utwa.filled(0)
    uzx = utwa*ex
    utwa = 0.5*(utwa[:,:,:,1:]+utwa[:,:,:,:-1])

    uzx = 0.5*(uzx[:,:,:,1:]+uzx[:,:,:,:-1])

    emforydiff = fh.variables['e'][0:,zs:ze,ys-1:ye+1,xs:xe]
    elmforydiff = 0.5*(emforydiff[:,0:-1,:,:]+emforydiff[:,1:,:,:])
    ey = np.diff(elmforydiff,axis=2)/dycvforydiff

    vh = fh2.variables['vh'][0:,zs:ze,ys-1:ye,xs:xe]
    h_cv = fh.variables['h_Cv'][0:,zs:ze,ys-1:ye,xs:xe]
    h_cv = np.ma.masked_array(h_cv,mask=(h_cv<1e-3))
    vtwa = vh/dycuforydiff/h_cv
    vzy = vtwa*ey
    vtwa = 0.5*(vtwa[:,:,1:,:]+vtwa[:,:,:-1,:])

    vzy = 0.5*(vzy[:,:,1:,:]+vzy[:,:,:-1,:])

    hwm_v = fh.variables['hw_Cv'][0:,zs:ze,ys-1:ye,xs:xe]
    wb = hwm_v
    wzb = 0.5*(wb[:,:,1:,:]+wb[:,:,:-1,:])
    whash = uzx + vzy + wzb


    terms = np.ma.concatenate(( utwa[:,:,:,:,np.newaxis],
                                vtwa[:,:,:,:,np.newaxis],
                                whash[:,:,:,:,np.newaxis]), axis=4)

    termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)

    X = dimh[keepax[1]]
    Y = dimh[keepax[0]]
    if 1 in keepax:
        em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
        elm = 0.5*(em[:,:-1] + em[:,1:])
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        elm = np.ma.apply_over_axes(np.mean, elm, meanax)
        Y = elm.squeeze()
        X = np.meshgrid(X,dimh[1])[0]

    fh2.close()
    fh.close()

    P = termsm.squeeze()
    P = np.ma.filled(P.astype(float), np.nan)
    X = np.ma.filled(X.astype(float), np.nan)
    Y = np.ma.filled(Y.astype(float), np.nan)

    return X,Y,P

def plot_uv(geofil,vgeofil,fil,fil2,xstart,xend,
            ystart,yend,zs,ze,meanax,minperc = [5,3,0],xyasindices = False):

    X,Y,P = getuv(geofil,vgeofil,fil,fil2,xstart,xend,
            ystart,yend,zs,ze,meanax,xyasindices = False)

    
    P = np.ma.masked_array(P,mask=np.isnan(P))
    fig,ax = plt.subplots(P.shape[-1],1,sharex=True,sharey=True,
                          figsize=(6/(0.5*(1+np.sqrt(5))),6))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [r'$\widehat{u}$',r'$\widehat{v}$',r'$w^{\#}$']

    for i in range(P.shape[-1]):
        cmax = np.nanpercentile(P[:,:,i],[minperc[i],100-minperc[i]])
        cmax = np.max(np.fabs(cmax))
        axc = ax.ravel()[i]
        im = m6plot((X,Y,P[:,:,i]),axc,vmax=cmax,vmin=-cmax,
                    txt=lab[i], ylim=(-2500,0),cmap='RdBu_r',cbar=False)
        cb = fig.colorbar(im, ax=axc)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
                                                
        axc.set_ylabel('z (m)')
        if i == np.size(ax)-1:
            xdegtokm(axc,0.5*(ystart+yend))
