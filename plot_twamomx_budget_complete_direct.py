import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_twamomx_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False,xyasindices=False,calledfrompv=False):

    if not alreadysaved:
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
        zi = rdp1.getdims(fh)[2][0]
        dbl = -np.diff(zi)*9.8/1031
        if xyasindices:
            (xs,xe),(ys,ye) = (xstart,xend),(ystart,yend)
            _,_,dimu = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                    zs=zs,ze=ze,ts=0,te=None,xhxq='xq',yhyq='yh',zlzi='zl')
        else:
            (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                    slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
        Dforgetutwaforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[0]
        Dforgetutwaforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[0]
        Dforgethvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[0]
        dxt,dyt = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][6:8]
        dxcu,dycu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][0:2]
        dycuforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][1:2]
        dycuforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[2][1:2]
        dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][5]
        dyt1 = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][7]
        nt_const = dimu[0].size
        t0 = time.time()

        em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

        uh = fh.variables['uh'][0:,zs:ze,ys:ye,xs:xe]
        h_cu = fh.variables['h_Cu'][0:,zs:ze,ys:ye,xs:xe]
        utwa = uh/h_cu/dycu

        uhforxdiff = fh.variables['uh'][0:,zs:ze,ys:ye,xs-1:xe]
        h_cuforxdiff = fh.variables['h_Cu'][0:,zs:ze,ys:ye,xs-1:xe]
        utwaforxdiff = uhforxdiff/h_cuforxdiff/dycuforxdiff

        uhforydiff = fh.variables['uh'][0:,zs:ze,ys-1:ye+1,xs:xe]
        h_cuforydiff = fh.variables['h_Cu'][0:,zs:ze,ys-1:ye+1,xs:xe]
        utwaforydiff = uhforydiff/h_cuforydiff/dycuforydiff

        utwax = np.diff(utwaforxdiff.filled(0),axis=3)/dxt
        utwax = np.concatenate((utwax,-utwax[:,:,:,[-1]]),axis=3)
        utwax = 0.5*(utwax[:,:,:,0:-1] + utwax[:,:,:,1:])

        utway = np.diff(utwaforydiff,axis=2)/dybu
        utway = 0.5*(utway[:,:,0:-1,:] + utway[:,:,1:,:])

        humx = np.diff(uhforxdiff.filled(0),axis=3)/dxt
        humx = np.concatenate((humx,-humx[:,:,:,[-1]]),axis=3)
        humx = 0.5*(humx[:,:,:,0:-1] + humx[:,:,:,1:])

        hv_cu = fh.variables['hv_Cu'][0:,zs:ze,ys-1:ye+1,xs:xe]
        hvmy = np.diff(hv_cu,axis=2)/dyt1
        hvmy = 0.5*(hvmy[:,:,:-1,:] + hvmy[:,:,1:,:])

        huuxphuvym = fh.variables['twa_huuxpt'][0:,zs:ze,ys:ye,xs:xe] + fh.variables['twa_huvymt'][0:,zs:ze,ys:ye,xs:xe]
        huu = fh.variables['huu_T'][0:,zs:ze,ys:ye,xs:xe]
        huu = np.concatenate((huu,huu[:,:,:,-1:]),axis=3)
        huuxm = np.diff(huu,axis=3)/dxcu
        huvym = huuxphuvym - huuxm

        utwaforvdiff = np.concatenate((utwa[:,[0],:,:],utwa),axis=1)
        utwab = np.diff(utwaforvdiff,axis=1)/db[:,np.newaxis,np.newaxis]
        utwab = np.concatenate((utwab,np.zeros([utwab.shape[0],1,utwab.shape[2],utwab.shape[3]])),axis=1)
        utwab = 0.5*(utwab[:,0:-1,:,:] + utwab[:,1:,:,:])

        hwb_u = fh.variables['hwb_Cu'][0:,zs:ze,ys:ye,xs:xe]
        hwm_u = fh.variables['hw_Cu'][0:,zs:ze,ys:ye,xs:xe]

        esq = fh.variables['esq'][0:,zs:ze,ys:ye,xs:xe]
        edlsqm = (esq - elm**2)
        edlsqm = np.concatenate((edlsqm,edlsqm[:,:,:,[-1]]),axis=3)
        edlsqmx = np.diff(edlsqm,axis=3)/dxcu

        epfu = fh.variables['epfu'][0:,zs:ze,ys:ye,xs:xe]
        ecu = fh.variables['e_Cu'][0:,zs:ze,ys:ye,xs:xe]
        pfum = fh.variables['PFu'][0:,zs:ze,ys:ye,xs:xe]
        edpfudm = epfu - pfum*ecu

        hfvm = fh.variables['twa_hfv'][0:1,zs:ze,ys:ye,xs:xe]
        huwbm = fh.variables['twa_huwb'][0:1,zs:ze,ys:ye,xs:xe]
        hdiffum = fh.variables['twa_hdiffu'][0:1,zs:ze,ys:ye,xs:xe]
        hdudtviscm = fh.variables['twa_hdudtvisc'][0:1,zs:ze,ys:ye,xs:xe]
        fh.close()

        advx = utwa*utwax
        advy = hvm*utway/h_um
        advb = hwm_u*utwab/h_um
        cor = hfvm/h_um
        pfum = pfum

        xdivep1 = huuxm/h_um
        xdivep2 = -advx
        xdivep3 = -utwa*humx/h_um 
        xdivep4 = 0.5*edlsqmx*dbl[:,np.newaxis,np.newaxis]/h_um
        xdivep = (xdivep1 + xdivep2 + xdivep3 + xdivep4)

        ydivep1 = huvym/h_um
        ydivep2 = -advy
        ydivep3 = -utwa*hvmy/h_um
        ydivep = (ydivep1 + ydivep2 + ydivep3)

        bdivep1 = huwbm/h_um
        bdivep2 = -advb
        bdivep3 = -utwa*hwb_u/h_um 
        bdivep4 = np.diff(edpfudm,axis=1)/h_um
        bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4)
        X1twa = hdiffum/h_um
        X2twa = hdudtviscm/h_um

        terms = np.ma.concatenate(( -advx[:,:,:,:,np.newaxis],
                                    -advy[:,:,:,:,np.newaxis],
                                    -advb[:,:,:,:,np.newaxis],
                                    cor[:,:,:,:,np.newaxis],
                                    pfum[:,:,:,:,np.newaxis],
                                    -xdivep[:,:,:,:,np.newaxis],
                                    -ydivep[:,:,:,:,np.newaxis],
                                    -bdivep[:,:,:,:,np.newaxis],
                                    X1twa[:,:,:,:,np.newaxis],
                                    X2twa[:,:,:,:,np.newaxis]),
                                    axis=4)
        termsep = np.ma.concatenate((   xdivep1[:,:,:,:,np.newaxis],
                                        xdivep3[:,:,:,:,np.newaxis],
                                        xdivep4[:,:,:,:,np.newaxis],
                                        ydivep1[:,:,:,:,np.newaxis],
                                        ydivep3[:,:,:,:,np.newaxis],
                                        bdivep1[:,:,:,:,np.newaxis],
                                        bdivep3[:,:,:,:,np.newaxis],
                                        bdivep4[:,:,:,:,np.newaxis]),
                                        axis=4)

        terms[np.isinf(terms)] = np.nan
        termsep[np.isinf(termsep)] = np.nan
        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)
        termsepm = np.ma.apply_over_axes(np.nanmean, termsep, meanax)

        X = dimu[keepax[1]]
        Y = dimu[keepax[0]]
        if 1 in keepax:
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimu[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        Pep = termsepm.squeeze()
        Pep = np.ma.filled(Pep.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        if not calledfrompv: 
            np.savez('twamomx_complete_terms', X=X,Y=Y,P=P,Pep=Pep)
    else:
        npzfile = np.load('twamomx_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        Pep = npzfile['Pep']
        
    return (X,Y,P,Pep)

def getutwaforxdiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
    h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
    return ((h_u*u).filled(0), h_u.filled(0), (h_u*u*u).filled(0))

def getutwaforydiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
    h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
    return ((h_u*u).filled(0), h_u.filled(0))

def gethvforydiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    v = fh.variables['v'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatv = fh.variables['frhatv'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
    h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
    return (h_v*v).filled(0)

def plot_twamomx(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxscalefactor = 1,cmaxscalefactorforep=1, savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P,Pep = extract_twamomx_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))*cmaxscalefactor
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(5,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 7:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()

    cmax = np.nanmax(np.absolute(Pep))*cmaxscalefactorforep
    plt.figure()
    for i in range(Pep.shape[-1]):
        ax = plt.subplot(4,2,i+1)
        im = m6plot((X,Y,Pep[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 5:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfilep:
        plt.savefig(savfilep+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
