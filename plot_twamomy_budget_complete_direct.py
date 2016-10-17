import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_twamomy_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
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
            _,_,dimv = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                    zs=zs,ze=ze,ts=0,te=None,xhxq='xh',yhyq='yq',zlzi='zl')
        else:
            (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                    slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
        D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
        Dforgetutwaforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[0]
        Dforgetutwaforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[0]
        Dforgethvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[0]
        dxt,dyt = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][6:8]
        dxcv,dycv = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][2:4]
        dxcvforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][2:3]
        dxcvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[2][3:4]
        dxbu = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][4]
        dyt1 = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][7]
        nt_const = dimv[0].size
        t0 = time.time()

        em = fh.variables['e'][0:,zs:ze,ys:ye,xs:xe]
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

        vh = fh.variables['vh'][0:,zs:ze,ys:ye,xs:xe]
        h_cv = fh.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]
        h_vm = h_cv
        vtwa = vh/h_cv/dxcv

        vhforxdiff = fh.variables['vh'][0:,zs:ze,ys:ye,xs-1:xe]
        h_cvforxdiff = fh.variables['h_Cv'][0:,zs:ze,ys:ye,xs-1:xe]
        vtwaforxdiff = vhforxdiff/h_cvforxdiff/dxcvforxdiff
        vtwaforxdiff = np.concatenate((vtwaforxdiff,vtwaforxdiff[:,:,:,-1:]),axis=3)

        vhforydiff = fh.variables['vh'][0:,zs:ze,ys-1:ye+1,xs:xe]
        h_cvforydiff = fh.variables['h_Cv'][0:,zs:ze,ys-1:ye+1,xs:xe]
        vtwaforydiff = vhforydiff/h_cvforydiff/dxcvforydiff

        vtwax = np.diff(vtwaforxdiff.filled(0),axis=3)/dxbu
        vtwax = 0.5*(vtwax[:,:,:,0:-1] + vtwax[:,:,:,1:])

        vtway = np.diff(vtwaforydiff,axis=2)/dyt
        vtway = 0.5*(vtway[:,:,0:-1,:] + vtway[:,:,1:,:])

        hvmy = np.diff(vhforydiff.filled(0)/dxcvforydiff,axis=3)/dyt
        hvmy = 0.5*(hvmy[:,:,:-1,:] + hvmy[:,:,1:,:])

        hum = fh.variables['hu_Cv'][0:,zs:ze,ys:ye,xs-1:xe]
        humforxdiff = np.concatenate((hum,-hum[:,:,:,-1:]),axis=3)
        humx = np.diff(humforxdiff,axis=3)/dxbu
        humx = 0.5*(humx[:,:,:,:-1] + humx[:,:,:,1:])

        huvxphvvym = fh.variables['twa_huvxpt'][0:,zs:ze,ys:ye,xs:xe] + fh.variables['twa_hvvymt'][0:,zs:ze,ys:ye,xs:xe]
        hvv = fh.variables['hvv_T'][0:,zs:ze,ys:ye+1,xs:xe]
        hvvym = np.diff(hvv,axis=2)/dycv
        huvxm = huvxphvvym - hvvym

        vtwaforvdiff = np.concatenate((vtwa[:,[0],:,:],vtwa),axis=1)
        vtwab = np.diff(vtwaforvdiff,axis=1)/db[:,np.newaxis,np.newaxis]
        vtwab = np.concatenate((vtwab,np.zeros([vtwab.shape[0],1,vtwab.shape[2],vtwab.shape[3]])),axis=1)
        vtwab = 0.5*(vtwab[:,0:-1,:,:] + vtwab[:,1:,:,:])

        hwb_v = fh.variables['hwb_Cv'][0:,zs:ze,ys:ye,xs:xe]
        hwm_v = fh.variables['hw_Cv'][0:,zs:ze,ys:ye,xs:xe]

        emforydiff = fh.variables['e'][0:,zs:ze,ys:ye+1,xs:xe]
        elmforydiff = 0.5*(emforydiff[:,0:-1,:,:]+emforydiff[:,1:,:,:])
        esq = fh.variables['esq'][0:,zs:ze,ys:ye+1,xs:xe]
        edlsqm = (esq - elmforydiff**2)
        edlsqmy = np.diff(edlsqm,axis=2)/dycv

        epfv = fh.variables['epfv'][0:,zs:ze,ys:ye,xs:xe]
        ecv = fh.variables['e_Cv'][0:,zs:ze,ys:ye,xs:xe]
        pfvm = fh.variables['PFv'][0:,zs:ze,ys:ye,xs:xe]
        edpfvdm = epfv - pfvm*ecv
        edpfvdmb = np.diff(edpfvdm,axis=1)
        edpfvdmb = np.concatenate((edpfvdmb[:,:1,:,:],edpfvdmb,edpfvdmb[:,-1:,:,:]),axis=1)
        edpfvdmb = 0.5*(edpfvdmb[:,:-1,:,:] + edpfvdmb[:,1:,:,:])

        hmfum = fh.variables['twa_hmfu'][0:1,zs:ze,ys:ye,xs:xe]
        hvwbm = fh.variables['twa_hvwb'][0:1,zs:ze,ys:ye,xs:xe]
        hdiffvm = fh.variables['twa_hdiffv'][0:1,zs:ze,ys:ye,xs:xe]
        hdvdtviscm = fh.variables['twa_hdvdtvisc'][0:1,zs:ze,ys:ye,xs:xe]
        fh.close()

        advx = hum*vtwax
        advy = vtwa*vtway
        advb = hwm_v*vtwab/h_vm
        cor = hmfum/h_vm
        pfvm = pfvm

        xdivep1 = huvxm/h_vm
        xdivep2 = -advx
        xdivep3 = -vtwa*humx/h_vm 
        xdivep = (xdivep1 + xdivep2 + xdivep3)

        ydivep1 = hvvym/h_vm
        ydivep2 = -advy
        ydivep3 = -vtwa*hvmy/h_vm
        ydivep4 = 0.5*edlsqmy*dbl[:,np.newaxis,np.newaxis]/h_vm
        ydivep = (ydivep1 + ydivep2 + ydivep3 + ydivep4)

        bdivep1 = hvwbm/h_vm
        bdivep2 = -advb
        bdivep3 = -vtwa*hwb_v/h_vm 
        bdivep4 = edpfvdmb/h_vm
        bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4)
        Y1twa = hdiffvm/h_vm
        Y2twa = hdvdtviscm/h_vm

        terms = np.ma.concatenate(( -advx[:,:,:,:,np.newaxis],
                                    -advy[:,:,:,:,np.newaxis],
                                    -advb[:,:,:,:,np.newaxis],
                                    cor[:,:,:,:,np.newaxis],
                                    pfum[:,:,:,:,np.newaxis],
                                    -xdivep[:,:,:,:,np.newaxis],
                                    -ydivep[:,:,:,:,np.newaxis],
                                    -bdivep[:,:,:,:,np.newaxis],
                                    Y1twa[:,:,:,:,np.newaxis],
                                    Y2twa[:,:,:,:,np.newaxis]),
                                    axis=4)
        termsep = np.ma.concatenate((   xdivep1[:,:,:,:,np.newaxis],
                                        xdivep3[:,:,:,:,np.newaxis],
                                        ydivep1[:,:,:,:,np.newaxis],
                                        ydivep3[:,:,:,:,np.newaxis],
                                        ydivep4[:,:,:,:,np.newaxis],
                                        bdivep1[:,:,:,:,np.newaxis],
                                        bdivep3[:,:,:,:,np.newaxis],
                                        bdivep4[:,:,:,:,np.newaxis]),
                                        axis=4)

        terms[np.isinf(terms)] = np.nan
        termsep[np.isinf(termsep)] = np.nan
        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)
        termsepm = np.ma.apply_over_axes(np.nanmean, termsep, meanax)

        X = dimv[keepax[1]]
        Y = dimv[keepax[0]]
        if 1 in keepax:
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimv[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        Pep = termsepm.squeeze()
        Pep = np.ma.filled(Pep.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        if not calledfrompv: 
            np.savez('twamomy_complete_terms', X=X,Y=Y,P=P,Pep=Pep)
    else:
        npzfile = np.load('twamomy_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        Pep = npzfile['Pep']
        
    return (X,Y,P,Pep)

def plot_twamomy(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxscalefactor = 1,cmaxscalefactorforep=1, savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P,Pep = extract_twamomy_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
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