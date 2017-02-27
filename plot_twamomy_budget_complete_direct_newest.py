import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot,xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_twamomy_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False,xyasindices=False,calledfrompv=False, htol=1e-3):

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
        fh2 = mfdset(fil2)
        zi = rdp1.getdims(fh)[2][0]
        dbl = np.diff(zi)*9.8/1031
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
        dxcvforxdiff,dycvforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][2:4]
        dxcvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[2][3:4]
        dxbu,dybu = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][4:6]
        nt_const = dimv[0].size
        t0 = time.time()
        dt = fh.variables['average_DT'][:]
        dt = dt[:,np.newaxis,np.newaxis,np.newaxis]

        em = (fh2.variables['e'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

        vh = (fh.variables['vh_masked'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cv = (fh.variables['h_Cv'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cv[h_cv < htol] = np.nan
        h_vm = h_cv
        vtwa = vh/h_cv/dxcv

        vhforxdiff = (fh.variables['vh_masked'][0:,zs:ze,ys:ye,xs-1:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cvforxdiff = (fh.variables['h_Cv'][0:,zs:ze,ys:ye,xs-1:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cvforxdiff[h_cvforxdiff < htol] = np.nan
        vtwaforxdiff = vhforxdiff/h_cvforxdiff#/dxcvforxdiff
        vtwaforxdiff = np.concatenate((vtwaforxdiff,vtwaforxdiff[:,:,:,-1:]),axis=3)

        vhforydiff = (fh.variables['vh_masked'][0:,zs:ze,ys-1:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cvforydiff = (fh.variables['h_Cv'][0:,zs:ze,ys-1:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cvforydiff[h_cvforydiff < htol] = np.nan
        vtwaforydiff = vhforydiff/h_cvforydiff#/dxcvforydiff

        vtwax = np.diff(vtwaforxdiff,axis=3)/dxbu/dybu
        vtwax = 0.5*(vtwax[:,:,:,0:-1] + vtwax[:,:,:,1:])

        vtway = np.diff(vtwaforydiff,axis=2)/dyt/dxt
        vtway = 0.5*(vtway[:,:,0:-1,:] + vtway[:,:,1:,:])

        hvmy = np.diff(vhforydiff,axis=2)/dyt/dxt
        hvmy = 0.5*(hvmy[:,:,:-1,:] + hvmy[:,:,1:,:])

        uh = (fh.variables['uh_masked'][0:,zs:ze,ys:ye+1,xs-1:xe]*dt).filled(0).sum(axis=0,keepdims=True)/np.sum(dt)
        hum = 0.25*(uh[:,:,:-1,:-1] + uh[:,:,:-1,1:] + uh[:,:,1:,:-1] +
                uh[:,:,1:,1:])/dycv

        humx = np.diff(np.nan_to_num(uh),axis=3)/dxt/dyt
        humx = 0.5*(humx[:,:,:-1,:] + humx[:,:,1:,:])

        huvxphvvym = (fh.variables['twa_huvxpt'][0:,zs:ze,ys:ye,xs:xe]*dt +
                fh.variables['twa_hvvymt'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        #v = (fh.variables['v_masked'][0:,zs:ze,ys-1:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        #hvv = v*vhforydiff
        #hvvym = np.diff(hvv,axis=2)/dxt/dyt
        #hvvym = 0.5*(hvvym[:,:,:-1,:] + hvvym[:,:,1:,:])
        hvv = (fh.variables['hvv_T'][0:,zs:ze,ys:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)*dxt/np.sum(dt)
        hvvym = np.diff(hvv,axis=2)/dycv/dxcv
        huvxm = huvxphvvym - hvvym

        vtwaforvdiff = np.concatenate((vtwa[:,[0],:,:],vtwa),axis=1)
        vtwab = np.diff(vtwaforvdiff,axis=1)/db[:,np.newaxis,np.newaxis]
        vtwab = np.concatenate((vtwab,np.zeros([vtwab.shape[0],1,vtwab.shape[2],vtwab.shape[3]])),axis=1)
        vtwab = 0.5*(vtwab[:,0:-1,:,:] + vtwab[:,1:,:,:])

        hwb = (fh2.variables['wd'][0:,zs:ze,ys:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hwb = np.diff(hwb,axis=1)
        hwb_v = 0.5*(hwb[:,:,:-1,:] + hwb[:,:,1:,:])
        hwm_v = hwb_v*dbl[:,np.newaxis,np.newaxis]
        #hwb_v = fh.variables['hwb_Cv'][0:,zs:ze,ys:ye,xs:xe]
        #hwm_v = fh.variables['hw_Cv'][0:,zs:ze,ys:ye,xs:xe]

        esq = (fh.variables['esq'][0:,zs:ze,ys:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        emforydiff = (fh2.variables['e'][0:,zs:ze,ys:ye+1,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        elmforydiff = 0.5*(emforydiff[:,0:-1,:,:]+emforydiff[:,1:,:,:])
        edlsqm = (esq - elmforydiff**2)
        edlsqmy = np.diff(edlsqm,axis=2)/dycv

        hpfv = (fh.variables['twa_hpfv'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        pfvm = (fh2.variables['PFv'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        edpfvdmb = -hpfv + h_cv*pfvm - 0.5*edlsqmy*dbl[:,np.newaxis,np.newaxis]

        hmfum = (fh.variables['twa_hmfu'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hvwbm = (fh.variables['twa_hvwb'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hdiffvm = (fh.variables['twa_hdiffv'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hdvdtviscm = (fh.variables['twa_hdvdtvisc'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        fh2.close()
        fh.close()

        advx = hum*vtwax/h_vm
        advy = vtwa*vtway
        advb = hwm_v*vtwab/h_vm
        cor = hmfum/h_vm
        pfvm = pfvm

        xdivep1 = -huvxm/h_vm
        xdivep2 = -advx
        xdivep3 = -vtwa*humx/h_vm 
        xdivep = (xdivep1 + xdivep2 + xdivep3)

        ydivep1 = -hvvym/h_vm
        ydivep2 = -advy
        ydivep3 = -vtwa*hvmy/h_vm
        ydivep4 = 0.5*edlsqmy*dbl[:,np.newaxis,np.newaxis]/h_vm
        ydivep = (ydivep1 + ydivep2 + ydivep3 + ydivep4)

        bdivep1 = -hvwbm/h_vm
        bdivep2 = -advb
        bdivep3 = -vtwa*hwb_v/h_vm 
        bdivep4 = edpfvdmb/h_vm
        bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4)
        Y1twa = hdiffvm/h_vm
        Y2twa = hdvdtviscm/h_vm

        terms = np.concatenate(( -advx[:,:,:,:,np.newaxis],
                                    -advy[:,:,:,:,np.newaxis],
                                    -advb[:,:,:,:,np.newaxis],
                                    cor[:,:,:,:,np.newaxis],
                                    pfvm[:,:,:,:,np.newaxis],
                                    -xdivep[:,:,:,:,np.newaxis],
                                    -ydivep[:,:,:,:,np.newaxis],
                                    -bdivep[:,:,:,:,np.newaxis],
                                    Y1twa[:,:,:,:,np.newaxis],
                                    Y2twa[:,:,:,:,np.newaxis]),
                                    axis=4)
        termsep = np.concatenate((   -xdivep1[:,:,:,:,np.newaxis],
                                        -xdivep3[:,:,:,:,np.newaxis],
                                        -ydivep1[:,:,:,:,np.newaxis],
                                        -ydivep3[:,:,:,:,np.newaxis],
                                        -ydivep4[:,:,:,:,np.newaxis],
                                        -bdivep1[:,:,:,:,np.newaxis],
                                        -bdivep3[:,:,:,:,np.newaxis],
                                        -bdivep4[:,:,:,:,np.newaxis]),
                                        axis=4)

        termsm = np.nanmean(terms,meanax)
        termsepm = np.nanmean(termsep,meanax)

        X = dimv[keepax[1]]
        Y = dimv[keepax[0]]
        if 1 in keepax:
            em = np.mean(em,meanax)
            elm = np.mean(elm,meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimv[1])[0]

        P = termsm.squeeze()
        Pep = termsepm.squeeze()
        if not calledfrompv: 
            np.savez('twamomy_complete_terms', X=X,Y=Y,P=P,Pep=Pep)
    else:
        npzfile = np.load('twamomy_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        Pep = npzfile['Pep']
        
    return (X,Y,P,Pep)

def plot_twamomy(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxpercfactor = 1,cmaxpercfactorforep=1, 
        savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P,Pep = extract_twamomy_terms(geofil,vgeofil,fil,fil2,
                                        xstart,xend,ystart,yend,zs,ze,
                                        meanax, alreadysaved)
    P = np.ma.masked_array(P,mask=np.isnan(P))
    cmax = np.nanpercentile(P,[cmaxpercfactor,100-cmaxpercfactor])
    cmax = np.max(np.fabs(cmax))
    fig,ax = plt.subplots(np.int8(np.ceil(P.shape[-1]/2)),2,
                          sharex=True,sharey=True,figsize=(12, 9))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [ r'$-\hat{u}\hat{v}_{\tilde{x}}$', 
            r'$-\hat{v}\hat{v}_{\tilde{y}}$', 
            r'$-\hat{\varpi}\hat{v}_{\tilde{b}}$', 
            r'$-f\hat{u}$', 
            r'$-\overline{m_{\tilde{y}}}$', 
            r"""$-\frac{1}{\overline{h}}(\overline{h}\widehat{u^{\prime \prime}v^{\prime \prime}})_{\tilde{x}}$""", 
            r"""$-\frac{1}{\overline{h}}(\overline{h}\widehat{v^{\prime \prime}v^{\prime \prime}}+\frac{1}{2}\overline{\zeta^{\prime 2}})_{\tilde{y}}$""",
            r"""$-\frac{1}{\overline{h}}(\overline{h}\widehat{v^{\prime \prime}\varpi ^{\prime \prime}} + \overline{\zeta^\prime m_{\tilde{y}}^\prime})_{\tilde{b}}$""",
            r'$\widehat{Y^H}$', 
            r'$\widehat{Y^V}$']


    for i in range(P.shape[-1]):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,P[:,:,i]),axc,vmax=cmax,vmin=-cmax,
                txt=lab[i], ylim=(-2500,0),cmap='RdBu_r',cbar=False)
        
        if i % 2 == 0:
            axc.set_ylabel('z (m)')
        if i > np.size(ax)-3:
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

    im = m6plot((X,Y,np.sum(P,axis=2)),vmax=cmax,vmin=-cmax,cmap='RdBu_r',ylim=(-2500,0))
    if savfil:
        plt.savefig(savfil+'res.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

    Pep = np.ma.masked_array(Pep,mask=np.isnan(Pep))
    cmax = np.nanpercentile(Pep,[cmaxpercfactorforep,100-cmaxpercfactorforep])
    cmax = np.max(np.fabs(cmax))

    lab = [ r'$-\frac{(\overline{huv})_{\tilde{x}}}{\overline{h}}$',
            r'$\frac{\hat{v}(\overline{hu})_{\tilde{x}}}{\overline{h}}$',
            r'$-\frac{(\overline{hvv})_{\tilde{y}}}{\overline{h}}$',
            r'$\frac{\hat{v}(\overline{hv})_{\tilde{y}}}{\overline{h}}$',
            r"""-$\frac{1}{2\overline{h}}\overline{\zeta ^{\prime 2}}_{\tilde{y}}$""",
            r'$-\frac{(\overline{hv\varpi})_{\tilde{b}}}{\overline{h}}$',
            r'$\frac{\hat{v}(\overline{h\varpi})_{\tilde{b}}}{\overline{h}}$',
            r"""$-\frac{(\overline{\zeta^\prime m_{\tilde{y}}^\prime})_{\tilde{b}}}{\overline{h}}$"""]


    fig,ax = plt.subplots(np.int8(np.ceil(Pep.shape[-1]/2)),2,sharex=True,sharey=True,figsize=(12, 9))
    for i in range(Pep.shape[-1]):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,Pep[:,:,i]),axc,vmax=cmax,vmin=-cmax,
                txt=lab[i],cmap='RdBu_r', ylim=(-2500,0),cbar=False)
        if i % 2 == 0:
            axc.set_ylabel('z (m)')

        if i > np.size(ax)-3:
            xdegtokm(axc,0.5*(ystart+yend))
            
    fig.tight_layout()
    cb = fig.colorbar(im, ax=ax.ravel().tolist())
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    
    if savfilep:
        plt.savefig(savfilep+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
