import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc

def extract_twamomx_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
        fil3=None,alreadysaved=False,xyasindices=False,calledfrompv=False,htol=1e-3):

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
            _,_,dimu = rdp1.getdimsbyindx(fh,xs,xe,ys,ye,
                    zs=zs,ze=ze,ts=0,te=None,xhxq='xq',yhyq='yh',zlzi='zl')
        else:
            (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                    slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        sl = np.s_[:,:,ys:ye,xs:xe]
        slmy = np.s_[:,:,ys-1:ye,xs:xe]
        D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
        Dforgetutwaforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[0]
        Dforgetutwaforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[0]
        Dforgethvforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[0]
        dxt,dyt = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][6:8]
        dxcu,dycu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][0:2]
        dycuforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][1:2]
        dycuforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[2][1:2]
        dxbu,dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][4:6]
        aq1 = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye)[1][1]
        ah1 = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[1][0]
        dxcu1 = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[2][0]
        nt_const = dimu[0].size
        t0 = time.time()
        dt = fh.variables['average_DT'][:]
        dt = dt[:,np.newaxis,np.newaxis,np.newaxis]

        if fil3:
            fh3 = mfdset(fil3)
            slmytn = np.s_[-1:,:,ys-1:ye,xs:xe]
#            islayerdeep0 = fh3.variables['islayerdeep'][:,0,0,0].sum()
#            islayerdeep = (fh3.variables['islayerdeep'][slmy].filled(np.nan)).sum(axis=0,
#                                                                               keepdims=True)
            islayerdeep0 = fh3.variables['islayerdeep'][-1:,0,0,0]
            islayerdeep = (fh3.variables['islayerdeep'][slmytn].filled(np.nan))
            swash = (islayerdeep0 - islayerdeep)/islayerdeep0*100
            swash = 0.5*(swash[:,:,:-1,:] + swash[:,:,1:,:])
            fh3.close()
        else:
            swash = None

        em = (fh2.variables['e'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

        uh = (fh.variables['uh_masked'][0:,zs:ze,ys:ye,xs:xe].filled(np.nan)*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cu = (fh.variables['h_Cu'][0:,zs:ze,ys:ye,xs:xe].filled(0)*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cu[h_cu < htol] = np.nan
        h_um = h_cu
        utwa = uh/h_cu/dycu

        uhforxdiff = (fh.variables['uh_masked'][0:,zs:ze,ys:ye,xs-1:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cuforxdiff = (fh.variables['h_Cu'][0:,zs:ze,ys:ye,xs-1:xe]*dt).filled(0).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cuforxdiff[h_cuforxdiff < htol] = np.nan
        utwaforxdiff = uhforxdiff/h_cuforxdiff#/dycuforxdiff

        uhforydiff = (fh.variables['uh_masked'][0:,zs:ze,ys-1:ye+1,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cuforydiff = (fh.variables['h_Cu'][0:,zs:ze,ys-1:ye+1,xs:xe]*dt).filled(0).sum(axis=0,keepdims=True)/np.sum(dt)
        h_cuforydiff[h_cuforydiff < htol] = np.nan
        utwaforydiff = uhforydiff/h_cuforydiff#/dycuforydiff

        utwax = np.diff(np.nan_to_num(utwaforxdiff),axis=3)/dxt/dyt
        utwax = np.concatenate((utwax,-utwax[:,:,:,[-1]]),axis=3)
        utwax = 0.5*(utwax[:,:,:,0:-1] + utwax[:,:,:,1:])

        utway = np.diff(utwaforydiff,axis=2)/dxbu/dybu
        utway = 0.5*(utway[:,:,0:-1,:] + utway[:,:,1:,:])

        humx = np.diff(np.nan_to_num(uhforxdiff),axis=3)/dxt/dyt
        humx = np.concatenate((humx,-humx[:,:,:,[-1]]),axis=3)
        humx = 0.5*(humx[:,:,:,0:-1] + humx[:,:,:,1:])

        hvm = (fh.variables['vh_masked'][0:,zs:ze,ys-1:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hvm = np.concatenate((hvm,-hvm[:,:,:,-1:]),axis=3)
        hvm = 0.25*(hvm[:,:,:-1,:-1] + hvm[:,:,:-1,1:] + hvm[:,:,1:,:-1] +
                hvm[:,:,1:,1:])/dxcu

        hv = (fh.variables['vh_masked'][0:,zs:ze,ys-1:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hvmy = np.diff(hv,axis=2)/dxt/dyt
        hvmy = np.concatenate((hvmy,-hvmy[:,:,:,-1:]),axis=3)
        hvmy = 0.5*(hvmy[:,:,:,:-1] + hvmy[:,:,:,1:])

        huuxphuvym = (fh.variables['twa_huuxpt'][0:,zs:ze,ys:ye,xs:xe]*dt +
                fh.variables['twa_huvymt'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        #u = (fh.variables['u_masked'][0:,zs:ze,ys:ye,xs-1:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        huu = (fh.variables['huu_Cu'][0:,zs:ze,ys:ye,xs-1:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        huuxm = np.diff(np.nan_to_num(huu),axis=3)/dxt/dyt
        huuxm = np.concatenate((huuxm,-huuxm[:,:,:,-1:]),axis=3)
        huuxm = 0.5*(huuxm[:,:,:,:-1] + huuxm[:,:,:,1:])
#        huu = (fh.variables['huu_T'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)*dyt
#        huu = np.concatenate((huu,-huu[:,:,:,-1:]),axis=3)
#        huuxm = np.diff(huu,axis=3)/dxcu/dycu
        huvym = huuxphuvym + huuxm

        utwaforvdiff = np.concatenate((utwa[:,[0],:,:],utwa),axis=1)
        utwab = np.diff(utwaforvdiff,axis=1)/db[:,np.newaxis,np.newaxis]
        utwab = np.concatenate((utwab,np.zeros(utwab[:,:1,:,:].shape)),axis=1)
        utwab = 0.5*(utwab[:,0:-1,:,:] + utwab[:,1:,:,:])

        hwb = (fh2.variables['wd'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        hwb = np.diff(hwb,axis=1)
        hwb = np.concatenate((hwb,-hwb[:,:,:,-1:]),axis=3)
        hwb_u = 0.5*(hwb[:,:,:,:-1] + hwb[:,:,:,1:])
        hwm_u = hwb_u*dbl[:,np.newaxis,np.newaxis]

        esq = (fh.variables['esq'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
        edlsqm = (esq - elm**2)
        edlsqm = np.concatenate((edlsqm,edlsqm[:,:,:,-1:]),axis=3)
        edlsqmx = np.diff(edlsqm,axis=3)/dxcu

        hpfu = (fh.variables['twa_hpfu'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        pfum = (fh2.variables['PFu'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        edpfudmb = -hpfu + h_cu*pfum - 0.5*edlsqmx*dbl[:,np.newaxis,np.newaxis]

        hfvm = (fh.variables['twa_hfv'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        huwbm = (fh.variables['twa_huwb'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        hdiffum = (fh.variables['twa_hdiffu'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        hdudtviscm = (fh.variables['twa_hdudtvisc'][0:,zs:ze,ys:ye,xs:xe]*dt).filled(np.nan).sum(axis=0,keepdims=True)/np.sum(dt)
        fh2.close()
        fh.close()

        advx = utwa*utwax
        advy = hvm*utway/h_um
        advb = hwm_u*utwab/h_um
        cor = hfvm/h_um
        pfum = pfum

        xdivep1 = -huuxm/h_um
        xdivep2 = advx
        xdivep3 = utwa*humx/h_um 
        xdivep4 = -0.5*edlsqmx*dbl[:,np.newaxis,np.newaxis]/h_um
        xdivep = (xdivep1 + xdivep2 + xdivep3 + xdivep4)

        ydivep1 = huvym/h_um
        ydivep2 = advy
        ydivep3 = utwa*hvmy/h_um
        ydivep = (ydivep1 + ydivep2 + ydivep3)

        bdivep1 = huwbm/h_um
        bdivep2 = advb
        bdivep3 = utwa*hwb_u/h_um 
        bdivep4 = -edpfudmb/h_um
        bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4)
        X1twa = hdiffum/h_um
        X2twa = hdudtviscm/h_um

        terms = np.concatenate((-advx[:,:,:,:,np.newaxis],
                                -advy[:,:,:,:,np.newaxis],
                                -advb[:,:,:,:,np.newaxis],
                                cor[:,:,:,:,np.newaxis],
                                pfum[:,:,:,:,np.newaxis],
                                xdivep[:,:,:,:,np.newaxis],
                                ydivep[:,:,:,:,np.newaxis],
                                bdivep[:,:,:,:,np.newaxis],
                                X1twa[:,:,:,:,np.newaxis],
                                X2twa[:,:,:,:,np.newaxis]),
                                axis=4)
        termsep = np.concatenate((  xdivep1[:,:,:,:,np.newaxis],
                                    xdivep3[:,:,:,:,np.newaxis],
                                    xdivep4[:,:,:,:,np.newaxis],
                                    ydivep1[:,:,:,:,np.newaxis],
                                    ydivep3[:,:,:,:,np.newaxis],
                                    bdivep1[:,:,:,:,np.newaxis],
                                    bdivep3[:,:,:,:,np.newaxis],
                                    bdivep4[:,:,:,:,np.newaxis]),
                                    axis=4)

        termsm = np.nanmean(terms,axis=meanax,keepdims=True)
        termsepm = np.nanmean(termsep,axis=meanax,keepdims=True)

        X = dimu[keepax[1]]
        Y = dimu[keepax[0]]
        if 1 in keepax and not calledfrompv:
            em = np.nanmean(em,axis=meanax,keepdims=True)
            elm = np.nanmean(elm,axis=meanax,keepdims=True)
            z = np.linspace(-3000,0,100)
            Y = z
            P = getvaratzc5(termsm.astype(np.float32),
                    z.astype(np.float32),
                    em.astype(np.float32))
            Pep = getvaratzc5(termsepm.astype(np.float32),
                    z.astype(np.float32),
                    em.astype(np.float32))
            if fil3:
                swash = np.nanmean(swash,meanax,keepdims=True)
                swash = getvaratzc(swash.astype(np.float32),
                        z.astype(np.float32),
                        em.astype(np.float32)).squeeze()
        else:
            P = termsm.squeeze()
            Pep = termsepm.squeeze()
        if not calledfrompv: 
            np.savez('twamomx_complete_terms', X=X,Y=Y,P=P,Pep=Pep)
    else:
        npzfile = np.load('twamomx_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        Pep = npzfile['Pep']
        
    return (X,Y,P,Pep,swash)

def plot_twamomx(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
        fil3=None,cmaxpercfactor = 1,cmaxpercfactorforep=1, plotterms=[3,4,7],
        swashperc=1,savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P,Pep,swash = extract_twamomx_terms(geofil,vgeofil,fil,fil2,
                                        xstart,xend,ystart,yend,zs,ze,
                                        meanax, alreadysaved=alreadysaved,fil3=fil3)
    P = np.ma.masked_array(P,mask=np.isnan(P)).squeeze()
    cmax = np.nanpercentile(P,[cmaxpercfactor,100-cmaxpercfactor])
    cmax = np.max(np.fabs(cmax))
    fig,ax = plt.subplots(np.int8(np.ceil(len(plotterms)/2)),2,
                          sharex=True,sharey=True,figsize=(12, 2.5))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [ r'$-\hat{u}\hat{u}_{\tilde{x}}$',
            r'$-\hat{v}\hat{u}_{\tilde{y}}$',
            r'$-\hat{\varpi}\hat{u}_{\tilde{b}}$',
            r'$f\hat{v}$',
            r'$-\overline{m_{\tilde{x}}}$',
            r"""-$\frac{1}{\overline{h}}(\overline{h}\widehat{u ^{\prime \prime} u ^{\prime \prime} } +\frac{1}{2}\overline{\zeta ^{\prime 2}})_{\tilde{x}}$""",
            r"""-$\frac{1}{\overline{h}}(\overline{h}\widehat{u ^{\prime \prime} v ^{\prime \prime}})_{\tilde{y}}$""",
            r"""-$\frac{1}{\overline{h}}(\overline{h}\widehat{u ^{\prime \prime} \varpi ^{\prime \prime}} + \overline{\zeta ^\prime m_{\tilde{x}}^\prime})_{\tilde{b}}$""",
            r'$\widehat{X^H}$',
            r'$\widehat{X^V}$']

    for i,p in enumerate(plotterms):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,P[:,:,p]),axc,vmax=cmax,vmin=-cmax,ptype='imshow',
                txt=lab[p], ylim=(-2500,0),cmap='RdBu_r',cbar=False)
        if fil3:
            cs = axc.contour(X,Y,swash,np.array([swashperc]), colors='k')
        
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

    im = m6plot((X,Y,np.sum(P,axis=2)),vmax=cmax,vmin=-cmax,
            ptype='imshow',cmap='RdBu_r',ylim=(-2500,0))
    if savfil:
        plt.savefig(savfil+'res.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

    Pep = np.ma.masked_array(Pep,mask=np.isnan(Pep)).squeeze()
    cmax = np.nanpercentile(Pep,[cmaxpercfactorforep,100-cmaxpercfactorforep])
    cmax = np.max(np.fabs(cmax))

    lab = [ r'$-\frac{(\overline{huu})_{\tilde{x}}}{\overline{h}}$',
            r'$\frac{\hat{u}(\overline{hu})_{\tilde{x}}}{\overline{h}}$',
            r"""$-\frac{1}{2\overline{h}}\overline{\zeta ^{\prime 2}}_{\tilde{x}}$""",
            r'$-\frac{(\overline{huv})_{\tilde{y}}}{\overline{h}}$',
            r'$\frac{\hat{u}(\overline{hv})_{\tilde{y}}}{\overline{h}}$',
            r'$-\frac{(\overline{hu\varpi})_{\tilde{b}}}{\overline{h}}$',
            r'$\frac{\hat{u}(\overline{h\varpi})_{\tilde{b}}}{\overline{h}}$',
            r"""$-\frac{(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})_{\tilde{b}}}{\overline{h}}$"""]

    fig,ax = plt.subplots(np.int8(np.ceil(Pep.shape[-1]/2)),2,sharex=True,sharey=True,figsize=(12, 9))
    for i in range(Pep.shape[-1]):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,Pep[:,:,i]),axc,vmax=cmax,vmin=-cmax,ptype='imshow',
                txt=lab[i],cmap='RdBu_r', ylim=(-2500,0),cbar=False)
        if fil3:
            cs = axc.contour(X,Y,swash,np.array([swashperc]), colors='k')
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
