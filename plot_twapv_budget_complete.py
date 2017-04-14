import sys
import readParams_moreoptions as rdp1
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from plot_twamomx_budget_complete_direct_newest import extract_twamomx_terms
from plot_twamomy_budget_complete_direct_newest import extract_twamomy_terms
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc

def getutwa(fhgeo, fh, fh2, sl):
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]
    uh = (fh.variables['uh_masked'][sl].filled(0)*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    h_cu = fh.variables['h_Cu'][sl].filled(0).mean(axis=0,keepdims=True)
    h_cu = np.ma.masked_array(h_cu,mask=(h_cu<1e-3))
    dycu = fhgeo.variables['dyCu'][sl[2:]]
    utwa = uh/h_cu/dycu
    return utwa, h_cu

def getvtwa(fhgeo, fh, fh2, sl):
    dt = fh.variables['average_DT'][:]
    dt = dt[:,np.newaxis,np.newaxis,np.newaxis]
    vh = (fh.variables['vh_masked'][sl]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
    h_cv = fh.variables['h_Cv'][sl].mean(axis=0,keepdims=True)
    h_cv = np.ma.masked_array(h_cv,mask=(h_cv<1e-3))
    dxcv = fhgeo.variables['dxCv'][sl[2:]]
    vtwa = vh/dxcv/h_cv
    vtwa = np.concatenate((vtwa,-vtwa[:,:,:,-1:]),axis=3)
    h_cv = np.concatenate((h_cv,h_cv[:,:,:,-1:]),axis=3)
    return vtwa, h_cv

def getpv(fhgeo, fh, fh2, xs, xe, ys, ye, zs=0, ze=None):
    sl = np.s_[:,zs:ze,ys:ye,xs:xe]
    slpy = np.s_[:,zs:ze,ys:ye+1,xs:xe]

    utwa,h_cu = getutwa(fhgeo, fh, fh2, slpy)
    dybu = fhgeo.variables['dyBu'][sl[2:]]
    utway = np.diff(utwa,axis=2)/dybu

    vtwa,h_cv = getvtwa(fhgeo, fh, fh2, sl)
    dxbu = fhgeo.variables['dxBu'][sl[2:]]
    vtwax = np.diff(vtwa,axis=3)/dxbu

    h_q = 0.25*(h_cu[:,:,:-1,:] + h_cu[:,:,1:,:] +
            h_cv[:,:,:,:-1] + h_cv[:,:,:,1:])
    f = fhgeo.variables['f'][sl[2:]]
    pvhash = (f - utway + vtwax)/h_q
    return pvhash, h_q

def extract_twapv_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,fil3=None,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fhgeo = dset(geofil)
        fh = mfdset(fil)
        fh2 = mfdset(fil2)
        zi = rdp1.getdims(fh)[2][0]
        dbl = -np.diff(zi)*9.8/1031
        (xs,xe),(ys,ye),dimq = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yq')
        dxbu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][4]
        dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][5]
        aq = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[1][1]
        f = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[-1]
        nt_const = dimq[0].size

        pvhash,hq = getpv(fhgeo, fh, fh2, xs, xe, ys, ye)
        sl = np.s_[:,zs:ze,ys:ye,xs:xe]
        slpy = np.s_[:,zs:ze,ys:ye+1,xs:xe]
        dxcu = fhgeo.variables['dxCu'][slpy[2:]]
        dycv = fhgeo.variables['dyCv'][sl[2:]]
        dycv = np.concatenate((dycv,dycv[:,-1:]),axis=1)

        if fil3:
            fh3 = mfdset(fil3)
            sltn = np.s_[-1:,zs:ze,ys:ye,xs:xe]
            islayerdeep0 = fh3.variables['islayerdeep'][-1:,0,0,0]
            islayerdeep = (fh3.variables['islayerdeep'][sltn].filled(np.nan))
            swash = (islayerdeep0 - islayerdeep)/islayerdeep0*100
            fh3.close()
        else:
            swash = None

        xmom = extract_twamomx_terms(geofil,vgeofil,fil,fil2,xs,xe,ys,ye+1,zs,ze,(0,),
                alreadysaved=False,xyasindices=True,calledfrompv=True)[2]
        ymom = extract_twamomy_terms(geofil,vgeofil,fil,fil2,xs,xe,ys,ye,zs,ze,(0,),
                alreadysaved=False,xyasindices=True,calledfrompv=True)[2]

        xmom = xmom[np.newaxis,:,:,:,:]
        ymom = ymom[np.newaxis,:,:,:,:]
        ymom = np.concatenate((ymom,ymom[:,:,:,-1:]),axis=3)

        bxppvflx = np.sum(xmom[:,:,:,:,[0,1,3,4]],axis=4)
        pvhash1,_ = getpv(fhgeo, fh, fh2, xs, xe, ys-1, ye+1)
        sl1 = np.s_[:,zs:ze,ys-1:ye+1,xs:xe]
        vtwa1, h_cv1 = getvtwa(fhgeo,fh,fh2,sl1)
        vtwa1 = 0.5*(vtwa1[:,:,:,:-1] + vtwa1[:,:,:,1:])
        pvhashvtwa = pvhash1*vtwa1
        sl1 = np.s_[:,zs:ze,ys:ye+1,xs:xe]
        h_cu1 = fh.variables['h_Cu'][sl1].filled(np.nan).mean(axis=0,keepdims=True)
        h_cu1 = np.ma.masked_array(h_cu1,mask=(h_cu1<1e-3))
        pvflxx = h_cu1*(pvhashvtwa[:,:,:-1,:]+pvhashvtwa[:,:,1:,:])/2
        
        byppvflx = np.sum(ymom[:,:,:,:,[0,1,3,4]],axis=4)
        pvhash1,_ = getpv(fhgeo, fh, fh2, xs-1, xe, ys, ye)
        sl1 = np.s_[:,zs:ze,ys:ye+1,xs-1:xe]
        utwa1, h_cu1 = getutwa(fhgeo,fh,fh2,sl1)
        utwa1 = 0.5*(utwa1[:,:,:-1,:]+utwa1[:,:,1:,:])
        pvhashutwa = pvhash1*utwa1
        pvhashutwa[:,:,:,-1] = 0.0
        sl1 = np.s_[:,zs:ze,ys:ye,xs:xe]
        h_cv1 = fh.variables['h_Cv'][sl1].mean(axis=0,keepdims=True)
        h_cv1 = np.ma.masked_array(h_cv1,mask=(h_cv1<1e-3))
        pvflxy = h_cv1*(pvhashutwa[:,:,:,:-1]+pvhashutwa[:,:,:,1:])/2
        pvflxy = np.concatenate((pvflxy,pvflxy[:,:,:,-1:]),axis=3)

        bx = bxppvflx - pvflxx
        by = byppvflx + pvflxy

        xmom1 = xmom[:,:,:,:,[2,5,6,7,8,9]]
        xmom1 = np.concatenate((+pvflxx[:,:,:,:,np.newaxis], xmom1, bx[:,:,:,:,np.newaxis]), axis = 4)
        ymom1 = ymom[:,:,:,:,[2,5,6,7,8,9]]
        ymom1 = np.concatenate((-pvflxy[:,:,:,:,np.newaxis], ymom1, by[:,:,:,:,np.newaxis]), axis = 4) 

        #pv = (-np.diff(xmom*dxcu[:,:,np.newaxis],axis=2) + np.diff(ymom*dycv[:,:,np.newaxis],axis=3))/aq[:,:,np.newaxis]
        pv = -np.diff(xmom,axis=2)/dybu[:,:,np.newaxis] + np.diff(ymom,axis=3)/dxbu[:,:,np.newaxis]
        pv1x = -np.diff(xmom1*dxcu[:,:,np.newaxis],axis=2)/aq[:,:,np.newaxis]
        pv1y =  np.diff(ymom1*dycv[:,:,np.newaxis],axis=3)/aq[:,:,np.newaxis]

        slyp = np.s_[:,:,ys:ye+1,xs:xe]
        ah1 = fhgeo.variables['Ah'][slyp[2:]]
        slxmyp = np.s_[:,:,ys:ye+1,xs-1:xe]
        uh = fh2.variables['uh'][slxmyp].filled(0).mean(axis=0,keepdims=True)
        uhx = np.diff(uh,axis=3)/ah1
        uhx = np.concatenate((uhx,uhx[:,:,:,-1:]),axis=3)
        uhx = 0.25*(uhx[:,:,:-1,:-1] + uhx[:,:,:-1,1:] + uhx[:,:,1:,:-1] +
                uhx[:,:,1:,1:])
        pv1y[:,:,:,:,0] += pvhash*uhx

        slymp = np.s_[:,:,ys-1:ye+1,xs:xe]
        vh = fh2.variables['vh'][slymp].mean(axis=0,keepdims=True)
        vhy = np.diff(vh,axis=2)/ah1
        vhy = np.concatenate((vhy,vhy[:,:,:,-1:]),axis=3)
        vhy = 0.25*(vhy[:,:,:-1,:-1] + vhy[:,:,:-1,1:] + vhy[:,:,1:,:-1] +
                vhy[:,:,1:,1:])
        pv1x[:,:,:,:,0] += pvhash*vhy

        wd = fh2.variables['wd'][slyp].mean(axis=0,keepdims=True)
        wdb = np.diff(wd,axis=1)
        wdb = np.concatenate((wdb,wdb[:,:,:,-1:]),axis=3)
        wdb = 0.25*(wdb[:,:,:-1,:-1] + wdb[:,:,:-1,1:] + wdb[:,:,1:,:-1] +
                wdb[:,:,1:,1:])
        pv3 = pvhash*wdb
        pv3 = pv3[:,:,:,:,np.newaxis]
        #hq[hq<1] = np.nan
        pvnew = np.concatenate((pv1y[:,:,:,:,:1],
                                pv1x[:,:,:,:,:1],
                                pv3,
                                pv1x[:,:,:,:,1:-1],
                                pv1y[:,:,:,:,1:-1],
                                pv1x[:,:,:,:,-1:]+pv1y[:,:,:,:,-1:]),axis=4)/hq[:,:,:,:,np.newaxis]

        pv = np.ma.filled(pv.astype(np.float64), np.nan)
        pvnew = np.ma.filled(pvnew.astype(np.float64), np.nan)
        pvhash = np.ma.filled(pvhash.astype(np.float64), np.nan)

        pv = np.nanmean(pv,meanax,keepdims=True)
        pvnew = np.nanmean(pvnew,meanax,keepdims=True)
        pvhash = np.nanmean(pvhash,meanax,keepdims=True)

        X = dimq[keepax[1]]
        Y = dimq[keepax[0]]
        if 1 in keepax:
            dt = fh.variables['average_DT'][:]
            dt = dt[:,np.newaxis,np.newaxis,np.newaxis]
            em = (fh2.variables['e'][0:,zs:ze,ys:ye,xs:xe]*dt).sum(axis=0,keepdims=True)/np.sum(dt)
            em = np.nanmean(em, meanax,keepdims=True)
            z = np.linspace(-3000,0,100)
            Y = z
            P = getvaratzc5(pv.astype(np.float32),
                    z.astype(np.float32),
                    em.astype(np.float32)).squeeze()
            Pnew = getvaratzc5(pvnew.astype(np.float32),
                    z.astype(np.float32),
                    em.astype(np.float32)).squeeze()
            pvhash = getvaratzc(pvhash.astype(np.float32),
                    z.astype(np.float32),
                    em.astype(np.float32)).squeeze()
            if fil3:
                swash = np.nanmean(swash, meanax,keepdims=True)
                swash = getvaratzc(swash.astype(np.float32),
                        z.astype(np.float32),
                        em.astype(np.float32)).squeeze()
        else:
            P = pv.squeeze()
            Pnew = pvnew.squeeze()
            pvhash = pvhash.squeeze()
            swash = swash.squeeze()
        np.savez('twapv_complete_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twapv_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    fhgeo.close()
    fh.close()
    fh2.close()
    return (X,Y,P,pvhash,Pnew,swash)

def plot_twapv(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
        fil3=None,plotterms = [0,1,10,11,12,13], swashperc = 1,
        cmaxpercfactor = 1,cmaxpercfactorpvhash=15,cmaxpercfactorPnew=15, savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P,pvhash,Pnew,swash = extract_twapv_terms(geofil,vgeofil,fil,fil2,
            xstart,xend,ystart,yend,zs,ze,meanax,alreadysaved=alreadysaved,fil3=fil3)
    cmax = np.nanpercentile(P,[cmaxpercfactor,100-cmaxpercfactor])
    cmax = np.max(np.fabs(cmax))
    fig,ax = plt.subplots(np.int8(np.ceil(P.shape[-1]/2)),2,
                          sharex=True,sharey=True,figsize=(12, 9))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)',
            '(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)']
    labx = [ r'$(\hat{u}\hat{u}_{\tilde{x}})_{\tilde{y}}$', 
            r'$(\hat{v}\hat{u}_{\tilde{y}})_{\tilde{y}}$', 
            r'$(\hat{\varpi}\hat{u}_{\tilde{b}})_{\tilde{y}}$', 
            r'$(-f\hat{v})_{\tilde{y}}$', 
            r'$(\overline{m_{\tilde{x}}})_{\tilde{y}}$', 
            r"""$(\frac{1}{\overline{h}}(\overline{h}\widehat{u^{\prime \prime}u^{\prime \prime}}+\frac{1}{2}\overline{\zeta^{\prime 2}})_{\tilde{x}})_{\tilde{y}}$""", 
            r"""$(\frac{1}{\overline{h}}(\overline{h}\widehat{u^{\prime \prime}v^{\prime \prime}})_{\tilde{y}}$""",
            r"""$(\frac{1}{\overline{h}}(\overline{h}\widehat{u^{\prime \prime}\varpi^{\prime \prime}} + \overline{\zeta^{\prime}m_{\tilde{x}}^{\prime}})_{\tilde{b}})_{\tilde{y}}$""",
            r'$(-\widehat{X^H})_{\tilde{y}}$', 
            r'$(-\widehat{X^V})_{\tilde{y}}$']
    laby = [ r'$(-\hat{u}\hat{v}_{\tilde{x}})_{\tilde{x}}$', 
            r'$(-\hat{v}\hat{v}_{\tilde{y}})_{\tilde{x}}$', 
            r'$(-\hat{\varpi}\hat{v}_{\tilde{b}})_{\tilde{x}}$', 
            r'$(-f\hat{u})_{\tilde{x}}$', 
            r'$(-\overline{m_{\tilde{y}}})_{\tilde{x}}$', 
            r"""$(-\frac{1}{\overline{h}}(\overline{h}\widehat{u^{\prime \prime}v^{\prime \prime}})_{\tilde{x}})_{\tilde{x}}$""", 
            r"""$(-\frac{1}{\overline{h}}(\overline{h}\widehat{v^{\prime \prime}v^{\prime \prime}}+\frac{1}{2}\overline{\zeta^{\prime 2}})_{\tilde{y}})_{\tilde{x}}$""",
            r"""$(-\frac{1}{\overline{h}}(\overline{h}\widehat{v^{\prime \prime}\varpi^{\prime \prime}} + \overline{\zeta^{\prime}m_{\tilde{y}}^{\prime}})_{\tilde{b}})_{\tilde{x}}$""",
            r'$(\widehat{Y^H})_{\tilde{x}}$', 
            r'$(\widehat{Y^V})_{\tilde{x}}$']
    for i in range(P.shape[-1]):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,P[:,:,i]),axc,vmax=cmax,vmin=-cmax,ptype='imshow',
                txt=labx[i]+' + '+laby[i], ylim=(-2500,0),
                cmap='RdBu_r', cbar=False)
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

    im = m6plot((X,Y,np.sum(P,axis=2)),vmax=cmax,vmin=-cmax,ptype='imshow',cmap='RdBu_r',ylim=(-2500,0))
    if savfil:
        plt.savefig(savfil+'res.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

    fig,ax = plt.subplots(np.int8(np.ceil(len(plotterms)/2)),2,
                          sharex=True,sharey=True,figsize=(12,7))

    cmaxpvhash = np.nanpercentile(pvhash,
            [cmaxpercfactorpvhash,100-cmaxpercfactorpvhash])
    cmaxpvhash = np.max(np.fabs(cmaxpvhash))

    cmax = np.nanpercentile(Pnew,
            [cmaxpercfactorPnew,100-cmaxpercfactorPnew])
    cmax = np.max(np.fabs(cmax))

    lab = [ r"$-\hat{u}\Pi^{\#}_{\tilde{x}}$",
            r"$-\hat{v}\Pi^{\#}_{\tilde{y}}$",
            r"$\Pi^{\#}(\bar{h} \hat{\varpi})_{\tilde{b}}$",
            r"$\frac{(\hat{\varpi}\hat{u}_{\tilde{b}})_{\tilde{y}}}{\bar{h}}$",
            r"""$\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{u^{\prime \prime}u^{\prime \prime}}+\frac{1}{2}\overline{\zeta^{\prime 2}})_{\tilde{x}})_{\tilde{y}}$""", 
            r"""$\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{u^{\prime \prime}v^{\prime \prime}})_{\tilde{y}})_{\tilde{y}}$""",
            r"""$\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{u^{\prime \prime}\varpi^{\prime \prime}} + \overline{\zeta^{\prime}m_{\tilde{x}}^{\prime}})_{\tilde{b}})_{\tilde{y}}$""",
            r'$-\frac{1}{\bar{h}}(\widehat{X^H})_{\tilde{y}}$', 
            r'$-\frac{1}{\bar{h}}(\widehat{X^V})_{\tilde{y}}$',
            r'$-\frac{(\hat{\varpi}\hat{v}_{\tilde{b}})_{\tilde{x}}}{\bar{h}}$', 
            r"""$-\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{u^{\prime \prime}v^{\prime \prime}})_{\tilde{x}})_{\tilde{x}}$""", 
            #r"""$-\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{v^{\prime \prime}v^{\prime \prime}}+\frac{1}{2}\overline{\zeta^{\prime 2}})_{\tilde{y}})_{\tilde{x}}$""",
            r"""$-\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{v^{\prime \prime}v^{\prime \prime}})_{\tilde{y}})_{\tilde{x}}$""",
            #r"""$-\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\bar{h}\widehat{v^{\prime \prime}\varpi^{\prime \prime}} + \overline{\zeta^{\prime}m_{\tilde{y}}^{\prime}})_{\tilde{b}})_{\tilde{x}}$""",
            r"""$-\frac{1}{\bar{h}}(\frac{1}{\bar{h}}(\overline{\zeta^{\prime}m_{\tilde{y}}^{\prime}})_{\tilde{b}})_{\tilde{x}}$""",
            r'$\frac{1}{\bar{h}}(\widehat{Y^H})_{\tilde{x}}$', 
            r'$\frac{1}{\bar{h}}(\widehat{Y^V})_{\tilde{x}}$',
            r'$B_{\tilde{x} \tilde{y}} - B_{\tilde{y} \tilde{x}}$']

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    for i,p in enumerate(plotterms):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,Pnew[:,:,p]),axc,vmax=cmax,vmin=-cmax,ptype='imshow',
                ylim=(-1200,0), txt=lab[p], 
                cmap='RdBu_r', cbar=False)
        im2 = axc.contour(X,Y,pvhash,np.logspace(-6,-5.5,5),colors='k')
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
        plt.savefig(savfil+'Pnew.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
    im = m6plot((X,Y,np.sum(Pnew,axis=2)),ptype='imshow',vmax=cmax,vmin=-cmax,cmap='RdBu_r',ylim=(-2500,0))
    if savfil:
        plt.savefig(savfil+'Pnewres.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

    cmax = np.nanpercentile(pvhash,
            [cmaxpercfactorpvhash,100-cmaxpercfactorpvhash])
    cmax = np.max(np.fabs(cmax))
    im = m6plot((X,Y,pvhash),ptype='imshow',vmax=cmax,vmin=-cmax,cmap='RdBu_r',ylim=(-2500,0))
    if fil3:
        cs = axc.contour(X,Y,swash,np.array([swashperc]), colors='k')
    if savfil:
        plt.savefig(savfil+'pvhash.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
