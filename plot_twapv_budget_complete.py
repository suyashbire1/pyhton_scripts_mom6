import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from plot_twamomx_budget_complete_direct_newest import extract_twamomx_terms
from plot_twamomy_budget_complete_direct_newest import extract_twamomy_terms
from plot_twauvpv import getuv

def getpv(fhgeo, fh, fh2, xs, xe, ys, ye, zs=0, ze=None):
    sl = np.s_[:,zs:ze,ys:ye,xs:xe]
    slpy = np.s_[:,zs:ze,ys:ye+1,xs:xe]

    uh1 = fh2.variables['uh'][slpy]
    h_cu1 = fh.variables['h_Cu'][slpy]
    h_cu1 = np.ma.masked_array(h_cu1,mask=(h_cu1<1e-3))
    dycu1 = fhgeo.variables['dyCu'][slpy[2:]]
    utwa1 = uh1/h_cu1/dycu1
    dybu1 = fhgeo.variables['dyBu'][sl[2:]]
    utway = np.diff(utwa1,axis=2)/dybu1

    vh1 = fh2.variables['vh'][sl]
    h_cv1 = fh.variables['h_Cv'][sl]
    h_cv1 = np.ma.masked_array(h_cv1,mask=(h_cv1<1e-3))
    dxcv1 = fhgeo.variables['dxCv'][sl[2:]]
    vtwa1 = vh1/dxcv1/h_cv1
    vtwa1 = np.concatenate((vtwa1,-vtwa1[:,:,:,-1:]),axis=3)
    h_cv1 = np.concatenate((h_cv1,h_cv1[:,:,:,-1:]),axis=3)
    dxbu1 = fhgeo.variables['dxBu'][sl[2:]]
    vtwax = np.diff(vtwa1,axis=3)/dxbu1
    h_q1 = 0.25*(h_cu1[:,:,:-1,:] + h_cu1[:,:,1:,:] +
            h_cv1[:,:,:,:-1] + h_cv1[:,:,:,1:])
    f1 = fhgeo.variables['f'][sl[2:]]
    pvhash1 = (f1 - utway + vtwax)/h_q1
    return pvhash1

def extract_twapv_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
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
        f = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[-1]
        nt_const = dimq[0].size

        pvhash = getpv(fhgeo, fh, fh2, xs, xe, ys, ye)

        xmom = extract_twamomx_terms(geofil,vgeofil,fil,fil2,xs,xe,ys,ye+1,zs,ze,(0,),
                alreadysaved=False,xyasindices=True,calledfrompv=True)[2]
        ymom = extract_twamomy_terms(geofil,vgeofil,fil,fil2,xs,xe,ys,ye,zs,ze,(0,),
                alreadysaved=False,xyasindices=True,calledfrompv=True)[2]

        xmom = xmom[np.newaxis,:,:,:,:]
        ymom = ymom[np.newaxis,:,:,:,:]
        ymom = np.concatenate((ymom,-ymom[:,:,:,-1:]),axis=3)

        bxppvflx = np.sum(xmom[:,:,:,:,[0,1,3,4]],axis=4)
        pvhash1 = getpv(fhgeo, fh, fh2, xs, xe, ys-1, ye+1)
        sl1 = np.s_[:,zs:ze,ys-1:ye+1,xs:xe]
        vh1 = fh2.variables['vh'][sl1]
        h_cv1 = fh.variables['h_Cv'][sl1]
        h_cv1 = np.ma.masked_array(h_cv1,mask=(h_cv1<1e-3))
        dxcv1 = fhgeo.variables['dxCv'][sl1[2:]]
        vtwa1 = vh1/dxcv1/h_cv1
        vtwa1 = np.concatenate((vtwa1,-vtwa1[:,:,:,-1:]),axis=3)
        vtwa1 = 0.5*(vtwa1[:,:,:,:-1] + vtwa1[:,:,:,1:])
        pvhashvtwa = pvhash1*vtwa1
        sl1 = np.s_[:,zs:ze,ys:ye+1,xs:xe]
        h_cu1 = fh.variables['h_Cv'][sl1]
        h_cu1 = np.ma.masked_array(h_cu1,mask=(h_cu1<1e-3))
        pvflxx = h_cu1*(pvhashvtwa[:,:,:-1,:]+pvhashvtwa[:,:,1:,:])/2
        
        byppvflx = np.sum(ymom[:,:,:,:,[0,1,3,4]],axis=4)
        pvhash1 = getpv(fhgeo, fh, fh2, xs-1, xe, ys, ye)
        sl1 = np.s_[:,zs:ze,ys:ye+1,xs-1:xe]
        uh1 = fh2.variables['uh'][sl1].filled(0)
        h_cu1 = fh.variables['h_Cu'][sl1]
        h_cu1 = np.ma.masked_array(h_cu1,mask=(h_cu1<1e-3))
        dycu1 = fhgeo.variables['dyCu'][sl1[2:]]
        utwa1 = uh1/h_cu1/dycu1
        utwa1 = 0.5*(utwa1[:,:,:-1,:]+utwa1[:,:,1:,:])
        pvhashutwa = pvhash1*utwa1
        sl1 = np.s_[:,zs:ze,ys:ye,xs:xe]
        h_cv1 = fh.variables['h_Cv'][sl1]
        h_cv1 = np.ma.masked_array(h_cv1,mask=(h_cv1<1e-3))
        pvflxy = h_cv1*(pvhashutwa[:,:,:,:-1]+pvhashutwa[:,:,:,1:])/2
        pvflxy = np.concatenate((pvflxy,-pvflxy[:,:,:,-1:]),axis=3)

        xmom1 = xmom[:,:,:,:,[2,5,6,7,8,9]]
        ymom1 = ymom[:,:,:,:,[2,5,6,7,8,9]]

        pv = -np.diff(xmom,axis=2)/dybu[:,:,np.newaxis] + np.diff(ymom,axis=3)/dxbu[:,:,np.newaxis]
        pvmask = np.zeros(pv.shape,dtype=np.int8)
        pvmask[:,:,:,-1:] = 1
        pv = np.ma.masked_array(pv, mask=(pvmask==1))

        pv = np.ma.apply_over_axes(np.nanmean, pv, meanax)
        pvhash = np.ma.apply_over_axes(np.nanmean, pvhash, meanax)

        X = dimq[keepax[1]]
        Y = dimq[keepax[0]]
        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt_const
            for i in range(1,nt_const):
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt_const
                sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
                sys.stdout.flush()

            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimq[1])[0]

        P = pv.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        pvhash = pvhash.squeeze()
        pvhash = np.ma.filled(pvhash.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('twapv_complete_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twapv_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    fhgeo.close()
    fh.close()
    fh2.close()
    return (X,Y,P,pvhash)

def plot_twapv(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxpercfactor = 1,cmaxpercfactorpvhash=15, savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P,pvhash = extract_twapv_terms(geofil,vgeofil,fil,fil2,
            xstart,xend,ystart,yend,zs,ze,meanax, alreadysaved)
    cmax = np.nanpercentile(P,[cmaxpercfactor,100-cmaxpercfactor])
    cmax = np.max(np.fabs(cmax))
    fig,ax = plt.subplots(np.int8(np.ceil(P.shape[-1]/2)),2,
                          sharex=True,sharey=True,figsize=(12, 9))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    labx = [ r'$(\hat{u}\hat{u}_{\tilde{x}})_{\tilde{y}}$', 
            r'$(\hat{v}\hat{u}_{\tilde{y}})_{\tilde{y}}$', 
            r'$(\hat{\varpi}\hat{u}_{\tilde{b}})_{\tilde{y}}$', 
            r'$(-f\hat{v})_{\tilde{y}}$', 
            r'$(\overline{m_{\tilde{x}}})_{\tilde{y}}$', 
            r"""$(\frac{1}{\overline{h}}(\widehat{u''u''}+\frac{1}{2}\overline{\zeta ' ^2})_{\tilde{x}})_{\tilde{y}}$""", 
            r"""$(\frac{1}{\overline{h}}(\widehat{u''v''})_{\tilde{y}}$""",
            r"""$(\frac{1}{\overline{h}}(\widehat{u''\varpi ''} + \overline{\zeta 'm_{\tilde{x}}'})_{\tilde{b}})_{\tilde{y}}$""",
            r'$(-\widehat{X^H})_{\tilde{y}}$', 
            r'$(-\widehat{X^V})_{\tilde{y}}$']
    laby = [ r'$(-\hat{u}\hat{v}_{\tilde{x}})_{\tilde{x}}$', 
            r'$(-\hat{v}\hat{v}_{\tilde{y}})_{\tilde{x}}$', 
            r'$(-\hat{\varpi}\hat{v}_{\tilde{b}})_{\tilde{x}}$', 
            r'$(-f\hat{u})_{\tilde{x}}$', 
            r'$(-\overline{m_{\tilde{y}}})_{\tilde{x}}$', 
            r"""$(-\frac{1}{\overline{h}}(\widehat{u''v''})_{\tilde{x}})_{\tilde{x}}$""", 
            r"""$(-\frac{1}{\overline{h}}(\widehat{v''v''}+\frac{1}{2}\overline{\zeta ' ^2})_{\tilde{y}})_{\tilde{x}}$""",
            r"""$(-\frac{1}{\overline{h}}(\widehat{v''\varpi ''} + \overline{\zeta 'm_{\tilde{y}}'})_{\tilde{b}})_{\tilde{x}}$""",
            r'$(\widehat{Y^H})_{\tilde{x}}$', 
            r'$(\widehat{Y^V})_{\tilde{x}}$']
    for i in range(P.shape[-1]):
        axc = ax.ravel()[i]
        im = m6plot((X,Y,P[:,:,i]),axc,vmax=cmax,vmin=-cmax,
                txt=labx[i]+' + '+laby[i], ylim=(-2500,0),
                cmap='RdBu_r', cbar=False)
        
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

    cmax = np.nanpercentile(pvhash,
            [cmaxpercfactorpvhash,100-cmaxpercfactorpvhash])
    cmax = np.max(np.fabs(cmax))
    im = m6plot((X,Y,pvhash),vmax=cmax,vmin=-cmax,cmap='RdBu_r',ylim=(-2500,0))
    if savfil:
        plt.savefig(savfil+'pvhash.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
