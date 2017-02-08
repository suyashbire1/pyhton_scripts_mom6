import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from plot_twamomx_budget_complete_direct_newest import extract_twamomx_terms
from plot_twamomy_budget_complete_direct_newest import extract_twamomy_terms

def extract_twapv_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fhgeo = dset(geofil)
        fh = mfdset(fil)
        zi = rdp1.getdims(fh)[2][0]
        dbl = -np.diff(zi)*9.8/1031
        (xs,xe),(ys,ye),dimq = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yq')
        dxbu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][4]
        dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][5]
        nt_const = dimq[0].size
        fhgeo.close()
        
        xmom = extract_twamomx_terms(geofil,vgeofil,fil,fil2,xs,xe,ys,ye+1,zs,ze,(0,),
                alreadysaved=False,xyasindices=True,calledfrompv=True)[2]
        ymom = extract_twamomy_terms(geofil,vgeofil,fil,fil2,xs,xe,ys,ye,zs,ze,(0,),
                alreadysaved=False,xyasindices=True,calledfrompv=True)[2]

        xmom = xmom[np.newaxis,:,:,:,:]
        ymom = ymom[np.newaxis,:,:,:,:]
        ymom = np.concatenate((ymom,-ymom[:,:,:,-1:]),axis=3)

        pv = -np.diff(xmom,axis=2)/dybu[:,:,np.newaxis] + np.diff(ymom,axis=3)/dxbu[:,:,np.newaxis]
        pvmask = np.zeros(pv.shape,dtype=np.int8)
        pvmask[:,:,:,-1:] = 1
        pv = np.ma.masked_array(pv, mask=(pvmask==1))
        pv = np.ma.apply_over_axes(np.nanmean, pv, meanax)

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
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('twapv_complete_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twapv_complete_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_twapv(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxpercfactor = 1,cmaxpercfactorforep=1, savfil=None,savfilep=None,alreadysaved=False):
    X,Y,P = extract_twapv_terms(geofil,vgeofil,fil,fil2,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
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
