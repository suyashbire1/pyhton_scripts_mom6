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
        D  = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0]
        (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[1]
        Dforgetvtwaforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[0]
        Dforgetvtwaforydiff = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[0]
        Dforgethuforxdiff = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye+1)[0]
        dxt,dyt = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][6:8]
        dycv = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][3]
        dxbu = rdp1.getgeombyindx(fhgeo,xs-1,xe,ys,ye)[2][4]
        nt_const = dimv[0].size
        t0 = time.time()

        print('Reading data using loop...')
        v = fh.variables['v'][0:1,zs:ze,ys:ye,xs:xe]
        nt = np.ones(v.shape)*nt_const
        frhatv = fh.variables['frhatv'][0:1,zs:ze,ys:ye,xs:xe]
        h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
        h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
        nt[h_v<=1e-3] -= 1
        hvm = (h_v*v).filled(0)
        hvvm = (h_v*v*v).filled(0)
        h_vm = h_v.filled(0)

        hvmforxdiff, h_vmforxdiff = getvtwaforxdiff(fh,fhgeo,Dforgetvtwaforxdiff,0,xs-1,xe,ys,ye,zs,ze)
        hvmforydiff, h_vmforydiff, hvvmforydiff  = getvtwaforydiff(fh,fhgeo,Dforgetvtwaforydiff,0,xs,xe,ys-1,ye+1,zs,ze)

        cav = fh.variables['CAv'][0:1,zs:ze,ys:ye,xs:xe]
        gkev = fh.variables['gKEv'][0:1,zs:ze,ys:ye,xs:xe]
        rvxu = fh.variables['rvxu'][0:1,zs:ze,ys:ye,xs:xe]
        hmfum = (h_v*(cav - gkev - rvxu)).filled(0)
        
        pfvm = fh.variables['PFv'][0:1,zs:ze,ys:ye,xs:xe]
#        pfvm = np.ma.masked_array(pfvm,mask=(h_v<=1e-3).astype(int))
#        pfvm = pfvm.filled(0)
        
        hdvdtviscm = (h_v*fh.variables['dv_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)
        hdiffvm = (h_v*fh.variables['diffv'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)

        dvdtdia = fh.variables['dvdt_dia'][0:1,zs:ze,ys:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye+1,xs:xe]
        wd = np.diff(wd,axis=1)
        wdm = wd
        hvwbm = (v*(wd[:,:,0:-1,:]+wd[:,:,1:,:])/2 - h_v*dvdtdia).filled(0)

        uh = fh.variables['uh'][0:1,zs:ze,ys:ye+1,xs-1:xe]
        uh = np.ma.filled(uh.astype(float), 0)
        uhx = np.diff(uh,axis = 3)/ah
        huvxpTm = (v*(uhx[:,:,0:-1,:]+uhx[:,:,1:,:])/2 - h_v*rvxu).filled(0)

        vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye+1,xs:xe]
        vh = np.ma.filled(vh.astype(float), 0)
        vhy = np.diff(vh,axis = 2)/ah
        hvvymTm = (v*(vhy[:,:,0:-1,:]+vhy[:,:,1:,:])/2 - h_v*gkev).filled(0)

        u = fh.variables['u'][0:1,zs:ze,ys:ye+1,xs-1:xe]
        u = 0.25*(u[:,:,0:-1,0:-1] + u[:,:,1:,0:-1] + u[:,:,0:-1,1:] +
                u[:,:,1:,1:])
        hum = (h_v*u).filled(0)

        humforxdiff = gethuforxdiff(fh,fhgeo,Dforgethuforxdiff,0,xs-1,xe,ys,ye+1,zs,ze)

        wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye+1,xs:xe]
        hw = wd*dbi[:,np.newaxis,np.newaxis]
        hw = 0.5*(hw[:,0:-1,:,:] + hw[:,1:,:,:])
        hwm_v = 0.5*(hw[:,:,0:-1,:] + hw[:,:,1:,:])

        emforydiff = fh.variables['e'][0:1,zs:ze,ys:ye+1,xs:xe]/nt_const
        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt_const

        for i in range(1,nt_const):
            v = fh.variables['v'][i:i+1,zs:ze,ys:ye,xs:xe]
            frhatv = fh.variables['frhatv'][i:i+1,zs:ze,ys:ye,xs:xe]
            h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
            h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
            nt[h_v<=1e-3] -= 1
            hvm += (h_v*v).filled(0)
            h_vm += h_v.filled(0)

            hvforxdiff, h_vforxdiff = getvtwaforxdiff(fh,fhgeo,Dforgetvtwaforxdiff,i,xs-1,xe,ys,ye,zs,ze)
            hvforydiff, h_vforydiff, hvvforydiff  = getvtwaforydiff(fh,fhgeo,Dforgetvtwaforydiff,i,xs,xe,ys-1,ye+1,zs,ze)

            hvmforxdiff += hvforxdiff
            h_vmforxdiff += h_vforxdiff
            hvvmforydiff += hvvforydiff
            hvmforydiff += hvforydiff
            h_vmforydiff += h_vforydiff

            cav = fh.variables['CAv'][i:i+1,zs:ze,ys:ye,xs:xe]
            gkev = fh.variables['gKEv'][i:i+1,zs:ze,ys:ye,xs:xe]
            rvxu = fh.variables['rvxu'][i:i+1,zs:ze,ys:ye,xs:xe]
            hmfu = h_v*(cav - gkev - rvxu)
            hmfum += hmfu.filled(0)
            
            pfv = fh.variables['PFv'][i:i+1,zs:ze,ys:ye,xs:xe]
#            pfv = np.ma.masked_array(pfv,mask=(h_v<=1e-3).astype(int))
#            pfvm += pfv.filled(0)
            pfvm += pfv
            
            hdvdtvisc = h_v*fh.variables['dv_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe]
            hdvdtviscm += hdvdtvisc.filled(0)
            hdiffv = h_v*fh.variables['diffv'][i:i+1,zs:ze,ys:ye,xs:xe]
            hdiffvm += hdiffv.filled(0)

            dvdtdia = fh.variables['dvdt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            wd = np.diff(wd,axis=1)
            wdm += wd
            hvwb = (v*(wd[:,:,0:-1,:]+wd[:,:,1:,:])/2 - h_v*dvdtdia)
            hvwb = np.ma.masked_array(hvwb,mask=(h_v<=1e-3).astype(int))
            hvwbm += hvwb.filled(0)

            uh = fh.variables['uh'][0:1,zs:ze,ys:ye+1,xs-1:xe]
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            huvxpT = (v*(uhx[:,:,0:-1,:]+uhx[:,:,1:,:])/2 - h_v*rvxu).filled(0)
            huvxpTm += huvxpT

            vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye+1,xs:xe]
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            hvvymT = (v*(vhy[:,:,0:-1,:]+vhy[:,:,1:,:])/2 - h_v*gkev).filled(0)
            hvvymTm = hvvymT

            u = fh.variables['u'][0:1,zs:ze,ys:ye+1,xs-1:xe]
            u = 0.25*(u[:,:,0:-1,0:-1] + u[:,:,1:,0:-1] + u[:,:,0:-1,1:] +
                    u[:,:,1:,1:])
            hum += (h_v*u).filled(0)

            humforxdiff += gethuforxdiff(fh,fhgeo,Dforgethuforxdiff,i,xs-1,xe,ys,ye+1,zs,ze)

            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            hw = wd*dbi[:,np.newaxis,np.newaxis]
            hw = 0.5*(hw[:,0:-1,:,:] + hw[:,1:,:,:])
            hwm_v += 0.5*(hw[:,:,0:-1,:] + hw[:,:,1:,:])

            emforydiff += fh.variables['e'][i:i+1,zs:ze,ys:ye+1,xs:xe]/nt_const
            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt_const

            sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
            sys.stdout.flush()


        fhgeo.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
        elmforydiff = 0.5*(emforydiff[:,0:-1,:,:]+emforydiff[:,1:,:,:])

        vtwa = hvm/h_vm
        vtwaforxdiff = hvmforxdiff/h_vmforxdiff
        vtwaforydiff = hvmforydiff/h_vmforydiff

        vtwaforxdiff = np.concatenate((vtwaforxdiff,-vtwaforxdiff[:,:,:,-1:]),axis=3)
        vtwax = np.diff(vtwaforxdiff,axis=3)/dxbu
        vtwax = 0.5*(vtwax[:,:,:,0:-1] + vtwax[:,:,:,1:])

        vtway = np.diff(vtwaforydiff,axis=2)/dyt
        vtway = 0.5*(vtway[:,:,0:-1,:] + vtway[:,:,1:,:])

        humx = np.diff(humforxdiff,axis=3)/dxt
        humx = 0.5*(humx[:,:,0:-1,:] + humx[:,:,1:,:])

        hvmy = np.diff(hvmforydiff,axis=2)/dyt
        hvmy = 0.5*(hvmy[:,:,0:-1,:] + hvmy[:,:,1:,:])

        huvxphvvym = huvxpTm + hvvymTm
        hvvym = np.diff(hvvmforydiff,axis=2)/dyt
        hvvym = 0.5*(hvvym[:,:,0:-1,:] + hvvym[:,:,1:,:])
        huvxm = huvxphvvym - hvvym

        vtwaforvdiff = np.concatenate((vtwa[:,[0],:,:],vtwa),axis=1)
        vtwab = np.diff(vtwaforvdiff,axis=1)/db[:,np.newaxis,np.newaxis]
        vtwab = np.concatenate((vtwab,np.zeros([vtwab.shape[0],1,vtwab.shape[2],vtwab.shape[3]])),axis=1)
        vtwab = 0.5*(vtwab[:,0:-1,:,:] + vtwab[:,1:,:,:])

        hwb_v = 0.5*(wdm[:,:,0:-1,:] + wdm[:,:,1:,:])

        print('Calculating form drag using loop...')
        t0 = time.time()
        e = fh.variables['e'][0:1,zs:ze,ys:ye+1,xs:xe]
        el = 0.5*(e[:,0:-1,:,:] + e[:,1:,:,:])
        ed = e - emforydiff
        edl = el - elmforydiff
        edlsqm = (edl**2)
        pfv = fh.variables['PFv'][0:1,zs:ze,ys:ye,xs:xe]
        pfvd = pfv - pfvm/nt_const
        geta = 9.8*ed[:,:1,:,:]/1031
        getay = np.diff(geta,axis=2)/dycv
        pfvd = np.concatenate((-getay,pfvd,np.zeros([pfvd.shape[0],1,pfvd.shape[2],pfvd.shape[3]])),axis=1)
        pfvd = 0.5*(pfvd[:,0:-1,:,:] + pfvd[:,1:,:,:])
        ed = 0.5*(ed[:,:,0:-1,:] + ed[:,:,1:,:]) 
        edpfvdm = ed*pfvd
        for i in range(1,nt_const):
            e = fh.variables['e'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            el = 0.5*(e[:,0:-1,:,:] + e[:,1:,:,:])
            ed = e - emforydiff
            edl = el - elmforydiff
            edlsqm += (edl**2)
            pfv = fh.variables['PFv'][i:i+1,zs:ze,ys:ye,xs:xe]
            pfvd = pfv - pfvm/nt_const
            geta = 9.8*ed[:,:1,:,:]/1031
            getay = np.diff(geta,axis=2)/dycv
            pfvd = np.concatenate((-getay,pfvd,np.zeros([pfvd.shape[0],1,pfvd.shape[2],pfvd.shape[3]])),axis=1)
            pfvd = 0.5*(pfvd[:,0:-1,:,:] + pfvd[:,1:,:,:])
            ed = 0.5*(ed[:,:,0:-1,:] + ed[:,:,1:,:]) 
            edpfvdm += ed*pfvd

            sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
            sys.stdout.flush()

        print('Time taken for data reading: {}s'.format(time.time()-t0))
        fh.close()
            
        edlsqmy = np.diff(edlsqm,axis=2)/dycv
        advx = hum*vtwax/h_vm
        advy = vtwa*vtway
        advb = hwm_v*vtwab/h_vm
        cor = hmfum/h_vm
        pfvm = pfvm/nt_const

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
        bdivep4 = np.diff(edpfvdm,axis=1)/db*dbl/h_vm
        bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4)

        Y1twa = hdiffvm/h_vm
        Y2twa = hdvdtviscm/h_vm

        terms = np.ma.concatenate(( -advx[:,:,:,:,np.newaxis],
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

def getvtwaforxdiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    v = fh.variables['v'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatv = fh.variables['frhatv'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
    h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
    return ((h_v*v).filled(0), h_v.filled(0))

def getvtwaforydiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    v = fh.variables['v'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatv = fh.variables['frhatv'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
    h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
    return ((h_v*v).filled(0), h_v.filled(0), (h_v*v*v).filled(0))

def gethuforxdiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
    h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
    return (h_u*u).filled(0)

def plot_twamomy(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        cmaxscalefactor=1,cmaxscalefactorforep=1, savfil=None,savfilep=None,alreadysaved=False):
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
