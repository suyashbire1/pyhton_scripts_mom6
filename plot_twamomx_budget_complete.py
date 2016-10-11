import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_twamomx_terms(geofil,vgeofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False,xyasindices=False):

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
        dxcu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[2][0]
        dybu = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye+1)[2][5]
        nt_const = dimu[0].size
        t0 = time.time()

        print('Reading data using loop...')
        u = fh.variables['u'][0:1,zs:ze,ys:ye,xs:xe]
        nt = np.ones(u.shape)*nt_const
        frhatu = fh.variables['frhatu'][0:1,zs:ze,ys:ye,xs:xe]
        h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
        h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
        nt[h_u<=1e-3] -= 1
        hum = (h_u*u).filled(0)
        huum = (h_u*u*u).filled(0)
        h_um = h_u.filled(0)

        humforxdiff, h_umforxdiff, huumforxdiff = getutwaforxdiff(fh,fhgeo,Dforgetutwaforxdiff,0,xs-1,xe,ys,ye,zs,ze)
        humforydiff, h_umforydiff = getutwaforydiff(fh,fhgeo,Dforgetutwaforydiff,0,xs,xe,ys-1,ye+1,zs,ze)

        cau = fh.variables['CAu'][0:1,zs:ze,ys:ye,xs:xe]
        gkeu = fh.variables['gKEu'][0:1,zs:ze,ys:ye,xs:xe]
        rvxv = fh.variables['rvxv'][0:1,zs:ze,ys:ye,xs:xe]
        hfvm = h_u*(cau - gkeu - rvxv).filled(0)
        
        pfum = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]
#        pfum = np.ma.masked_array(pfum,mask=(h_u<=1e-3).astype(int))
#        pfum = pfum.filled(0)
        
        hdudtviscm = (h_u*fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)
        hdiffum = (h_u*fh.variables['diffu'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)

        dudtdia = fh.variables['dudt_dia'][0:1,zs:ze,ys:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        wd = np.diff(wd,axis=1)
        wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
        wdm = wd
        huwbm = (u*(wd[:,:,:,0:-1]+wd[:,:,:,1:])/2 - h_u*dudtdia).filled(0)

        uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs-1:xe]
        uh = np.ma.filled(uh.astype(float), 0)
        uhx = np.diff(uh,axis = 3)/ah
        uhx = np.concatenate((uhx,uhx[:,:,:,-1:]),axis=3)
        huuxpTm = (u*(uhx[:,:,:,0:-1]+uhx[:,:,:,1:])/2 - h_u*gkeu).filled(0)

        vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye,xs:xe]
        vh = np.ma.filled(vh.astype(float), 0)
        vhy = np.diff(vh,axis = 2)/ah
        vhy = np.concatenate((vhy,vhy[:,:,:,-1:]),axis=3)
        huvymTm = (u*(vhy[:,:,:,0:-1]+vhy[:,:,:,1:])/2 - h_u*rvxv).filled(0)

        v = fh.variables['v'][0:1,zs:ze,ys-1:ye,xs:xe]
        v = np.concatenate((v,-v[:,:,:,[-1]]),axis=3)
        v = 0.25*(v[:,:,0:-1,0:-1] + v[:,:,1:,0:-1] + v[:,:,0:-1,1:] +
                v[:,:,1:,1:])
        hvm = (h_u*v).filled(0)

        hvmforydiff = gethvforydiff(fh,fhgeo,Dforgethvforydiff,0,xs,xe,ys-1,ye,zs,ze)

        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
        hw = wd*dbi[:,np.newaxis,np.newaxis]
        hw = 0.5*(hw[:,0:-1,:,:] + hw[:,1:,:,:])
        hwm_u = 0.5*(hw[:,:,:,0:-1] + hw[:,:,:,1:])


        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt_const

        for i in range(1,nt_const):
            u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
            frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
            h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
            nt[h_u<=1e-3] -= 1
            hum += (h_u*u).filled(0)
            h_um += h_u.filled(0)

            huforxdiff, h_uforxdiff, huuforxdiff = getutwaforxdiff(fh,fhgeo,Dforgetutwaforxdiff,i,xs-1,xe,ys,ye,zs,ze)
            huforydiff, h_uforydiff = getutwaforydiff(fh,fhgeo,Dforgetutwaforydiff,i,xs,xe,ys-1,ye+1,zs,ze)

            humforxdiff += huforxdiff
            h_umforxdiff += h_uforxdiff
            huumforxdiff += huuforxdiff
            humforydiff += huforydiff
            h_umforydiff += h_uforydiff

            cau = fh.variables['CAu'][i:i+1,zs:ze,ys:ye,xs:xe]
            gkeu = fh.variables['gKEu'][i:i+1,zs:ze,ys:ye,xs:xe]
            rvxv = fh.variables['rvxv'][i:i+1,zs:ze,ys:ye,xs:xe]
            hfv = h_u*(cau - gkeu - rvxv)
            hfvm += hfv.filled(0)
            
            pfu = fh.variables['PFu'][i:i+1,zs:ze,ys:ye,xs:xe]
#            pfu = np.ma.masked_array(pfu,mask=(h_u<=1e-3).astype(int))
#            pfum += pfu.filled(0)
            pfum += pfu
            
            hdudtvisc = h_u*fh.variables['du_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe]
            hdudtviscm += hdudtvisc.filled(0)
            hdiffu = h_u*fh.variables['diffu'][i:i+1,zs:ze,ys:ye,xs:xe]
            hdiffum += hdiffu.filled(0)

            dudtdia = fh.variables['dudt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = np.diff(wd,axis=1)
            wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
            wdm += wd
            huwb = (u*(wd[:,:,:,0:-1]+wd[:,:,:,1:])/2 - h_u*dudtdia)
            huwb = np.ma.masked_array(huwb,mask=(h_u<=1e-3).astype(int))
            huwbm += huwb.filled(0)

            uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhx = np.concatenate((uhx,uhx[:,:,:,-1:]),axis=3)
            huuxpT = (u*(uhx[:,:,:,0:-1]+uhx[:,:,:,1:])/2 - h_u*gkeu)
            huuxpT = np.ma.masked_array(huuxpT,mask=(h_u<=1e-3).astype(int))
            huuxpTm += huuxpT.filled(0)

            vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhy = np.concatenate((vhy,vhy[:,:,:,-1:]),axis=3)
            huvymT = (u*(vhy[:,:,:,0:-1]+vhy[:,:,:,1:])/2 - h_u*rvxv)
            huvymT = np.ma.masked_array(huvymT,mask=(h_u<=1e-3).astype(int))
            huvymTm += huvymT.filled(0)

            v = fh.variables['v'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            v = np.concatenate((v,-v[:,:,:,[-1]]),axis=3)
            v = 0.25*(v[:,:,0:-1,0:-1] + v[:,:,1:,0:-1] + v[:,:,0:-1,1:] +
                    v[:,:,1:,1:])
            hv = h_u*v
            hvm += hv.filled(0)

            hvmforydiff += gethvforydiff(fh,fhgeo,Dforgethvforydiff,i,xs,xe,ys-1,ye,zs,ze)

            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
            hw = wd*dbi[:,np.newaxis,np.newaxis]
            hw = 0.5*(hw[:,0:-1,:,:] + hw[:,1:,:,:])
            hwm_u += 0.5*(hw[:,:,:,0:-1] + hw[:,:,:,1:])

            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt_const

            sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
            sys.stdout.flush()


        fhgeo.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])

        utwa = hum/h_um
        utwaforxdiff = humforxdiff/h_umforxdiff
        utwaforydiff = humforydiff/h_umforydiff

        utwaforxdiff[np.isnan(utwaforxdiff)] = 0
        utwax = np.diff(utwaforxdiff,axis=3)/dxt
        utwax = np.concatenate((utwax,-utwax[:,:,:,[-1]]),axis=3)
        utwax = 0.5*(utwax[:,:,:,0:-1] + utwax[:,:,:,1:])

        utway = np.diff(utwaforydiff,axis=2)/dybu
        utway = 0.5*(utway[:,:,0:-1,:] + utway[:,:,1:,:])

        humx = np.diff(humforxdiff,axis=3)/dxt
        humx = np.concatenate((humx,-humx[:,:,:,[-1]]),axis=3)
        humx = 0.5*(humx[:,:,:,0:-1] + humx[:,:,:,1:])

        hvmy = np.diff(hvmforydiff,axis=2)/dyt
        hvmy = np.concatenate((hvmy,-hvmy[:,:,:,[-1]]),axis=3)
        hvmy = 0.5*(hvmy[:,:,:,0:-1] + hvmy[:,:,:,1:])

        huuxphuvym = huuxpTm + huvymTm
        huuxm = np.diff(huumforxdiff,axis=3)/dxt
        huuxm = np.concatenate((huuxm,-huuxm[:,:,:,[-1]]),axis=3)
        huuxm = 0.5*(huuxm[:,:,:,0:-1] + huuxm[:,:,:,1:])
        huvym = huuxphuvym - huuxm

        utwaforvdiff = np.concatenate((utwa,np.zeros([utwa.shape[0],1,utwa.shape[2],utwa.shape[3]])),axis=1)
        utwab = np.diff(utwaforvdiff,axis=1)/db[:,np.newaxis,np.newaxis]
        utwab = np.concatenate((utwab,np.zeros([utwab.shape[0],1,utwab.shape[2],utwab.shape[3]])),axis=1)
        utwab = 0.5*(utwab[:,0:-1,:,:] + utwab[:,1:,:,:])

        hwb_u = 0.5*(wdm[:,:,:,0:-1] + wdm[:,:,:,1:])

        print('Calulcating form drag using loop...')
        t0 = time.time()
        e = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]
        el = 0.5*(e[:,0:-1,:,:] + e[:,1:,:,:])
        ed = e - em
        edl = el - elm
        edlsqm = (edl**2)
        pfu = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]
        pfud = pfu - pfum/nt_const
        pfud = np.concatenate((np.zeros([pfud.shape[0],1,pfud.shape[2],pfud.shape[3]]),
            pfud,np.zeros([pfud.shape[0],1,pfud.shape[2],pfud.shape[3]])),axis=1)
        pfud = 0.5*(pfud[:,0:-1,:,:] + pfud[:,1:,:,:])
        ed = np.concatenate((ed,-ed[:,:,:,-1:]),axis=3)
        ed = 0.5*(ed[:,:,:,0:-1] + ed[:,:,:,1:]) 
        edpfudm = ed*pfud
        for i in range(1,nt_const):
            e = fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]
            el = 0.5*(e[:,0:-1,:,:] + e[:,1:,:,:])
            ed = e - em
            edl = el - elm
            edlsqm += (edl**2)
            pfu = fh.variables['PFu'][i:i+1,zs:ze,ys:ye,xs:xe]
            pfud = pfu - pfum/nt_const
            pfud = np.concatenate((np.zeros([pfud.shape[0],1,pfud.shape[2],pfud.shape[3]]),
                pfud,np.zeros([pfud.shape[0],1,pfud.shape[2],pfud.shape[3]])),axis=1)
            pfud = 0.5*(pfud[:,0:-1,:,:] + pfud[:,1:,:,:])
            ed = np.concatenate((ed,-ed[:,:,:,-1:]),axis=3)
            ed = 0.5*(ed[:,:,:,0:-1] + ed[:,:,:,1:]) 
            edpfudm += ed*pfud

            sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
            sys.stdout.flush()

        print('Time taken for data reading: {}s'.format(time.time()-t0))
        fh.close()
            
        edlsqm = np.concatenate((edlsqm,edlsqm[:,:,:,[-1]]),axis=3)
        edlsqmx = np.diff(edlsqm,axis=3)/dxcu
        advx = utwa*utwax
        advy = hvm*utway/h_um
        advb = hwm_u*utwab/h_um
        cor = hfvm/h_um
        pfum = pfum/nt_const

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
        cmaxscalefactor = 1, savfil=None,savfilep=None,alreadysaved=False):
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

    cmax = np.nanmax(np.absolute(Pep))*cmaxscalefactor
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
