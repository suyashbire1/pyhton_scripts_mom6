import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time

def extract_twamomx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fhgeo = dset(geofil)
        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
        D1 = rdp1.getgeombyindx(fhgeo,xs,xe,ys-1,ye+1)[0]
        dxt = rdp1.getgeombyindx(fhgeo,xs+1,xe,ys,ye)[2][6]
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

        hum1, h_um1 = getutwaforydiff(fh,fhgeo,D1,0,xs,xe,ys-1,ye+1,zs,ze)

        cau = fh.variables['CAu'][0:1,zs:ze,ys:ye,xs:xe]
        gkeu = fh.variables['gKEu'][0:1,zs:ze,ys:ye,xs:xe]
        rvxv = fh.variables['rvxv'][0:1,zs:ze,ys:ye,xs:xe]
        hfvm = h_u*(cau - gkeu - rvxv).filled(0)
        
        pfum = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]
        pfum = np.ma.masked_array(pfum,mask=(h_u<=1e-3).astype(int))
        pfum = pfum.filled(0)
        
        hdudtviscm = (h_u*fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)
        hdiffum = (h_u*fh.variables['diffu'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)

        dudtdia = fh.variables['dudt_dia'][0:1,zs:ze,ys:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        wd = np.diff(wd,axis=1)
        wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
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

        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt_const

        for i in range(1,nt_const):
            u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
            frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
            h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
            nt[h_u<=1e-3] -= 1
            hum += (h_u*u).filled(0)
            huum += (h_u*u*u).filled(0)
            h_um += h_u.filled(0)

            hu1, h_u1 = getutwaforydiff(fh,fhgeo,D1,i,xs,xe,ys-1,ye+1,zs,ze)
            hum1 += hu1
            h_um1 += h_u1

            cau = fh.variables['CAu'][i:i+1,zs:ze,ys:ye,xs:xe]
            gkeu = fh.variables['gKEu'][i:i+1,zs:ze,ys:ye,xs:xe]
            rvxv = fh.variables['rvxv'][i:i+1,zs:ze,ys:ye,xs:xe]
            hfv = h_u*(cau - gkeu - rvxv)
            hfvm += hfv.filled(0)
            
            pfu = fh.variables['PFu'][i:i+1,zs:ze,ys:ye,xs:xe]
            pfu = np.ma.masked_array(pfu,mask=(h_u<=1e-3).astype(int))
            pfum += pfu.filled(0)
            
            hdudtvisc = h_u*fh.variables['du_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe]
            hdudtviscm += hdudtvisc.filled(0)
            hdiffu = h_u*fh.variables['diffu'][i:i+1,zs:ze,ys:ye,xs:xe]
            hdiffum += hdiffu.filled(0)

            dudtdia = fh.variables['dudt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = np.diff(wd,axis=1)
            wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
            huwb = (u*(wd[:,:,:,0:-1]+wd[:,:,:,1:])/2 - h_u*dudtdia)
            huwb = np.ma.masked_array(huwb,mask=(h_u<=1e-3).astype(int))
            huwbm += huwb.filled(0)

            uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhx = np.concatenate((uhx,uhx[:,:,:,-1:]),axis=3)
            huuxpT = (u*(uhx[:,:,:,0:-1]+uhx[:,:,:,1:])/2 - h_u*gkeu)
            huuxpT = np.ma.masked_array(huuxpT,mask=(h_u<=1e-3).astype(int))
            huuxpTm = huuxpT.filled(0)

            vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhy = np.concatenate((vhy,vhy[:,:,:,-1:]),axis=3)
            huvymT = (u*(vhy[:,:,:,0:-1]+vhy[:,:,:,1:])/2 - h_u*rvxv)
            huvymT = np.ma.masked_array(huvymT,mask=(h_u<=1e-3).astype(int))
            huvymTm = huvymT.filled(0)

            v = fh.variables['v'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            v = np.concatenate((v,-v[:,:,:,[-1]]),axis=3)
            v = 0.25*(v[:,:,0:-1,0:-1] + v[:,:,1:,0:-1] + v[:,:,0:-1,1:] +
                    v[:,:,1:,1:])
            hv = h_u*v
            hvm += hv.filled(0)

            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt_const


        fh.close()
        fhgeo.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        utwa = hum/h_um
        utwax = np.diff(utwa,axis=3)/dxt
        utwax = np.concatenate((utwax,np.zeros(np.shape(utwax[:,:,:,[0]]))),axis=3)

        utwa1 = hum1/h_um1
        utway = np.diff(utwa1,axis=2)/dybu
        utway = 0.5*(utway[:,:,0:-1,:] + utway[:,:,1:,:])

        huuxphuvym = huuxpTm + huvymTm
        huuxm = np.diff(huum,axis=3)/dxt
        huuxm = np.concatenate((huuxm,np.zeros(np.shape(huuxm[:,:,:,[0]]))),axis=3)
        huvym = huuxphuvym - huuxm

        print(utwax.shape,utway.shape)

#        terms = np.ma.concatenate(( hagum[:,:,:,:,np.newaxis],
#                                    -huwbm[:,:,:,:,np.newaxis],
#                                    -huuxpTm[:,:,:,:,np.newaxis],
#                                    -huvymTm[:,:,:,:,np.newaxis],
#                                    hdudtviscm[:,:,:,:,np.newaxis],
#                                    hdiffum[:,:,:,:,np.newaxis]),
#                                    axis=4)/nt[:,:,:,:,np.newaxis]
#
#        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)
#
#        X = dimu[keepax[1]]
#        Y = dimu[keepax[0]]
#        if 1 in keepax:
#            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
#            em = np.ma.apply_over_axes(np.mean, em, meanax)
#            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
#            Y = elm.squeeze()
#            X = np.meshgrid(X,dimu[1])[0]
#
#        P = termsm.squeeze()
#        P = np.ma.filled(P.astype(float), np.nan)
#        X = np.ma.filled(X.astype(float), np.nan)
#        Y = np.ma.filled(Y.astype(float), np.nan)
#        np.savez('twamomx_terms', X=X,Y=Y,P=P)
#    else:
#        npzfile = np.load('twamomx_terms.npz')
#        X = npzfile['X']
#        Y = npzfile['Y']
#        P = npzfile['P']
#        
#    return (X,Y,P)

def getutwaforydiff(fh,fhgeo,D,i,xs,xe,ys,ye,zs,ze):
    u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
    frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
    h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
    h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))
    return ((h_u*u).filled(0), h_u.filled(0))

def plot_twamomx(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil1=None,savfil2=None,alreadysaved=False):
    X,Y,P = extract_twamomx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(3,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 3:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil1:
        plt.savefig(savfil1+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()

    P1 = np.concatenate((P[:,:,0:2],np.sum(P[:,:,2:4],axis=2,keepdims=True)
        ,P[:,:,4:]),axis=2)
    cmax = np.nanmax(np.absolute(P1))
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)']
    for i in range(P1.shape[-1]):
        ax = plt.subplot(3,2,i+1)
        im = m6plot((X,Y,P1[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 3:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil2:
        plt.savefig(savfil2+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P1,axis=2)),Zmax=cmax)
        plt.show()
