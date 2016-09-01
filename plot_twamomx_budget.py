import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time

def extract_twamomx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
#        _,(vys,vye),_ = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
#                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
#        (xs,xe),(ys,ye),_ = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
#                slat=ystart, nlat=yend,zs=zs,ze=ze)
        D, (ah,aq) = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0:2]
        nt = dimu[0].size
        t0 = time.time()

        print('Reading data using loop...')
        u = fh.variables['u'][0:1,zs:ze,ys:ye,xs:xe]
        frhatu = fh.variables['frhatu'][0:1,zs:ze,ys:ye,xs:xe]
        h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
        h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))

        cau = fh.variables['CAu'][0:1,zs:ze,ys:ye,xs:xe]
        gkeu = fh.variables['gKEu'][0:1,zs:ze,ys:ye,xs:xe]
        rvxv = fh.variables['rvxv'][0:1,zs:ze,ys:ye,xs:xe]
        pfu = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]
        hagum = h_u*(cau - gkeu - rvxv + pfu)/nt
        
        hdudtviscm = h_u*fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]/nt
        hdiffum = h_u*fh.variables['diffu'][0:1,zs:ze,ys:ye,xs:xe]/nt

        dudtdia = fh.variables['dudt_dia'][0:1,zs:ze,ys:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        wd = np.diff(wd,axis=1)
        wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
        huwbm = (u*(wd[:,:,:,0:-1]+wd[:,:,:,1:])/2 - h_u*dudtdia)/nt

        uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs-1:xe]
        uh = np.ma.filled(uh.astype(float), 0)
        uhx = np.diff(uh,axis = 3)/ah
        uhx = np.concatenate((uhx,uhx[:,:,:,-1:]),axis=3)
        huuxpTm = (u*(uhx[:,:,:,0:-1]+uhx[:,:,:,1:])/2 - h_u*gkeu)/nt

        vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye,xs:xe]
        vh = np.ma.filled(vh.astype(float), 0)
        vhy = np.diff(vh,axis = 2)/ah
        vhy = np.concatenate((vhy,vhy[:,:,:,-1:]),axis=3)
        huvymTm = (u*(vhy[:,:,:,0:-1]+vhy[:,:,:,1:])/2 - h_u*rvxv)/nt

        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt

        for i in range(1,nt):
            u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs:xe]
            frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys:ye,xs:xe]
            h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
            h_u = np.ma.masked_array(h_u,mask=(h_u<=1e-3).astype(int))

            cau = fh.variables['CAu'][i:i+1,zs:ze,ys:ye,xs:xe]
            gkeu = fh.variables['gKEu'][i:i+1,zs:ze,ys:ye,xs:xe]
            rvxv = fh.variables['rvxv'][i:i+1,zs:ze,ys:ye,xs:xe]
            pfu = fh.variables['PFu'][i:i+1,zs:ze,ys:ye,xs:xe]
            hagum += h_u*(cau - gkeu - rvxv + pfu)/nt
            
            hdudtviscm += h_u*fh.variables['du_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
            hdiffum += h_u*fh.variables['diffu'][i:i+1,zs:ze,ys:ye,xs:xe]/nt

            dudtdia = fh.variables['dudt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = np.diff(wd,axis=1)
            wd = np.concatenate((wd,wd[:,:,:,-1:]),axis=3)
            huwbm += (u*(wd[:,:,:,0:-1]+wd[:,:,:,1:])/2 - h_u*dudtdia)/nt

            uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhx = np.concatenate((uhx,uhx[:,:,:,-1:]),axis=3)
            huuxpTm += (u*(uhx[:,:,:,0:-1]+uhx[:,:,:,1:])/2 - h_u*gkeu)/nt

            vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhy = np.concatenate((vhy,vhy[:,:,:,-1:]),axis=3)
            huvymTm += (u*(vhy[:,:,:,0:-1]+vhy[:,:,:,1:])/2 - h_u*rvxv)/nt
            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()
            
        fh.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        terms = np.ma.concatenate(( hagum[:,:,:,:,np.newaxis],
                                    -huwbm[:,:,:,:,np.newaxis],
                                    -huuxpTm[:,:,:,:,np.newaxis],
                                    -huvymTm[:,:,:,:,np.newaxis],
                                    hdudtviscm[:,:,:,:,np.newaxis],
                                    hdiffum[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)

        X = dimu[keepax[1]]
        Y = dimu[keepax[0]]
        if 1 in keepax:
            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimu[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('twamomx_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twamomx_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_twamomx(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,alreadysaved=False):
    X,Y,P = extract_twamomx_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))
    plt.figure()
    ax = plt.subplot(3,2,1)
    im = m6plot((X,Y,P[:,:,0]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,2)
    im = m6plot((X,Y,P[:,:,1]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,3)
    im = m6plot((X,Y,P[:,:,2]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,4)
    im = m6plot((X,Y,P[:,:,3]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,5)
    im = m6plot((X,Y,P[:,:,4]),ax,Zmax=cmax)
    ax = plt.subplot(3,2,6)
    im = m6plot((X,Y,P[:,:,5]),ax,Zmax=cmax)

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()
