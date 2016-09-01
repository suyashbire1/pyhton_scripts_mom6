import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time

def extract_twamomy_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
#        (xs,xe),(ys,ye),dimu = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
#                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq')
        (xs,xe),(ys,ye),dimv = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,yhyq='yq')
#        (xs,xe),(ys,ye),_ = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
#                slat=ystart, nlat=yend,zs=zs,ze=ze)
        D, (ah,aq) = rdp1.getgeombyindx(geofil,xs,xe,ys,ye)[0:2]
        nt_const = dimv[0].size
        t0 = time.time()

        print('Reading data using loop...')
        v = fh.variables['v'][0:1,zs:ze,ys:ye,xs:xe]
        nt = np.ones(v.shape)*nt_const
        frhatv = fh.variables['frhatv'][0:1,zs:ze,ys:ye,xs:xe]
        h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
        h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
        nt[h_v<=1e-3] -= 1

        cav = fh.variables['CAv'][0:1,zs:ze,ys:ye,xs:xe]
        gkev = fh.variables['gKEv'][0:1,zs:ze,ys:ye,xs:xe]
        rvxu = fh.variables['rvxu'][0:1,zs:ze,ys:ye,xs:xe]
        pfv = fh.variables['PFv'][0:1,zs:ze,ys:ye,xs:xe]
        hagvm = (h_v*(cav - gkev - rvxu + pfv)).filled(0)

        
        hdvdtviscm = (h_v*fh.variables['dv_dt_visc'][0:1,zs:ze,
            ys:ye,xs:xe]).filled(0)
        hdiffvm = (h_v*fh.variables['diffv'][0:1,zs:ze,ys:ye,xs:xe]).filled(0)

        dvdtdia = fh.variables['dvdt_dia'][0:1,zs:ze,ys:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        wd = np.diff(wd,axis=1)
        wd = np.concatenate((wd,wd[:,:,-1:,:]),axis=2)
        hvwbm = ((v*(wd[:,:,0:-1,:]+wd[:,:,1:,:])/2 - h_v*dvdtdia)).filled(0)

        uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs-1:xe]
        uh = np.ma.filled(uh.astype(float), 0)
        uhx = np.diff(uh,axis = 3)/ah
        uhx = np.concatenate((uhx,uhx[:,:,-1:,:]),axis=2)
        huvxpTm = ((v*(uhx[:,:,0:-1,:]+uhx[:,:,1:,:])/2 - h_v*rvxu)).filled(0)

        vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye,xs:xe]
        vh = np.ma.filled(vh.astype(float), 0)
        vhy = np.diff(vh,axis = 2)/ah
        vhy = np.concatenate((vhy,vhy[:,:,-1:,:]),axis=2)
        hvvymTm = ((v*(vhy[:,:,0:-1,:]+vhy[:,:,1:,:])/2 - h_v*gkev)).filled(0)

        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt_const

        for i in range(1,nt_const):
            v = fh.variables['v'][i:i+1,zs:ze,ys:ye,xs:xe]
            frhatv = fh.variables['frhatv'][i:i+1,zs:ze,ys:ye,xs:xe]
            h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
            h_v = np.ma.masked_array(h_v,mask=(h_v<=1e-3).astype(int))
            nt[h_v<=1e-3] -= 1

            cav = fh.variables['CAv'][i:i+1,zs:ze,ys:ye,xs:xe]
            gkev = fh.variables['gKEv'][i:i+1,zs:ze,ys:ye,xs:xe]
            rvxu = fh.variables['rvxu'][i:i+1,zs:ze,ys:ye,xs:xe]
            pfv = fh.variables['PFv'][i:i+1,zs:ze,ys:ye,xs:xe]
            hagvm += (h_v*(cav - gkev - rvxu + pfv)).filled(0)
            
            hdvdtviscm += (h_v*fh.variables['dv_dt_visc'][i:i+1,zs:ze,
                ys:ye,xs:xe]).filled(0)
            hdiffvm += (h_v*fh.variables['diffv'][i:i+1,zs:ze,
                ys:ye,xs:xe]).filled(0)

            dvdtdia = fh.variables['dvdt_dia'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]
            wd = np.diff(wd,axis=1)
            wd = np.concatenate((wd,wd[:,:,-1:,:]),axis=2)
            hvwbm += ((v*(wd[:,:,0:-1,:]+wd[:,:,1:,:])/2 -
                h_v*dvdtdia)).filled(0)

            uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhx = np.concatenate((uhx,uhx[:,:,-1:,:]),axis=2)
            huvxpTm += ((v*(uhx[:,:,0:-1,:]+uhx[:,:,1:,:])/2 -
                h_v*rvxu)).filled(0)

            vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhy = np.concatenate((vhy,vhy[:,:,-1:,:]),axis=2)
            hvvymTm += ((v*(vhy[:,:,0:-1,:]+vhy[:,:,1:,:])/2 -
                h_v*gkev)).filled(0)
            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt_const

            sys.stdout.write('\r'+str(int((i+1)/nt_const*100))+'% done...')
            sys.stdout.flush()
            
        fh.close()
        print('Time taken for data reading: {}s'.format(time.time()-t0))

        terms = np.ma.concatenate(( hagvm[:,:,:,:,np.newaxis],
                                    -hvwbm[:,:,:,:,np.newaxis],
                                    -huvxpTm[:,:,:,:,np.newaxis],
                                    -hvvymTm[:,:,:,:,np.newaxis],
                                    hdvdtviscm[:,:,:,:,np.newaxis],
                                    hdiffvm[:,:,:,:,np.newaxis]),
                                    axis=4)/nt[:,:,:,:,np.newaxis]

        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)

        X = dimv[keepax[1]]
        Y = dimv[keepax[0]]
        if 1 in keepax:
            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimv[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('twamomy_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('twamomy_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_twamomy(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,alreadysaved=False):
    X,Y,P = extract_twamomy_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
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
