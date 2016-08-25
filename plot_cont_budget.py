import sys
import readParams_moreoptions as rdp1
from getvaratz import *
import matplotlib.pyplot as plt
from mom_plot import m6plot

def extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
      loop=True,alreadysaved=False):

    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        D, (ah,aq) = rdp1.getgeom(geofil,wlon=xstart,
                elon=xend,slat=ystart,nlat=yend)[0:2]
        time = rdp1.getdims(fil)[3]
        nt = time.size
        #D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp1.getgeom(geofil)
        #(xh,yh), (xq,yq), (zi,zl), time = rdp1.getdims(fil)

        if loop:
            print('Reading data using loop...')
            dimu,uh = rdp1.getvar('uh',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,xhxq='xq',ts=0,te=1)
            dimv,vh = rdp1.getvar('vh',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,yhyq='yq',ts=0,te=1)
            dime,em = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,ts=0,te=1,zlzi='zi')
            dimh,dhdtm = rdp1.getvar('dhdt',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,ts=0,te=1)
            wd = rdp1.getvar('wd',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                    zs=zs,ze=ze,ts=0,te=1,zlzi='zi')[1]
            print(uh.shape,vh.shape,wd.shape,dhdtm.shape,em.shape)

            em /= nt

            dhdtm /= nt

            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(uh,axis = 3)/ah
            uhxm = uhx/nt

            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(vh,axis = 2)/ah
            vhym = vhy/nt

            wdm = np.diff(wd,axis=1)/nt

            for i in range(1,nt):
                dimu,uh = rdp1.getvar('uh',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,xhxq='xq',ts=0,te=1)
                dimv,vh = rdp1.getvar('vh',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,yhyq='yq',ts=0,te=1)
                dime,e = rdp1.getvar('e',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,ts=0,te=1,zlzi='zi')
                dimh,dhdt = rdp1.getvar('dhdt',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,ts=0,te=1)
                wd = rdp1.getvar('wd',fil,wlon=xstart,elon=xend,slat=ystart,nlat=yend,
                        zs=zs,ze=ze,ts=0,te=1,zlzi='zi')[1]

                em += e/nt

                dhdtm += dhdt/nt

                uh = np.ma.filled(uh.astype(float), 0)
                uhx = np.diff(uh,axis = 3)/ah
                uhxm += uhx/nt

                vh = np.ma.filled(vh.astype(float), 0)
                vhy = np.diff(vh,axis = 2)/ah
                vhym += vhy/nt

                wdm += np.diff(wd,axis=1)/nt

                sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
                sys.stdout.flush()

        terms = np.concatenate((dhdtm[:,:,:,:,np.newaxis],
                                uhxm[:,:,:,:,np.newaxis],
                                vhym[:,:,:,:,np.newaxis],
                                wdm[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.mean, terms, meanax)
        elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
        em = np.ma.apply_over_axes(np.mean, em, meanax)
        elm = np.ma.apply_over_axes(np.mean, elm, meanax)

        X = dimh[keepax[1]]
        Y = dimh[keepax[0]]
        if 1 in keepax:
            Y = elm.squeeze()
            X = np.meshgrid(X,dimh[1])[0]

        P = termsm.squeeze()
        np.savez('cb_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('cb_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_cb(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,loop=True,alreadysaved=False):
    X,Y,P = extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            loop,alreadysaved)
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
    im = m6plot((X,Y,np.sum(P,axis=2)),ax,Zmax=cmax)
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
