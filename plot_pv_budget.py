import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time

def extract_pvterms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,alreadysaved=False):


    if not alreadysaved:
        keepax = ()
        for i in range(4):
            if i not in meanax:
                keepax += (i,)

        fh = mfdset(fil)
        (xs,xe),(ys,ye),dimq = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
                slat=ystart, nlat=yend,zs=zs,ze=ze,xhxq='xq',yhyq='yq')
        _,_,dimqp1 = rdp1.getdimsbyindx(fh,xs,xe,ys-1,ye,zs=zs,ze=ze,xhxq='xq',yhyq='yq')
        D, (ah,aq),(dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp1.getgeombyindx(geofil
                ,xs,xe,ys,ye)[0:3]
        D_u = rdp1.getgeombyindx(geofil,xs,xe,ys-1,ye)[0]
        D_v = rdp1.getgeombyindx(geofil,xs-1,xe,ys,ye)[0]
        nt = dimq[0].size
        omega = 2*np.pi/24/3600 + 2*np.pi/24/3600/365
        R = 6378000
        f_q = 2*omega*np.sin(dimq[2]*np.pi/180)
        beta = 2*omega*np.cos(dimq[2][None,:,None]*np.pi/180)/R
        t0 = time.time()

        frhatu = fh.variables['frhatu'][0:1,zs:ze,ys-1:ye,xs:xe]
        frhatv = fh.variables['frhatv'][0:1,zs:ze,ys:ye,xs-1:xe]
        h_u = frhatu*D_u[np.newaxis,np.newaxis,:,:]
        h_v = frhatv*D_v[np.newaxis,np.newaxis,:,:]
        h_q = 0.25*(h_u[:,:,0:-1,:] + h_u[:,:,1:,:] + h_v[:,:,:,0:-1] + h_v[:,:,:,1:])

        h = fh.variables['h'][0:1,zs:ze,ys:ye,xs:xe]
        u = fh.variables['u'][0:1,zs:ze,ys:ye,xs-1:xe]
        v = fh.variables['v'][0:1,zs:ze,ys-1:ye,xs:xe]
        uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs-1:xe]
        vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        cau = fh.variables['CAu'][0:1,zs:ze,ys:ye+1,xs:xe]
        gkeu = fh.variables['gKEu'][0:1,zs:ze,ys:ye+1,xs:xe]
        rvxv = fh.variables['rvxv'][0:1,zs:ze,ys:ye+1,xs:xe]
        pfu = fh.variables['PFu'][0:1,zs:ze,ys:ye+1,xs:xe]
        dudtvisc = fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye+1,xs:xe]
        diffu = fh.variables['diffu'][0:1,zs:ze,ys:ye+1,xs:xe]
        cav = fh.variables['CAv'][0:1,zs:ze,ys:ye,xs:xe+1]
        gkev = fh.variables['gKEv'][0:1,zs:ze,ys:ye,xs:xe+1]
        rvxu = fh.variables['rvxu'][0:1,zs:ze,ys:ye,xs:xe+1]
        pfv = fh.variables['PFv'][0:1,zs:ze,ys:ye,xs:xe+1]
        diffv = fh.variables['diffv'][0:1,zs:ze,ys:ye,xs:xe+1]
        dvdtvisc = fh.variables['dv_dt_visc'][0:1,zs:ze,ys:ye,xs:xe+1]

        uadv = -gkeu - rvxv
        uadv = np.ma.filled(uadv.astype(float), 0)
        uadvym = h_q*(np.diff(uadv,axis=2)/dybu)/nt

        vadv = -gkev - rvxu
        vadv = np.ma.filled(vadv.astype(float), 0)
        vadvxm = h_q*(diffx_bound(vadv,dxbu))/nt

        fv = cau - gkeu - rvxv
        fv = np.ma.filled(fv.astype(float), 0)
        fvym = h_q*(np.diff(fv,axis=2)/dybu)/nt

        fu = -cav + gkev + rvxu
        fu = np.ma.filled(fu.astype(float), 0)
        fuxm = h_q*(diffx_bound(fu,dxbu))/nt

        Y1 = diffv
        Y1 = np.ma.filled(Y1.astype(float), 0)
        Y1xm = h_q*(diffx_bound(Y1,dxbu))/nt

        Y2 = dvdtvisc
        Y2 = np.ma.filled(Y2.astype(float), 0)
        Y2xm = h_q*(diffx_bound(Y2,dxbu))/nt

#        hbetavm = (h_v*beta*v)/nt

        u = np.ma.filled(u.astype(float), 0)
        ux = np.diff(u,axis=3)/dxt

        v = np.ma.filled(v.astype(float), 0)
        vy = np.diff(v,axis=2)/dyt

        uh = np.ma.filled(uh.astype(float), 0)
        uh_x = np.diff(uh,axis=3)/ah
        uhxm = f_q[np.newaxis,np.newaxis,:,np.newaxis]*(uh_x - h*ux)/nt

        vh = np.ma.filled(vh.astype(float), 0)
        vh_y = np.diff(vh,axis=2)/ah
        vhym = f_q[np.newaxis,np.newaxis,:,np.newaxis]*(vh_y - h*vy)/nt

        fdiapycm = f_q[np.newaxis,np.newaxis,:,np.newaxis]*np.diff(wd,axis=1)/nt

        if 1 in keepax:
            em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt

        for i in range(1,nt):
            frhatu = fh.variables['frhatu'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            frhatv = fh.variables['frhatv'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            h_u = frhatu*D_u[np.newaxis,np.newaxis,:,:]
            h_v = frhatv*D_v[np.newaxis,np.newaxis,:,:]
            h_q = 0.25*(h_u[:,:,0:-1,:] + h_u[:,:,1:,:] + h_v[:,:,:,0:-1] + h_v[:,:,:,1:])

            h = fh.variables['h'][i:i+1,zs:ze,ys:ye,xs:xe]
            u = fh.variables['u'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            v = fh.variables['v'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
            vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
            wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]
            cau = fh.variables['CAu'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            gkeu = fh.variables['gKEu'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            rvxv = fh.variables['rvxv'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            pfu = fh.variables['PFu'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            dudtvisc = fh.variables['du_dt_visc'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            diffu = fh.variables['diffu'][i:i+1,zs:ze,ys:ye+1,xs:xe]
            cav = fh.variables['CAv'][i:i+1,zs:ze,ys:ye,xs:xe+1]
            gkev = fh.variables['gKEv'][i:i+1,zs:ze,ys:ye,xs:xe+1]
            rvxu = fh.variables['rvxu'][i:i+1,zs:ze,ys:ye,xs:xe+1]
            pfv = fh.variables['PFv'][i:i+1,zs:ze,ys:ye,xs:xe+1]
            diffv = fh.variables['diffv'][i:i+1,zs:ze,ys:ye,xs:xe+1]
            dvdtvisc = fh.variables['dv_dt_visc'][i:i+1,zs:ze,ys:ye,xs:xe+1]

            uadv = -gkeu - rvxv
            uadv = np.ma.filled(uadv.astype(float), 0)
            uadvym += h_q*(np.diff(uadv,axis=2)/dybu)/nt

            vadv = -gkev - rvxu
            vadv = np.ma.filled(vadv.astype(float), 0)
            vadvxm += h_q*(diffx_bound(vadv,dxbu))/nt

            fv = cau - gkeu - rvxv
            fv = np.ma.filled(fv.astype(float), 0)
            fvym += h_q*(np.diff(fv,axis=2)/dybu)/nt

            fu = -cav + gkev + rvxu
            fu = np.ma.filled(fu.astype(float), 0)
            fuxm += h_q*(diffx_bound(fu,dxbu))/nt

            Y1 = diffv
            Y1 = np.ma.filled(Y1.astype(float), 0)
            Y1xm += h_q*(diffx_bound(Y1,dxbu))/nt

            Y2 = dvdtvisc
            Y2 = np.ma.filled(Y2.astype(float), 0)
            Y2xm += h_q*(diffx_bound(Y2,dxbu))/nt

#            hbetavm += (h_v*beta*v)/nt

            u = np.ma.filled(u.astype(float), 0)
            ux = np.diff(u,axis=3)/dxt

            v = np.ma.filled(v.astype(float), 0)
            vy = np.diff(v,axis=2)/dyt

            uh = np.ma.filled(uh.astype(float), 0)
            uh_x = np.diff(uh,axis=3)/ah
            uhxm += f_q[np.newaxis,np.newaxis,:,np.newaxis]*(uh_x - h*ux)/nt

            vh = np.ma.filled(vh.astype(float), 0)
            vh_y = np.diff(vh,axis=2)/ah
            vhym += f_q[np.newaxis,np.newaxis,:,np.newaxis]*(vh_y - h*vy)/nt

            fdiapycm += f_q[np.newaxis,np.newaxis,:,np.newaxis]*np.diff(wd,axis=1)/nt

            if 1 in keepax:
                em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt

            sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
            sys.stdout.flush()

        fh.close()
        print('Time taken for reading: {}'.format(time.time()-t0))

        terms = np.ma.concatenate((uadvym[:,:,:,:,np.newaxis],
            vadvxm[:,:,:,:,np.newaxis],
            Y1xm[:,:,:,:,np.newaxis],
            Y2xm[:,:,:,:,np.newaxis],
            fdiapycm[:,:,:,:,np.newaxis],
            uhxm[:,:,:,:,np.newaxis],
            vhym[:,:,:,:,np.newaxis]),axis=4)

        termsm = np.ma.apply_over_axes(np.nanmean, terms, meanax)

        X = dimq[keepax[1]]
        Y = dimq[keepax[0]]
        if 1 in keepax:
            elm = 0.5*(em[:,0:-1,:,:]+em[:,1:,:,:])
            em = np.ma.apply_over_axes(np.mean, em, meanax)
            elm = np.ma.apply_over_axes(np.mean, elm, meanax)
            Y = elm.squeeze()
            X = np.meshgrid(X,dimq[1])[0]

        P = termsm.squeeze()
        P = np.ma.filled(P.astype(float), np.nan)
        X = np.ma.filled(X.astype(float), np.nan)
        Y = np.ma.filled(Y.astype(float), np.nan)
        np.savez('pv_terms', X=X,Y=Y,P=P)
    else:
        npzfile = np.load('pv_terms.npz')
        X = npzfile['X']
        Y = npzfile['Y']
        P = npzfile['P']
        
    return (X,Y,P)

def plot_pv(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,alreadysaved=False):
    X,Y,P = extract_pvterms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
            alreadysaved)
    cmax = np.nanmax(np.absolute(P))
    plt.figure()
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
    for i in range(P.shape[-1]):
        ax = plt.subplot(4,2,i+1)
        im = m6plot((X,Y,P[:,:,i]),ax,Zmax=cmax,titl=ti[i])
        if i % 2:
            ax.set_yticklabels([])
        else:
            plt.ylabel('z (m)')

        if i > 5:
            plt.xlabel('x from EB (Deg)')
        else:
            ax.set_xticklabels([])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),Zmax=cmax)
        plt.show()

def diffx_bound(var,delta):
    try:
        varx = np.diff(var,axis=3)/delta
    except:
        var = np.concatenate((var, -var[:,:,:,[-1]]),axis=3)
        varx = np.diff(var,axis=3)/delta
    return varx

def diffy_bound(var,delta):
    try:
        vary = np.diff(var,axis=2)/delta
    except:
        var = np.concatenate((var, -var[:,:,[-1],:]),axis=2)
        vary = np.diff(var,axis=2)/delta
    return vary
