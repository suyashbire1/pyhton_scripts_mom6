import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot import m6plot
import numpy as np
from netCDF4 import MFDataset as mfdset
import time
import xarray as xr

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
        D, (ah,aq),(dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp1.getgeombyindx(geofil
                ,xs,xe,ys,ye)[0:3]
        nt = dimq[0].size
        omega = 2*np.pi/24/3600 + 2*np.pi/24/3600/365
        R = 6378000
        f_q = 2*omega*np.sin(dimq[2]*np.pi/180)
        beta = 2*omega*np.cos(dimq[2][None,:,None]*np.pi/180)/R
        t0 = time.time()

        frhatu = fh.variables['frhatu'][0:1,zs:ze,ys:ye,xs:xe]
        frhatv = fh.variables['frhatv'][0:1,zs:ze,ys:ye,xs:xe]
        h_u = frhatu*D[np.newaxis,np.newaxis,:,:]
        h_v = frhatv*D[np.newaxis,np.newaxis,:,:]
        h_q = 0.25*(h_u + np.roll(h_u,-1,axis=2) + h_v + np.roll(h_v,-1,axis=3))

        h = fh.variables['h'][0:1,zs:ze,ys:ye,xs:xe]
        u = fh.variables['u'][0:1,zs:ze,ys:ye,xs:xe]
        v = fh.variables['v'][0:1,zs:ze,ys:ye,xs:xe]
        uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs:xe]
        vh = fh.variables['vh'][0:1,zs:ze,ys:ye,xs:xe]
        wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]
        cau = fh.variables['CAu'][0:1,zs:ze,ys:ye,xs:xe]
        gkeu = fh.variables['gKEu'][0:1,zs:ze,ys:ye,xs:xe]
        rvxv = fh.variables['rvxv'][0:1,zs:ze,ys:ye,xs:xe]
        pfu = fh.variables['PFu'][0:1,zs:ze,ys:ye,xs:xe]
        cav = fh.variables['CAv'][0:1,zs:ze,ys:ye,xs:xe]
        gkev = fh.variables['gKEv'][0:1,zs:ze,ys:ye,xs:xe]
        rvxu = fh.variables['rvxu'][0:1,zs:ze,ys:ye,xs:xe]
        pfv = fh.variables['PFv'][0:1,zs:ze,ys:ye,xs:xe]
        diffu = fh.variables['diffu'][0:1,zs:ze,ys:ye,xs:xe]
        diffv = fh.variables['diffv'][0:1,zs:ze,ys:ye,xs:xe]
        dudtvisc = fh.variables['du_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]
        dvdtvisc = fh.variables['dv_dt_visc'][0:1,zs:ze,ys:ye,xs:xe]

        uadv = -gkeu - rvxv
        uadv = np.ma.filled(uadv.astype(float), 0)
        uadvy = np.diff(np.concatenate((uadv,uadv[:,:,[-1],:]),axis=2),axis = 2)/dybu

        vadv = -gkev - rvxu
        vadv = np.ma.filled(vadv.astype(float), 0)
        vadvx = np.diff(np.concatenate((vadv,vadv[:,:,:,[-1]]),axis=3),axis = 3)/dxbu

        fv = cau - gkeu - rvxv
        fv = np.ma.filled(fv.astype(float), 0)
        fvy = np.diff(np.concatenate((fv,-fv[:,:,[-1],:]),axis=2),axis = 2)/dybu

        fu = -cav + gkev + rvxu
        fu = np.ma.filled(fu.astype(float), 0)
        fux = np.diff(np.concatenate((fu,-fu[:,:,:,[-1]]),axis=3),axis = 3)/dxbu

        Y1 = diffv
        Y1 = np.ma.filled(Y1.astype(float), 0)
        Y1x = np.diff(np.concatenate((Y1,Y1[:,:,:,[-1]]),axis=3),axis = 3)/dxbu

        Y2 = dvdtvisc
        Y2 = np.ma.filled(Y2.astype(float), 0)
        Y2x = np.diff(np.concatenate((Y2,Y2[:,:,:,[-1]]),axis=3),axis = 3)/dxbu

        hbetav = h_v*beta*v

        u = np.ma.filled(u.astype(float), 0)
        ux = np.diff(np.concatenate((u[:,:,:,[-1]],u),axis=3),axis = 3)/dxt

        v = np.ma.filled(v.astype(float), 0)
        vy = np.diff(np.concatenate((v[:,:,[-1],:],v),axis=2),axis = 2)/dyt

        uh = np.ma.filled(uh.astype(float), 0)
        uh_x = np.diff(np.concatenate((uh[:,:,:,[-1]],uh),axis=3),axis = 3)/ah
        uhx = uh_x - h*ux 

        vh = np.ma.filled(vh.astype(float), 0)
        vh_y = np.diff(np.concatenate((vh[:,:,[-1],:],vh),axis=2),axis = 2)/ah
        vhy = vh_y - h*vy 

        fdiapycm = f_q*np.diff(wd,axis=0)

        print('done!')
