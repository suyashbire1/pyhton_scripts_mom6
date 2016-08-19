import numpy as np
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset

def getgeom(filename):
    """
    Usage: 
    | D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f = getgeom(filename)
    | This fuction returns the depth of the domain D, 
    | cell areas at T and Bu points ah and aq, resp,
    | grid spacing at Cu, Cv, Bu and T points,
    | and f at Bu points.
    """
    fhgeo = dset(filename, mode='r')
    D = fhgeo.variables['D'][:]
    ah = fhgeo.variables['Ah'][:]
    aq = fhgeo.variables['Aq'][:]
    dxcu = fhgeo.variables['dxCu'][:]
    dycu = fhgeo.variables['dyCu'][:]
    dxcv = fhgeo.variables['dxCv'][:]
    dycv = fhgeo.variables['dyCv'][:]
    dxbu = fhgeo.variables['dxBu'][:]
    dybu = fhgeo.variables['dyBu'][:]
    dxt = fhgeo.variables['dxT'][:]
    dyt = fhgeo.variables['dyT'][:]
    f = fhgeo.variables['f'][:]
    fhgeo.close()
    return D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f

def getdims(filename):
    fh = mfdset(filename)
    xq = fh.variables['xq'][:]
    yq = fh.variables['yq'][:]
    time = fh.variables['Time'][:]
    xh = fh.variables['xh'][:]
    yh = fh.variables['yh'][:]
    zi = fh.variables['zi'][:]
    zl = fh.variables['zl'][:]
    fh.close()
    return (xh,yh), (xq,yq), (zi,zl), time

def getuvhe(filename,i):
    fh = mfdset(filename)
    h = fh.variables['h'][i,:,:,:] 
    e = fh.variables['e'][i,:,:,:] 
    el = (e[0:-1,:,:] + e[1:,:,:])/2;
    u = fh.variables['u'][i,:,:,:]
    v = fh.variables['v'][i,:,:,:]
    fh.close()
    return (u,v,h,e,el)

