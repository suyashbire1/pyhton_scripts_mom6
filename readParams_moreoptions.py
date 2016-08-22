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
    """
    Usage:
    | (xh,yh), (xq,yq), (zi,zl), time = getdims(filename)
    | This function returns all the dimensions from any MOM6 output file.
    """
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

def getvar(var,filename,wlon=-25,elon=0,slat=10,nlat=60,
        zs=0,ze=None,ts=0,te=None, xhxq='xh',yhyq='yh',zlzi='zl'):
    """
    Usage:
    | var = getvar(var,filename,wlon,elon,slat,nlat,zs,ze,ts,te,xhxq,yhyq)
    | This function extracts a slice from a MOM6 output file in netcdf format.
    | The x and y limits for slicing are given in degrees, whereas depth and
    | time are given in index numbers.
    | ** Future: Add capability for striding.
    """
    (xh,yh), (xq,yq), (zi,zl), time = getdims(filename)
    x = eval(xhxq)
    y = eval(yhyq)
    z = eval(zlzi)
    xs = (x >= wlon).nonzero()[0][0]
    xe = (x <= elon).nonzero()[0][-1]
    ys = (y >= slat).nonzero()[0][0]
    ye = (y <= nlat).nonzero()[0][-1]
    if te:
        if ze:
            rvar = fh.variables[var][ts:te+1,zs:ze+1,ys:ye+1,xs:xe+1]
            rz = z[zs:ze+1]
            rt = time[ts:te+1]
        else:
            rvar = fh.variables[var][ts:te+1,zs:,ys:ye+1,xs:xe+1]
            rz = z[zs:]
            rt = time[ts:te+1]
    else:
        if ze:
            rvar = fh.variables[var][ts:,zs:ze+1,ys:ye+1,xs:xe+1]
            rz = z[zs:ze+1]
            rt = time[ts:]
        else:
            rvar = fh.variables[var][ts:,zs:,ys:ye+1,xs:xe+1]
            rz = z[zs:]
            rt = time[ts:]

    rx = x[xs:xe+1]
    ry = y[ys:ye+1]
    fh = mfdset(filename)
    return (rt,rz,ry,rx), rvar 
    fh.close()

