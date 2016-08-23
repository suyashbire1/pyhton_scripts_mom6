import numpy as np
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset

def getgeom(filename,wlon=-25,elon=0,slat=10,nlat=60):
    """
    Usage: 
    | D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f = getgeom(filename)
    | This fuction returns the depth of the domain D, 
    | cell areas at T and Bu points ah and aq, resp,
    | grid spacing at Cu, Cv, Bu and T points,
    | and f at Bu points.
    """
    fhgeo = dset(filename, mode='r')
    lath = fhgeo.variables['lath'][:]
    lonh = fhgeo.variables['lonh'][:]
    xs = (lonh >= wlon).nonzero()[0][0]
    xe = (lonh <= elon).nonzero()[0][-1]
    ys = (lath >= slat).nonzero()[0][0]
    ye = (lath <= nlat).nonzero()[0][-1]
    D = fhgeo.variables['D'][ys:ye+1,xs:xe+1]
    ah = fhgeo.variables['Ah'][ys:ye+1,xs:xe+1]
    aq = fhgeo.variables['Aq'][ys:ye+1,xs:xe+1]
    dxcu = fhgeo.variables['dxCu'][ys:ye+1,xs:xe+1]
    dycu = fhgeo.variables['dyCu'][ys:ye+1,xs:xe+1]
    dxcv = fhgeo.variables['dxCv'][ys:ye+1,xs:xe+1]
    dycv = fhgeo.variables['dyCv'][ys:ye+1,xs:xe+1]
    dxbu = fhgeo.variables['dxBu'][ys:ye+1,xs:xe+1]
    dybu = fhgeo.variables['dyBu'][ys:ye+1,xs:xe+1]
    dxt = fhgeo.variables['dxT'][ys:ye+1,xs:xe+1]
    dyt = fhgeo.variables['dyT'][ys:ye+1,xs:xe+1]
    f = fhgeo.variables['f'][ys:ye+1,xs:xe+1]
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
    fh = mfdset(filename)
    rvar = fh.variables[var][ts:te,zs:ze,ys:ye+1,xs:xe+1]
    rz = z[zs:ze]
    rt = time[ts:te]
    rx = x[xs:xe+1]
    ry = y[ys:ye+1]
    return (rt,rz,ry,rx), rvar 
    fh.close()

