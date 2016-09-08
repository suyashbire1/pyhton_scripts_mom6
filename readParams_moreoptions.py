import numpy as np
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset

def getgeom(filename,wlon=-25,elon=0,slat=10,nlat=60,
        xadditional=False,yadditional=False):
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
    latq = fhgeo.variables['latq'][:]
    lonq = fhgeo.variables['lonq'][:]
    xsh = (lonh >= wlon).nonzero()[0][0]
    xeh = (lonh <= elon).nonzero()[0][-1]
    ysh = (lath >= slat).nonzero()[0][0]
    yeh = (lath <= nlat).nonzero()[0][-1]
    xsq = (lonq >= wlon).nonzero()[0][0]
    xeq = (lonq <= elon).nonzero()[0][-1]
    ysq = (latq >= slat).nonzero()[0][0]
    yeq = (latq <= nlat).nonzero()[0][-1]
    if xadditional:
        xsh -= 1
        xsq -= 1
    if yadditional:
        ysh -= 1
        ysq -= 1
    D = fhgeo.variables['D'][ysh:yeh+1,xsh:xeh+1]
    ah = fhgeo.variables['Ah'][ysh:yeh+1,xsh:xeh+1]
    aq = fhgeo.variables['Aq'][ysq:yeq+1,xsq:xeq+1]
    dxcu = fhgeo.variables['dxCu'][ysh:yeh+1,xsq:xeq+1]
    dycu = fhgeo.variables['dyCu'][ysh:yeh+1,xsq:xeq+1]
    dxcv = fhgeo.variables['dxCv'][ysq:yeq+1,xsh:xeh+1]
    dycv = fhgeo.variables['dyCv'][ysq:yeq+1,xsh:xeh+1]
    dxbu = fhgeo.variables['dxBu'][ysq:yeq+1,xsq:xeq+1]
    dybu = fhgeo.variables['dyBu'][ysq:yeq+1,xsq:xeq+1]
    dxt = fhgeo.variables['dxT'][ysh:yeh+1,xsh:xeh+1]
    dyt = fhgeo.variables['dyT'][ysh:yeh+1,xsh:xeh+1]
    f = fhgeo.variables['f'][ysq:yeq+1,xsq:xeq+1]
    fhgeo.close()
    return D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f

def getgeombyindx(filename,xs,xe,ys,ye):
    """
    Usage: 
    | D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f = getgeom(filename)
    | This fuction returns the depth of the domain D, 
    | cell areas at T and Bu points ah and aq, resp,
    | grid spacing at Cu, Cv, Bu and T points,
    | and f at Bu points.
    """
    fhgeo = dset(filename, mode='r')
    D = fhgeo.variables['D'][ys:ye,xs:xe]
    ah = fhgeo.variables['Ah'][ys:ye,xs:xe]
    aq = fhgeo.variables['Aq'][ys:ye,xs:xe]
    dxcu = fhgeo.variables['dxCu'][ys:ye,xs:xe]
    dycu = fhgeo.variables['dyCu'][ys:ye,xs:xe]
    dxcv = fhgeo.variables['dxCv'][ys:ye,xs:xe]
    dycv = fhgeo.variables['dyCv'][ys:ye,xs:xe]
    dxbu = fhgeo.variables['dxBu'][ys:ye,xs:xe]
    dybu = fhgeo.variables['dyBu'][ys:ye,xs:xe]
    dxt = fhgeo.variables['dxT'][ys:ye,xs:xe]
    dyt = fhgeo.variables['dyT'][ys:ye,xs:xe]
    f = fhgeo.variables['f'][ys:ye,xs:xe]
    fhgeo.close()
    return D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt), f

def getdims(fh):
    """
    Usage:
    | (xh,yh), (xq,yq), (zi,zl), time = getdims(filename)
    | This function returns all the dimensions from any MOM6 output file.
    """
    try:
        xq = fh.variables['xq'][:]
    except KeyError:
        xq = None
    try:
        yq = fh.variables['yq'][:]
    except KeyError:
        yq = None
    try:
        time = fh.variables['Time'][:]
    except KeyError:
        time = None
    try:
        xh = fh.variables['xh'][:]
    except KeyError:
        xh = None
    try:
        yh = fh.variables['yh'][:]
    except KeyError:
        yh = None
    try:
        zi = fh.variables['zi'][:]
    except KeyError:
        zi = None
    try:
        zl = fh.variables['zl'][:]
    except KeyError:
        zl = None

    return (xh,yh), (xq,yq), (zi,zl), time

def getvar(var,fh,filename,wlon=-25,elon=0,slat=10,nlat=60,
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
    rvar = fh.variables[var][ts:te,zs:ze,ys:ye+1,xs:xe+1]
    rz = z[zs:ze]
    rt = time[ts:te]
    rx = x[xs:xe+1]
    ry = y[ys:ye+1]
    return (rt,rz,ry,rx), rvar 

def getlatlonindx(fh,wlon=-25,elon=0,slat=10,nlat=60,
        zs=0,ze=None,ts=0,te=None,xhxq='xh',yhyq='yh',zlzi='zl'):
    (xh,yh), (xq,yq), (zi,zl), time = getdims(fh)
    x = eval(xhxq)
    y = eval(yhyq)
    z = eval(zlzi)
    xs = (x >= wlon).nonzero()[0][0]
    xe = (x <= elon).nonzero()[0][-1]+1
    ys = (y >= slat).nonzero()[0][0]
    ye = (y <= nlat).nonzero()[0][-1]+1
    rz = z[zs:ze]
    rt = time[ts:te]
    rx = x[xs:xe]
    ry = y[ys:ye]
    return (xs,xe),(ys,ye),(rt,rz,ry,rx)
