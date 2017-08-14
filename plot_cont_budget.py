import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
from pym6 import Domain, Variable, Plotter
import importlib
importlib.reload(Domain)
importlib.reload(Variable)
importlib.reload(Plotter)
gv = Variable.GridVariable

def extract_cb_terms_pym6(initializer):

    domain = Domain.Domain(initializer)

    with mfdset(initializer.fil) as fh, mfdset(initializer.fil2) as fh2:

        vhy = gv('vh',domain,'vl',fh2,fh,plot_loc='hl').ysm().read_array(filled=0).ddx(2,div_by_area=True)
        uhx = gv('uh',domain,'ul',fh2,fh,plot_loc='hl').xsm().read_array(filled=0).ddx(3,div_by_area=True)
        wd = gv('wd',domain,'hi',fh2).read_array().o1diff(1)
#    budgetlist = [-uhx*(1/domain.dyT[uhx._slice[2:]]),-vhy*(1/domain.dxT[vhy._slice[2:]]),-wd]
    budgetlist = [-uhx,-vhy,-wd]
    lab = [ r'$(\bar{h}\hat{u})_{\tilde{x}}$',
            r'$(\bar{h}\hat{v})_{\tilde{y}}$',
            r'$(\bar{h}\hat{\varpi})_{\tilde{b}}$']
    for i,var in enumerate(budgetlist):
        var.name = lab[i]
    return budgetlist

def plot_cb_pym6(initializer,perc=99):
    budgetlist = extract_cb_terms_pym6(initializer)
    with mfdset(initializer.fil) as fh, mfdset(initializer.fil2) as fh2:
        e = gv('e',budgetlist[0].dom,'hi',fh2).read_array()
    plot_kwargs = dict(cmap='RdBu_r')
    plotter_kwargs = dict(zcoord=True,z=initializer.z,e=e,isop_mean=True)

    fig = Plotter.budget_plot(budgetlist,initializer.meanax,
                              plot_kwargs=plot_kwargs,
                              plotter_kwargs=plotter_kwargs,
                              perc=perc,individual_cbars=False)
    return fig

def extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax):

    keepax = ()
    for i in range(4):
        if i not in meanax:
            keepax += (i,)

    fh = mfdset(fil)
    (xs,xe),(ys,ye),dimh = rdp1.getlatlonindx(fh,wlon=xstart,elon=xend,
            slat=ystart, nlat=yend,zs=zs,ze=ze)
    fhgeo = dset(geofil)
    D, (ah,aq) = rdp1.getgeombyindx(fhgeo,xs,xe,ys,ye)[0:2]
    fhgeo.close()
    nt = dimh[0].size
    t0 = time.time()

    uh = fh.variables['uh'][0:1,zs:ze,ys:ye,xs-1:xe]
    vh = fh.variables['vh'][0:1,zs:ze,ys-1:ye,xs:xe]
    em = fh.variables['e'][0:1,zs:ze,ys:ye,xs:xe]/nt
    #dhdtm = fh.variables['dhdt'][0:1,zs:ze,ys:ye,xs:xe]/nt
    wd = fh.variables['wd'][0:1,zs:ze,ys:ye,xs:xe]

    uh = np.ma.filled(uh.astype(float), 0)
    uhx = np.diff(uh,axis = 3)/ah
    uhxm = uhx/nt

    vh = np.ma.filled(vh.astype(float), 0)
    vhy = np.diff(vh,axis = 2)/ah
    vhym = vhy/nt

    wdm = np.diff(wd,axis=1)/nt

    for i in range(1,nt):
        uh = fh.variables['uh'][i:i+1,zs:ze,ys:ye,xs-1:xe]
        vh = fh.variables['vh'][i:i+1,zs:ze,ys-1:ye,xs:xe]
        em += fh.variables['e'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
        #dhdtm += fh.variables['dhdt'][i:i+1,zs:ze,ys:ye,xs:xe]/nt
        wd = fh.variables['wd'][i:i+1,zs:ze,ys:ye,xs:xe]

        uh = np.ma.filled(uh.astype(float), 0)
        uhx = np.diff(uh,axis = 3)/ah
        uhxm += uhx/nt

        vh = np.ma.filled(vh.astype(float), 0)
        vhy = np.diff(vh,axis = 2)/ah
        vhym += vhy/nt

        wdm += np.diff(wd,axis=1)/nt

        sys.stdout.write('\r'+str(int((i+1)/nt*100))+'% done...')
        sys.stdout.flush()

    fh.close()
    print('Total reading time: {}s'.format(time.time()-t0))

    terms = np.ma.concatenate(( uhxm[:,:,:,:,np.newaxis],
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
    P = np.ma.filled(P.astype(float), np.nan)
    X = np.ma.filled(X.astype(float), np.nan)
    Y = np.ma.filled(Y.astype(float), np.nan)
    np.savez('cb_terms', X=X,Y=Y,P=P)

    return (X,Y,P)


def plot_cb(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,
        savfil=None,perc=99):
    X,Y,P = extract_cb_terms(geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax)
    cmax = np.nanpercentile(np.absolute(P),perc)
    fig,axc = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10, 3))
    ti = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    lab = [ r'$(\bar{h}\hat{u})_{\tilde{x}}$',
            r'$(\bar{h}\hat{v})_{\tilde{y}}$',
            r'$(\bar{h}\hat{\varpi})_{\tilde{b}}$']

    for i in range(P.shape[-1]):
        ax = axc.ravel()[i]
        im = m6plot((X,Y,P[:,:,i]),ax,vmax=cmax,vmin=-cmax,#ptype='imshow',
                txt=lab[i], ylim=(-2500,0),cmap='RdBu_r',cbar=False)

        xdegtokm(ax,0.5*(ystart+yend))
        if i == 0:
            ax.set_ylabel('z (m)')
        ax.set_ylim(-1500,0)

    fig.tight_layout()
    cb = fig.colorbar(im, ax=axc.ravel().tolist())
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w',
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        im = m6plot((X,Y,np.sum(P,axis=2)),vmax=cmax,vmin=-cmax,cmap='RdBu_r')
        plt.show()
