import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
import copy
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc, getTatzc, getTatzc2
from pym6 import Domain, Variable, Plotter
import importlib
importlib.reload(Domain)
importlib.reload(Variable)
importlib.reload(Plotter)
gv = Variable.GridVariable


def plot_eddy_velocities(initializer, ts=0, te=1, z=None):
    if z:
        initializer.z = z

    domain = Domain.Domain(initializer)
    initializer_instantaneous = copy.copy(initializer)
    initializer_instantaneous.ts = ts
    initializer_instantaneous.te = te
    domain1 = Domain.Domain(initializer_instantaneous)

    with mfdset(initializer.fil) as fh, mfdset(
            initializer.fil2) as fh2, mfdset(initializer.fil3) as fh3:
        e_cu = (gv('e', domain, 'hi', fh2, fh, plot_loc='ui')
                .xep().read_array(extend_kwargs={'method': 'mirror'})
                .move_to('ui'))
        ur = gv('uh_masked',
                domain,
                'ul',
                fh2,
                fh,
                plot_loc='ul',
                divisor='h_Cu',
                name=r'$\hat{u}$').read_array(
                    divide_by_dy=True, filled=0)
        u = gv('u', domain1, 'ul', fh3, plot_loc='ul', name=r'$u$').read_array(
            filled=0, tmean=False)
        udd = u - ur
        udd = udd.toz(initializer.z, e=e_cu)
        udd.name = r"$u''$"
        u = u.toz(initializer.z, e=e_cu)
        um = gv(
            'u', domain, 'ul', fh2, fh, plot_loc='ul',
            name=r'$\bar{u}$').read_array(filled=0).toz(initializer.z, e=e_cu)
        ud = u - um
        ud.name = r"$u'$"

        e_cv = (gv('e', domain, 'hi', fh2, fh, plot_loc='vi')
                .yep().read_array(extend_kwargs={'method': 'mirror'})
                .move_to('vi'))
        vr = gv('vh_masked',
                domain,
                'vl',
                fh2,
                fh,
                plot_loc='vl',
                divisor='h_Cv',
                name=r'$\hat{v}$').read_array(
                    divide_by_dy=True, filled=0)
        v = gv('v', domain1, 'vl', fh3, plot_loc='vl', name=r'$v$').read_array(
            filled=0, tmean=False)
        vdd = v - vr
        vdd = vdd.toz(initializer.z, e=e_cv)
        vdd.name = r"$v''$"
        v = v.toz(initializer.z, e=e_cv)
        vm = gv(
            'v', domain, 'vl', fh2, fh, plot_loc='vl',
            name=r'$\bar{v}$').read_array(filled=0).toz(initializer.z, e=e_cv)
        vd = v - vm
        vd.name = r"$v'$"

    budgetlist = [u, ud, udd, v, vd, vdd]
    ax, fig = Plotter.budget_plot(
        budgetlist,
        initializer.meanax,
        ncols=3,
        plot_kwargs=dict(cmap='RdBu_r'),
        individual_cbars=False)

    #        fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    #        u.plot(
    #            'nanmean',
    #            initializer.meanax,
    #            ax=ax[0, 0],
    #            cmap='RdBu_r',
    #            cbar=True)
    #        ud.plot(
    #            'nanmean',
    #            initializer.meanax,
    #            ax=ax[0, 1],
    #            cmap='RdBu_r',
    #            cbar=True)
    #        udd.plot(
    #            'nanmean',
    #            initializer.meanax,
    #            ax=ax[0, 2],
    #            cmap='RdBu_r',
    #            cbar=True)
    #        v.plot(
    #            'nanmean',
    #            initializer.meanax,
    #            ax=ax[1, 0],
    #            cmap='RdBu_r',
    #            cbar=True)
    #        vd.plot(
    #            'nanmean',
    #            initializer.meanax,
    #            ax=ax[1, 1],
    #            cmap='RdBu_r',
    #            cbar=True)
    #        vdd.plot(
    #            'nanmean',
    #            initializer.meanax,
    #            ax=ax[1, 2],
    #            cmap='RdBu_r',
    #            cbar=True)

    return fig
