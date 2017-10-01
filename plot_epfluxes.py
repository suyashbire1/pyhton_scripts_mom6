import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc
from pym6 import Domain, Variable, Plotter
import importlib
importlib.reload(Domain)
importlib.reload(Variable)
importlib.reload(Plotter)
gv = Variable.GridVariable
import copy


def extract_ep_terms_pym6(initializer):

    domain = Domain.Domain(initializer)

    plot_loc = 'vl'
    with mfdset(initializer.fil) as fh, mfdset(initializer.fil2) as fh2:

        #initializer.z = np.linspace(-500, 0, 20)
        e = (gv('e', domain, 'hi', fh2, fh, plot_loc=plot_loc)
             .yep().read_array(
                 extend_kwargs={'method': 'mirror'}).move_to('vi'))
        h = (gv('h_Cv', domain, plot_loc, fh2, fh, plot_loc=plot_loc)
             .read_array(filled=0))
        vr = (gv('vh',
                 domain,
                 plot_loc,
                 fh2,
                 fh,
                 plot_loc=plot_loc,
                 divisor='h_Cv').read_array(
                     divide_by_dx=True, filled=0))
        hum = (gv('uh', domain, 'ul', fh2, fh, plot_loc=plot_loc).xsm().yep()
               .read_array(
                   divide_by_dy=True,
                   filled=0,
                   extend_kwargs={'method': 'vorticity'})
               .move_to('hl').move_to(plot_loc))
        huvm = (gv('huv_Bu', domain, 'ql', fh2, fh, plot_loc=plot_loc).xsm()
                .read_array(
                    filled=0,
                    extend_kwargs={'method': 'vorticity'}).move_to(plot_loc))
        uv_e_flux = (huvm - vr * hum) / h
        uv_e_flux = uv_e_flux.mean(axis=initializer.meanax).toz(initializer.z,
                                                                e=e)
        ex = (gv('e', domain, 'hi', fh2, fh, plot_loc=plot_loc)
              .xsm().xep().yep().read_array(extend_kwargs={'method': 'mirror'})
              .ddx(3).move_to('hi').move_to('vi').move_to('vl'))
        ex = ex.mean(axis=initializer.meanax).toz(initializer.z, e=e)

        e1 = (gv('e', domain, 'hi', fh2, fh, plot_loc=plot_loc)
              .yep().read_array(extend_kwargs={'method': 'mirror'})
              .move_to('hl'))
        esq = (gv('esq', domain, 'hl', fh2, fh, plot_loc=plot_loc)
               .yep().read_array(extend_kwargs={'method': 'mirror'}))
        edlsqm = esq - e1 * e1
        edlsqmy = edlsqm.ddx(2)
        hpfv = (gv('twa_hpfv', domain, plot_loc, fh2, fh, plot_loc=plot_loc)
                .read_array(filled=0))
        pfvm = (gv('PFv', domain, plot_loc, fh2, fh, plot_loc=plot_loc)
                .read_array(filled=0))
        edpfvdmb = -hpfv + h * pfvm - edlsqmy * domain.db * 0.5
        edpfvdmb1 = copy.copy(edpfvdmb)
        edpfvdmb1 = edpfvdmb1.mean(axis=initializer.meanax).toz(initializer.z,
                                                                e=e)
        edpfvdmb = edpfvdmb / domain.db
        edpfvdm = edpfvdmb.vert_integral()
        edpfvdm = edpfvdm.mean(axis=initializer.meanax).toz(initializer.z, e=e)

    flux_list = [uv_e_flux, edpfvdm]
    lab = [
        r"""$\widehat{u^{\prime \prime}v^{\prime \prime}}(\hat{i} + \bar{\zeta_{\tilde{x}}}\hat{k})$""",
        r"""$\overline{\zeta^\prime m_{\tilde{y}}^\prime} \hat{k}$""",
        r"""$\widehat{u^{\prime \prime}v^{\prime \prime}}(\hat{i} + \bar{\zeta_{\tilde{x}}}\hat{k}) +\overline{\zeta^\prime m_{\tilde{y}}^\prime} \hat{k}$"""
    ]

    dz = np.diff(initializer.z)[0]
    dx = np.radians(
        np.diff(domain.lonh[ex._plot_slice[3, 0]:ex._plot_slice[3, 1]])[0]
    ) * 6378000 * np.cos(
        np.radians(0.5 * (initializer.slat + initializer.nlat)))
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 5))
    units = 'xy'
    angles = 'xy'
    q1 = ax[0].quiver(
        domain.lonh[ex._plot_slice[3, 0]:ex._plot_slice[3, 1]],
        initializer.z[::2],
        uv_e_flux.values.squeeze()[::2] / dx * np.sqrt(dx**2 + dz**2),
        uv_e_flux.values.squeeze()[::2] * ex.values.squeeze()[::2] / dz *
        np.sqrt(dx**2 + dz**2),
        units=units)
    #        angles=angles)
    tx = ax[0].text(0.05, 0.1, lab[0], transform=ax[0].transAxes)
    tx.set_fontsize(15)
    tx.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
    ax[0].quiverkey(
        q1, 0.75, 1.05, 1e-2, r'$1e-2$', labelpos='E', coordinates='axes')
    xdegtokm(ax[0], 0.5 * (initializer.slat + initializer.nlat))
    ax[0].set_ylabel('z (m)')
    q2 = ax[1].quiver(
        domain.lonh[ex._plot_slice[3, 0]:ex._plot_slice[3, 1]],
        initializer.z[::2],
        np.zeros(edpfvdm.values.squeeze()[::2].shape),
        edpfvdm.values.squeeze()[::2] / dx * np.sqrt(dx**2 + dz**2),
        units=units)
    #        angles=angles)
    tx = ax[1].text(0.05, 0.1, lab[1], transform=ax[1].transAxes)
    tx.set_fontsize(15)
    tx.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
    ax[1].quiverkey(
        q2, 0.75, 1.05, 1e-3, r'$1e-3$', labelpos='E', coordinates='axes')
    xdegtokm(ax[1], 0.5 * (initializer.slat + initializer.nlat))
    q3 = ax[2].quiver(
        domain.lonh[ex._plot_slice[3, 0]:ex._plot_slice[3, 1]],
        initializer.z[::2],
        np.zeros(edpfvdm.values.squeeze()[::2].shape) +
        uv_e_flux.values.squeeze()[::2] / dx * np.sqrt(dx**2 + dz**2),
        (edpfvdm.values.squeeze()[::2] + uv_e_flux.values.squeeze()[::2] *
         ex.values.squeeze()[::2]) / dz * np.sqrt(dx**2 + dz**2),
        units=units)
    #        angles=angles)
    tx = ax[2].text(0.05, 0.1, lab[2], transform=ax[2].transAxes)
    tx.set_fontsize(15)
    tx.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
    ax[2].quiverkey(
        q3, 0.75, 1.05, 1e-2, r'$1e-2$', labelpos='E', coordinates='axes')
    xdegtokm(ax[2], 0.5 * (initializer.slat + initializer.nlat))

    return fig
