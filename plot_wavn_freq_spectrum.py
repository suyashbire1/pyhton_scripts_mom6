import sys
import readParams_moreoptions as rdp1
import matplotlib.pyplot as plt
from mom_plot1 import m6plot, xdegtokm
import numpy as np
from netCDF4 import MFDataset as mfdset, Dataset as dset
import time
import pyximport
pyximport.install()
from getvaratzc import getvaratzc5, getvaratzc, getTatzc, getTatzc2
from pym6 import Domain, Variable, Plotter
import importlib
importlib.reload(Domain)
importlib.reload(Variable)
importlib.reload(Plotter)
gv = Variable.GridVariable


def plot_wf_spectrum(initializer, perc=99):

    domain = Domain.Domain(initializer)

    with mfdset(initializer.fil) as fh:
        e = gv('e', domain, 'hi', fh).read_array(tmean='anom')
        fig, ax = plt.subplots(1, 2)
        e.plot(
            'nanmean',
            initializer.meanax,
            ax=ax[0],
            cbar=True,
            contour=False,
            plot_kwargs=dict(cmap='RdBu_r'))

        initializer.ls = 0
        initializer.le = None
        domain1 = Domain.Domain(initializer)
        h = gv('e', domain1, 'hi', fh).read_array().o1diff(1).values
        omega = 2 * np.pi * (1 / 24 / 3600 + 1 / 365 / 24 / 3600)
        lr = np.sum(np.sqrt(h * domain.db) / np.pi / (
            2 * omega *
            np.sin(np.radians(0.5 * (initializer.slat + initializer.nlat)))),
                    axis=1,
                    keepdims=True)
        lr = np.mean(lr, axis=(0, 1, 2, 3)).squeeze()

        sig = e.mean(axis=2).values.squeeze()
        fx, fx1, ft, Rsq = cross_spectrum(sig, 5 * 24 * 3600, 4000)

        vmax = np.percentile(Rsq, 99.9)
        im = ax[1].pcolormesh(fx1, ft, Rsq, cmap='Reds', vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax[1])
        beta = 2 * omega * np.cos(
            np.radians(0.5 * (initializer.slat + initializer.nlat
                              ))) / domain.R_earth
        w = -beta * fx / (fx**2 + lr**-2)
        ax[1].plot(-fx, -w, 'k')
    return fig, ax


def plot_wf_spectrum_vort(initializer, perc=99):

    domain = Domain.Domain(initializer)

    with mfdset(initializer.fil) as fh:
        e = gv('e', domain, 'hi', fh, plot_loc='qi').xsm().ysm().read_array(
            tmean=False,
            extend_kwargs=dict(method='mirror')).move_to('ui').move_to('qi')
        uy = gv('u', domain, 'ul', fh, plot_loc='ql').yep().read_array(
            tmean=False, filled=0,
            extend_kwargs=dict(method='vorticity')).ddx(2)
        vx = gv('v', domain, 'vl', fh, plot_loc='ql').xep().read_array(
            tmean=False, filled=0,
            extend_kwargs=dict(method='vorticity')).ddx(3)
        vort = (vx - uy).toz([-1], e=e)
        vort.name = ''
        fig, ax = plt.subplots(1, 2)
        vort.plot(
            'nanmean',
            initializer.meanax,
            perc=98,
            contour=False,
            cbar=True,
            plot_kwargs=dict(cmap='RdBu_r'),
            ax=ax[0])

        initializer.ls = 0
        initializer.le = None
        domain1 = Domain.Domain(initializer)
        h = gv('e', domain1, 'hi', fh).read_array().o1diff(1).values
        omega = 2 * np.pi * (1 / 24 / 3600 + 1 / 365 / 24 / 3600)
        lr = np.sum(np.sqrt(h * domain.db) / np.pi / (
            2 * omega *
            np.sin(np.radians(0.5 * (initializer.slat + initializer.nlat)))),
                    axis=1,
                    keepdims=True)
        lr = np.mean(lr, axis=(0, 1, 2, 3)).squeeze()

        sig = vort.mean(axis=2).values.squeeze()
        fx, fx1, ft, Rsq = cross_spectrum(sig, 5 * 24 * 3600, 4000)

        vmax = np.percentile(Rsq, 99.9)
        im = ax[1].pcolormesh(fx1, ft, Rsq, cmap='Reds', vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax[1])
        beta = 2 * omega * np.cos(
            np.radians(0.5 * (initializer.slat + initializer.nlat
                              ))) / domain.R_earth
        w = -beta * fx / (fx**2 + lr**-2)
        ax[1].plot(-fx, -w, 'k')
    return fig, ax


def cross_spectrum(sig, dt, dx):
    sig -= np.mean(sig, axis=0, keepdims=True)
    sig -= np.mean(sig, axis=1, keepdims=True)

    ft = np.fft.rfftfreq(sig.shape[0], d=dt)
    fx = np.fft.rfftfreq(sig.shape[1], d=dx)
    H = np.fft.rfft(sig, axis=0)
    ck, sk = np.real(H), np.imag(H)
    cw = np.fft.rfft(ck, axis=1)
    Akw, Bkw = np.real(cw), np.imag(cw)
    sw = np.fft.rfft(sk, axis=1)
    akw, bkw = np.real(sw), np.imag(sw)

    Rsqplus = 0.25 * ((Akw - bkw)**2 + (akw + Bkw)**2)
    Rsqminus = 0.25 * ((Akw + bkw)**2 + (akw - Bkw)**2)

    fx1 = np.concatenate((-np.flipud(fx[1:]), fx))
    Rsq = np.concatenate((np.fliplr(Rsqplus[:, 1:]), Rsqminus), axis=1)
    return fx, fx1, ft, Rsq
