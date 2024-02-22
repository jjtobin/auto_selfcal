"""
Version0 by Ryan Loomis (?/?/22)
Version1 with updated by Rich Teague / Ryan Loomis (8/8/22)

All functions required to align several EBs.
"""
import casatasks
import casatools
import numpy as np
from numba import njit
from astropy import units as u
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord, FK5
import os
import shutil
import matplotlib.pyplot as plt
import warnings
from scipy import constants
import emcee
import dynesty
import dynesty.utils
import dynesty.plotting
import scipy.stats
import glob
import xml.etree.ElementTree as ET


skycoord_frames = {'ICRS':'icrs','J2000':FK5(equinox='J2000')}


def calculate_phase_shift(grid_vis, grid_nvis, grid_uu, grid_vv, mu_RA,
                          mu_DEC):
    """
    Apply a phase shifted to a single channel of gridded visibilities.

    Args:
        grid_vis (array): 2D array of gridded visibilities.
        grid_nvis (array): 2D array of number of gridded visibilities in cell.
        grid_uu (array): 2D array of grid u coordinates.
        grid_vv (array): 2D array of grid v coordinates.
        mu_RA (float): Offset in right ascension from phase center in [arcsec].
        mu_DEC (float): Offset in declination from phase center in [arcsec].

    Returns:
        shifted_grid_vis (array): 2D array of phase-shifted gridded
            visibilities.
    """
    phase_shifts = grid_uu * mu_RA * constants.arcsec + grid_vv * mu_DEC * constants.arcsec
    phase_shifts = np.exp(-2.0 * np.pi * 1.0j * phase_shifts)
    shifted_grid_vis = grid_vis * phase_shifts
    shifted_grid_vis[grid_nvis < 1] = 0.0 + 0.0j
    return shifted_grid_vis

def calculate_phase_difference(grid_vis1, grid_vis2, grid_wgts1, grid_wgts2,
                               phase_rms1, phase_rms2, grid_nvis, grid_model_vis1, grid_model_vis2):
    """
    Calculate the phase difference (no amplitude) between two sets of gridded
    visibilities, ``grid_vis1`` and ``grid_vis2``.

    Args:
        grid_vis1 (array): 2D array of gridded visibilities.
        grid_vis2 (array): 2D array of gridded visibilities.
        grid_wgts1 (array): 2D array of gridded weights for ``grid_vis1``.
        grid_wgts2 (array): 2D array of gridded weights for ``grid_vis2``.
        grid_nvis (array): 2D array of number of gridded visibilites in cell.

    Returns:
        phase_difference (array): 2D array of the phase differences between
            ``grid_vis1`` and ``grid_vis2``.
    """
    angle1, angle2 = np.angle(grid_vis1), np.angle(grid_vis2)

    phase_unc1 = np.minimum(np.sqrt(1./grid_wgts1 * 1./np.abs(grid_model_vis1)**2), np.repeat(2*np.pi, grid_wgts1.shape))
    phase_wgts1 = 1./(phase_unc1**2 + phase_rms1**2)
    phase_unc2 = np.minimum(np.sqrt(1./grid_wgts2 * 1./np.abs(grid_model_vis2)**2), np.repeat(2*np.pi, grid_wgts2.shape))
    phase_wgts2 = 1./(phase_unc2**2 + phase_rms2**2)

    phase_difference = np.minimum(2.0 * np.pi - np.abs(angle2 - angle1),
                                  np.abs(angle2 - angle1))**2
    phase_difference *= 1./(1./phase_wgts1 + 1./phase_wgts2)
    return phase_difference


def calculate_full_phase_difference(grid_vis1, grid_vis2, grid_wgts1,
                                    grid_wgts2, grid_nvis):
    """
    Calculate the full phase difference (including amplitude) between two sets
    of gridded visibilities, ``grid_vis1`` and ``grid_vis2``.

    Args:
        grid_vis1 (array): 2D array of gridded visibilities.
        grid_vis2 (array): 2D array of gridded visibilities.
        grid_wgts1 (array): 2D array of gridded weights for ``grid_vis1``.
        grid_wgts2 (array): 2D array of gridded weights for ``grid_vis2``.
        grid_nvis (array): 2D array of number of gridded visibilites in cell.

    Returns:
        phase_difference (array): 2D array of the phase differences between
            ``grid_vis1`` and ``grid_vis2``.
    """
    phase_difference = np.abs(grid_vis2 - grid_vis1)**2
    phase_difference *= 1./(1./grid_wgts1 + 1./grid_wgts2) * 0.5
    return phase_difference


@njit(fastmath=True)
def grid(grid_vis, grid_nvis, grid_wgts, grid_model_vis, uu, vv, du, dv, npix, vis, wgts, model_vis):
    """
    TBD

    Args:
        grid_vis (array): 2D array of gridded visibilities.
        grid_nvis (array): 2D array of number of gridded visibilities in cell.
        grid_wgts (array): 2D array of gridded weights.
        uu (array):
        vv (array):
        du (array):
        dv (array)
        npix (array):
        vis (array):
        wgts (array):

    Returns:
        grid_vis (array): 2D array of gridded visibilities.
        grid_nvis (array): 2D array of number of gridded visibilities in cell.
        grid_wgts (array): 2D array of gridded weights.
    """
    for i in np.arange(uu.size):
        uidx_a = int(npix / 2.0 + uu[i] / du + 0.5)
        uidx_b = int(npix / 2.0 - uu[i] / du + 0.5)
        vidx_a = int(npix / 2.0 + vv[i] / dv + 0.5)
        vidx_b = int(npix / 2.0 - vv[i] / dv + 0.5)
        grid_vis[uidx_a, vidx_a] += vis[i]*wgts[i]*0.5
        grid_vis[uidx_b, vidx_b] += np.conjugate(vis[i])*wgts[i]*0.5
        grid_model_vis[uidx_a, vidx_a] += model_vis[i]*wgts[i]*0.5
        grid_model_vis[uidx_b, vidx_b] += np.conjugate(model_vis[i])*wgts[i]*0.5
        grid_wgts[uidx_a, vidx_a] += wgts[i]*0.5  # *0.5 because we also add the complex conjugate, but that doesn't add data.
        grid_wgts[uidx_b, vidx_b] += wgts[i]*0.5  # *0.5 because we also add the complex conjugate, but that doesn't add data.
        grid_nvis[uidx_a, vidx_a] += 1
        grid_nvis[uidx_b, vidx_b] += 1
    return grid_vis, grid_nvis, grid_wgts, grid_model_vis


def ingest_ms(base_ms, target, npix, cell_size, grid_needs_to_cover_all_data, spwid=0, datacolumn="DATA"):
    """
    Ingest a measurement set and grid onto a regular grid with ``npix`` cells,
    each with a size ``cell_size`` in [arcsec].

    Args:
        base_ms (str): Measurement set to ingest.
        npix (int): Number of pixels for the grid.
        cell_size (float): Cell size in [arcsec] for the grid.
        grid_needs_to_cover_all_data (bool): if True, make sure that grid cover all data
        spwid (int): The spectral window to ingest; defaults to 0.

    Returns:
        grid_vis (array): 2D array of gridded visibilities.
        grid_nvis (array): 2D array of number of gridded visibilities in cell.
        grid_uu (array): 2D array of grid u coordinates.
        grid_vv (array): 2D array of grid v coordinates.
        grid_wgts (array): 2D array of gridded weights.
    """

    # Use CASA table tools to get required columns.

    tb = casatools.table()
    tb.open(base_ms+"/SPECTRAL_WINDOW")
    chan_freqs_all = tb.getvarcol("CHAN_FREQ")
    tb.close()
    chan_freqs = np.concatenate([chan_freqs_all["r"+str(spw+1)] for spw in spwid])

    tb.open(base_ms)
    if np.ndim(spwid) == 0:
        spwid = [spwid]

    # this is an assumption that is valid for exoALMA data, but not in general
    msmd = casatools.msmetadata()
    msmd.open(base_ms)
    for field_id in msmd.fieldsforname(target):
        subt = tb.query("FIELD_ID=="+str(field_id))
        if subt.nrows() > 0:
            field_id = str(field_id)
            break
    subt.close()
    msmd.close()
    
    flag, weight, data = [], [], []
    model = []
    for ispw, spw in enumerate(spwid):
        data_desc_id = str(spw)

        subt = tb.query("DATA_DESC_ID=="+data_desc_id+" && FIELD_ID=="+field_id)
        if ispw == 0:
            uvw = subt.getcol("UVW")
            ant1 = subt.getcol("ANTENNA1")
            ant2 = subt.getcol("ANTENNA2")
        else:
            assert np.all(uvw == subt.getcol("UVW"))
            assert np.all(ant1 == subt.getcol("ANTENNA1"))
            assert np.all(ant2 == subt.getcol("ANTENNA2"))

        flag += [subt.getcol("FLAG")]
        data += [subt.getcol(datacolumn)]
        weight += [subt.getcol("WEIGHT_SPECTRUM")]
        model += [subt.getcol("MODEL_DATA")]
        subt.close()
    tb.close()

    flag = np.concatenate(flag, axis=1)
    data = np.concatenate(data, axis=1)
    weight = np.concatenate(weight, axis=1)
    model = np.concatenate(model, axis=1)

    # Define visibilities and weights.

    vis = (data[0, :] + data[1, :]) / 2.0
    wgts = 0.5 * (weight[0, :] + weight[1, :]) # 0.5 because using the wrong channel width after Hanning smoothing?
    model_vis = (model[0, :] + model[1, :]) / 2.0

    # Break out the u, v spatial frequencies, convert from m to lambda.

    uu = uvw[0, :][:, np.newaxis] * chan_freqs[:, 0] / constants.c
    vv = uvw[1, :][:, np.newaxis] * chan_freqs[:, 0] / constants.c

    # Toss out the autocorrelation placeholders.

    xc = np.where(ant1 != ant2)[0]
    uu, vv, vis, wgts, model_vis = uu[xc], vv[xc], vis[:, xc], wgts[:, xc], model_vis[:, xc]

    # Remove flagged visibilities.

    flag = np.where(flag.sum(axis=0) > 0, False, True)
    uu, vv, vis, wgts, model_vis = uu.T[flag], vv.T[flag], vis[flag], wgts[flag], model_vis[flag]

    # Define grid in uv space.

    dl = cell_size * constants.arcsec
    dm = cell_size * constants.arcsec
    du = 1.0 / npix / dl
    dv = 1.0 / npix / dm

    # Empty arrays to hold gridded data.

    grid_vis = np.zeros((npix, npix)).astype('complex')
    grid_model_vis = np.zeros((npix, npix)).astype('complex')
    grid_wgts = np.zeros((npix, npix))
    grid_nvis = np.zeros((npix, npix))
    grid_uu, grid_vv = np.mgrid[-npix/2:npix/2:1,-npix/2:npix/2:1]
    grid_uu *= du
    grid_vv *= dv
    #sometimes mgrid gives the wrong shape, i.e. one element too much
    #for example, cell_size=0.1 and npix=100 leads to grid_uu.shape=(101,101),
    #which then crashes the code
    if not grid_uu.shape == (npix,npix) or not grid_vv.shape == (npix,npix):
        raise RuntimeError('please choose a slightly different npix')

    #toss away data that falls outside of the grid:
    min_uu,max_uu = np.min(grid_uu),np.max(grid_uu)
    min_vv,max_vv = np.min(grid_vv),np.max(grid_vv)
    inside_grid = (min_uu<uu) & (uu<max_uu) & (min_vv<vv) & (vv<max_vv)
    if not np.all(inside_grid):
        warnings.warn(f'some data of {base_ms} are outside your uv grid')
        if grid_needs_to_cover_all_data:
            raise ValueError('grid does not cover all data')
    vis = vis[inside_grid]
    model_vis = model_vis[inside_grid]
    wgts = wgts[inside_grid]
    uu = uu[inside_grid]
    vv = vv[inside_grid]

    # Grid the data and return.
    grid_vis, grid_nvis, grid_wgts, grid_model_vis = grid(grid_vis,
                                          grid_nvis,
                                          grid_wgts,
                                          grid_model_vis,
                                          uu,
                                          vv,
                                          du,
                                          dv,
                                          npix,
                                          vis,
                                          wgts,
                                          model_vis)

    return grid_vis, grid_nvis, grid_uu, grid_vv, grid_wgts, grid_model_vis


def calculate_likelihood(x, data, cell_size):
    """
    Calculate the likelihood using a full phase difference after phase shifting
    the second measurement set in ``data``.

    RICH -- Is this actually the likelihood, or are we just minimizing the
            aggregate phase difference between the two datasets?

    Args:
        x (list): A list containing the phase shift, ``(mu_RA, mu_DEC)``.
        data (list): A list containing a list of ``grid_vis``, ``grid_nvis``,
            ``grid_uu``, ``grid_vv`` and ``grid_wgts`` for the two datasets.

    Returns:
        likelihood (float): Likelihood value.
    """

    # Unpack the data.

    ms1_data, ms2_data = data

    # Apply a phase center shift to the second measurement set.

    shifted_data = calculate_phase_shift(grid_vis=ms2_data[0],
                                         grid_nvis=ms2_data[1],
                                         grid_uu=ms2_data[2],
                                         grid_vv=ms2_data[3],
                                         mu_RA=x[0],
                                         mu_DEC=x[1])

    # Calculate the likelihood and return.

    likelihood = calculate_phase_difference(grid_vis1=ms1_data[0],
                                                 grid_vis2=shifted_data,
                                                 grid_wgts1=ms1_data[4],
                                                 grid_wgts2=ms2_data[4],
                                                 phase_rms1=ms1_data[6],
                                                 phase_rms2=ms2_data[6],
                                                 grid_nvis=ms1_data[1],
                                                 grid_model_vis1=ms1_data[5],
                                                 grid_model_vis2=ms2_data[5])

    likelihood = np.sum(np.abs(likelihood))
    lnprior = (np.array(x)**2).sum()/(cell_size/4)**2
    return likelihood + lnprior

def log_likelihood(x, data, cell_size):
    return -0.5*calculate_likelihood(x, data, cell_size)

def plot_grid_nvis(ax,grid_nvis,grid_uu,grid_vv,vmin=None,vmax=None):
    #to avoid divide by zero warnings, fill upt with very small number:
    log_grid_nvis = np.log10(np.where(grid_nvis>0,grid_nvis,1e-10))
    if vmin is None:
        assert np.all(grid_nvis[grid_nvis>0] >= 1)
        vmin = -0.5
    img = ax.pcolormesh(grid_uu,grid_vv,log_grid_nvis,vmin=vmin,vmax=vmax)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    return img

def find_offset(reference_ms, offset_ms, target, aquareport='', npix=1024, cell_size=0.01, 
                spwid=0, fail_silently=False,verbose=False,plot_uv_grid=False,
                uv_grid_plot_filename=None):
    """
    Find the offset between ``offset_ms`` and ``reference_ms`` by minimizing
    the aggregate phase angle and amplitude.

    Args:
        reference_ms (str): The reference measurement set.
        offset_ms (str): The measurement set to use to derive an offset.
        npix (optional[int]): Number of pixels in the grid.
        cell_size (optional[float]): The grid cell size in [arcsec].
        spwid (optional[int]): The spectral window to evaluate; defaults to 0.
        fail_silently (optional[bool]): If ``True`` return a null offset if the
            minimization fails, otherwise raise a ``RuntimeError``.
        verbose (bool): whether to print out info 
        plot_uv_grid (bool): whether to plot an overview of the uv grid
        uv_grid_plot_filename (str): filename out output uv grid plot
    Returns:
        offset (list): A list specifying the right ascension and declination
            offsets in [arcsec].
    """

    #to calculate the offset, we need ms in the same reference frame
    #thus, we convert to J2000 if necessary
    input_ms = {'ref':reference_ms,'offset':offset_ms}
    temporary_ms = []
    ms_for_offset_calculation = {}
    for ms_ID,ms in input_ms.items():
        phase_center = get_phase_center(measurement_set=ms, target=target)
        frame = get_coord_frame(phase_center)
        if frame == 'J2000':
            if verbose:
                print(f'{ms_ID} ms {ms} is already in J2000, no need to transform')
            ms_for_offset_calculation[ms_ID] = ms
        elif frame == 'ICRS':
            if verbose:
                print(f'{ms_ID} ms {ms} is in ICRS, going to use copy in J2000 to calculate offset')
            assert ms[-3:] == '.ms'
            J2000_ms = ms[:-3] + '_J2000.ms'
            if os.path.isdir(J2000_ms):
                shutil.rmtree(J2000_ms)
            sky_coord_phase_center = get_skycoord(phase_center)
            sky_coord_phase_center = sky_coord_phase_center.transform_to(
                                                  skycoord_frames['J2000'])
            phase_center_J2000 = f'J2000 {sky_coord_phase_center.to_string("hmsdms")}'
            casatasks.fixvis(vis=str(ms),outputvis=J2000_ms, field=target,phasecenter=phase_center_J2000)
            temporary_ms.append(J2000_ms)
            ms_for_offset_calculation[ms_ID] = J2000_ms
        else:
            raise RuntimeError(f'unknown frame {frame}')

    # Ingest the required measurement sets. Will return a list of ``grid_vis``,
    # ``grid_nvis``, ``grid_uu``, ``grid_vv`` and ``grid_wgts``.

    ms1 = ingest_ms(base_ms=ms_for_offset_calculation['ref'], target=target, 
                    npix=npix,cell_size=cell_size,grid_needs_to_cover_all_data=False,spwid=spwid[0])
    ms2 = ingest_ms(base_ms=ms_for_offset_calculation['offset'], target=target, 
                    npix=npix,cell_size=cell_size, grid_needs_to_cover_all_data=True,spwid=spwid[1])

    # Define the overlap between the two measurement sets.
    overlap = np.logical_and(ms1[1].real >= 1, ms2[1].real >= 1).astype('int')
    print(overlap.sum()/(ms1[1].real >= 1).sum(), overlap.sum()/(ms2[1].real >= 1).sum())

    if plot_uv_grid:
        fig,axes = plt.subplots(2,2,constrained_layout=True)
        vmax = np.log10(np.max((ms1[1],ms2[1])))
        images = []
        for ID,ms_filename,ms,ax in zip(('ref','offset'),(reference_ms,offset_ms),
                                        (ms1,ms2),axes[0,:]):
            ax.set_title(f'{ms_filename} ({ID})',fontsize=8)
            img = plot_grid_nvis(ax=ax,grid_nvis=ms[1],grid_uu=ms[2],grid_vv=ms[3],
                                 vmax=vmax)
            images.append(img)
            #n_nonempty_grid_points = np.sum(ms[1]>0)
            #print(f'{ID}: there are {n_nonempty_grid_points} uv grid points containing data')
        fig.colorbar(images[1],ax=axes,label='log(number of vis points)',
                     location='top',shrink=0.6)
        axes[1,0].set_title('overlap')
        axes[1,0].pcolormesh(ms1[2],ms1[3],overlap,cmap='Greys')
        for ax in axes.ravel():
            ax.set_xlabel('grid u [lambda]')
            ax.set_ylabel('grid v [lambda]')
            ax.set_aspect('equal')
        axes[1,1].remove()
        if uv_grid_plot_filename is not None:
            fig.savefig(uv_grid_plot_filename)

        plt.clf()
        plt.close(fig)

    # Mask out all the cells where there is no overlap. Note that the np.clip()
    # is to avoid RuntimeWarnings when dividing by zero. These grid points will
    # be masked out by overlap anyway.

    ms1 = [ms1[0] / np.clip(ms1[4], a_min=1.0, a_max=None) * overlap,
           ms1[1] * overlap, ms1[2], ms1[3], ms1[4], ms1[5] / np.clip(ms1[4], a_min=1.0, a_max=None) * overlap]
    ms2 = [ms2[0] / np.clip(ms2[4], a_min=1.0, a_max=None) * overlap,
           ms2[1] * overlap, ms2[2], ms2[3], ms2[4], ms2[5] / np.clip(ms2[4], a_min=1.0, a_max=None) * overlap]

    ms1 = [entry[overlap > 0] for entry in ms1]
    ms2 = [entry[overlap > 0] for entry in ms2]

    # Derive the offset. The starting point x0 is picked as a fraction (chosen as 1/6)
    #of the "resolution" expected from the longest baseline of the offset data set
    contains_data = ms2[1] > 0
    max_uu_vv = np.max(np.sqrt(ms2[2][contains_data]**2 + ms2[3][contains_data]**2))
    #x0 = np.array((1/max_uu_vv/constants.arcsec,1/max_uu_vv/constants.arcsec)) / 6
    x0 = np.array([cell_size,cell_size])
    #print('x0: ',x0)

    if aquareport != '':
        tree = ET.parse(aquareport)
        root = tree.getroot()

        median_phase_rms = {}
        for value in root.find('QaPerStage'):
            if value.attrib['Name'] == 'hifa_spwphaseup':
                for child in value:
                    if "stability" in child.attrib['Reason']:
                        for val in child.attrib['Reason'].split(" "):
                            if "deg" in val:
                                phase_rms = float(val[0:-3])
                            if "uid" in val:
                                ms_file = val.replace('.ms.','_targets.selfcal.ms')
                        median_phase_rms[ms_file] = phase_rms

        # Calculate the phase uncertainties on the data and add in the residual phase uncertainties after gain calibration.
        for ms_name, ms in zip([reference_ms, offset_ms],[ms1, ms2]):
            uvdist = np.sqrt(ms[2]**2 + ms[3]**2) * 1.0e-3
            phase_rms = np.pi/180.*median_phase_rms[offset_ms]*(uvdist / 300.)**0.6
            ms += [phase_rms]
    else:
        for ms in [ms1, ms2]:
            ms += [np.repeat(0.,ms[0].shape)]

    print("==========================================")
    print("Minimize")
    print("==========================================")
    res = minimize(fun=calculate_likelihood,x0=x0,args=([ms1, ms2], cell_size),method='L-BFGS-B')
    print(res)

    xx, yy = np.linspace(-2.,2.,200), np.linspace(-2.,2.,200)
    likelihood = np.empty(200**2).reshape((200,200))
    for i in range(200):
        for j in range(200):
            likelihood[i,j] = log_likelihood([xx[i],yy[j]], [ms1,ms2], cell_size)

    plt.imshow(likelihood, origin="lower", interpolation="none")
    plt.colorbar()
    plt.contour(likelihood, levels=likelihood.max()-scipy.stats.chi2.isf(q=[0.003,0.05,0.32], df=2), colors="white", linestyles='--')
    plt.tight_layout()
    plt.savefig(target+'_'+offset_ms+'_likelihood.png')
    plt.clf()
    plt.close()


    uncertainty = np.sqrt(np.diag(res.hess_inv.todense()))
    for i in range(len(res.x)):
        print('x^{0} = {1:12.4e} ± {2:.1e}'.format(i, res.x[i], uncertainty[i]))

    print("==========================================")
    print("Emcee")
    print("==========================================")

    ndim, nwalkers = 2, 50
    p0 = np.random.uniform(-x0[0],x0[0], (nwalkers,ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=([ms1, ms2], cell_size))
    sampler.run_mcmc(p0, 1000)

    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["m", "b", "log(f)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    fig.savefig(target+"_"+offset_ms+"_stepsplot.png")
    plt.clf()
    plt.close(fig)

    samples = sampler.get_chain(flat=True, discard=500)

    print(np.median(samples, axis=0))
    print(np.std(samples, axis=0))

    print("==========================================")
    print("Dynesty")
    print("==========================================")

    def prior_transform(u):
        return (2.*u - 1.)*5.
        #return scipy.stats.norm.ppf(u)

    sampler = dynesty.NestedSampler(log_likelihood, prior_transform, 2, nlive=500, logl_args=([ms1, ms2], cell_size))

    sampler.run_nested()

    fig, ax = dynesty.plotting.traceplot(sampler.results, show_titles=True, trace_cmap='viridis', connect=True, connect_highlight=range(5))
    fig.savefig(target+"_"+offset_ms+"_traceplot.png")
    plt.clf()
    plt.close(fig)

    samples = dynesty.utils.resample_equal(sampler.results.samples, sampler.results.importance_weights())

    print(np.median(samples, axis=0))
    print(np.std(samples, axis=0))

    print("==========================================")

    # Return the offset, otherwise if this fails, either raise an error or
    # returns a null offset.

    if not res.success:
        if fail_silently:
            return [0.0, 0.0]
        else:
            print(res)
            raise RuntimeError
    for temp_ms in temporary_ms:
        if verbose:
            print(f'going to delete temporary ms {temp_ms}')
        shutil.rmtree(temp_ms)
    """
    return res.x
    """
    return np.median(samples, axis=0)


def get_phase_center(measurement_set, target):
    """
    Read the phase center for the given measurement set.

    Args:
        measurement_set (str): Measurement to grab the phase center from.

    Returns:
        phase_center (str): Coordinates of the phase center.
    """

    msmd = casatools.msmetadata()
    msmd.open(measurement_set)
    phase_center_data = msmd.phasecenter(fieldid=msmd.fieldsforname(target)[0])
    msmd.close()
    if phase_center_data['refer'] == "ICRS":
        frame = "icrs"
    elif phase_center_data['refer'] == "J2000":
        frame = FK5(equinox="J2000")
    else:
        raise ValueError(f'unknown frame {phase_center_data["refer"]}')

    # Convert the reference coordinate into a parsable string.
    for i in range(2):
        assert phase_center_data[f'm{i}']['unit'] == 'rad'
    c = SkyCoord(phase_center_data['m0']['value'], phase_center_data['m1']['value'],
                 frame=frame, unit=u.rad)
    return phase_center_data['refer'] + " " + c.to_string('hmsdms')


def get_coord_frame(coord):
    if coord[:4] == 'ICRS':
        frame = 'ICRS'
    elif coord[:5] == 'J2000':
        frame = 'J2000'
    else:
        raise RuntimeError(f'unable to determine reference frame of {coord}')
    return frame


def get_skycoord(coord):
    frame = get_coord_frame(coord)
    skycoord_input = coord.replace(f'{frame} ', '')
    splitted_skycoord_input = skycoord_input.split(" ")
    c = SkyCoord(splitted_skycoord_input[0],splitted_skycoord_input[1],
                 frame=skycoord_frames[frame])
    return c


def generate_shifted_coords(original_coord,offset,return_J2000):
    """
    For a given reference coordinate, ``original_coord``, and a list of RA and
    Dec offsets, ``offset``, calculate the new, shifted coordinate.

    NOTE: Due to quirks of CASA, some functions only work with J2000
          coordinates. Thus, there is an option to get the output in J2000
    Args:
        original_coord (str): Initial coordinate in ICRS or J2000 'frame hmsdms' format, for
            example: ``"ICRS 12h43m12.159252s +85d52m12.952837s"``.
        offset (tuple): Tuple of RA and Dec offset in [arcsec], for example:
            ``[0.01, -0.02]``.
        return_J2000 (bool): if True, output will be in J2000, if False, output will
                             be same as input

    Returns:
        shifted_coord (str): Shifted coordinate 'hmsdms' format.
    """

    original_frame = get_coord_frame(original_coord)

    c = get_skycoord(original_coord)

    # Apply the RA and Dec offsets.
    ra_offset, dec_offset = offset
    c.data.lon[()] = c.ra + ra_offset / 3600.0 / np.cos(c.dec.rad) * u.degree
    c.data.lat[()] = c.dec + dec_offset / 3600.0 * u.degree
    c.cache.clear()

    if original_frame == 'ICRS' and return_J2000:
        shifted_coord = c.transform_to(skycoord_frames['J2000']).to_string('hmsdms')
        shifted_coord = 'J2000 ' + shifted_coord
    else:
        shifted_coord = f'{original_frame} ' + c.to_string('hmsdms')
    return shifted_coord


def update_phase_center(vis, target, new_phase_center, ref_phase_center,
                        suffix='_shift'):
    """
    Apply the updated phase center to the provided measurement set. This will
    update both the phase center using 'fixvis' with ``new_phase_center`` and
    the phase center coordinates using 'fixplanets' with ``ref_phase_center``.

    Args:
        vis (str): Visibility set to update the phase center of.
        target (str or int): Target source to update phase center of.
        new_phase_center (str): New phase center to apply in J2000.
        ref_phase_center (str): Reference phase center to update in J2000.
        suffix (optional[str]): Suffix to add prior to '.ms' for the new MS.
    """
    for pc in (new_phase_center,ref_phase_center):
        assert get_coord_frame(pc) == 'J2000'
    shifted_vis = vis.replace('.ms', suffix+'.ms')
    if os.path.exists(shifted_vis):
        input_vis = shifted_vis
    else:
        input_vis = vis
    casatasks.fixvis(vis=str(input_vis), outputvis=str(shifted_vis), field=target,
                     phasecenter=new_phase_center)
    casatasks.fixplanets(vis=str(shifted_vis), field=target, fixuvw=False,
                         direction=ref_phase_center)


def align_measurement_sets(reference_ms, align_ms, target, aquareport='', align_offsets=None,npix=1024,
                           cell_size=0.01,spwid=0,plot_uv_grid=False,
                           plot_file_template=None,suffix='_shift'):
    """
    Using ``reference_ms`` as the truth, align all meausrement sets in
    ``align_ms``. This includes calculating the RA and Dec offset between the
    two measurement sets, calcuating the updated phase center coordinate, and
    then appling this phase center shift to the data.

    Args:
        reference_ms (str): The MS to use as a the fixed reference point.
        align_ms (str or list): The MS to align to the reference MS.
        target(str or int): The target to align.
        align_offsets (optional[list]): list of offsets to be used for the alignment.
            Each element corresponds to an element of align_ms.
            If None, offsets will be calculated.
        npix (optional[int]): Number of pixels in the grid.
        cell_size (optional[float]): Cell size in [arcsec] for the grid.
        spwid (optional[int]): The spectral window to align based on; defaults to 0.
        plot_uv_grid (bool): whether to plot an overview of the uv grid
        plot_file_template (str): template to produce output file of uv grid plot
    """

    # Use the reference MS as the phase center for all the shifted MSs. Note
    # the call to generate_shifted_coords() is to convert from ICRS to J2000 for
    # the fixplanets() call later which only uses J2000.

    source_phase_center = get_phase_center(measurement_set=reference_ms, target=target)
    source_phase_center = generate_shifted_coords(
                              original_coord=source_phase_center,offset=[0.0,0.0],
                              return_J2000=True)

    # Cycle through each measurement set and find the offset and then update
    # the phase center and replace the coordinates to match that of the
    # reference MS.
    align_ms = np.atleast_1d(align_ms)
    if align_offsets is not None:
        assert len(align_offsets) == len(align_ms),\
                'number of provided offsets does not correspond to number of ms'
    zero_offset = np.zeros(2)
    calculated_offsets = {}
    for i,ms in enumerate(align_ms):
        is_ref_ms = (ms == reference_ms)
        if is_ref_ms:
            if align_offsets is not None:
                assert np.all(align_offsets[i] == zero_offset),\
                                   'offset of ref ms has to be [0,0]'
            offset = zero_offset
        else:
            if align_offsets is not None:
                offset = align_offsets[i]
            else:
                if plot_file_template is None:
                    uv_grid_plot_filename = None
                else:
                    directory,file_template = os.path.split(plot_file_template)
                    uv_grid_plot_filename = os.path.join(directory,f'{ms}_{file_template}')
                offset = find_offset(reference_ms=reference_ms,offset_ms=ms,npix=npix,
                                     target=target,aquareport=aquareport,cell_size=cell_size,spwid=spwid,
                                     plot_uv_grid=plot_uv_grid,
                                     uv_grid_plot_filename=uv_grid_plot_filename)

        calculated_offsets[ms] = offset

        ms_phase_center = get_phase_center(measurement_set=ms, target=target)
        shifted = generate_shifted_coords(original_coord=ms_phase_center,
                                          offset=offset,return_J2000=True)
        if align_offsets is None:
            print(f'#New coordinates for {target} in {ms}')
            if is_ref_ms:
                print('#no shift, reference MS.\n')
            else:
                print('#requires a shift of [{:.5g},{:.5g}]\n'.format(*offset))
        else:
            print(f'applying shift {offset} to {target} in {ms}')
        update_phase_center(vis=ms,target=target,new_phase_center=shifted,
                            ref_phase_center=source_phase_center,suffix=suffix)

    return calculated_offsets


def find_disk_center(ms,npix=1024,cell_size=0.01,spwid=0,plot_diagnostics=False,
                     diagnostic_plot_filename=None):
    '''
    Find the disk center by searching for the phase center shift that minimizes
    the sum of imaginaries. This relies on the assumption that the source is
    point symmetric.

    Parameters
    ----------
    ms : str
        input measurement set
    npix : int, optional
        number of pixels for uv grid. The default is 1024.
    cell_size : float, optional
        cell size of the uv grid. The default is 0.01.
    spwid : int, optional
        id of the spw to be used. The default is 0.
    plot_diagnostics : bool, optional
        Whether or not to plot diagnostics. The default is False.
    diagnostic_plot_filename : bool, optional
        Filepath to save the diagnostics plot. The default is None (i.e. do not save).

    Returns
    -------
    numpy array
        offset in arcsec

    '''
    ms_data = ingest_ms(base_ms=ms,npix=npix,cell_size=cell_size,
                        grid_needs_to_cover_all_data=True,spwid=spwid)
    def to_minimize(offset):
        shifted_data = calculate_phase_shift(grid_vis=ms_data[0],
                                             grid_nvis=ms_data[1],
                                             grid_uu=ms_data[2],
                                             grid_vv=ms_data[3],
                                             mu_RA=offset[0],
                                             mu_DEC=offset[1])
        return np.sum(np.abs(np.imag(shifted_data)))
    grid_nvis = ms_data[1]
    grid_uu = ms_data[2]
    grid_vv = ms_data[3]
    contains_data = grid_nvis >= 1
    #I think the reason for the periodicity seen in the sum of imaginaries is as follows:
    #the sum of the imaginaries can be written (considering offset in one dimension only)
    #sum_k( sin(phi_k+2*pi*mu*u_k) )
    #now consider a larger offset mu+dmu:
    #sum_k( sin(phi_k+2*pi*(mu+dmu)*u_k) ) = sum_k( sin(phi_k+2*pi*(mu+dmu)*(u0+du_k)) )
    #where we have written uk = u0+du_k with u0 = min(abs(u_k))
    #sum_k( sin(phi_k+2*pi*mu*u_k+2*pi*dmu*u0+2*pi*dmu*du_k) )
    #since sin is pi-periodic, if we put dmu=1/(2*u0), this becomes
    #sum_k( sin(phi_k+2*pi*mu*u_k+2*pi*dmu*du_k) )
    #so it's almost the same as the original expression, expect for 2*pi*dmu*du_k
    #which is small for small values of dmu (?)
    min_abs_u = np.min(np.abs(grid_uu[contains_data]))
    min_abs_v = np.min(np.abs(grid_vv[contains_data]))
    pseudo_period_mu_ra = 1/(2*min_abs_u) / constants.arcsec
    pseudo_period_mu_dec = 1/(2*min_abs_v) / constants.arcsec
    #fit within a little less than half the period
    bounds = [0.45*np.array((-pseudo_period_mu_ra,pseudo_period_mu_ra)),
              0.45*np.array((-pseudo_period_mu_dec,pseudo_period_mu_dec))]
    for i,coord in enumerate(('ra','dec')):
        print(f'considered {coord} offsets: {bounds[i][0]:.4g} - {bounds[i][1]:.4g} arcsec')
    res = minimize(fun=to_minimize,x0=[-0.1,-0.7],method='L-BFGS-B',bounds=bounds)
    assert res.success,'minimization failed'
    fitted_offset = res.x
    if plot_diagnostics:
        fig,axes = plt.subplots(1,2,constrained_layout=True)
        mean_psuedo_period = np.mean((pseudo_period_mu_ra,pseudo_period_mu_dec))
        test_offsets = np.linspace(-mean_psuedo_period,mean_psuedo_period,50)
        MU_RA,MU_DEC = np.meshgrid(test_offsets,test_offsets,indexing='ij')
        summed_imaginaries = np.empty_like(MU_RA)
        for i,mu_ra in enumerate(test_offsets):
            for j,mu_dec in enumerate(test_offsets):
                summed_imaginaries[i,j] = to_minimize([mu_ra,mu_dec])
        axes[0].plot(*fitted_offset,marker='x',color='white')
        imag_img = axes[0].pcolormesh(MU_RA,MU_DEC,np.log10(summed_imaginaries))
        axes[0].set_xlabel('RA offset [arcsec]')
        axes[0].set_ylabel('DEC offset [arcsec]')
        fig.colorbar(imag_img,ax=axes[0],label='log(sum(abs(imag(shifted_data))))',
                     location='top',shrink=0.8)
        rect = plt.Rectangle(xy=(bounds[0][0],bounds[1][0]),
                             width=np.diff(bounds[0])[0],
                             height=np.diff(bounds[1])[0],fill=False,color='white')
        axes[0].add_patch(rect)
        grid_img = plot_grid_nvis(ax=axes[1],grid_nvis=grid_nvis,grid_uu=grid_uu,
                                  grid_vv=grid_vv,vmin=None,vmax=None)
        fig.colorbar(grid_img,ax=axes[1],label='log(number of vis points)',
                     location='top',shrink=0.8)
        for ax in axes:
            ax.set_aspect('equal')
        if diagnostic_plot_filename is not None:
            fig.savefig(diagnostic_plot_filename)
    phase_center = get_phase_center(measurement_set=ms)
    disk_center = generate_shifted_coords(
                              original_coord=phase_center,
                              offset=fitted_offset,return_J2000=False)
    return {'fitted offset':fitted_offset,'disk center':disk_center}


if __name__ == '__main__':

    vis_folder = '/lustre/cv/projects/exoALMA/ALMA_PL_calibrated_data/J1604-2130/COMBINED/self_calibration'
    '''
    vis_name = 'J1604_SB_EB1_initcont.ms'
    npix=102
    cell_size=0.1
    # vis_name = 'J1604_LB_EB1_initcont.ms'
    # npix=1024
    # cell_size=0.01
    # vis_name = 'J1604_ACA_EB1_initcont.ms'
    # npix=30
    # cell_size=1
    vis = os.path.join(vis_folder,vis_name)
    offset = find_disk_center(ms=vis,npix=npix,cell_size=cell_size,
                              plot_diagnostics=True,spwid=1)
    print(offset)
    '''
    
    reference_ms = os.path.join(vis_folder,'J1604_LB_EB2_initcont_selfcal.ms')
    align_ms = os.path.join(vis_folder,'J1604_LB_EB0_initcont_selfcal.ms')
    align_measurement_sets(reference_ms=reference_ms,
                           align_ms=align_ms,npix=1024,
                           cell_size=0.01,spwid=1,plot_uv_grid=True,
                           plot_file_template=None)
