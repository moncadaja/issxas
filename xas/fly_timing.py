"""Systematic fly-scan region durations from the processing bin grid.

A fly scan is later binned by :func:`xas.bin.bin` onto a fixed grid, so the
quantity that sets data quality is the acquisition time per processing bin.
Each trajectory region (pre-edge, XANES, post-edge) gets
``duration = time_per_bin * number_of_processing_bins_in_the_region``; the
time-per-bin policy lives in ``fly_timing.json``.
"""
import json
import importlib.resources

import numpy as np

from xas.bin import xas_energy_grid
from xas.xray import e2k, k2e

REGIONS = ('preedge', 'xanes', 'postedge')


def load_fly_timing(path=None):
    if path is None:
        path = importlib.resources.files('xas') / 'fly_timing.json'
    with open(str(path)) as fp:
        return json.load(fp)


def xanes_spacing(e0, config):
    for threshold, spacing in config['bin_grid']['xanes_spacing_eV']:
        if threshold is None or e0 < threshold:
            return spacing
    return config['bin_grid']['xanes_spacing_eV'][-1][1]


def processing_grid(e0, energy_lo, energy_hi, config):
    """Centers of the xas.bin grid that fall inside [energy_lo, energy_hi] (absolute eV)."""
    grid = config['bin_grid']
    centers = xas_energy_grid(np.array([energy_lo, energy_hi]), e0, grid['edge_start_eV'], grid['edge_end_eV'],
                              grid['preedge_spacing_eV'], xanes_spacing(e0, config), grid['exafs_spacing_k'])
    centers = np.asarray(centers)
    return centers[(centers >= energy_lo) & (centers <= energy_hi)]


def region_bin_counts(e0, preedge_start, xanes_start, xanes_end, exafs_end_k, config):
    """Number of processing bins in each trajectory region; bounds relative to e0, EXAFS end in k."""
    exafs_end = k2e(exafs_end_k, 0)
    centers = processing_grid(e0, e0 + preedge_start, e0 + exafs_end, config) - e0
    return {'preedge': int(np.sum((centers >= preedge_start) & (centers < xanes_start))),
            'xanes': int(np.sum((centers >= xanes_start) & (centers < xanes_end))),
            'postedge': int(np.sum((centers >= xanes_end) & (centers <= exafs_end)))}


def time_per_bin_for_mode(mode, config, count_rate=None):
    """Seconds per processing bin for each region; HERFD uses target counts / count rate when available."""
    if mode == 'HERFD':
        herfd = config['HERFD']
        if count_rate is not None and count_rate > 0:
            tau = herfd['target_counts_per_bin'] / count_rate
            return {region: tau for region in REGIONS}, f'{herfd["target_counts_per_bin"]} counts per bin at {count_rate:.0f} cps'
        exafs = config['time_per_bin_s']['EXAFS']
        return {region: exafs[region] * herfd['fallback_multiplier'] for region in REGIONS}, \
            f'no count rate: {herfd["fallback_multiplier"]}x the EXAFS policy'
    return dict(config['time_per_bin_s'][mode]), f'{mode} policy'


def suggest_durations(mode, e0, preedge_start, xanes_start, xanes_end, exafs_end_k, count_rate=None, config=None):
    """Region durations (s) for the 'standard' double-sine fly trajectory.

    Returns dict with ``durations``, ``bins``, ``time_per_bin_s`` and ``source``.
    """
    config = config or load_fly_timing()
    bins = region_bin_counts(e0, preedge_start, xanes_start, xanes_end, exafs_end_k, config)
    tau, source = time_per_bin_for_mode(mode, config, count_rate)
    floor = config.get('min_region_duration_s', 0.0)
    durations = {region: float(max(round(bins[region] * tau[region], 1), floor)) for region in REGIONS}
    return {'durations': durations, 'bins': bins, 'time_per_bin_s': tau, 'source': source}


def implied_time_per_bin(e0, preedge_start, xanes_start, xanes_end, exafs_end_k, durations, config=None):
    """Calibration helper: seconds per bin that an existing trajectory's durations correspond to."""
    config = config or load_fly_timing()
    bins = region_bin_counts(e0, preedge_start, xanes_start, xanes_end, exafs_end_k, config)
    return {region: durations[region] / bins[region] if bins[region] else float('nan') for region in REGIONS}


def realized_time_per_bin(time, energy, e0, config=None):
    """Histogram of a trajectory's time onto the processing grid: (bin centers in eV, seconds per bin)."""
    config = config or load_fly_timing()
    time = np.asarray(time, dtype=float)
    energy = np.asarray(energy, dtype=float)
    centers = processing_grid(e0, energy.min(), energy.max(), config)
    if centers.size < 2:
        return centers, np.zeros_like(centers)
    edges = np.hstack((centers[0] - 0.5 * (centers[1] - centers[0]),
                       centers[:-1] + 0.5 * np.diff(centers),
                       centers[-1] + 0.5 * (centers[-1] - centers[-2])))
    dt = np.gradient(time)
    seconds, _ = np.histogram(energy, edges, weights=np.abs(dt))
    return centers, seconds
