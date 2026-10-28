# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:02:23 2023

@author: dleshchev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import xraydb
import copy
import xlsxwriter
import json
import importlib.resources


def load_crystal_library(path=None):
    """Read the analyzer crystal library (default: xas/crystal_library.json).

    Returns ``(geometries, crystals)``: ``geometries`` is the ordered list of
    geometry labels; each crystal dict has ``material``, ``hkl``, ``color``,
    ``notes``, ``inventory`` ({geometry label: count}) and ``at_iss``
    (True when any count is non-zero).
    """
    if path is None:
        path = importlib.resources.files('xas') / 'crystal_library.json'
    with open(str(path)) as fp:
        data = json.load(fp)
    labels = {g['key']: g['label'] for g in data['geometries']}
    crystals = []
    for entry in data['crystals']:
        inventory = {labels[key]: int(entry['inventory'].get(key, 0)) for key in labels}
        crystals.append({'material': entry['material'],
                         'hkl': list(entry['hkl']),
                         'color': entry.get('color', 'k'),
                         'notes': entry.get('notes', ''),
                         'inventory': inventory,
                         'at_iss': any(count > 0 for count in inventory.values())})
    return list(labels.values()), crystals


# %%

class Source:
    x, y, z = 0.0, 0.0, 0.0

    def __init__(self, Ds, Ls, nx=15, ny=15, nz=15):
        self.Ds = Ds
        self.Ls = Ls
        self.nxyz = [nx, ny, nz]
        self.get_grid()

    def get_grid(self):
        Ss = self.Ds / 2.355
        Rs = Ss * 3  # most of the intensity is covered in 3-sigma

        _x_ = np.linspace(-Rs, Rs, self.nxyz[0])
        _y_ = np.linspace(-Rs, Rs, self.nxyz[1]) - self.y
        _z_ = np.linspace(-self.Ls, self.Ls, self.nxyz[2]) - self.z
        XYZ_list = []
        I_list = []
        for x in _x_:
            for y in _y_:
                r = np.sqrt(x ** 2 + y ** 2)
                ksi = r / Ss
                for z in _z_:
                    if r < Rs:
                        XYZ_list.append([x - self.x,
                                         y - self.y,
                                         z - self.z])
                        I_list.append(np.exp(- 0.5 * ksi ** 2))

        self.XYZ = np.array(XYZ_list)
        self.I = np.array(I_list)
        self.I /= np.sum(self.I)


class Crystal:

    def __init__(self, R, r, hkl, kind):
        self.R = R
        self.r = r
        self.hkl = hkl
        self.kind = kind
        self.d = self.lat_const / self.refl_order

    @property
    def lat_const(self):
        if self.kind == 'Si':
            return 5.431020511
        elif self.kind == 'Ge':
            return 5.658
        elif self.kind == 'SiO2':
            return 8.514 / 2

    @property
    def refl_order(self):
        h, k, l = self.hkl
        return np.sqrt(h ** 2 + k ** 2 + l ** 2)

    def E2L(self, E):
        return 12.3984 / E

    def L2E(self, L):
        return 12.3984 / L

    def bragg_angle(self, E):
        L = self.E2L(E)
        return np.arcsin(L / (2 * self.d))

    def bragg_energy(self, ba):
        L = 2 * self.d * np.sin(ba)
        return self.L2E(L)

    def place_E(self, E, src):
        self.E = E
        self.ba = self.bragg_angle(E)
        self._place()

    def place_ba(self, ba, src):
        self.ba = ba
        self.E = self.bragg_energy(ba)
        self._place()

    def _place(self):
        self.Ox = self.R / 2 * np.cos(np.pi / 2 - self.ba) + src.x
        self.Oy = self.R / 2 * np.sin(np.pi / 2 - self.ba) - src.y
        self.Oz = 0

        self.OOx = src.x
        self.OOy = 2 * self.Oy
        self.OOz = self.Oz

        self.x = self.Ox * 2
        self.y = src.y
        self.z = src.z

        ksi = self.R / 2 * np.sqrt(2 - 2 * np.cos(2 * self.ba))
        self.Dx = self.x - ksi * np.cos(np.pi - 2 * self.ba)
        self.Dy = ksi * np.sin(np.pi - 2 * self.ba)
        self.Dz = src.z

        phi = np.linspace(0, np.pi * 2, 361)
        self.rowland_circle_x = self.Ox + self.R / 2 * np.cos(phi)
        self.rowland_circle_y = self.Oy + self.R / 2 * np.sin(phi)
        self.rowland_circle_z = src.z

    #        self.get_grid()
    #        self.observed_resolution()

    def get_grid(self):

        _y_ = np.linspace(-self.r, self.r, self.nyz[0])
        _z_ = np.linspace(-self.r, self.r, self.nyz[1])
        XYZ_list = []
        NXYZ_list = []
        for y in _y_:
            for z in _z_:

                rho = np.sqrt(y ** 2 + z ** 2)
                #                print(y, z, rho, self.r)
                if rho <= self.r:
                    x = self.OOx - np.sqrt(self.R ** 2 - rho ** 2)
                    y += self.OOy
                    z += self.OOz
                    XYZ_list.append([x, y, z])
                    n = np.array([self.OOx - x, self.OOy - y, self.OOz - z])
                    n /= np.linalg.norm(n)
                    NXYZ_list.append(n)

        self.XYZ = np.array(XYZ_list)
        self.NXYZ = np.array(NXYZ_list)

    def observed_resolution(self):
        energy_list = []
        I_list = []
        for src_xyz, src_i in zip(src.XYZ, src.I):
            for cr_xyz, cr_nxyz in zip(self.XYZ, self.NXYZ):
                # vector between each source point and crystal point
                a = src_xyz - cr_xyz
                a_norm = a / np.linalg.norm(a)
                ba_loc = np.pi / 2 - np.arccos(np.sum(a_norm * cr_nxyz))
                energy_loc = self.bragg_energy(ba_loc)
                energy_list.append(energy_loc)
                I_list.append(src_i)

        energy = np.array(energy_list)
        I = np.array(I_list)
        e_edges = np.linspace(energy.min(), energy.max(), 31)
        self.e_res = e_edges[:-1] + np.diff(e_edges) / 2
        self.ba_res = self.bragg_angle(self.e_res)
        IRF = []
        for e_0, e_1 in zip(e_edges[:-1], e_edges[1:]):
            mask = (energy >= e_0) & (energy < e_1)
            IRF.append(np.sum(I[mask]))
        self.i_res = np.array(IRF)
        self.i_res /= np.sum(self.i_res)


# line_dict =


src = Source(50e-3, 1)

# cr = Crystal(1000, 50, [8, 0, 0], 'Ge')
# cr.place_E(9.1751, src)
# print(np.rad2deg(cr.ba))

# cr.observed_resolution()


# 82.817 80.720

# 81.700 - 81.690
# %% THE ISS ATLAS

# XLIM = (3500, 24500)
# XLIM = (3700, 12700)
# XLIM = (11700, 24700)

cr = Crystal(1000, 50, [1, 1, 1], 'Si')


def make_figure_with_small_font(*args, fontsize=7, **kwargs):
    font = {'weight': 'normal',
            'size': fontsize}

    matplotlib.rc('font', **font)
    plt.figure(*args, **kwargs)


def _annotate_line(z, line_pars, all_lines):
    annotate_group = line_pars['annotate_group']
    if len(annotate_group) > 0:
        #        all_lines = xraydb.xray_lines(int(z))
        energies = [all_lines[line].energy for line in line_pars['annotate_group'] if line in all_lines.keys()]
        energy = np.mean(energies)
        label = xraydb.atomic_symbol(int(z)) + '\n' + line_pars['annotate_label']
        plt.text(energy, line_pars['ytext'], label, va='center', ha='center', color=line_pars['color'])


def plot_emission_lines(ylim=(0, 8),
                        z_list=[],
                        lines_dict={},
                        fmt='-', lw=0.5):
    output_list = []
    for z in z_list:
        all_lines = xraydb.xray_lines(int(z))
        for line_key, line_pars in lines_dict.items():
            if line_key in all_lines.keys():
                energy = all_lines[line_key].energy
                if (energy > XLIM[0]) and (energy < XLIM[1]):
                    plt.plot([energy, energy], line_pars['ylim'], fmt, lw=lw, color=line_pars['color'])
                    _annotate_line(z, line_pars, all_lines)
                    output_list.append({'element': xraydb.atomic_symbol(int(z)),
                                        'line_edge': line_key,
                                        'energy': energy,
                                        'color': line_pars['color']})
    return output_list


def plot_edge_lines(ylim=(0, 8),
                    z_list=[],
                    edge_dict={},
                    fmt='-', lw=0.5):
    output_list = []
    for z in z_list:
        all_lines = xraydb.xray_edges(int(z))
        for line_key, line_pars in edge_dict.items():
            if line_key in all_lines.keys():
                energy = all_lines[line_key].energy
                if (energy > XLIM[0]) and (energy < XLIM[1]):
                    plt.plot([energy, energy], line_pars['ylim'], fmt, lw=lw, color=line_pars['color'])
                    _annotate_line(z, line_pars, all_lines)
                    output_list.append({'element': xraydb.atomic_symbol(int(z)),
                                        'line_edge': line_key,
                                        'energy': energy,
                                        'color': line_pars['color']})
    return output_list


def refl_is_good(hkl):
    is_odd = all([(i % 2 == 1) for i in hkl])
    is_even = all([(i % 2 == 0) for i in hkl])
    if is_odd:
        return True
    if is_even:
        if (sum(hkl) % 4) == 0:
            return True
    return False


def compute_allowed_reflections(crystal, nmin=1, nmax=10):
    crystal_list = []
    h, k, l = crystal['hkl']
    for n in range(nmin, nmax):
        new_hkl = [n * h, n * k, n * l]
        if refl_is_good(new_hkl):
            new_crystal = copy.deepcopy(crystal)
            new_crystal['hkl'] = new_hkl
            crystal_list.append(new_crystal)
    return crystal_list


def crystal_temp_factor(crystal, ba, energy):
    if crystal['material'] == 'Si':
        B_RT = 0.4632
    elif crystal['material'] == 'Ge':
        B_RT = 0.5661
    else:
        raise Exception('AAAA')

    L = cr.E2L(energy)
    ksi = np.sin(ba) / L
    M = -B_RT * ksi ** 2
    return np.exp(M)


# def _crystal_reflectivity(crystal, hkl, bragg_deg, energy_ev):
#     TF = crystal_temp_factor(crystal, bragg_deg, energy_ev)
#     dw = xraydb.darwin_width(energy_ev, crystal, hkl, polarization='u')
#     return TF * np.trapezoid(dw.intensity, dw.dtheta*1e6)

def crystal_reflectivity(crystal, ba, energy):
    TF = crystal_temp_factor(crystal, ba, energy / 1e3)
    dw = xraydb.darwin_width(energy, crystal['material'], crystal['hkl'], polarization='u')
    return TF * np.trapezoid(dw.intensity, dw.dtheta * 1e6)


def crystal_str(crystal):
    return crystal['material'] + '(' + ','.join([str(i) for i in crystal['hkl']]) + ')'


def generate_all_allowed_reflections(materials=['Si', 'Ge'], max_index=12):
    """
    Generate all allowed reflections for diamond-structure crystals (Si, Ge).
    
    For diamond structure, allowed reflections are:
    - All indices odd (h,k,l all odd)
    - All indices even AND (h+k+l) divisible by 4
    
    Parameters
    ----------
    materials : list
        List of crystal materials (default: ['Si', 'Ge'])
    max_index : int
        Maximum Miller index to consider (default: 12)
    
    Returns
    -------
    list
        List of reflection dictionaries with 'material' and 'hkl' keys
    """
    reflections = []
    
    # Generate unique hkl combinations (h >= k >= l >= 0, at least one > 0)
    for h in range(0, max_index + 1):
        for k in range(0, h + 1):
            for l in range(0, k + 1):
                if h == 0 and k == 0 and l == 0:
                    continue  # Skip (0,0,0)
                
                hkl = [h, k, l]
                if refl_is_good(hkl):
                    for material in materials:
                        reflections.append({
                            'material': material,
                            'hkl': hkl,
                            'at_iss': False,  # Will be updated when matching with library
                            'color': 'b' if material == 'Si' else 'r'
                        })
    
    return reflections


def _normalize_hkl(hkl):
    """Normalize hkl to canonical form (sorted descending) for comparison."""
    return tuple(sorted(hkl, reverse=True))


def _is_in_library(material, hkl, crystal_library):
    """
    Check if a reflection (or its base crystal) exists in the crystal library.
    
    Returns tuple: (in_library: bool, at_iss: bool, library_crystal: dict or None)
    """
    normalized_hkl = _normalize_hkl(hkl)
    
    # Build set of all reflections from library (including higher orders)
    for crystal in crystal_library:
        if crystal['material'] != material:
            continue
        
        # Get all allowed reflections from this base crystal
        crystal_reflections = compute_allowed_reflections(crystal, nmin=1, nmax=20)
        for refl in crystal_reflections:
            if _normalize_hkl(refl['hkl']) == normalized_hkl:
                return True, crystal['at_iss'], crystal
    
    return False, False, None


def get_best_crystals_for_emission(element, emission_line, crystal_library=None, 
                                   n_results=5, min_bragg_angle=65, max_bragg_angle=88,
                                   curvature_radius=1000, crystal_radius=50, max_index=12):
    """
    Find the best analyzer crystals for a given element and emission line.
    
    Computes ALL possible Si and Ge crystal reflections, then matches against
    the crystal library to indicate which ones are available.
    
    Parameters
    ----------
    element : str or int
        Element symbol (e.g., 'Fe', 'Cu') or atomic number (e.g., 26, 29)
    emission_line : str
        Emission line name (e.g., 'Ka1', 'Ka2', 'Kb1', 'La1', 'Lb1')
    crystal_library : list, optional
        List of crystal dictionaries with 'material', 'hkl', 'at_iss', 'color' keys.
        Used to tag which crystals are in the library. If None, all crystals are tagged
        as not in library.
    n_results : int, optional
        Number of top crystals to return (default: 5, use None for all)
    min_bragg_angle : float, optional
        Minimum Bragg angle in degrees (default: 65)
    max_bragg_angle : float, optional
        Maximum Bragg angle in degrees (default: 88)
    curvature_radius : float, optional
        Crystal curvature radius in mm (default: 1000)
    crystal_radius : float, optional
        Crystal radius in mm (default: 50)
    max_index : int, optional
        Maximum Miller index to consider when generating reflections (default: 12)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'element': element symbol
        - 'emission_line': emission line name
        - 'energy': emission line energy in eV
        - 'crystals': list of crystal dicts sorted by reflectivity (highest first)
          Each crystal dict contains: 
            - 'material': 'Si' or 'Ge'
            - 'hkl': [h, k, l] indices
            - 'hkl_str': formatted string like 'Si(4,4,0)'
            - 'bragg_angle': Bragg angle in degrees
            - 'reflectivity': integrated reflectivity
            - 'energy_resolution': theoretical energy resolution FWHM in eV (Darwin width)
            - 'in_library': True if crystal is in the provided crystal_library
            - 'at_iss': True if crystal is at ISS (from library)
            - 'rank': ranking position (1 = best)
        - 'error': error message if any, None otherwise
    
    Examples
    --------
    >>> result = get_best_crystals_for_emission('Fe', 'Ka1', crystal_library)
    >>> print(result['crystals'][0])  # Best crystal
    >>> print(result['crystals'][1])  # Second best crystal
    
    >>> # Get all matching crystals without library
    >>> result = get_best_crystals_for_emission('Cu', 'Ka1', n_results=None)
    """
    if crystal_library is None:
        crystal_library = []
    
    # Convert element to atomic number if string
    if isinstance(element, str):
        z = xraydb.atomic_number(element)
        element_symbol = element
    else:
        z = int(element)
        element_symbol = xraydb.atomic_symbol(z)
    
    # Get emission line energy
    all_lines = xraydb.xray_lines(z)
    if emission_line not in all_lines:
        return {
            'element': element_symbol,
            'emission_line': emission_line,
            'energy': None,
            'crystals': [],
            'error': f"Emission line '{emission_line}' not found for {element_symbol}. "
                     f"Available lines: {list(all_lines.keys())}"
        }
    
    energy = all_lines[emission_line].energy  # in eV
    
    # Generate ALL possible Si and Ge reflections
    all_reflections = generate_all_allowed_reflections(materials=['Si', 'Ge'], max_index=max_index)
    
    # Find crystals that can reach this energy within Bragg angle limits
    matching_crystals = []
    seen_crystals = set()  # To avoid duplicates
    
    for reflection in all_reflections:
        material = reflection['material']
        hkl = reflection['hkl']
        
        # Create crystal object to compute Bragg angle
        cr_obj = Crystal(curvature_radius, crystal_radius, hkl, material)
        
        try:
            cr_obj.place_E(energy / 1000, src)  # energy in keV
        except:
            continue  # Skip if energy is out of range for this crystal
        
        bragg_angle = np.rad2deg(cr_obj.ba)
        
        # Check if Bragg angle is valid (not NaN and within limits)
        if np.isnan(bragg_angle):
            continue
        if bragg_angle < min_bragg_angle or bragg_angle > max_bragg_angle:
            continue
        
        # Create unique key to avoid duplicates
        crystal_key = (material, tuple(sorted(hkl, reverse=True)))
        if crystal_key in seen_crystals:
            continue
        seen_crystals.add(crystal_key)
        
        # Check if in library
        in_library, at_iss, library_crystal = _is_in_library(material, hkl, crystal_library)
        
        # Calculate reflectivity and energy resolution
        try:
            reflectivity = crystal_reflectivity(reflection, cr_obj.ba, energy)
            # Get Darwin width for energy resolution
            dw = xraydb.darwin_width(energy, material, hkl, polarization='u')
            energy_resolution = dw.energy_fwhm  # FWHM in eV
        except:
            continue  # Skip if calculation fails
        
        matching_crystals.append({
            'material': material,
            'hkl': hkl,
            'hkl_str': crystal_str(reflection),
            'bragg_angle': round(bragg_angle, 3),
            'reflectivity': reflectivity,
            'energy_resolution': round(energy_resolution, 3),  # FWHM in eV
            'in_library': in_library,
            'at_iss': at_iss,
            'inventory': dict(library_crystal['inventory']) if library_crystal else {},
            'library_crystal': crystal_str(library_crystal) if library_crystal else '',
        })
    
    # Sort by reflectivity (highest first)
    matching_crystals.sort(key=lambda x: x['reflectivity'], reverse=True)
    
    # Add rank to each crystal
    for i, crystal in enumerate(matching_crystals):
        crystal['rank'] = i + 1
    
    # Return top n_results
    return {
        'element': element_symbol,
        'emission_line': emission_line,
        'energy': energy,
        'crystals': matching_crystals[:n_results] if n_results else matching_crystals,
        'error': None
    }


def print_best_crystals(element, emission_line, crystal_library=None, n_results=5):
    """
    Print a formatted table of the best crystals for an emission line.
    
    Parameters
    ----------
    element : str or int
        Element symbol or atomic number
    emission_line : str
        Emission line name (e.g., 'Ka1', 'Kb1')
    crystal_library : list, optional
        List of crystal dictionaries. If provided, tags crystals that are in the library.
    n_results : int
        Number of results to display (default: 5)
    """
    result = get_best_crystals_for_emission(element, emission_line, crystal_library, n_results=n_results)
    
    if result['error']:
        print(f"Error: {result['error']}")
        return result
    
    print(f"\n{result['element']} {result['emission_line']} @ {result['energy']:.1f} eV")
    print("-" * 95)
    print(f"{'Rank':<6}{'Crystal':<15}{'Bragg Angle':<14}{'Reflectivity':<14}{'ΔE (eV)':<12}{'In Library':<12}{'At ISS':<8}{'Inventory'}")
    print("-" * 95)
    
    for crystal in result['crystals']:
        in_lib_str = "Yes" if crystal['in_library'] else "No"
        at_iss_str = "Yes" if crystal['at_iss'] else "-"
        inventory_str = ', '.join(f'{geometry}: {count}' for geometry, count in crystal['inventory'].items() if count)
        print(f"{crystal['rank']:<6}{crystal['hkl_str']:<15}{crystal['bragg_angle']:<14.2f}{crystal['reflectivity']:<14.4f}{crystal['energy_resolution']:<12.3f}{in_lib_str:<12}{at_iss_str:<8}{inventory_str}")
    
    return result


def update_crystal_energy_range(cr, crystal, angular_range_deg=[65, 70, 75, 80, 85, 88], lowest_angle=65):
    energy_range = []
    for tth in angular_range_deg:
        cr.place_ba(np.deg2rad(tth), src)
        energy = cr.E * 1000
        energy_range.append(energy)
    energy_range = np.array(energy_range)

    cr.place_ba(np.deg2rad(lowest_angle), src)
    highest_energy = cr.E * 1000

    crystal['E_min'] = np.min(energy_range)
    crystal['E_max'] = np.max(energy_range)
    crystal['E_range'] = energy_range
    crystal['tth_range'] = angular_range_deg
    crystal['tth_min_desperate'] = lowest_angle
    crystal['E_max_desperate'] = highest_energy
    crystal['reflectivity_range'] = [crystal_reflectivity(crystal, np.deg2rad(ba), E) for ba, E in
                                     zip(angular_range_deg, energy_range)]


#    crystal_reflectivity(reflection, ba, _energy)


def annotate_crystal(crystal, plot_height, dy_txt=0.25):  # , fontsize=12):
    e_min = np.max([crystal['E_min'], XLIM[0]])
    e_max = np.min([crystal['E_max_desperate'], XLIM[1]])
    energy = np.mean([e_min, e_max])
    y_txt = plot_height + 0.5  # dy_txt
    label = crystal_str(crystal)
    plt.text(energy, y_txt, label, va='center', ha='center', color=crystal['color'],
             fontweight='bold')  # , fontsize=fontsize)


def plot_crystal_energy_range(crystal, plot_height, lw_crystal=2, notch=0.15, lw_notch=1.0, dy_txt=0.25, annotate=True,
                              check_e_range=True, color=None):
    _color = color if color is not None else crystal['color']
    fmt = _color
    if crystal['at_iss']:
        fmt += '-'
    else:
        fmt += ':'

    if check_e_range:
        plot_flag = (crystal['E_max'] > XLIM[0]) and (crystal['E_min'] < XLIM[1])
    else:
        plot_flag = True

    if plot_flag:
        plt.plot([crystal['E_min'], crystal['E_max']],
                 [plot_height, plot_height],
                 fmt, lw=lw_crystal)
        plt.plot([crystal['E_max'], crystal['E_max_desperate']],
                 [plot_height, plot_height],
                 fmt, lw=lw_crystal, alpha=0.5)

        if annotate: annotate_crystal(crystal, plot_height, dy_txt=dy_txt)
        for E_point in crystal['E_range']:
            plt.plot([E_point, E_point],
                     [plot_height - notch, plot_height + notch],
                     '-', color=_color, lw=lw_notch)
        plt.plot([crystal['E_max_desperate'], crystal['E_max_desperate']],
                 [plot_height - notch, plot_height + notch],
                 '-', color=_color, lw=lw_notch, alpha=0.5)


crystal_radius = 50
curvature_radius = 1000
# curvature_radius = 500


# make_figure_with_small_font(1, figsize=(60//2.54, 30/2.54), clear=True)
# make_figure_with_small_font(1, figsize=(45//2.54, 15/2.54), clear=True, fontsize=10)


def plot_atlas(crystal_library, emin=4300, emax=12700, plot_reflectivity=True):
    global XLIM
    XLIM = (emin, emax)
    y_lo = 0
    y_hi = len(crystal_library)
    dy = 0.55
    _z_list = list(range(22, 36)) + list(range(36, 51)) + list(range(57, 71)) + list(range(73, 83)) + [89, 90, 92, 94]
    _lines_dict = {
        'Ka1': {'color': 'k', 'annotate_group': ['Ka1', 'Ka2'], 'annotate_label': 'Ka', 'ytext': (y_hi + dy * 6.25),
                'ylim': [y_lo, (y_hi + dy * 5.00)]},
        'Ka2': {'color': 'k', 'annotate_group': [], 'annotate_label': 'Ka', 'ytext': (y_hi + dy * 6.25),
                'ylim': [y_lo, (y_hi + dy * 5.00)]},
        'Kb1': {'color': 'm', 'annotate_group': ['Kb1'], 'annotate_label': 'Kb', 'ytext': (y_hi + dy * 3.75),
                'ylim': [y_lo, (y_hi + dy * 2.5)]},
        #                   'Kb5' : {'color' : 'm', 'annotate_group' : [],             'annotate_label': 'Kb', 'ytext' : (y_hi + dy * 1), 'ylim' : [y_lo, (y_hi + dy * 0)]},
        #                   'Kb2' : {'color' : 'c', 'annotate_group' : ['Kb2', 'Kb4'], 'annotate_label': 'Kb', 'ytext' : (y_hi + dy * 1), 'ylim' : [y_lo, (y_hi + dy * 0)]},
        #                   'Kb4' : {'color' : 'c', 'annotate_group' : [],             'annotate_label': 'Kb', 'ytext' : (y_hi + dy * 1), 'ylim' : [y_lo, (y_hi + dy * 0)]},
        'La1': {'color': 'g', 'annotate_group': ['La1'], 'annotate_label': 'La', 'ytext': (y_lo - dy * 2.25),
                'ylim': [y_lo - dy * 1.0, (y_hi)]}}

    emission_lines_list = plot_emission_lines(ylim=(y_lo, y_hi), z_list=_z_list, lines_dict=_lines_dict, lw=0.75)
    #    _z_list_edge = list(range(74, 81)) + [82, 83]
    #    _edge_dict = {'L3' : {'color' : 'c', 'annotate_group' : ['L3'], 'annotate_label': 'L3', 'ytext' : (y_lo - dy * 4.75), 'ylim' : [y_lo - dy * 3, (y_hi + dy * 0)]}}

    _z_list_edge = list(range(22, 36)) + list(range(36, 51))
    _edge_dict = {
        'K': {'color': 'tab:orange', 'annotate_group': ['K'], 'annotate_label': 'K', 'ytext': (y_hi + dy * 1.25),
              'ylim': [y_lo, (y_hi + dy * 0.00)]}}
    edges_list_1 = plot_edge_lines(ylim=(y_lo, y_hi), z_list=_z_list_edge, edge_dict=_edge_dict, lw=0.75)

    _z_list_edge = list(range(57, 71)) + list(range(73, 83)) + [89, 90, 92, 94]
    _edge_dict = {'L3': {'color': 'c', 'annotate_group': ['L3'], 'annotate_label': 'L3', 'ytext': (y_lo - dy * 4.75),
                         'ylim': [y_lo - dy * 3.00, (y_hi + dy * 0)]}}
    edges_list_2 = plot_edge_lines(ylim=(y_lo, y_hi), z_list=_z_list_edge, edge_dict=_edge_dict, lw=0.75)

    edges_list = edges_list_1 + edges_list_2

    reflection_library = []

    _yticks = []
    for i, crystal in enumerate(crystal_library):
        _yticks.append(crystal_str(crystal))
        crystal_reflections = compute_allowed_reflections(crystal, nmin=1, nmax=20)
        for crystal_refl in crystal_reflections:
            cr = Crystal(curvature_radius, crystal_radius, crystal_refl['hkl'], crystal_refl['material'])
            update_crystal_energy_range(cr, crystal_refl)
            plot_crystal_energy_range(crystal_refl, i, dy_txt=dy)
            crystal_refl['y_plot'] = i
            reflection_library.append(crystal_refl)

    for line in (emission_lines_list + edges_list):
        if 'reflections' not in line.keys():
            line['reflections'] = []
        for reflection in reflection_library:
            _energy = line['energy']
            refl_emin = reflection['E_min']
            refl_emax = reflection['E_max']
            if (_energy >= refl_emin) & (_energy <= refl_emax) and (refl_emin > XLIM[0]):  # and (refl_emax < XLIM[1]):
                cr = Crystal(curvature_radius, crystal_radius, reflection['hkl'], reflection['material'])
                cr.place_E(_energy / 1000, src)
                ba = cr.ba
                line['reflections'].append({'bragg_angle': np.rad2deg(ba),
                                            'reflectivity': crystal_reflectivity(reflection, ba, _energy),
                                            **reflection})
        _idx = np.argsort([refl['reflectivity'] for refl in line['reflections']])[::-1]
        line['reflections'] = np.array(line['reflections'])[_idx].tolist()
        #            print(line['element'], line['line_edge'], reflection['material'], reflection['hkl'],
        #                  _energy, np.rad2deg(ba), crystal_reflectivity(reflection, ba, _energy))
        if line['reflections']:
            if line['line_edge'] in ['Ka1', 'Kb1', 'La1', 'K', 'L3']:
                fmt_common = {'markeredgecolor': 'w', 'markeredgewidth': 0.75, 'alpha': 0.75}
                fmt_top = {'ms': 15, **fmt_common}
                fmt_second_top = {'ms': 8, **fmt_common}
                if plot_reflectivity:
                    plt.plot(_energy, line['reflections'][0]['y_plot'], '*', color=line['color'], **fmt_top)
                    if len(line['reflections']) > 1:
                        plt.plot(_energy, line['reflections'][1]['y_plot'], 'o', color=line['color'], **fmt_second_top)

    plt.xlim(XLIM)
    plt.ylim(y_lo - 3.25, y_hi + 4.5)
    plt.yticks(np.arange(y_lo, y_hi), labels=_yticks)
    plt.xticks(np.arange(np.ceil(XLIM[0] / 1000), np.floor(XLIM[1] / 1000) + 1) * 1000)
    plt.xlabel('Energy, eV')
    plt.grid()
    return reflection_library, emission_lines_list, edges_list, fmt_top, fmt_second_top


CRYSTAL_GEOMETRIES, crystal_library = load_crystal_library()

if __name__ == '__main__':
    make_figure_with_small_font(1, figsize=(11, 8.5), clear=True, fontsize=6)

    plt.subplot(211)
    reflection_library, emission_lines_list, edges_list, fmt_top, fmt_second_top = plot_atlas(crystal_library, emin=4300,
                                                                                              emax=12700)

    plt.subplot(212)
    _, emission_lines_list2, edges_list2, _, _ = plot_atlas(crystal_library, emin=11700, emax=24700)

    # plt.grid()
    plt.tight_layout()

    emission_lines_list += emission_lines_list2
    edges_list += edges_list2

    # aux


    plt.subplot(212)
    plt.plot(23800, -3.35 * 1 / 3, 'k*', **fmt_top)
    plt.text(23900, -3.35 * 1 / 3, '#1 reflectivity', va='center')
    plt.plot(23800, -3.35 * 2 / 3, 'ko', **fmt_second_top)
    plt.text(23900, -3.35 * 2 / 3, '#2 reflectivity', va='center')

    ax1 = plt.figure(1).add_axes([0.69, 0.055, 0.2, 0.04], frameon=True)
    # ax1.plot([crystal_refl['E_min'], crystal_refl['E_max']], [0, 0])
    plot_crystal_energy_range(reflection_library[-1], 1, annotate=False, check_e_range=False, color='k')

    for _e, _tth in zip(reflection_library[-1]['E_range'].tolist() + [reflection_library[-1]['E_max_desperate']],
                        reflection_library[-1]['tth_range'] + [reflection_library[-1]['tth_min_desperate']]):
        if _tth == 88:
            _ee = _e - 200
        else:
            _ee = _e
        plt.text(_ee, 1 - 0.5, str(_tth), va='center', ha='center', fontsize=6)

    plt.text((reflection_library[-1]['E_min'] + reflection_library[-1]['E_max_desperate']) / 2, 1 - 0.85, 'Bragg angle',
             va='center', ha='center', fontsize=6)

    ax1.set_ylim(1 - 0.8 - 0.3, 1 + 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.savefig('ISS_analyzer_atlas.png', dpi=600)
    plt.savefig('ISS_analyzer_atlas.pdf')
    # %%
    make_figure_with_small_font(2, figsize=(30 / 2.54, 12 / 2.54), clear=True, fontsize=6)
    reflection_library, emission_lines_list, edges_list, fmt_top, fmt_second_top = plot_atlas(crystal_library, emin=4300,
                                                                                              emax=13700,
                                                                                              plot_reflectivity=True)

    plt.yticks(plt.gca().get_yticks(), labels=[])

    plt.tight_layout()

    ax1 = plt.figure(2).add_axes([0.235, 0.125, 0.2, 0.08], frameon=True)
    # ax1.plot([crystal_refl['E_min'], crystal_refl['E_max']], [0, 0])
    plot_crystal_energy_range(reflection_library[-1], 1, annotate=False, check_e_range=False, color='k')

    for _e, _tth in zip(reflection_library[-1]['E_range'].tolist() + [reflection_library[-1]['E_max_desperate']],
                        reflection_library[-1]['tth_range'] + [reflection_library[-1]['tth_min_desperate']]):
        if _tth == 88:
            _ee = _e - 200
        else:
            _ee = _e
        plt.text(_ee, 1 - 0.5, str(_tth), va='center', ha='center', fontsize=6)

    plt.text((reflection_library[-1]['E_min'] + reflection_library[-1]['E_max_desperate']) / 2,
             1 - 0.95,
             'Bragg angle, deg', va='center', ha='center', fontsize=6)

    ax1.set_ylim(1 - 0.8 - 0.3, 1 + 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # plt.savefig(r'C:\Users\denis\Dropbox\ISS_Spectrometer_paper\Review_document\ISS_analyzer_atlas2.png', dpi=600)

    # %%

    plt.figure(5, clear=True)

    def plot_reflection_library_with_reflectivity(reflection_library, emission_lines_list, emin=4300, emax=13700):
        for i, refl in enumerate(reflection_library):

            energy = refl['E_range']
            if (energy.min() >= emin) and (energy.max() <= emax):
                reflectivity = refl['reflectivity_range']
                color = refl['color']
                plt.semilogy(energy, reflectivity, color=color)
        for line in emission_lines_list:

            energy = line['energy']
            if (energy >= emin) and (energy <= emax):
                plt.vlines(energy, 2, 1e3, colors=line['color'])

    plot_reflection_library_with_reflectivity(reflection_library, emission_lines_list)
# %%

from xraydb.xray import PLANCK_HC, DarwinWidth, R0
from xraydb import f0, f1_chantler, f2_chantler


def darwin_width(energy, crystal='Si', hkl=(1, 1, 1), a=None,
                 polarization='s', ignore_f2=False, ignore_f1=False, m=1, dtheta_pad=0):
    """darwin width for a crystal reflection and energy

    Args:
      energy (float):    X-ray energy in eV
      crystal (string):  name of crystal (one of 'Si', 'Ge', or 'C') ['Si']
      hkl (tuple):       h, k, l for reflection  [(1, 1, 1)]
      a (float or None): lattice constant [None - use built-in value]
      polarization ('s','p', 'u'): mono orientation relative to X-ray polarization ['s']
      ignore_f1 (bool):  ignore contribution from f1 - dispersion (False)
      ignore_f2 (bool):  ignore contribution from f2 - absorption (False)
      m (int):           order of reflection    [1]

    Returns:

      A named tuple 'DarwinWidth' with the following fields

        `theta`:        float, nominal Bragg angle, in rad,

        `theta_offset`: float, angular offset from Bragg angle, in rad,

        `theta_width`:  float, estimated angular Darwin width, in rad,

        `theta_fwhm`:   float, estimated FWHM of angular intensity, in rad,

        `energy_width`: float, estimated angular Darwin width, in rad,

        `energy_fwhm`:  float, estimated FWHM of energy intensity, in eV,

        `zeta`:         nd-array of Zeta parameter (delta_Lambda / Lambda),

        `dtheta`:       nd-array of angles away from Bragg angle, theta in rad,

        `denergy`:      nd-array of energies away from Bragg energy, in eV,

        `intensity`:    nd-array of reflected intensity

    Notes:

     1. This follows the calculation from section 6.4 of
        Elements of Modern X-ray Physics, 2nd Edition
        J Als-Nielsen, and D. McMorrow.

     2. Only diamond structures (Si, Ge, diamond) are currently supported.
        Default values of lattice constant `a` are in Angstroms: 5.4309 for Si,
        5.6578, for 'Ge', and 3.567 for 'C'.

     3. The `theta_width` and `energy_width` values will closely match the
        width of the intensity profile that would = 1 when ignoring the
        effect of absorption.  These are the values commonly reported as
        'Darwin Width'.  The value reported for `theta_fwhm' and
        `energy_fwhm` are larger than this by sqrt(9/8) ~= 1.06.

     4. Polarization can be 's', 'p', 'u',  or None. 's' means vertically
        deflecting crystal and a horizontally-polarized source, as for most
        synchrotron beamlines. 'p' is for a horizontally-deflecting crystal.
        'u' or None is for unpolarized light, as for most fluorescence/emission.

    Examples:
        >>> dw = darwin_width(10000, crystal='Si', hkl=(1, 1, 1))
        >>> dw.theta_width, dw.energy_width
        (2.8593683930207114e-05, 1.4177346002236872)

    """
    lattice_constants = {'Si': 5.4309, 'Ge': 5.6578, 'C': 3.567}

    h_, k_, l_ = hkl
    hklsum = (h_ + k_ + l_)
    if hklsum % 4 == 0 and (h_ % 2 == 0 and k_ % 2 == 0 and l_ % 2 == 0):
        eqr = 8
    elif (h_ % 2 == 1 and k_ % 2 == 1 and l_ % 2 == 1):  # all odd
        eqr = 4 * np.sqrt(2)
    else:
        raise ValueError("hkl must sum to 4 or be all odd")

    if a is None:
        a = lattice_constants[crystal.title()]
    dspace = a / np.sqrt(h_ * h_ + k_ * k_ + l_ * l_)
    lambd = PLANCK_HC / energy
    if lambd > 2 * dspace:
        return DarwinWidth(theta=np.nan, theta_offset=np.nan,
                           theta_width=np.nan, theta_fwhm=np.nan,
                           energy_width=np.nan, energy_fwhm=np.nan,
                           zeta=[], dtheta=[], denergy=[], intensity=[])

    theta = np.arcsin(lambd / (2 * dspace))
    q = 0.5 / dspace
    f1 = f2 = 0
    if not ignore_f1:
        f1 = f1_chantler(crystal, energy)
    if not ignore_f2:
        f2 = f2_chantler(crystal, energy)

    gscale = 2 * (dspace) ** 2 * R0 / (m * a ** 3)

    if polarization is None or polarization.startswith('u'):  # unpolarized
        gscale *= (1 + abs(np.cos(2 * theta))) / 2.0
    elif polarization.startswith('p'):
        gscale *= abs(np.cos(2 * theta))

    g0 = gscale * 8 * (f0(crystal, 0)[0] + f1 - 1j * f2)
    g = gscale * eqr * (f0(crystal, q)[0] + f1 - 1j * f2)

    total = abs(2 * g / (m * np.pi))
    fwhm = total * 3 / (2 * np.sqrt(2))  # where do A-N&M get this factor?

    zeta_offset = g0.real / np.pi
    theta_offset = np.tan(theta) * zeta_offset

    # as a check, the following formula from L Berman (and X0h doc)
    # will give identical results as theta_width. [sin(2x)= 2sin(x)*cos(x)]
    # dw_lb = 2*R0*lambd**2 * eqr*abs(f0(crystal, q)[0] + f1 - 1j*f2)/(m*np.pi*a**3* np.sin(2*theta))

    #  hueristic zeta range and step sizes for crystals:
    sz = zeta_offset

    zeta = np.concatenate((np.arange(-(1.5 + dtheta_pad) * sz, 0, 0.05 * total),
                           np.arange(0, 2 * sz, 0.01 * total),
                           np.arange(2 * sz, (3.5 + dtheta_pad) * sz, 0.05 * total)))
    xc = (m * np.pi * zeta - g0) / g
    _p = np.where(xc.real > 1)[0]
    _n = np.where(xc.real < -1)[0]

    r = (xc - 1j * np.sqrt(1 - xc ** 2))
    r[_p] = (xc - np.sqrt(xc ** 2 - 1))[_p]
    r[_n] = (xc + np.sqrt(xc ** 2 - 1))[_n]

    return DarwinWidth(theta=theta,
                       theta_offset=theta_offset,
                       theta_width=total * np.tan(theta),
                       theta_fwhm=fwhm * np.tan(theta),
                       energy_width=total * energy,
                       energy_fwhm=fwhm * energy,
                       zeta=zeta,
                       dtheta=zeta * np.tan(theta),
                       denergy=-zeta * energy,
                       intensity=abs(r * r.conjugate()))


# shamelessly taken from xraydb:
lattice_constants = {'Si': 5.4309, 'Ge': 5.6578, 'C': 3.567}


def bragg2e(ba_deg, crystal, hkl):
    h_, k_, l_ = hkl
    dspace = lattice_constants[crystal] / np.sqrt(h_ * h_ + k_ * k_ + l_ * l_)
    lambd = (2 * dspace) * np.sin(np.deg2rad(ba_deg))
    return (12398.4 / lambd)


def e2bragg(energy, crystal, hkl):
    h_, k_, l_ = hkl
    dspace = lattice_constants[crystal] / np.sqrt(h_ * h_ + k_ * k_ + l_ * l_)
    lambd = (12398.4 / energy)
    return np.rad2deg(np.arcsin(lambd / (2 * dspace)))


_B_RT = {'Si': 0.4632, 'Ge': 0.5661}


def crystal_temp_factor_str(crystal, bragg_deg, energy_ev):
    wavelength = 12398.4 / energy_ev
    bragg = np.deg2rad(bragg_deg)
    ksi = np.sin(bragg) / wavelength
    M = _B_RT[crystal] * (ksi ** 2)
    return np.exp(-M)


def _crystal_reflectivity(crystal, hkl, bragg_deg, energy_ev):
    TF = crystal_temp_factor_str(crystal, bragg_deg, energy_ev)
    dw = xraydb.darwin_width(energy_ev, crystal, hkl, polarization='u')
    return TF * np.trapezoid(dw.intensity, dw.dtheta * 1e6)


def crystal_reflectivity_str(crystal, hkl, energy_ev):
    bragg_deg = e2bragg(energy_ev, crystal, hkl)
    TF = crystal_temp_factor_str(crystal, bragg_deg, energy_ev)
    dw = xraydb.darwin_width(energy_ev, crystal, hkl, polarization='u')
    return TF * np.trapezoid(dw.intensity, dw.dtheta * 1e6)


def plot_reflectivity_curve(crystal, hkl, energy_ev, *args, **kwargs):
    dw = xraydb.darwin_width(energy_ev, crystal, hkl, polarization='u')
    _r = crystal_reflectivity_str(crystal, hkl, energy_ev)
    label = f'{crystal}-{hkl}, I={_r:0.3f}'
    plt.semilogy(dw.dtheta, dw.intensity, *args, label=label, **kwargs)


if __name__ == '__main__':
    plt.figure(2, clear=True)
    plot_reflectivity_curve('Si', [10, 4, 2], 12658)
    plot_reflectivity_curve('Ge', [8, 8, 0], 12658)
    plt.legend()

    # print('Si-1042:', crystal_reflectivity_str('Si', [10, 4, 2], 12658))
    # print('Ge-880:', crystal_reflectivity_str('Ge', [8, 8, 0], 12658))

    # %%

    line_data = [['Element', 'Edge/Line', 'Energy (eV)', 'Crystal', 'Bragg Angle (deg)', 'Integrated Reflectivity', '']]

    for line in (emission_lines_list + edges_list):
        for reflection in line['reflections']:
            d = [line['element'],
                 line['line_edge'],
                 line['energy'],
                 reflection['material'] + ' ' '(' + ','.join([str(i) for i in reflection['hkl']]) + ')',
                 np.round(reflection['bragg_angle'], 3),
                 reflection['reflectivity'],
                 reflection['at_iss']]
            line_data.append(d)
            print(d)

    workbook = xlsxwriter.Workbook(r'C:\work\ISS beamline\ISS_spectrometers\ISS_Analyzer_Atlas_ext.xlsx')
    worksheet = workbook.add_worksheet()

    cell_format_at_iss = workbook.add_format({'font_color': 'black'})
    cell_format_not_at_iss = workbook.add_format({'font_color': 'gray'})
    row0 = 0
    col0 = 0
    for i, item_row in enumerate(line_data):
        for j, item in enumerate(item_row[:-1]):
            if item_row[-1]:
                _format = cell_format_at_iss
            else:
                _format = cell_format_not_at_iss

            worksheet.write(i + row0, j + col0, item, _format)
    workbook.close()
