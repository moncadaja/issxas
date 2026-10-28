"""Ion-chamber gas mixing and photon-flux estimates for the ISS beamline.

Pure functions on top of xraydb; no device or GUI dependencies so the same
math serves the Bluesky prepare-beamline plan and the XLive status widget.
"""
import numpy as np
from xraydb import get_material, ionchamber_fluxes, material_mu

IC_LENGTH_CM = 14.5
PIPS_SI_THICKNESS_CM = 0.03
MFC_FULL_SCALE_V = 5.0
MFC_FULL_SCALE_SCCM = 2000.0  # both MFCs: 5 V = 2000 sccm, so the 5 V purge is 2000 sccm total
MFC_TRICKLE_FRACTION = 0.1  # after the purge the flows drop to 0.1x (0.5 V = 200 sccm total)
MFC_TOLERANCE_V = 0.05  # readback tolerance of the MFC signals; a setpoint below it is indistinguishable from 0

# ordered from the most transparent to the most absorbing gas; append 'Ar' when its MFC exists
GASES = ('He', 'N2')
XRAYDB_NAMES = {'He': 'helium', 'N2': 'nitrogen', 'Ar': 'argon'}


def gas_mu(gas, energy):
    """Photoabsorption coefficient of a fill gas at 1 atm, in 1/cm."""
    formula, density = get_material(XRAYDB_NAMES[gas])
    return float(material_mu(formula, energy, density=density, kind='photo'))


def gas_absorption(fractions, energy, length=IC_LENGTH_CM):
    """Fraction of the beam photoabsorbed over ``length`` cm of the mixture."""
    mu = sum(frac * gas_mu(gas, energy) for gas, frac in fractions.items())
    return 1.0 - float(np.exp(-mu * length))


def gas_fractions_for_absorption(energy, absorption, length=IC_LENGTH_CM, gases=GASES):
    """Volume fractions of ``gases`` giving ``absorption`` (0-1) over ``length`` cm at ``energy`` eV.

    Mixes the two lightest adjacent gases that bracket the target. Returns
    ``(fractions, clamped)``; ``clamped`` is True when even the heaviest gas
    cannot absorb that much, in which case the mixture is 100 % of it.
    """
    mu_needed = -np.log(1.0 - absorption) / length
    mus = [gas_mu(gas, energy) for gas in gases]
    fractions = {gas: 0.0 for gas in gases}
    if mu_needed <= mus[0]:
        fractions[gases[0]] = 1.0
        return fractions, False
    for (gas_low, mu_low), (gas_high, mu_high) in zip(zip(gases, mus), zip(gases[1:], mus[1:])):
        if mu_needed <= mu_high:
            x_high = (mu_needed - mu_low) / (mu_high - mu_low)
            fractions[gas_high] = float(x_high)
            fractions[gas_low] = float(1.0 - x_high)
            return fractions, False
    fractions[gases[-1]] = 1.0
    return fractions, True


def mfc_setpoints(fractions, total_v=MFC_FULL_SCALE_V, min_v=MFC_TOLERANCE_V):
    """Purge setpoints: volume fraction x ``total_v`` per gas (the setpoints add up to ``total_v``).

    A component below the readback tolerance is set to exactly 0.
    """
    return {gas: (float(frac * total_v) if frac * total_v >= min_v else 0.0) for gas, frac in fractions.items()}


def mfc_trickle_setpoints(setpoints, fraction=MFC_TRICKLE_FRACTION, min_v=MFC_TOLERANCE_V):
    """Flows kept during the experiment: ``fraction`` x the purge setpoints, same composition.

    If that would put a non-zero component below ``min_v`` the whole set is
    scaled up so the smallest component is exactly ``min_v``.
    """
    trickle = {gas: v * fraction for gas, v in setpoints.items()}
    nonzero = [v for v in trickle.values() if v > 0]
    if nonzero and min(nonzero) < min_v:
        scale = min_v / min(nonzero)
        trickle = {gas: v * scale for gas, v in trickle.items()}
    return {gas: float(v) for gas, v in trickle.items()}


def volts_to_sccm(volts, full_scale_v=MFC_FULL_SCALE_V, full_scale_sccm=MFC_FULL_SCALE_SCCM):
    return volts / full_scale_v * full_scale_sccm


def composition_from_flows(flows):
    """Volume fractions from MFC readbacks (volts, equal full scale); None when nothing flows."""
    total = sum(flows.values())
    if total <= 0:
        return None
    return {gas: float(v / total) for gas, v in flows.items()}


def _xraydb_gas(fractions):
    return {XRAYDB_NAMES[gas]: frac for gas, frac in fractions.items() if frac > 0}


def ion_chamber_flux(volts, gain_exponent, energy, fractions, length=IC_LENGTH_CM):
    """xraydb IonChamberFluxes for an offset-corrected amplifier output ``volts`` at gain 10**gain_exponent V/A."""
    return ionchamber_fluxes(gas=_xraydb_gas(fractions), volts=volts, length=length, energy=energy,
                             sensitivity=10.0 ** (-gain_exponent))


def pips_flux(volts, gain_exponent, energy, thickness=PIPS_SI_THICKNESS_CM):
    """xraydb IonChamberFluxes for a silicon PIPS diode of ``thickness`` cm."""
    return ionchamber_fluxes(gas='Si', volts=volts, length=thickness, energy=energy,
                             sensitivity=10.0 ** (-gain_exponent))
