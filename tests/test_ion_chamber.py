import math

import pytest

from xas.ion_chamber import (composition_from_flows, gas_absorption, gas_fractions_for_absorption,
                             ion_chamber_flux, mfc_setpoints, mfc_trickle_setpoints, pips_flux, volts_to_sccm)


@pytest.mark.parametrize('energy, n2_fraction', [(4966, 0.07), (7112, 0.216), (10000, 0.635)])
def test_four_percent_mix(energy, n2_fraction):
    fractions, clamped = gas_fractions_for_absorption(energy, 0.04)
    assert not clamped
    assert fractions['N2'] == pytest.approx(n2_fraction, abs=0.002)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert gas_absorption(fractions, energy) == pytest.approx(0.04, rel=1e-6)


@pytest.mark.parametrize('energy', [13000, 17000, 25000])
def test_clamps_to_pure_n2_when_unreachable(energy):
    fractions, clamped = gas_fractions_for_absorption(energy, 0.04)
    assert clamped and fractions == {'He': 0.0, 'N2': 1.0}
    assert gas_absorption(fractions, energy) < 0.04


def test_tiny_target_is_pure_helium():
    fractions, clamped = gas_fractions_for_absorption(7112, 1e-5)
    assert not clamped and fractions['He'] == 1.0


def test_setpoints_and_trickle():
    setpoints = mfc_setpoints({'He': 0.795, 'N2': 0.205})
    assert sum(setpoints.values()) == pytest.approx(5.0)
    assert setpoints['N2'] == pytest.approx(1.025)
    trickle = mfc_trickle_setpoints(setpoints)
    assert trickle == {'He': pytest.approx(0.3975), 'N2': pytest.approx(0.1025)}
    assert sum(trickle.values()) == pytest.approx(0.5)
    assert all(isinstance(v, float) for v in list(setpoints.values()) + list(trickle.values()))


def test_trickle_rescales_to_the_tolerance_floor():
    setpoints = mfc_setpoints({'He': 0.93, 'N2': 0.07})  # Ti K at 4 %: 0.1x would give N2 0.035 V
    trickle = mfc_trickle_setpoints(setpoints)
    assert trickle['N2'] == pytest.approx(0.05)
    assert trickle['He'] / trickle['N2'] == pytest.approx(setpoints['He'] / setpoints['N2'])
    assert sum(trickle.values()) > 0.5
    assert mfc_trickle_setpoints({'He': 5.0, 'N2': 0.0}) == {'He': pytest.approx(0.5), 'N2': 0.0}
    assert volts_to_sccm(0.5) == pytest.approx(200.0)


def test_component_below_tolerance_is_zero():
    assert mfc_setpoints({'He': 0.995, 'N2': 0.005}) == {'He': pytest.approx(4.975), 'N2': 0.0}
    assert mfc_setpoints({'He': 0.98, 'N2': 0.02})['N2'] == pytest.approx(0.1)


def test_composition_from_flows():
    assert composition_from_flows({'He': 1.5, 'N2': 3.5}) == {'He': pytest.approx(0.3), 'N2': pytest.approx(0.7)}
    assert composition_from_flows({'He': 0.0, 'N2': 0.0}) is None


def test_flux_magnitudes():
    fluxes = ion_chamber_flux(1.0, 5, 10000, {'He': 0.0, 'N2': 1.0})
    assert fluxes.incident == pytest.approx(1.75e12, rel=0.02)
    assert fluxes.transmitted == pytest.approx(1.63e12, rel=0.02)
    assert ion_chamber_flux(2.0, 5, 10000, {'N2': 1.0}).incident == pytest.approx(2 * fluxes.incident)
    assert ion_chamber_flux(1.0, 6, 10000, {'N2': 1.0}).incident == pytest.approx(fluxes.incident / 10)
    assert pips_flux(1.0, 5, 7000).incident == pytest.approx(1.66e10, rel=0.02)
    assert math.isfinite(pips_flux(1.0, 5, 7000).transmitted)
