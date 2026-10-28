import numpy as np
import pytest

from xas.fly_timing import (implied_time_per_bin, load_fly_timing, realized_time_per_bin, region_bin_counts,
                            suggest_durations)

STANDARD = dict(preedge_start=-200, xanes_start=-30, xanes_end=30, exafs_end_k=16)


def test_standard_trajectory_is_reproduced_at_fe_k():
    result = suggest_durations('EXAFS', 7112, **STANDARD)
    assert result['durations'] == pytest.approx({'preedge': 4.0, 'xanes': 6.1, 'postedge': 20.1}, abs=0.15)
    assert result['bins'] == {'preedge': 34, 'xanes': 255, 'postedge': 340}


def test_shorter_k_range_only_shortens_the_postedge():
    full = suggest_durations('EXAFS', 7112, **STANDARD)['durations']
    short = suggest_durations('EXAFS', 7112, -200, -30, 30, 12)['durations']
    assert short['preedge'] == full['preedge'] and short['xanes'] == full['xanes']
    assert short['postedge'] < full['postedge']


def test_xanes_policy_triples_edge_time():
    exafs = suggest_durations('EXAFS', 7112, **STANDARD)['durations']
    xanes = suggest_durations('XANES', 7112, **STANDARD)['durations']
    assert xanes['xanes'] == pytest.approx(3 * exafs['xanes'], abs=0.2)
    assert xanes['preedge'] == exafs['preedge'] and xanes['postedge'] == exafs['postedge']


def test_herfd_uses_count_rate_or_fallback():
    config = load_fly_timing()
    with_rate = suggest_durations('HERFD', 7112, **STANDARD, count_rate=800.0, config=config)
    assert with_rate['time_per_bin_s']['xanes'] == pytest.approx(config['HERFD']['target_counts_per_bin'] / 800.0)
    fallback = suggest_durations('HERFD', 7112, **STANDARD, count_rate=None, config=config)
    assert fallback['time_per_bin_s']['preedge'] == pytest.approx(
        config['HERFD']['fallback_multiplier'] * config['time_per_bin_s']['EXAFS']['preedge'])
    assert 'no count rate' in fallback['source']


def test_coarser_xanes_bins_at_high_energy():
    assert region_bin_counts(20000, **STANDARD, config=load_fly_timing())['xanes'] < \
        region_bin_counts(7112, **STANDARD, config=load_fly_timing())['xanes']


def test_implied_time_per_bin_round_trips():
    tau = implied_time_per_bin(7112, **STANDARD, durations={'preedge': 4, 'xanes': 6, 'postedge': 20})
    assert tau['preedge'] == pytest.approx(0.1176, abs=1e-3)


def test_realized_time_per_bin_conserves_total_time():
    t = np.linspace(0, 30, 30001)
    e = 7112 + np.interp(t, [0, 4, 10, 30], [-200, -30, 30, 975])
    centers, seconds = realized_time_per_bin(t, e, 7112)
    assert seconds.sum() == pytest.approx(30, abs=0.5)
    assert np.median(seconds[(centers >= 7082) & (centers < 7142)]) < np.median(seconds[centers < 7082])
