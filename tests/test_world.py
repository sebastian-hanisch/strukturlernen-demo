"""Welt: Struktur, Daten aus festen Strömen, Bevölkerungsgrößen von Hand, Treue-Spielraum gegen einen zweiten Weg, Zufalls-DAGs."""

import itertools

import numpy as np
import pytest

import stl_world as W
from kausal_generator import SplitMix64


def test_default_world_structure():
    w = W.World()
    assert w.parents == ((), (0,), (1,), (1, 2, 4), ()) and w.effect == 0.6 and w.var_z == pytest.approx(1.64)
    b = w.matrix
    assert b[W.Z, W.S] == 0.8 and b[W.X, W.Z] == 0.8 and b[W.Y, W.X] == 0.6 and b[W.Y, W.Z] == 0.6 and b[W.Y, W.W] == 0.6 and np.count_nonzero(b) == 5
    assert W.NAMES == ("S", "Z", "X", "Y", "W") and (W.S, W.Z, W.X, W.Y, W.W) == (0, 1, 2, 3, 4)


def test_a_zero_coefficient_removes_the_arrow():
    assert W.World(a_sz=0.0).parents[W.Z] == () and W.World(b_wy=0.0).parents[W.Y] == (1, 2)


def test_simulation_is_deterministic_and_has_the_right_shape():
    a = W.simulate(W.World(), 50, SplitMix64(5))
    assert a.shape == (50, 5) and np.array_equal(a, W.simulate(W.World(), 50, SplitMix64(5))) and not np.array_equal(a, W.simulate(W.World(), 50, SplitMix64(6)))


def test_population_covariance_by_hand():
    cov = W.population_cov(W.World().matrix)
    assert cov[W.S, W.S] == pytest.approx(1.0) and cov[W.Z, W.Z] == pytest.approx(1.64) and cov[W.S, W.Z] == pytest.approx(0.8)
    assert cov[W.Z, W.X] == pytest.approx(1.312) and cov[W.X, W.X] == pytest.approx(2.0496) and cov[W.X, W.Y] == pytest.approx(0.6 * 2.0496 + 0.6 * 1.312)
    assert cov[W.W, W.W] == pytest.approx(1.0) and cov[W.W, W.X] == pytest.approx(0.0)


def test_simulated_covariance_matches_the_population():
    data = W.simulate(W.World(), 200000, SplitMix64(11))
    assert np.cov(data, rowvar=False) == pytest.approx(W.population_cov(W.World().matrix), abs=0.04)


def test_the_two_simulators_agree_in_distribution():
    w = W.World()
    a = np.cov(W.simulate(w, 100000, SplitMix64(1)), rowvar=False)
    b = np.cov(W.simulate_linear(w.matrix, 100000, SplitMix64(2)), rowvar=False)
    assert a == pytest.approx(b, abs=0.06)


def test_quadratic_arrow_is_uncorrelated_with_z_and_centred():
    w = W.World(quadratic_zy=True)
    data = W.simulate(w, 200000, SplitMix64(3))
    z = data[:, W.Z]
    g = (z * z - w.var_z) / w.var_z
    assert g.mean() == pytest.approx(0.0, abs=0.01) and np.corrcoef(z, g)[0, 1] == pytest.approx(0.0, abs=0.01)
    assert abs(np.corrcoef(z, data[:, W.Y] - 0.6 * data[:, W.X])[0, 1]) < 0.01      # Z → Y trägt keine lineare Korrelation (Anteil von Y ohne X)


def test_cancellation_makes_z_and_y_uncorrelated():
    """Cov(Z, Y) = Var(Z) · (b_zy + a_zx · b_xy): bei b_zy = −0,48 genau null."""
    cov = W.population_cov(W.World(b_zy=-0.48).matrix)
    assert cov[W.Z, W.Y] == pytest.approx(0.0, abs=1e-12) and cov[W.S, W.Y] == pytest.approx(0.0, abs=1e-12)
    assert W.faithfulness_margins(W.World(b_zy=-0.48).matrix)[(W.Z, W.Y)] == pytest.approx(0.0, abs=1e-12)


def test_partial_correlation_population_hand_values():
    cov = W.population_cov(W.World().matrix)
    assert W.partial_corr_population(cov, W.S, W.X, (W.Z,)) == pytest.approx(0.0, abs=1e-12)           # S ⟂ X | Z
    assert W.partial_corr_population(cov, W.S, W.Y, (W.Z,)) == pytest.approx(0.0, abs=1e-12)
    r_sz = 0.8 / (1.0 * 1.64 ** 0.5)
    assert W.partial_corr_population(cov, W.S, W.Z) == pytest.approx(r_sz)


def _margin_by_regression(b):
    """Zweiter Weg zum Treue-Spielraum: Korrelation der Regressionsresiduen aus der Bevölkerungskovarianz."""
    cov = W.population_cov(b)
    d = b.shape[0]
    out = {}
    for j in range(d):
        for i in range(d):
            if b[j, i] == 0:
                continue
            rest = [k for k in range(d) if k not in (i, j)]
            best = 9.0
            for r in range(len(rest) + 1):
                for cond in itertools.combinations(rest, r):
                    c = list(cond)
                    if c:
                        beta_i = np.linalg.solve(cov[np.ix_(c, c)], cov[c, i])
                        beta_j = np.linalg.solve(cov[np.ix_(c, c)], cov[c, j])
                        cii = cov[i, i] - cov[c, i] @ beta_i
                        cjj = cov[j, j] - cov[c, j] @ beta_j
                        cij = cov[i, j] - cov[c, i] @ beta_j
                    else:
                        cii, cjj, cij = cov[i, i], cov[j, j], cov[i, j]
                    best = min(best, abs(cij / np.sqrt(cii * cjj)))
            out[(i, j)] = best
    return out


def test_faithfulness_margin_matches_a_second_way_and_the_stated_values():
    for world in (W.World(), W.World(a_zx=0.7, b_xy=1.0, b_zy=0.8), W.World(b_zy=-0.3)):
        mine, other = W.faithfulness_margins(world.matrix), _margin_by_regression(world.matrix)
        assert set(mine) == set(other) and all(mine[k] == pytest.approx(other[k], abs=1e-9) for k in mine)
    assert W.faithfulness_margin(W.World().matrix) == pytest.approx(0.2668, abs=1e-4)
    assert W.faithfulness_margin(W.World(a_zx=0.7, b_xy=1.0, b_zy=0.8).matrix) == pytest.approx(0.0485, abs=1e-4)
    assert W.faithfulness_margins(W.World(a_zx=0.7, b_xy=1.0, b_zy=0.8).matrix)[(W.Z, W.X)] == pytest.approx(0.0485, abs=1e-4)
    assert W.faithfulness_margin(np.zeros((3, 3))) == 1.0


def test_random_dag_is_acyclic_deterministic_and_in_range():
    b = W.random_dag(8, 0.5, SplitMix64(4))
    assert np.array_equal(b, W.random_dag(8, 0.5, SplitMix64(4))) and np.all(np.triu(b) == 0) and not np.array_equal(b, W.random_dag(8, 0.5, SplitMix64(5)))
    mags = np.abs(b[b != 0])
    assert len(mags) > 0 and mags.min() >= 0.5 and mags.max() <= 1.5
    rng = SplitMix64(9)
    counts = [np.count_nonzero(W.random_dag(10, 0.3, rng)) for _ in range(300)]
    assert np.mean(counts) == pytest.approx(0.3 * 45, abs=1.0)
    assert np.count_nonzero(W.random_dag(6, 0.0, SplitMix64(1))) == 0 and np.count_nonzero(W.random_dag(6, 1.1, SplitMix64(1))) == 15


def test_parents_of_matrix():
    b = np.zeros((3, 3))
    b[1, 0], b[2, 0], b[2, 1] = 1.0, 2.0, 3.0
    assert W.parents_of_matrix(b) == ((), (0,), (0, 1))


def test_quadratic_world_keeps_the_mean_of_y_at_zero():
    """g(Z) = (Z² − Var Z) / Var Z ist zentriert: Mit dem Pfeil Z → Y als g bleibt der Mittelwert von Y null (die Simulation selbst, nicht eine Nachrechnung der Formel)."""
    data = W.simulate(W.World(quadratic_zy=True), 200000, SplitMix64(8))
    assert data[:, W.Y].mean() == pytest.approx(0.0, abs=0.03) and data[:, W.Z].mean() == pytest.approx(0.0, abs=0.01)


def test_margin_of_random_dags_matches_the_second_way_including_arrows_into_the_last_node():
    rng = SplitMix64(21)
    checked = 0
    for _ in range(12):
        b = W.random_dag(5, 0.5, rng)
        if b[4].any():
            mine, other = W.faithfulness_margins(b), _margin_by_regression(b)
            assert set(mine) == set(other) and all(mine[k] == pytest.approx(other[k], abs=1e-9) for k in mine)
            checked += 1
    assert checked >= 4
