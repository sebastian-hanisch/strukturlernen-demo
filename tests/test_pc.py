"""PC von Grund auf: Teilkorrelation und Fisher-z von Hand und gegen zweite Wege, Skelett, v-Strukturen, Meek-Regeln, Vorwissen, Aufzählung der Klasse, Wirkungsbereich."""

import itertools
import math

import numpy as np
import pytest
from scipy.stats import norm

import dsep_graph as G
import stl_pc as P
import stl_world as W
from kausal_generator import SplitMix64

CHAIN = ((), (0,), (1,))                 # A → B → C
COLLIDER = ((), (0, 2), ())              # A → B ← C
FORK = ((1,), (), (1,))                  # A ← B → C


def test_partial_correlation_by_hand(corr_hand):
    assert P.partial_correlation(corr_hand, 0, 1, (2,)) == pytest.approx(1 / 24)
    assert P.partial_correlation(corr_hand, 0, 1, ()) == pytest.approx(0.5)
    assert P.partial_correlation(corr_hand, 0, 2, (1,)) == pytest.approx((0.6 - 0.5 * 0.8) / math.sqrt((1 - 0.25) * (1 - 0.64)))


def test_partial_correlation_equals_the_correlation_of_regression_residuals():
    """Zweiter Weg: Residuen der Regressionen von i und j auf die Bedingungsmenge."""
    data = W.simulate(W.World(), 4000, SplitMix64(3))
    corr = np.corrcoef(data, rowvar=False)
    for i, j, cond in ((1, 3, (2,)), (0, 2, (1, 3)), (4, 3, (1, 2)), (0, 3, ())):
        z = np.column_stack([np.ones(len(data))] + [data[:, c] for c in cond])
        res_i = data[:, i] - z @ np.linalg.lstsq(z, data[:, i], rcond=None)[0]
        res_j = data[:, j] - z @ np.linalg.lstsq(z, data[:, j], rcond=None)[0]
        assert P.partial_correlation(corr, i, j, cond) == pytest.approx(np.corrcoef(res_i, res_j)[0, 1], abs=1e-9)


def test_fisher_z_p_value_by_hand_and_against_scipy():
    # r = 0,3, n = 103, keine Bedingung: Prüfgröße √100 · artanh(0,3) = 3,0952, p = 0,001967
    assert P.fisher_z_pvalue(0.3, 103, 0) == pytest.approx(0.001967, abs=1e-6)
    assert math.sqrt(100) * math.atanh(0.3) == pytest.approx(3.0952, abs=1e-4)
    # kritischer Betrag von r bei α = 0,05: tanh(1,959964 / 10) = 0,1935
    assert math.tanh(1.959964 / 10) == pytest.approx(0.1935, abs=1e-4) and P.fisher_z_pvalue(0.1935, 103, 0) == pytest.approx(0.05, abs=1e-3)
    for r, n, k in ((0.1, 50, 1), (0.45, 200, 2), (0.02, 5000, 0), (-0.3, 103, 0)):
        stat = math.sqrt(n - k - 3) * abs(math.atanh(r))
        assert P.fisher_z_pvalue(r, n, k) == pytest.approx(2 * (1 - norm.cdf(stat)), abs=1e-12)
    assert P.fisher_z_pvalue(0.0, 100, 0) == 1.0 and P.fisher_z_pvalue(0.9, 100, 0) < 1e-30


def test_fisher_z_test_decision_follows_alpha(corr_hand):
    strict = P.fisher_z_test(corr_hand, 100, 0.001)
    loose = P.fisher_z_test(corr_hand, 100, 0.99)
    assert strict(0, 1, (2,)) is True and loose(0, 1, (2,)) is False        # r = 1/24: p ≈ 0,68
    assert P.fisher_z_test(corr_hand, 100, 0.05)(0, 1, ()) is False           # r = 0,5 bei n = 100: abgelehnt, also nicht unabhängig


def test_chain_has_no_v_structure_and_stays_undirected():
    g = P.pc(3, P.oracle_test(CHAIN)).graph
    assert g == (frozenset(), frozenset({(0, 1), (1, 2)})) and len(P.class_members(CHAIN)) == 3 and P.vstructs(CHAIN) == frozenset()


def test_fork_is_in_the_class_of_the_chain():
    assert P.true_cpdag(FORK) == P.true_cpdag(CHAIN) and set(P.class_members(FORK)) == set(P.class_members(CHAIN)) and len(P.class_members(FORK)) == 3


def test_collider_is_directed_and_alone():
    g = P.pc(3, P.oracle_test(COLLIDER)).graph
    assert g == (frozenset({(0, 1), (2, 1)}), frozenset()) and P.class_members(COLLIDER) == [COLLIDER] and P.vstructs(COLLIDER) == frozenset({(0, 1, 2)})


def test_complete_graph_on_three_nodes_is_one_class_of_six():
    tri = ((), (0,), (0, 1))
    assert len(P.class_members(tri)) == 6 and P.true_cpdag(tri) == (frozenset(), frozenset({(0, 1), (0, 2), (1, 2)}))


def test_meek_r1_directs_the_edge_after_a_collider():
    parents = ((), (0, 2), (), (1,))         # A → B ← C, B → D
    g = P.pc(4, P.oracle_test(parents)).graph
    assert g == (frozenset({(0, 1), (2, 1), (1, 3)}), frozenset())


def test_meek_r2_closes_the_triangle():
    parents = ((), (0, 3), (0, 1), ())       # 0 → 1 ← 3, 1 → 2, 0 → 2: R1 gibt 1 → 2, R2 dann 0 → 2
    parents = ((), (0, 3), (0, 1), ())
    g = P.pc(4, P.oracle_test(parents)).graph
    assert g == (frozenset({(0, 1), (3, 1), (1, 2), (0, 2)}), frozenset()) and g == P.cpdag_by_enumeration(parents)


def test_meek_r3_needs_the_third_rule():
    parents = ((), (0, 2, 3), (0,), (0,))    # a = 0, b = 1, c = 2, d = 3: a → b, a → c, a → d, c → b, d → b
    res = P.pc(4, P.oracle_test(parents))
    assert res.graph == (frozenset({(2, 1), (3, 1), (0, 1)}), frozenset({(0, 2), (0, 3)})) and res.graph == P.cpdag_by_enumeration(parents)


def test_roots_orient_their_edges_outward_and_meek_propagates():
    parents = W.World().parents
    open_graph = P.pc(5, P.oracle_test(parents)).graph
    known = P.pc(5, P.oracle_test(parents), roots=(W.S, W.W)).graph
    assert open_graph[1] == frozenset({(0, 1), (1, 2)}) and known == (frozenset({(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)}), frozenset())


def test_world_cpdag_and_class_by_hand():
    parents = W.World().parents
    assert parents == ((), (0,), (1,), (1, 2, 4), ())
    assert P.true_cpdag(parents) == (frozenset({(1, 3), (2, 3), (4, 3)}), frozenset({(0, 1), (1, 2)}))
    assert sorted(P.class_members(parents)) == sorted([((), (0,), (1,), (1, 2, 4), ()), ((1,), (), (1,), (1, 2, 4), ()), ((1,), (2,), (), (1, 2, 4), ())])


def test_oracle_pc_equals_enumeration_for_all_three_node_dags():
    for parents in G.all_dags(3):
        assert P.true_cpdag(parents) == P.cpdag_by_enumeration(parents), parents


def test_the_class_of_a_dag_equals_its_statement_class_for_three_nodes():
    classes = G.equivalence_classes(3)
    assert len(classes) == 11 and sum(len(v) for v in classes.values()) == 25
    for dags in classes.values():
        for t in dags:
            assert set(P.class_members(t)) == set(dags)


def test_shd_and_pair_states():
    g1 = (frozenset({(0, 1)}), frozenset({(1, 2)}))
    g2 = (frozenset({(1, 0)}), frozenset({(1, 2), (0, 2)}))
    assert P.pair_state(g1, 0, 1) == "f" and P.pair_state(g1, 1, 0) == "r" and P.pair_state(g1, 1, 2) == "u" and P.pair_state(g1, 0, 2) == "-"
    assert P.shd(g1, g1, 3) == 0 and P.shd(g1, g2, 3) == 2 and P.shd(g2, g1, 3) == 2
    assert P.skeleton_of(g1) == frozenset({frozenset({0, 1}), frozenset({1, 2})}) and P.dag_graph(CHAIN) == (frozenset({(0, 1), (1, 2)}), frozenset())


def test_effect_adjusted_by_hand_for_the_world():
    cov = W.population_cov(W.World().matrix)
    assert cov[W.Z, W.X] == pytest.approx(0.8 * 1.64) and cov[W.X, W.X] == pytest.approx(0.64 * 1.64 + 1.0)
    first, second, third = P.class_members(W.World().parents)
    values = sorted(P.effect_adjusted(cov, t, W.X, W.Y) for t in (first, second, third))
    assert values == pytest.approx([0.6, 0.6, 0.6 + 0.6 * 1.312 / 2.0496]) and 0.6 + 0.6 * 1.312 / 2.0496 == pytest.approx(0.9841, abs=1e-4)


def test_class_effects_of_the_cpdag_equals_the_members():
    cov = W.population_cov(W.World().matrix)
    graph = P.true_cpdag(W.World().parents)
    assert sorted(round(e, 6) for e in P.class_effects(graph, cov, W.X, W.Y)) == [0.6, 0.6, 0.984075]
    known = P.pc(5, P.oracle_test(W.World().parents), roots=(W.S, W.W)).graph
    assert [round(e, 6) for e in P.class_effects(known, cov, W.X, W.Y)] == [0.6]


def test_class_dags_refuses_a_class_that_is_too_large():
    path = tuple(() if v == 0 else (v - 1,) for v in range(14))
    graph = P.true_cpdag(path)
    assert len(graph[1]) == 13 and P.class_dags(graph, 14) is None and P.class_effects(graph, np.eye(14), 0, 13) is None
    assert len(P.class_dags(P.true_cpdag(CHAIN), 3)) == 3


def test_class_dags_is_empty_for_a_contradictory_output():
    cyc = (frozenset({(0, 1), (1, 2), (2, 0)}), frozenset())
    assert P.class_dags(cyc, 3) == []
    assert P.class_effects(cyc, np.eye(3), 0, 2) == []


def test_stable_skeleton_does_not_depend_on_the_order():
    for seed in range(8):
        data = W.simulate_linear(W.random_dag(6, 0.4, SplitMix64(seed)), 300, SplitMix64(100 + seed))
        test = P.fisher_z_test(np.corrcoef(data, rowvar=False), 300, 0.05)
        orders = [list(np.random.default_rng(k).permutation(6)) for k in range(5)]
        assert len({P.skeleton_of(P.pc(6, test, stable=True, order=o).graph) for o in orders}) == 1, seed


def test_the_original_version_can_depend_on_the_order():
    for seed in range(200):
        data = W.simulate_linear(W.random_dag(6, 0.4, SplitMix64(seed)), 100, SplitMix64(1000 + seed))
        test = P.fisher_z_test(np.corrcoef(data, rowvar=False), 100, 0.05)
        results = {P.skeleton_of(P.pc(6, test, stable=False, order=list(np.random.default_rng(k).permutation(6))).graph) for k in range(6)}
        if len(results) > 1:
            return
    raise AssertionError("die ursprüngliche Fassung sollte bei wenigen Daten von der Reihenfolge abhängen")


def test_a_data_run_on_the_world_finds_the_class():
    data = W.simulate(W.World(), 3000, SplitMix64(7))
    corr = np.corrcoef(data, rowvar=False)
    res = P.pc(5, P.fisher_z_test(corr, 3000, 0.01))
    assert res.graph == P.true_cpdag(W.World().parents) and res.conflicts == 0 and res.sep[(0, 2)] == {1} and res.sep[(0, 3)] == {1} and res.sep[(4, 1)] == set()


def test_all_pairs_are_tested_with_the_empty_set_first():
    calls = []

    def indep(i, j, cond):
        calls.append((i, j, tuple(cond)))
        return False

    P.pc(3, indep)
    assert calls[:3] == [(0, 1, ()), (0, 2, ()), (1, 0, ())] or {c for c in calls if not c[2]} >= {(0, 1, ()), (0, 2, ()), (1, 2, ())}
    assert all(len(c[2]) <= 1 for c in calls) and any(len(c[2]) == 1 for c in calls)
    assert list(itertools.combinations(range(3), 2)) == [(0, 1), (0, 2), (1, 2)]


def test_class_dags_accepts_exactly_twelve_undirected_edges():
    path13 = tuple(() if v == 0 else (v - 1,) for v in range(13))
    graph = P.true_cpdag(path13)
    assert len(graph[1]) == 12 and len(P.class_dags(graph, 13)) == 13          # ein Baum: je Wahl der Wurzel ein Diagramm
