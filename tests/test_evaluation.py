"""Auswertung: Analyse, Kantenurteile, die vier Experimente in kleinem Maßstab (Struktur und Identitäten), der Baukasten für Äquivalenzklassen."""

import numpy as np
import pytest

import stl_constants as C
import stl_evaluation as ev
import stl_pc as P
import stl_world as W
from kausal_generator import SplitMix64


def test_number_formats():
    assert ev.de(1.5) == "1,50" and ev.de(-0.25, 3) == "−0,250" and ev.signed(0.5) == "+0,50" and ev.signed(-0.5) == "−0,50" and ev.signed(0.0) == "+0,00" and ev.pct(0.123, 1) == "12,3 %"


def test_settings_properties():
    s = ev.Settings()
    assert s.world == W.World() and s.stable and s.obs == (0, 1, 2, 3, 4) and s.roots == ()
    assert ev.Settings(version="ursprünglich").stable is False and ev.Settings(observed="nein").obs == (0, 2, 3, 4)
    assert ev.Settings(knowledge="Wurzeln S und W").roots == (W.S, W.W) and ev.Settings(form="quadratisch").world.quadratic_zy is True
    assert ev.Settings(a_sz=0.5, b_wy=0.3).world == W.World(a_sz=0.5, b_wy=0.3)


def test_edge_status_covers_every_kind():
    dag = frozenset({(0, 1), (1, 2), (3, 2)})
    learned = (frozenset({(0, 1), (2, 1), (2, 3)}), frozenset({(1, 4)}))
    got = {(a, b): (k, st) for a, b, k, st in ev.edge_status(learned, dag)}
    assert got[(0, 1)] == ("gerichtet", "richtig") and got[(2, 1)] == ("gerichtet", "falsch gerichtet") and got[(2, 3)] == ("gerichtet", "falsch gerichtet")
    assert got[(1, 4)] == ("ungerichtet", "überzählig") and len(got) == 4
    learned2 = (frozenset({(0, 1)}), frozenset({(1, 2)}))
    got2 = {(a, b): (k, st) for a, b, k, st in ev.edge_status(learned2, dag)}
    assert got2[(1, 2)] == ("ungerichtet", "offen") and got2[(3, 2)] == ("fehlt", "fehlt") and got2[(0, 1)][1] == "richtig"
    assert ev.edge_status((frozenset(), frozenset({(0, 3)})), dag)[0] == (0, 3, "ungerichtet", "überzählig")


def test_oracle_graph_for_the_world_and_for_hidden_nachfrage():
    parents = W.World().parents
    assert ev.oracle_graph(parents, (0, 1, 2, 3, 4)) == (frozenset({(1, 3), (2, 3), (4, 3)}), frozenset({(0, 1), (1, 2)}))
    assert ev.oracle_graph(parents, (0, 1, 2, 3, 4), roots=(0, 4)) == (frozenset({(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)}), frozenset())
    assert ev.oracle_graph(parents, (0, 2, 3, 4)) == (frozenset({(0, 3), (2, 3), (4, 3)}), frozenset({(0, 2)}))      # Z verborgen: S–X und S → Y als Spur der verborgenen Ursache


def test_run_pc_maps_back_to_world_nodes():
    data = W.simulate(W.World(), 3000, SplitMix64(5))
    g, conflicts = ev.run_pc(W.World(), data, (0, 2, 3, 4), 0.01, True, ())
    assert all(v in (0, 2, 3, 4) for e in g[0] | g[1] for v in e) and (2, 3) in g[0] and conflicts == 0


def test_analysis_of_the_default_setting():
    a = ev.analyse(ev.Settings())
    assert a.parents == W.World().parents and a.truth == 0.6 and a.learned == a.oracle == P.true_cpdag(a.parents) and a.exact and a.shd_oracle == 0 and a.conflicts == 0
    assert a.class_size == 3 and len(a.effects) == 3 and len(a.dags) == 3 and a.obs == (0, 1, 2, 3, 4) and a.contains is True
    assert a.margin == pytest.approx(0.2668, abs=1e-4) and a.missing == [] and a.extra == [] and all(st in ("richtig", "offen") for _, _, _, st in a.edge_status)
    assert ev.analyse(ev.Settings()).learned == a.learned


def test_effects_of_the_analysis_follow_the_class_by_hand():
    a = ev.analyse(ev.Settings(n=10000))
    assert sorted(round(e, 1) for e in a.effects) == [0.6, 0.6, 1.0] and min(a.effects) == pytest.approx(0.6, abs=0.04) and max(a.effects) == pytest.approx(0.984, abs=0.04)
    known = ev.analyse(ev.Settings(n=10000, knowledge="Wurzeln S und W"))
    assert known.class_size == 1 and known.effects == [pytest.approx(0.6, abs=0.04)] and known.learned[1] == frozenset() and known.exact


def test_hidden_nachfrage_gives_a_clean_graph_with_the_wrong_effect():
    a = ev.analyse(ev.Settings(observed="nein", n=5000))
    assert a.obs == (0, 2, 3, 4) and (W.X, W.Y) in a.learned[0] and a.exact and a.contains is False
    assert (W.S, W.Y) in a.extra or (W.S, W.X) in a.extra
    assert min(a.effects) > 0.8 and all(v in (0, 2, 3, 4) for e in a.learned[0] | a.learned[1] for v in e)


def test_cancellation_gives_a_false_collider_and_a_missing_edge():
    a = ev.analyse(ev.Settings(b_zy=-0.48))
    assert not a.exact and (W.Z, W.Y) in a.missing and (W.Z, W.X) in a.learned[0] and (W.Y, W.X) in a.learned[0] and ev._false_colliders(a.learned, a.oracle) == [(W.Z, W.X, W.Y)]
    assert a.margins[(W.Z, W.Y)] == pytest.approx(0.0, abs=1e-12) and a.margin == pytest.approx(0.0, abs=1e-12)


def test_quadratic_arrow_is_missing_for_any_sample_size():
    for n in (500, 5000):
        a = ev.analyse(ev.Settings(form="quadratisch", n=n))
        assert (W.Z, W.Y) in a.missing and a.oracle[0] >= {(1, 3)}


def test_the_original_version_gives_the_same_graph_in_the_easy_case():
    s = ev.Settings(version="ursprünglich")
    assert ev.analyse(s).learned == ev.analyse(ev.Settings()).learned


def test_cancellation_matrix_by_hand():
    b = ev.cancellation_matrix(1.0)
    assert b[1, 0] == 1.0 and b[2, 1] == 1.0 and b[2, 0] == -1.0
    cov = W.population_cov(b)
    assert cov[0, 2] == pytest.approx(0.0, abs=1e-12) and ev.cancellation_matrix(0.0)[2, 0] == 0.0 and W.population_cov(ev.cancellation_matrix(0.5))[0, 2] == pytest.approx(0.5)


def test_sample_experiment_structure_and_identities(small_sample):
    r = small_sample
    assert [(x["alpha"], x["n"]) for x in r["rows"]] == [(a, n) for a in (0.01, 0.05) for n in (100, 300, 1000)] and r["reps"] == 10 and len(r["unfaithful"]) == 3
    for x in r["rows"] + r["unfaithful"]:
        assert 0 <= x["exact"] <= x["skeleton"] <= 1 and 0 <= x["xy"] <= 1 and x["shd"] >= 0 and x["missing"] >= 0 and x["extra"] >= 0
        assert x["exact"] * 10 == pytest.approx(round(x["exact"] * 10)) and (x["exact"] == 1 or x["shd"] > 0)
    assert r["margin"] == pytest.approx(0.2668, abs=1e-4) and r["margin_unfaithful"] == pytest.approx(0.0485, abs=1e-4) and set(r["margins_unfaithful"]) == {(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)}


def test_open_experiment_structure_and_identities(small_open):
    r = small_open
    assert r["class_size"] == 3 and r["pop_effects"] == pytest.approx([0.6, 0.984075]) and r["know_effects"] == [0.6] and r["truth"] == 0.6
    assert r["truth_graph"] == (frozenset({(1, 3), (2, 3), (4, 3)}), frozenset({(0, 1), (1, 2)})) and r["know_graph"][1] == frozenset()
    assert [(x["d"], x["degree"]) for x in r["random"]] == [(6, 1.0), (6, 2.0)]
    assert all(0 <= x["undirected"] <= 1 and 0 <= x["full"] <= 1 and x["with_edges"] <= 10 for x in r["random"])
    modes = {(x["n"], x["mode"]): x for x in r["rows"]}
    assert set(modes) == {(n, m) for n in (300, 1000) for m in ("ohne", "mit Vorwissen")}
    assert all(modes[(n, "mit Vorwissen")]["width"] == pytest.approx(0.0) for n in (300, 1000)) and all(modes[(n, "ohne")]["width"] > 0.3 for n in (300, 1000))


def test_assumption_experiment_structure_and_identities(small_assumptions):
    r = small_assumptions
    assert len(r["cancel"]) == 6 and len(r["nonlinear"]) == 4 and len(r["hidden"]) == 1 and r["truth"] == 0.6
    for x in r["cancel"]:
        # Struktur: Wird X–Y mit leerer Trennmenge entfernt, ist M nicht in ihr, also entsteht der Collider; bei λ = 0 gibt es keinen direkten Pfeil (das Fehlen der Kante ist richtig)
        assert x["collider"] == (0.0 if x["lam"] == 0.0 else x["missing"])
    at1 = {x["n"]: x for x in r["cancel"] if x["lam"] == 1.0}
    assert at1[500]["missing"] >= 0.7 and at1[5000]["missing"] >= 0.7
    assert all(x["missing"] >= 0.8 for x in r["nonlinear"] if x["quadratic"]) and all(x["missing"] == 0.0 for x in r["nonlinear"] if not x["quadratic"])
    h = r["hidden"][0]
    assert h["main"] == ("S–X", "S→Y", "W→Y", "X→Y") and h["inside"] == 0.0 and h["lo"] > 0.8


def test_oracle_experiment_counts():
    o = ev.oracle_experiment()
    assert (o["dags"], o["classes"], o["enum_equal"], o["pc_equal"], o["alone"]) == (543, 185, 543, 543, 59)
    assert o["sizes"] == {1: 59, 2: 48, 3: 36, 4: 19, 6: 4, 8: 12, 10: 6, 24: 1} and sum(k * v for k, v in o["sizes"].items()) == 543 and sum(o["sizes"].values()) == 185


def test_random_experiment_structure(small_procedure):
    r = small_procedure["random"]
    assert r["dags"] == 5 and r["orderings"] == 3 and [x["n"] for x in r["rows"]] == [100, 1000]
    for x in r["rows"]:
        assert 0 <= x["exact"] <= 1 and x["skel_stab"] == 0.0 and x["skel_orig"] >= x["skel_stab"] and x["shd"] >= 0 and x["out_stab"] >= x["skel_stab"] and x["out_orig"] >= x["skel_orig"]


def test_builder_parents_and_states():
    assert ev.builder_parents({(0, 1): 1, (1, 2): 1, (2, 3): 2}) == ((), (0,), (1, 3), ())
    assert ev.builder_parents({(0, 1): 2, (0, 2): 0}) == ((1,), (), (), ())
    assert ev.BUILDER_NAMES == ("A", "B", "C", "D") and len(ev.BUILDER_PAIRS) == 6 and ev.BUILDER_STATES == ("keine Kante", "→", "←")


def test_builder_result_for_named_examples():
    chain = ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 1}))
    assert chain["acyclic"] and chain["size"] == 3 and chain["cpdag"] == (frozenset(), frozenset({(0, 1), (1, 2)})) and chain["vstructs"] == [] and chain["edges"] == 2 and chain["dags"] == 543 and chain["classes"] == 185
    collider = ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 2}))
    assert collider["size"] == 1 and collider["cpdag"] == (frozenset({(0, 1), (2, 1)}), frozenset()) and collider["vstructs"] == [(0, 1, 2)]
    with_d = ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 2, (1, 3): 1}))
    assert with_d["size"] == 1 and (1, 3) in with_d["cpdag"][0]
    cyc = ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 1, (0, 2): 2}))
    assert cyc == {"acyclic": False}
    empty = ev.builder_result(ev.builder_parents({}))
    assert empty["size"] == 1 and empty["edges"] == 0 and empty["same_statements"] == 1


def test_builder_size_equals_the_statement_class_for_every_state():
    import itertools

    checked = 0
    for combo in itertools.product(range(3), repeat=6):
        b = ev.builder_result(ev.builder_parents(dict(zip(ev.BUILDER_PAIRS, combo))))
        if not b["acyclic"]:
            continue
        assert b["size"] == b["same_statements"] and len(b["members"]) == b["size"]
        checked += 1
    assert checked == 543
