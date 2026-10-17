"""Dauerhaftes Orakel: Der PC mit d-Separation als Test (also mit den wahren Unabhängigkeiten) gibt für jedes Diagramm den CPDAG aus, den die Aufzählung der Klasse (gleiches Skelett,
gleiche v-Strukturen, azyklisch) liefert; die Aufzählung stimmt mit den Klassen aus den Aussagenmengen (dsep_graph) überein. Alle 543 Diagramme mit vier Knoten, dazu Zufalls-DAGs."""

import numpy as np

import dsep_graph as G
import stl_pc as P
import stl_world as W
from kausal_generator import SplitMix64


def test_all_543_four_node_dags():
    classes = G.equivalence_classes(4)
    assert sum(len(v) for v in classes.values()) == 543 and len(classes) == 185
    enum_ok = pc_ok = 0
    for dags in classes.values():
        for t in dags:
            enum_ok += set(P.class_members(t)) == set(dags)
            pc_ok += P.true_cpdag(t) == P.cpdag_by_enumeration(t)
    assert enum_ok == 543 and pc_ok == 543


def test_random_dags_with_five_and_six_nodes():
    rng = SplitMix64(7)
    for d, p, count in ((5, 0.4, 60), (5, 0.6, 60), (6, 0.35, 40)):
        for _ in range(count):
            t = W.parents_of_matrix(W.random_dag(d, p, rng))
            assert P.true_cpdag(t) == P.cpdag_by_enumeration(t), t


def test_the_oracle_is_the_limit_of_the_data_run():
    """Mit sehr vielen Daten gibt der Fisher-z-Lauf denselben Graphen wie das Orakel (leichte Welt, α = 0,01)."""
    world = W.World()
    data = W.simulate(world, 40000, SplitMix64(3))
    corr = np.corrcoef(data, rowvar=False)
    assert P.pc(5, P.fisher_z_test(corr, 40000, 0.001)).graph == P.true_cpdag(world.parents)


def test_oracle_with_roots_equals_the_enumeration_plus_the_roots():
    """Vorwissen: Mit Wurzeln ist das Ergebnis das, was bleibt, wenn man nur Diagramme der Klasse zulässt, in denen die Wurzeln keine Eltern haben."""
    rng = SplitMix64(11)
    checked = 0
    for _ in range(80):
        b = W.random_dag(5, 0.5, rng)
        t = W.parents_of_matrix(b)
        roots = tuple(v for v in range(5) if not t[v])[:2]
        if not roots:
            continue
        members = [m for m in P.class_members(t) if all(not m[r] for r in roots)]
        assert members, t
        assert P.pc(5, P.oracle_test(t), roots=roots).graph == P.cpdag_from_members(members), (t, roots)
        checked += 1
    assert checked > 40
