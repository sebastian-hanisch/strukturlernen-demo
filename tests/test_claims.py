"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil) oder sind exakte Rechnungen; Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte Stelle ändern, deshalb
Toleranzen von einer halben Einheit der gedruckten Stelle. Was Konstruktion ist, steht ausdrücklich dabei: dass die Klasse das wahre Diagramm enthält (der Wirkungsbereich also die Wahrheit „enthält“),
dass Kante fehlt und falscher Collider zugleich auftreten (Struktur), dass das stabile Skelett nicht von der Reihenfolge abhängt (Konstruktion) und dass der Orakel-PC den Aufzählungs-CPDAG trifft (Prüfung
der Implementierung, kein Befund über die Welt).
"""

import math

import numpy as np
import pytest

import stl_constants as C
import stl_evaluation as ev
import stl_pc as P
import stl_world as W


def test_default_sample():
    a = ev.analyse(ev.Settings())
    assert a.truth == 0.6 and a.exact and len(a.learned[0]) == 3 and len(a.learned[1]) == 2 and a.class_size == 3 and a.margin == pytest.approx(0.267, abs=0.0006)
    assert min(a.effects) == pytest.approx(0.616, abs=0.0006) and max(a.effects) == pytest.approx(0.985, abs=0.0006) and a.contains is True
    assert ev.verdict_instance(a)[0] == "success" and "Die Wirkung des Faktors ist ein Bereich: 0,62 bis 0,99 (3 Diagramme passen)" in ev.verdict_instance(a)[1]


def test_hand_values_of_the_text():
    assert math.sqrt(100) * math.atanh(0.3) == pytest.approx(3.095, abs=0.0006) and P.fisher_z_pvalue(0.3, 103, 0) == pytest.approx(0.001967, abs=0.0000006)
    assert math.tanh(1.959964 / 10) == pytest.approx(0.1935, abs=0.00006)
    assert P.partial_correlation(np.array([[1, 0.5, 0.6], [0.5, 1, 0.8], [0.6, 0.8, 1]]), 0, 1, (2,)) == pytest.approx(0.0417, abs=0.00006)
    assert len(P.class_members(((), (0,), (1,)))) == 3 and len(P.class_members(((), (0, 2), ()))) == 1 and len(P.class_members(((), (0,), (0, 1)))) == 6
    cov = W.population_cov(W.World().matrix)
    assert cov[W.Z, W.X] == pytest.approx(1.312) and cov[W.X, W.X] == pytest.approx(2.0496) and 0.6 + 0.6 * 1.312 / 2.0496 == pytest.approx(0.9841, abs=0.00006)
    assert W.population_cov(ev.cancellation_matrix(1.0))[0, 2] == pytest.approx(0.0, abs=1e-12)


def test_experiment_1_sample_and_alpha(full_sample):
    by = {(r["alpha"], r["n"]): r for r in full_sample["rows"]}
    assert full_sample["reps"] == 300 and C.EXP_REPS == 300 and len(by) == 10
    for key, count in {(0.01, 100): 149, (0.01, 300): 295, (0.01, 1000): 297, (0.01, 3000): 299, (0.01, 10000): 296, (0.05, 100): 222, (0.05, 300): 278, (0.05, 1000): 276, (0.05, 3000): 280, (0.05, 10000): 281}.items():
        assert round(by[key]["exact"] * 300) == count, key
    assert by[(0.05, 10000)]["extra"] == pytest.approx(0.07, abs=0.0006) and by[(0.05, 300)]["extra"] == pytest.approx(0.05, abs=0.0006) and by[(0.05, 1000)]["extra"] == pytest.approx(0.0833, abs=0.0006)
    assert max(by[(0.01, n)]["extra"] for n in (300, 1000, 3000, 10000)) == pytest.approx(0.0133, abs=0.0006) and by[(0.01, 100)]["missing"] == pytest.approx(0.5567, abs=0.0006)
    assert all(by[(0.05, n)]["xy"] >= 0.94 for n in (300, 1000, 3000, 10000)) and all(by[(0.01, n)]["xy"] >= 0.99 for n in (300, 1000, 3000, 10000))
    assert by[(0.01, 100)]["xy"] == pytest.approx(0.7167, abs=0.0006)


def test_experiment_1_unfaithful_world(full_sample):
    u = {r["n"]: r for r in full_sample["unfaithful"]}
    assert full_sample["margin"] == pytest.approx(0.267, abs=0.0006) and full_sample["margin_unfaithful"] == pytest.approx(0.048, abs=0.0006)
    assert [round(u[n]["exact"] * 300) for n in (300, 1000, 3000)] == [42, 131, 247] and [round(u[n]["zx_missing"] * 300) for n in (300, 1000, 3000)] == [252, 164, 33]
    assert "55 % der Läufe" in ev.verdict_sample(full_sample)[1] and "(Treue-Spielraum 0,05 gegen 0,27)" in ev.verdict_sample(full_sample)[1]


def test_experiment_2_what_stays_open(full_open):
    r = {(x["d"], x["degree"]): x for x in full_open["random"]}
    assert [round(r[(d, 1.0)]["undirected"], 3) for d in (6, 8, 10)] == [0.562, 0.564, 0.565] and [round(r[(d, 2.0)]["undirected"], 3) for d in (6, 8, 10)] == [0.417, 0.364, 0.306]
    assert [round(r[(d, 3.0)]["undirected"], 3) for d in (6, 8, 10)] == [0.383, 0.237, 0.232]
    assert [round(r[(d, 2.0)]["full"], 3) for d in (6, 8, 10)] == [0.115, 0.05, 0.065] and [round(x["full"], 3) for x in full_open["random"] if x["degree"] == 1.0] == [0.149, 0.071, 0.045]
    assert max(x["full"] for x in full_open["random"]) == pytest.approx(0.149, abs=0.0006) and min(x["full"] for x in full_open["random"]) == pytest.approx(0.04, abs=0.0006)
    assert full_open["class_size"] == 3 and full_open["pop_effects"] == pytest.approx([0.6, 0.984075]) and full_open["know_effects"] == [0.6]


def test_experiment_2_effect_range(full_open):
    by = {(x["n"], x["mode"]): x for x in full_open["rows"]}
    for n, right in ((300, 0.98), (1000, 0.99), (3000, 0.9833)):
        assert by[(n, "ohne")]["right"] == pytest.approx(right, abs=0.0006)
    assert [round(by[(n, "ohne")]["width"], 2) for n in (300, 1000, 3000)] == [0.38, 0.38, 0.39] and [by[(n, "mit Vorwissen")]["width"] for n in (300, 1000, 3000)] == [0.0, 0.0, 0.0]
    assert by[(1000, "ohne")]["lo"] == pytest.approx(0.600, abs=0.0006) and by[(1000, "ohne")]["hi"] == pytest.approx(0.985, abs=0.0006) and by[(1000, "mit Vorwissen")]["lo"] == pytest.approx(0.600, abs=0.0006)
    # Konstruktion: Die Klasse enthält das wahre Diagramm, der Bereich also die Wahrheit (bis auf den Stichprobenfehler); der Befund ist die Breite
    assert all(by[(n, m)]["inside"] >= 0.97 for n in (300, 1000, 3000) for m in ("ohne", "mit Vorwissen"))
    assert "mehr Daten verengen ihn in dieser Welt nicht" in ev.verdict_open(full_open)[1] and "ist die Breite 0,00, die Wirkung 0,60" in ev.verdict_open(full_open)[1]


def test_experiment_3_cancellation(full_assumptions):
    c = {(x["n"], x["lam"]): x for x in full_assumptions["cancel"]}
    for lam, missing in ((0.0, 0.96), (0.5, 0.0), (0.8, 0.1267), (0.9, 0.6833), (1.0, 0.96), (1.1, 0.6667), (1.2, 0.1467), (1.5, 0.0)):
        assert c[(500, lam)]["missing"] == pytest.approx(missing, abs=0.0006), lam
    for lam, missing in ((0.0, 0.92), (0.5, 0.0), (0.8, 0.0), (0.9, 0.0), (1.0, 0.9733), (1.1, 0.0), (1.2, 0.0), (1.5, 0.0)):
        assert c[(5000, lam)]["missing"] == pytest.approx(missing, abs=0.0006), lam
    # Struktur, kein zweiter Befund: Fehlt X–Y mit leerer Trennmenge, ist M nicht in ihr, und PC setzt X → M ← Y; bei λ = 0 ist das Fehlen der Kante richtig
    assert all(x["collider"] == (0.0 if x["lam"] == 0.0 else x["missing"]) for x in full_assumptions["cancel"]) and c[(500, 0.0)]["right"] == pytest.approx(0.96, abs=0.0006) and c[(5000, 0.0)]["right"] == pytest.approx(0.92, abs=0.0006)
    assert c[(500, 1.0)]["right"] == pytest.approx(0.04, abs=0.0006) and c[(5000, 1.0)]["right"] == pytest.approx(0.0267, abs=0.0006)


def test_experiment_3_nonlinear_and_hidden(full_assumptions):
    nl = {(x["quadratic"], x["n"]): x for x in full_assumptions["nonlinear"]}
    assert [round(nl[(True, n)]["missing"], 3) for n in (300, 1000, 5000, 20000)] == [0.98, 0.977, 0.97, 0.973] and all(nl[(False, n)]["missing"] == 0.0 for n in (300, 1000, 5000, 20000))
    assert [round(nl[(True, n)]["exact"], 3) for n in (300, 1000, 5000, 20000)] == [0.013, 0.023, 0.03, 0.027] and min(nl[(False, n)]["exact"] for n in (300, 1000, 5000, 20000)) == pytest.approx(0.9767, abs=0.0006)
    h = {x["n"]: x for x in full_assumptions["hidden"]}
    assert all(x["main"] == ("S–X", "S→Y", "W→Y", "X→Y") for x in h.values()) and h[1000]["main_share"] == pytest.approx(0.9933, abs=0.0006) and h[5000]["main_share"] == pytest.approx(0.9967, abs=0.0006)
    assert h[1000]["xy"] == pytest.approx(0.9967, abs=0.0006) and h[1000]["lo"] == pytest.approx(0.894, abs=0.0006) and h[1000]["hi"] == pytest.approx(0.985, abs=0.0006) and h[5000]["lo"] == pytest.approx(0.892, abs=0.0006)
    assert h[1000]["inside"] == 0.0 and h[5000]["inside"] == 0.0                     # Der Bereich eines falschen Graphen verfehlt die Wahrheit; der Befund ist, dass PC nicht warnt (keine Kante bidirektional, kein Konflikt)
    text = ev.verdict_assumptions(full_assumptions)[1]
    assert "99,7 % der Läufe dieselbe Ausgabe" in text and "λ = 0,9: 68 % gegen 0 %" in text and "in 97 % der Läufe auch bei n = 20 000 (linear: 0 %)" in text


def test_experiment_4_procedure(full_procedure):
    o = full_procedure["oracle"]
    assert (o["dags"], o["classes"], o["enum_equal"], o["pc_equal"], o["alone"]) == (543, 185, 543, 543, 59) and o["sizes"] == {1: 59, 2: 48, 3: 36, 4: 19, 6: 4, 8: 12, 10: 6, 24: 1}
    rows = {x["n"]: x for x in full_procedure["random"]["rows"]}
    assert [rows[n]["exact"] for n in (100, 1000, 10000)] == pytest.approx([0.235, 0.475, 0.67], abs=0.0006) and [round(rows[n]["shd"], 2) for n in (100, 1000, 10000)] == [3.38, 2.27, 1.27]
    assert [rows[n]["skel_orig"] for n in (100, 1000, 10000)] == pytest.approx([0.135, 0.085, 0.05], abs=0.0006) and [rows[n]["skel_stab"] for n in (100, 1000, 10000)] == [0.0, 0.0, 0.0]     # Konstruktion der stabilen Fassung
    assert [rows[n]["out_orig"] for n in (100, 1000, 10000)] == pytest.approx([0.545, 0.345, 0.195], abs=0.0006) and [rows[n]["out_stab"] for n in (100, 1000, 10000)] == pytest.approx([0.475, 0.315, 0.185], abs=0.0006)
    text = ev.verdict_procedure(full_procedure)[1]
    assert "(543 von 543)" in text and "in 24 % (n = 100) bis 67 % (n = 10 000)" in text and "(14 % bei n = 100, 5 % bei n = 10 000)" in text and "ursprünglich in 20 %, stabil in 18 %" in text
