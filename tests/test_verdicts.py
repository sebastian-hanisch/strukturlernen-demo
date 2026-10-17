"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import stl_evaluation as ev


def test_instance_verdict_in_the_default_case_is_a_success_with_a_range():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_instance(a)
    assert kind == "success" and "PC findet die Klasse: 3 Kanten gerichtet, 2 ungerichtet." in text and "ist ein Bereich: " in text and "(3 Diagramme passen)" in text and "die Wahrheit ist 0,60" in text
    assert ev.de(min(a.effects)) in text and ev.de(max(a.effects)) in text and "Vorwissen schon" in text


def test_instance_verdict_with_roots_names_a_single_effect():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(knowledge="Wurzeln S und W")))
    assert kind == "success" and "5 Kanten gerichtet, 0 ungerichtet" in text and "ist eindeutig: " in text and "Vorwissen schon" not in text


def test_instance_verdict_names_the_loose_alpha():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(alpha=0.05, n=3000)))
    assert "Mit α = 0,05 gibt es auch bei vielen Daten gelegentlich eine überzählige Kante" in text or kind == "warning"


def test_instance_verdict_hidden_nachfrage_warns_without_a_hint_in_the_output():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(observed="nein")))
    assert kind == "warning" and "Die Nachfrage Z ist nicht gemessen" in text and "liegt nicht darin; nichts in der Ausgabe warnt davor" in text and "überzählig davon:" in text


def test_instance_verdict_cancellation_names_the_false_collider_and_the_margin():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(b_zy=-0.48)))
    assert kind == "warning" and "fehlende Kanten: Z–Y" in text and "falsch gerichtet: Y→X" in text and "ein Collider, den es nicht gibt (Z → X ← Y)" in text and "Treue-Spielraum der wahren Kante Z–Y ist nur 0,000" in text


def test_instance_verdict_untruthful_world_names_a_missing_edge_and_the_margin():
    for seed in range(35, 80):
        a = ev.analyse(ev.Settings(a_zx=0.7, b_xy=1.0, b_zy=0.8, alpha=0.05, seed=seed))
        if a.missing:
            kind, text = ev.verdict_instance(a)
            assert kind == "warning" and "fehlende Kanten: Z–X" in text and "Treue-Spielraum der wahren Kante Z–X ist nur 0,048" in text
            return
    raise AssertionError("die fast untreue Welt sollte die Kante Z–X verlieren")


def test_instance_verdict_nonlinear_arrow_names_the_test():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(form="quadratisch")))
    assert kind == "warning" and "fehlende Kanten: Z–Y" in text and "Fisher-z-Test sieht nur lineare Zusammenhänge" in text


def test_instance_verdict_few_data_mentions_the_weak_edges():
    for seed in range(35, 100):
        a = ev.analyse(ev.Settings(n=100, seed=seed))
        if a.missing and a.margins.get(a.missing[0], 1.0) >= 0.1:
            kind, text = ev.verdict_instance(a)
            assert kind == "warning" and "Mit so wenigen Standorten fallen schwache Kanten aus dem Skelett" in text
            return
    raise AssertionError("bei 100 Standorten sollte eine Kante fehlen")


def test_instance_verdict_for_an_extra_edge():
    for seed in range(35, 400):
        a = ev.analyse(ev.Settings(alpha=0.1, n=3000, seed=seed))
        if a.extra and not a.missing:
            kind, text = ev.verdict_instance(a)
            assert kind == "warning" and "überzählige Kanten: " in text and "fehlende Kanten" not in text
            return
    raise AssertionError("mit α = 0,1 sollte gelegentlich eine Kante zu viel entstehen")


def test_effect_range_text_for_the_special_cases():
    a = ev.analyse(ev.Settings())
    a.effects = None
    assert ev.effect_range_text(a).startswith("Die Klasse ist zu groß")
    a.effects = []
    assert ev.effect_range_text(a).startswith("Kein Diagramm der Klasse passt")
    a.effects = [0.6, 0.6000001]
    assert ev.effect_range_text(a) == "Die Wirkung des Faktors ist eindeutig: 0,60"


def test_sample_verdict(small_sample):
    kind, text = ev.verdict_sample(small_sample)
    rows = {(r["alpha"], r["n"]): r for r in small_sample["rows"]}
    assert kind == "info" and f"in {ev.pct(rows[(0.01, 100)]['exact'])} (n = 100)" in text and f"{ev.pct(rows[(0.01, 1000)]['exact'])} (n = 1 000)" in text and "überzählige Kanten je Lauf" in text
    assert f"{ev.de(small_sample['margin_unfaithful'], 2)} gegen {ev.de(small_sample['margin'], 2)}" in text and "verwirft PC Z–X bei n = 1 000 in" in text


def test_open_verdict(small_open):
    kind, text = ev.verdict_open(small_open)
    assert kind == "info" and "In der Welt passen 3 Diagramme; die Wirkung ist 0,60 oder 0,98 (Wahrheit 0,60)" in text and "mehr Daten verengen ihn nicht" in text and "die Wirkung 0,60" in text and "Breite 0,00" in text


def test_assumption_verdict(small_assumptions):
    kind, text = ev.verdict_assumptions(small_assumptions)
    assert kind == "info" and "(λ = 1)" in text and "setzt PC einen Collider, den es nicht gibt" in text and "S–X, S→Y, W→Y, X→Y" in text and "liegt in 0 % der Läufe darin" in text and "(linear: 0 %)" in text


def test_procedure_verdict(small_procedure):
    kind, text = ev.verdict_procedure(small_procedure)
    assert kind == "info" and "alle 543 Diagramme mit vier Knoten genau den CPDAG der Aufzählung aus (543 von 543)" in text and "stimmt mit den 185 Klassen" in text and "in der stabilen nie" in text


def test_builder_verdicts():
    chain = ev.verdict_builder(ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 1})))
    assert chain[0] == "info" and "3 Diagramme gehören zusammen" in chain[1] and "2 von 2 Kanten bleiben ungerichtet" in chain[1] and "185 Klassen" in chain[1]
    collider = ev.verdict_builder(ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 2})))
    assert collider[0] == "success" and "allein: Alle 2 Kanten" in collider[1] and "1 v-Struktur" in collider[1]
    cycle = ev.verdict_builder(ev.builder_result(ev.builder_parents({(0, 1): 1, (1, 2): 1, (0, 2): 2})))
    assert cycle[0] == "error" and "Kreis" in cycle[1]
    empty = ev.verdict_builder(ev.builder_result(ev.builder_parents({})))
    assert empty[0] == "info" and "alle Größen unabhängig" in empty[1] and "1 von 543" in empty[1]
