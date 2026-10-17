"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import pytest

import stl_constants as C
import stl_evaluation as ev
import stl_pc as P
import stl_visualization as V
import stl_world as W


def _arrows(fig):
    return [ann for ann in fig.layout.annotations if ann.showarrow]


def test_true_diagram_draws_every_arrow_and_marks_x_and_y():
    parents = W.World().parents
    fig = V.build_true_diagram(parents)
    assert len(_arrows(fig)) == sum(len(p) for p in parents) == 5
    nodes = {t.text[0]: t for t in fig.data if t.mode == "markers+text"}
    assert set(nodes) == set(W.NAMES) and nodes["X"].marker.line.color == C.COLOR_GOOD and nodes["Y"].marker.line.color == C.COLOR_TRUTH


def test_true_diagram_with_hidden_node_has_no_arrows_to_or_from_it():
    fig = V.build_true_diagram(W.World().parents, hidden=(W.Z,))
    assert len(_arrows(fig)) == 2 and any("(verborgen)" in t.text[0] for t in fig.data if t.mode == "markers+text")


def test_learned_diagram_colors_follow_the_verdict():
    a = ev.analyse(ev.Settings())
    fig = V.build_learned_diagram(a)
    colors = {ann.arrowcolor for ann in _arrows(fig)}
    assert colors == {C.COLOR_GOOD} and len(_arrows(fig)) == 3
    lines = [t for t in fig.data if t.mode == "lines" and t.x[0] is not None]
    assert len(lines) == 2 and all(t.line.color == C.COLOR_CONTROL for t in lines)
    assert {t.name for t in fig.data if t.name} == {"richtig (gerichtet)", "richtig, Richtung offen"}


def test_learned_diagram_shows_wrong_missing_and_extra_edges():
    a = ev.analyse(ev.Settings(b_zy=-0.48))
    fig = V.build_learned_diagram(a)
    assert C.COLOR_TRUTH in {ann.arrowcolor for ann in _arrows(fig)}                     # Y → X falsch gerichtet
    assert any(t.line.dash == "dash" for t in fig.data if t.mode == "lines" and t.line.color == C.COLOR_NEUTRAL)
    names = {t.name for t in fig.data if t.name}
    assert "falsch gerichtet" in names and "fehlt" in names
    hidden = V.build_learned_diagram(ev.analyse(ev.Settings(observed="nein")))
    assert C.COLOR_BAD in {ann.arrowcolor for ann in _arrows(hidden)} and any("(verborgen)" in t.text[0] for t in hidden.data if t.mode == "markers+text")


def test_edge_table_by_hand():
    a = ev.analyse(ev.Settings())
    t = V.build_edge_table(a)
    assert list(t.columns) == ["Kante", "PC", "Urteil", "Treue-Spielraum"] and len(t) == 5
    row = t[t["Kante"] == "S – Z"].iloc[0]
    assert row["PC"] == "ungerichtet" and row["Urteil"] == "richtig, Richtung offen" and row["Treue-Spielraum"] == "0,442"
    assert t[t["Kante"] == "X → Y"].iloc[0]["Urteil"] == "richtig (gerichtet)"
    V.style_edge_table(V.build_edge_table(ev.analyse(ev.Settings(b_zy=-0.48))))
    miss = V.build_edge_table(ev.analyse(ev.Settings(b_zy=-0.48)))
    assert (miss["PC"] == "nicht gefunden").sum() == 1 and miss[miss["PC"] == "nicht gefunden"].iloc[0]["Treue-Spielraum"] == "0,000"


def test_summary_table_by_hand():
    a = ev.analyse(ev.Settings())
    t = V.build_summary_table(a)
    assert t.iloc[0]["Wert"] == "3 / 2" and t.iloc[1]["Wert"] == "ja" and t.iloc[4]["Wert"] == "3" and t.iloc[5]["Wert"] == "0,60" and "bis" in t.iloc[6]["Wert"]
    assert V.build_summary_table(ev.analyse(ev.Settings(b_zy=-0.48))).iloc[1]["Wert"].startswith("nein (Strukturabstand ")


def test_effect_chart_and_class_table_agree():
    a = ev.analyse(ev.Settings())
    fig = V.build_effect_chart(a)
    xs = sorted(x for t in fig.data if t.mode == "markers+text" for x in t.x)
    assert len(xs) == 2 and xs == pytest.approx(sorted({round(e, 3) for e in a.effects})) and fig.layout.shapes[0].x0 == pytest.approx(0.6)
    t = V.build_class_table(a)
    assert len(t) == 3 and list(t.columns) == ["Diagramm", "Eltern von X", "Wirkung"] and sorted(t["Wirkung"]) == sorted(ev.de(e, 2) for e in a.effects)
    assert set(t["Eltern von X"]) == {"Z", "keine"}
    a.effects, a.dags = None, None
    assert len(V.build_class_table(a)) == 0 and V.build_effect_chart(a) is not None


def test_experiment_figures_and_tables_build(small_sample, small_open, small_assumptions, small_procedure):
    assert V.build_sample_chart(small_sample) is not None and len(V.build_sample_table(small_sample)) == 6 and len(V.build_unfaithful_table(small_sample)) == 3
    assert V.build_open_chart(small_open) is not None and len(V.build_open_table(small_open)) == 2 and len(V.build_range_table(small_open)) == 4
    assert V.build_cancel_chart(small_assumptions) is not None and len(V.build_cancel_table(small_assumptions)) == 6 and len(V.build_nonlinear_table(small_assumptions)) == 4 and len(V.build_hidden_table(small_assumptions)) == 1
    assert V.build_procedure_chart(small_procedure) is not None and len(V.build_random_table(small_procedure)) == 2 and len(V.build_oracle_table(small_procedure)) == 8


def test_cancel_chart_has_one_line_per_sample_size_and_the_table_formats(small_assumptions):
    fig = V.build_cancel_chart(small_assumptions)
    assert [t.name for t in fig.data] == ["falscher Collider, n = 500", "falscher Collider, n = 5000"]
    t = V.build_cancel_table(small_assumptions)
    row = t[(t["n"] == 500) & (t["λ"] == "1,0")].iloc[0]
    assert row["Kante X–Y fehlt"] == row["falscher Collider X → M ← Y"]


def test_oracle_table_counts_the_diagrams(small_procedure):
    t = V.build_oracle_table(small_procedure)
    assert t["Diagramme"].sum() == 543 and t["Klassen"].sum() == 185 and list(t["Klassengröße"])[0] == 1


def test_builder_diagram_draws_arrows_for_directed_and_lines_for_undirected_edges():
    parents = ((), (0,), (1, 3), ())
    b = ev.builder_result(parents)
    own = V.build_builder_diagram(P.dag_graph(parents), "Dein Diagramm")
    assert len(_arrows(own)) == 3 and len([t for t in own.data if t.mode == "markers+text"]) == 4
    cp = V.build_builder_diagram(b["cpdag"], "CPDAG")
    assert len(_arrows(cp)) == 2 and len([t for t in cp.data if t.mode == "lines"]) == 1
    table = V.build_builder_table(b)
    assert len(table) == b["size"] and V.build_builder_table({"acyclic": False}).empty
