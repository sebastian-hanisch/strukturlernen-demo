"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import stl_constants as C
import stl_evaluation as ev
import stl_world as W
from stl_evaluation import Analysis, de, pct

POS = {W.S: (0.0, 1.0), W.Z: (0.34, 1.0), W.X: (0.34, 0.0), W.Y: (0.80, 0.5), W.W: (1.0, 1.0)}
STATUS_COLORS = {"richtig": C.COLOR_GOOD, "offen": C.COLOR_CONTROL, "falsch gerichtet": C.COLOR_TRUTH, "überzählig": C.COLOR_BAD, "fehlt": C.COLOR_NEUTRAL}
STATUS_LABELS = {"richtig": "richtig (gerichtet)", "offen": "richtig, Richtung offen", "falsch gerichtet": "falsch gerichtet", "überzählig": "überzählig", "fehlt": "fehlt"}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segment(p0, p1, r: float = 0.11):
    """Anfang und Ende einer Linie, an den Knotenrändern gekürzt."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    return (p0[0] + ux * r, p0[1] + uy * r), (p1[0] - ux * r, p1[1] - uy * r)


def _draw_edges(fig: go.Figure, pos: dict, directed, undirected, color_of, dashed=()):
    for a, b in directed:
        s, e = _segment(pos[a], pos[b])
        c, w = color_of(a, b)
        fig.add_annotation(x=e[0], y=e[1], ax=s[0], ay=s[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=w, arrowcolor=c, text="")
    for a, b in undirected:
        s, e = _segment(pos[a], pos[b], 0.1)
        c, w = color_of(a, b)
        fig.add_trace(go.Scatter(x=[s[0], e[0]], y=[s[1], e[1]], mode="lines", line=dict(color=c, width=w), hoverinfo="skip", showlegend=False))
    for a, b in dashed:
        s, e = _segment(pos[a], pos[b], 0.1)
        fig.add_trace(go.Scatter(x=[s[0], e[0]], y=[s[1], e[1]], mode="lines", line=dict(color=C.COLOR_NEUTRAL, width=2, dash="dash"), hoverinfo="skip", showlegend=False))


def _draw_nodes(fig: go.Figure, pos: dict, names: dict, hidden=(), outlines=None):
    outlines = outlines or {}
    for v, (x, y) in pos.items():
        is_hidden = v in hidden
        color = outlines.get(v, "#1b1b1b")
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[names[v] + (" (verborgen)" if is_hidden else "")], textposition="middle center", textfont=dict(size=11 if is_hidden else 12),
                                 marker=dict(size=64, color="rgba(47,107,101,0.05)" if is_hidden else "rgba(47,107,101,0.12)", line=dict(width=2 if not is_hidden else 1, color=color, dash=None)),
                                 hoverinfo="skip", showlegend=False))


def _layout_graph(fig: go.Figure, height: int = 330):
    fig.update_xaxes(visible=False, range=[-0.2, 1.2])
    fig.update_yaxes(visible=False, range=[-0.3, 1.3])
    return _fixed(fig, height, showlegend=True)


def build_true_diagram(parents: tuple, hidden=()) -> go.Figure:
    """Das wahre Diagramm: alle Pfeile grau; X grün und Y orange umrandet; eine verborgene Größe gestrichelt ohne Pfeile."""
    fig = go.Figure()
    edges = [(p, c) for c, ps in enumerate(parents) for p in ps if p not in hidden and c not in hidden]
    _draw_edges(fig, POS, edges, [], lambda a, b: ("#6b6b6b", 3))
    _draw_nodes(fig, POS, dict(enumerate(W.NAMES)), hidden, {W.X: C.COLOR_GOOD, W.Y: C.COLOR_TRUTH})
    fig.update_layout(showlegend=False)
    return _layout_graph(fig).update_layout(showlegend=False)


def build_learned_diagram(a: Analysis) -> go.Figure:
    """Der von PC gelernte Graph: Pfeile und ungerichtete Kanten nach Status gefärbt (richtig, richtig mit offener Richtung, falsch gerichtet, überzählig); fehlende Kanten gestrichelt grau."""
    s = a.settings
    fig = go.Figure()
    status = {(x, y): st for x, y, k, st in a.edge_status if k != "fehlt"}
    present = set()

    def color_of(x, y):
        st = status[(x, y)]
        present.add(st)
        return STATUS_COLORS[st], 4 if st != "richtig" else 3

    _draw_edges(fig, POS, sorted(a.learned[0]), sorted(a.learned[1]), color_of, dashed=a.missing)
    _draw_nodes(fig, POS, dict(enumerate(W.NAMES)), hidden=tuple(v for v in range(5) if v not in s.obs), outlines={W.X: C.COLOR_GOOD, W.Y: C.COLOR_TRUTH})
    for st in ("richtig", "offen", "falsch gerichtet", "überzählig"):
        if st in present:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=STATUS_COLORS[st], width=4), name=STATUS_LABELS[st]))
    if a.missing:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLOR_NEUTRAL, width=2, dash="dash"), name=STATUS_LABELS["fehlt"]))
    return _layout_graph(fig)


def build_edge_table(a: Analysis) -> pd.DataFrame:
    rows = []
    for x, y, kind, st in a.edge_status:
        shown = f"{W.NAMES[x]} → {W.NAMES[y]}" if kind == "gerichtet" else f"{W.NAMES[x]} – {W.NAMES[y]}"
        margin = a.margins.get((x, y), a.margins.get((y, x)))
        rows.append({"Kante": shown, "PC": {"gerichtet": "gerichtet", "ungerichtet": "ungerichtet", "fehlt": "nicht gefunden"}[kind], "Urteil": STATUS_LABELS[st], "Treue-Spielraum": "" if margin is None else de(margin, 3)})
    return pd.DataFrame(rows)


def style_edge_table(df: pd.DataFrame):
    def row_style(row):
        bad = row["Urteil"] in (STATUS_LABELS["überzählig"], STATUS_LABELS["falsch gerichtet"], STATUS_LABELS["fehlt"])
        return ["background-color: rgba(192,57,43,0.15)" if bad else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_summary_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    rows = [
        {"Größe": "Gelernt: gerichtete / ungerichtete Kanten", "Wert": f"{len(a.learned[0])} / {len(a.learned[1])}"},
        {"Größe": "Gleich dem Ergebnis mit unendlich vielen Daten (Orakel-PC)", "Wert": "ja" if a.exact else f"nein (Strukturabstand {a.shd_oracle})"},
        {"Größe": "Konflikte bei der Orientierung", "Wert": str(a.conflicts)},
        {"Größe": "Kleinster Treue-Spielraum der wahren Kanten", "Wert": de(a.margin, 3)},
        {"Größe": "Diagramme in der Klasse der Ausgabe", "Wert": "zu viele" if a.class_size is None else str(a.class_size)},
        {"Größe": "Wahre Wirkung des Faktors", "Wert": de(a.truth, 2)},
    ]
    if a.effects:
        rows.append({"Größe": "Mögliche Wirkungen über die Klasse (kleinste bis größte)", "Wert": f"{de(min(a.effects), 2)} bis {de(max(a.effects), 2)}"})
    rows.append({"Größe": "Standorte, Niveau α, Fassung", "Wert": f"{s.n}, {de(s.alpha, 3)}, {s.version}"})
    return pd.DataFrame(rows)


def build_effect_chart(a: Analysis) -> go.Figure:
    """Die möglichen Wirkungen: ein Punkt je Wert, der aus einem Diagramm der Klasse folgt; orange senkrecht die Wahrheit."""
    fig = go.Figure()
    if a.effects:
        vals = sorted({round(e, 3) for e in a.effects})
        if len(vals) > 1:
            fig.add_trace(go.Scatter(x=[min(vals), max(vals)], y=["Wirkung", "Wirkung"], mode="lines", line=dict(color=C.COLOR_NEUTRAL, width=6), opacity=0.4, hoverinfo="skip", showlegend=False))
        counts = {v: sum(1 for e in a.effects if round(e, 3) == v) for v in vals}
        ok = a.contains
        fig.add_trace(go.Scatter(x=vals, y=["Wirkung"] * len(vals), mode="markers+text", text=[f"{de(v, 2)} ({counts[v]}×)" for v in vals], textposition="top center",
                                 marker=dict(size=14, color=C.COLOR_GOOD if ok else C.COLOR_BAD, symbol="diamond"), hovertemplate="%{x:.3f}<extra></extra>", showlegend=False))
    fig.add_vline(x=a.truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(a.truth, 2)}", annotation_position="top")
    fig.update_xaxes(title="mögliche Wirkung des hohen Faktors (Punkte je Einheit Faktor)")
    return _fixed(fig, 240, showlegend=False)


def build_class_table(a: Analysis) -> pd.DataFrame:
    if not a.dags:
        return pd.DataFrame(columns=["Diagramm", "Eltern von X", "Wirkung"])
    names = [W.NAMES[v] for v in a.obs]
    xi = a.obs.index(W.X)
    rows = []
    for k, t in enumerate(a.dags, 1):
        edges = ", ".join(f"{names[p]}→{names[c]}" for c, ps in enumerate(t) for p in ps)
        rows.append({"Diagramm": f"{k}: {edges}", "Eltern von X": ", ".join(names[p] for p in t[xi]) or "keine", "Wirkung": de(a.effects[k - 1], 2)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experiment 1

def build_sample_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for alpha, color in ((0.01, C.COLOR_GOOD), (0.05, C.COLOR_CONTROL)):
        rows = [r for r in res["rows"] if r["alpha"] == alpha]
        fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[100 * r["exact"] for r in rows], mode="lines+markers", name=f"leichte Welt, α = {de(alpha, 2)}", line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(x=[r["n"] for r in res["unfaithful"]], y=[100 * r["exact"] for r in res["unfaithful"]], mode="lines+markers", name="fast untreue Welt, α = 0,05", line=dict(color=C.COLOR_BAD, width=3, dash="dash")))
    fig.update_xaxes(title="Standorte n", type="log")
    fig.update_yaxes(title="CPDAG genau getroffen (% der Läufe)", range=[-2, 104])
    return _fixed(fig, 380)


def build_sample_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"α": de(r["alpha"], 2), "n": r["n"], "CPDAG genau": pct(r["exact"]), "Skelett genau": pct(r["skeleton"]), "X → Y gerichtet": pct(r["xy"]), "Strukturabstand": de(r["shd"], 2),
                          "fehlende Kanten": de(r["missing"], 2), "überzählige Kanten": de(r["extra"], 2)} for r in res["rows"]])


def build_unfaithful_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"n": r["n"], "CPDAG genau": pct(r["exact"]), "Kante Z–X fehlt": pct(r["zx_missing"]), "Strukturabstand": de(r["shd"], 2), "überzählige Kanten": de(r["extra"], 2)} for r in res["unfaithful"]])


# ----------------------------------------------------------------------------- Experiment 2

def build_open_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    degrees = sorted({r["degree"] for r in res["random"]})
    for deg in degrees:
        rows = [r for r in res["random"] if r["degree"] == deg]
        fig.add_trace(go.Bar(x=[f"{r['d']} Knoten" for r in rows], y=[100 * r["undirected"] for r in rows], name=f"mittlerer Grad {de(deg, 0)}"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="ungerichtete Kanten im CPDAG (%)")
    return _fixed(fig, 360)


def build_open_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Knoten": r["d"], "mittlerer Grad": de(r["degree"], 0), "Kanten im Mittel": de(r["edges"], 1), "ungerichtet": pct(r["undirected"]), "vollständig gerichtete Graphen": pct(r["full"])} for r in res["random"]])


def build_range_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"n": r["n"], "Vorwissen": r["mode"], "richtig gelernt": pct(r["right"]), "Wirkungsbereich im Mittel": f"{de(r['lo'], 2)} bis {de(r['hi'], 2)}", "Breite": de(r["width"], 2),
                          "enthält die Wahrheit (±0,15)": pct(r["inside"])} for r in res["rows"]])


# ----------------------------------------------------------------------------- Experiment 3

def build_cancel_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for n in sorted({r["n"] for r in res["cancel"]}):
        rows = [r for r in res["cancel"] if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["lam"] for r in rows], y=[100 * r["collider"] for r in rows], mode="lines+markers", name=f"falscher Collider, n = {n}", line=dict(width=3)))
    fig.update_xaxes(title="λ: Anteil des Umwegs, den der direkte Pfeil aufhebt (1 = genau)")
    fig.update_yaxes(title="Läufe mit falschem Collider (%)", range=[-2, 104])
    return _fixed(fig, 360)


def build_cancel_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"n": r["n"], "λ": de(r["lam"], 1), "Kante X–Y fehlt": pct(r["missing"]), "falscher Collider X → M ← Y": pct(r["collider"]), "CPDAG richtig": pct(r["right"])} for r in res["cancel"]])


def build_nonlinear_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Z → Y": "quadratisch" if r["quadratic"] else "linear", "n": r["n"], "Kante Z–Y fehlt": pct(r["missing"]), "CPDAG genau": pct(r["exact"])} for r in res["nonlinear"]])


def build_hidden_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"n": r["n"], "häufigste Ausgabe": ", ".join(r["main"]), "Anteil der Läufe": pct(r["main_share"]), "X → Y gerichtet": pct(r["xy"]), "Wirkungsbereich im Mittel": f"{de(r['lo'], 2)} bis {de(r['hi'], 2)}",
                          "enthält die Wahrheit (±0,15)": pct(r["inside"])} for r in res["hidden"]])


# ----------------------------------------------------------------------------- Experiment 4

def build_procedure_chart(res: dict) -> go.Figure:
    rows = res["random"]["rows"]
    fig = go.Figure()
    labels = [f"n = {r['n']}" for r in rows]
    fig.add_trace(go.Bar(x=labels, y=[100 * r["out_orig"] for r in rows], name="Ausgabe, ursprünglich", marker_color=C.COLOR_NEUTRAL))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["out_stab"] for r in rows], name="Ausgabe, stabil", marker_color=C.COLOR_CONTROL))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["skel_orig"] for r in rows], name="Skelett, ursprünglich", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["skel_stab"] for r in rows], name="Skelett, stabil", marker_color=C.COLOR_GOOD))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Datensätze, in denen die Reihenfolge der Variablen etwas ändert (%)")
    return _fixed(fig, 380)


def build_random_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"n": r["n"], "CPDAG genau": pct(r["exact"]), "Strukturabstand": de(r["shd"], 2), "Skelett hängt ab (ursprünglich)": pct(r["skel_orig"]), "Skelett hängt ab (stabil)": pct(r["skel_stab"]),
                          "Ausgabe hängt ab (ursprünglich)": pct(r["out_orig"]), "Ausgabe hängt ab (stabil)": pct(r["out_stab"])} for r in res["random"]["rows"]])


def build_oracle_table(res: dict) -> pd.DataFrame:
    o = res["oracle"]
    return pd.DataFrame([{"Klassengröße": k, "Klassen": v, "Diagramme": k * v} for k, v in o["sizes"].items()])


# ----------------------------------------------------------------------------- Baukasten

BUILDER_POS = {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.0, 0.0), 3: (1.0, 0.0)}


def build_builder_diagram(graph: tuple, title: str = "") -> go.Figure:
    """Ein Graph auf den vier Knoten A bis D: Pfeile für gerichtete, Linien für ungerichtete Kanten."""
    fig = go.Figure()
    names = dict(enumerate(P_NAMES))
    _draw_edges(fig, BUILDER_POS, sorted(graph[0]), sorted(graph[1]), lambda a, b: (C.COLOR_GOOD if (a, b) in graph[0] else C.COLOR_CONTROL, 3))
    _draw_nodes(fig, BUILDER_POS, names)
    fig.update_xaxes(visible=False, range=[-0.35, 1.35])
    fig.update_yaxes(visible=False, range=[-0.35, 1.35])
    return _fixed(fig, 280, showlegend=False, title=dict(text=title, font=dict(size=13)))


P_NAMES = ev.BUILDER_NAMES


def build_builder_table(b: dict) -> pd.DataFrame:
    if not b.get("acyclic"):
        return pd.DataFrame(columns=["Diagramm in der Klasse"])
    return pd.DataFrame([{"Diagramm in der Klasse": ", ".join(f"{P_NAMES[p]}→{P_NAMES[c]}" for c, ps in enumerate(t) for p in ps) or "keine Kanten"} for t in b["members"]])
