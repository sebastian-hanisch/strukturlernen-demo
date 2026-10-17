"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import stl_world as W
from stl_evaluation import Analysis, de, effect_range_text, signed, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "α": "alpha", "λ": "lambda", "≤": "<=", "≥": ">=", "±": "+-"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Das Diagramm aus Daten lernen: Was PC findet"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Pfeilstärken S->Z {de(s.a_sz, 2)}, Z->X {de(s.a_zx, 2)}, X->Y {de(s.b_xy, 2)}, Z->Y {de(s.b_zy, 2)} ({s.form}), W->Y {de(s.b_wy, 2)}; {s.n} Standorte, Niveau alpha = {de(s.alpha, 3)}, "
                            f"PC {s.version}, Vorwissen: {s.knowledge}, Nachfrage beobachtet: {s.observed}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for x, y, kind, status in a.edge_status:
        arrow = "->" if kind == "gerichtet" else "-"
        pdf.multi_cell(0, 4.6, _s(f"Kante {W.NAMES[x]} {arrow} {W.NAMES[y]}: {kind}, {status}."), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 4.6, _s(f"Gleich dem Orakel-Ergebnis: {'ja' if a.exact else 'nein, Strukturabstand ' + str(a.shd_oracle)}. Kleinster Treue-Spielraum der wahren Kanten: {de(a.margin, 3)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 4.6, _s(f"{effect_range_text(a)}; wahre Wirkung {de(a.truth, 2)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: strukturlernen-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
