"""Das Diagramm aus Daten lernen: Was PC findet und was offen bleibt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dreizehntes Stück der Linie Kausale Diagramme und Adjustierung. Der PC-Algorithmus baut aus Unabhängigkeitstests ein Diagramm bis zur Markov-Äquivalenzklasse; was offen bleibt, macht die Wirkung zu einem Bereich,
und leise Annahmen (Treue, Linearität, nichts Verborgenes) bestimmen, ob die Klasse stimmt.
Kern: stl_pc.py, stl_world.py, stl_evaluation.py; Diagramme: dsep_graph.py (wortgleich aus den Vorgängern).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import stl_constants as C
import stl_pc as P
from stl_evaluation import (
    BUILDER_NAMES,
    BUILDER_PAIRS,
    BUILDER_STATES,
    Settings,
    analyse,
    assumption_experiment,
    builder_parents,
    builder_result,
    de,
    open_experiment,
    procedure_experiment,
    sample_experiment,
    verdict_assumptions,
    verdict_builder,
    verdict_instance,
    verdict_open,
    verdict_procedure,
    verdict_sample,
)
from stl_pdf_export import build_pdf
from stl_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from stl_visualization import (
    build_builder_diagram,
    build_builder_table,
    build_cancel_chart,
    build_cancel_table,
    build_class_table,
    build_edge_table,
    build_effect_chart,
    build_hidden_table,
    build_learned_diagram,
    build_nonlinear_table,
    build_open_chart,
    build_open_table,
    build_oracle_table,
    build_procedure_chart,
    build_random_table,
    build_range_table,
    build_sample_chart,
    build_sample_table,
    build_summary_table,
    build_true_diagram,
    build_unfaithful_table,
    style_edge_table,
)

st.set_page_config(page_title="Das Diagramm aus Daten – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"
PAIR_KEYS = {p: f"bld_{BUILDER_NAMES[p[0]]}{BUILDER_NAMES[p[1]]}" for p in BUILDER_PAIRS}
BUILDER_EXAMPLES = {
    "Kette A → B → C": {(0, 1): 1, (1, 2): 1},
    "Collider A → B ← C": {(0, 1): 1, (1, 2): 2},
    "Collider mit Abkömmling": {(0, 1): 1, (1, 2): 2, (1, 3): 1},
    "Dreieck": {(0, 1): 1, (1, 2): 1, (0, 2): 1},
    "Baum mit Abzweig": {(0, 1): 1, (1, 2): 1, (1, 3): 1},
    "Kreis (kein Diagramm)": {(0, 1): 1, (1, 2): 1, (0, 2): 2},
}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _sample():
    return sample_experiment()


@st.cache_data(show_spinner=SPINNER)
def _open():
    return open_experiment()


@st.cache_data(show_spinner=SPINNER)
def _assumptions():
    return assumption_experiment()


@st.cache_data(show_spinner=SPINNER)
def _procedure():
    return procedure_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def set_builder_example(name):
    ex = BUILDER_EXAMPLES[name]
    for pair, key in PAIR_KEYS.items():
        st.session_state[key] = BUILDER_STATES[ex.get(pair, 0)]


def experiment_button(label, key):
    if st.button(label, key=f"{key}_run"):
        st.session_state[f"{key}_ready"] = True
    return st.session_state.get(f"{key}_ready", False)


st.title("🔍 Das Diagramm aus Daten lernen")
st.markdown(
    """
**Kann man das Diagramm aus den Daten lesen?** Dies ist das dreizehnte Stück der Linie **Kausale Diagramme und Adjustierung**. Die Stücke davor haben das Diagramm **vorausgesetzt**; hier soll es aus Messungen entstehen.
Der **PC-Algorithmus** testet, welche Größen gegeben welche anderen unabhängig sind, wirft die Kanten dazu hinaus und richtet die Pfeile, die sich aus **v-Strukturen** (Collidern) ergeben. Mehr verraten Unabhängigkeitstests bei linear-gaußschen Daten nicht:
Es bleibt eine **Äquivalenzklasse** von Diagrammen, und die Wirkung des Faktors ist dann ein **Bereich** statt einer Zahl. Ob die Klasse stimmt, hängt an Annahmen, die leise brechen.

Weiter unten: der **Baukasten für Äquivalenzklassen**, vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man das Ergebnis", expanded=True):
    st.markdown(
        """
1. **Wahres Diagramm** (links): Saison S → Nachfrage Z → Faktor X → Servicegrad Y, dazu Z → Y und Wetter W → Y. Gesucht ist die Wirkung von X auf Y.
2. **Von PC gelernt** (rechts): **grün** richtig, **blau** richtig, aber die Richtung bleibt offen, **orange** falsch gerichtet, **rot** überzählig, **grau gestrichelt** fehlende Kante.
3. **Die Klasse:** Alle Diagramme, die dieselben Unabhängigkeiten haben, passen zu den Daten. Aus jedem folgt eine Wirkung (Anpassung an die Eltern von X); der **Wirkungsbereich** reicht vom kleinsten bis zum größten Wert.
4. **Vorwissen** (Saison und Wetter sind Wurzeln) richtet weitere Pfeile und kann den Bereich auf eine Zahl bringen.
5. **Annahmen:** Treue (keine zufällige Aufhebung), linear und Gauß (Fisher-z-Test), nichts Verborgenes. PC prüft sie nicht; die Experimente zeigen, was geschieht, wenn sie brechen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
names = list(C.PRESETS)
for chunk in (names[:4], names[4:]):
    cols = st.columns(4)
    for col, name in zip(cols, chunk):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Pfeilstärken der Welt**")
    a_sz = st.slider("Saison → Nachfrage", *bounds("asz_slider"), step=C.COEF_STEP, key="asz_slider", help="Koeffizient im linearen Modell mit Einheitsrauschen; 0 entfernt den Pfeil.")
    a_zx = st.slider("Nachfrage → Faktor", *bounds("azx_slider"), step=C.COEF_STEP, key="azx_slider", help="Der Disponent wählt den hohen Faktor nach der Nachfrage.")
    b_xy = st.slider("Faktor → Servicegrad (die Wirkung)", *bounds("bxy_slider"), step=C.COEF_STEP, key="bxy_slider", help="Die gesuchte Wirkung des hohen Faktors.")
    b_zy = st.slider("Nachfrage → Servicegrad", *bounds("bzy_slider"), step=C.COEF_STEP, key="bzy_slider", help="Negativ und genau −(Nachfrage → Faktor) · (Faktor → Servicegrad) hebt den Umweg über den Faktor auf.")
    b_wy = st.slider("Wetter → Servicegrad", *bounds("bwy_slider"), step=C.COEF_STEP, key="bwy_slider", help="Wetter wirkt nur auf den Servicegrad: Es bildet mit der Nachfrage eine v-Struktur.")
    st.markdown("**Daten und Verfahren**")
    n = st.slider("Standorte", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Zeilen der Datenmatrix.")
    alpha = st.select_slider("Niveau α des Tests", options=C.ALPHAS, key="alpha_slider", help="Eine Kante wird entfernt, wenn der Test auf Unabhängigkeit nicht ablehnt (p-Wert über α). α ist die Fehlalarmrate je Test.")
    version = st.radio("PC-Fassung", C.VERSIONS, key="version_radio", help="Stabil: die Nachbarschaften werden je Stufe eingefroren, das Skelett hängt dann nicht von der Reihenfolge der Variablen ab.")
    knowledge = st.radio("Vorwissen", C.KNOWLEDGE, key="knowledge_radio", help="Wurzeln: Nichts wirkt auf Saison und Wetter. Das richtet ihre Kanten nach außen.")
    st.markdown("**Was PC sieht**")
    observed = st.radio("Nachfrage gemessen?", C.YES_NO, key="observed_radio", help="Bei „nein“ fehlt die Spalte der Nachfrage: eine verborgene Ursache von Faktor und Servicegrad.")
    form = st.radio("Form des Pfeils Nachfrage → Servicegrad", C.FORMS, key="form_radio", help="Quadratisch: symmetrisch um null, gegeben den Faktor ist die lineare Teilkorrelation null. Der Test sieht nur Linearität.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"asz_slider": round(float(a_sz), 2), "azx_slider": round(float(a_zx), 2), "bxy_slider": round(float(b_xy), 2), "bzy_slider": round(float(b_zy), 2), "bwy_slider": round(float(b_wy), 2),
                   "n_slider": int(n), "alpha_slider": alpha, "version_radio": version, "knowledge_radio": knowledge, "observed_radio": observed, "form_radio": form, "seed_input": int(seed)})

settings = Settings(round(float(a_sz), 2), round(float(a_zx), 2), round(float(b_xy), 2), round(float(b_zy), 2), round(float(b_wy), 2), int(n), float(alpha), version, knowledge, observed, form, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahre Wirkung des Faktors", de(a.truth, 2), delta=f"Kleinster Treue-Spielraum {de(a.margin, 3)}", delta_color="off", help="Der Koeffizient von X in Y. Der Treue-Spielraum ist die kleinste Teilkorrelation einer wahren Kante über alle Bedingungsmengen.")
c2.metric("Von PC gelernt", f"{len(a.learned[0])} gerichtet, {len(a.learned[1])} offen", delta=("gleich dem Ergebnis mit unendlich vielen Daten" if a.exact else f"Strukturabstand zum Orakel-Ergebnis {a.shd_oracle}"), delta_color="off",
          help="Gerichtete und ungerichtete Kanten der Ausgabe; das Orakel-Ergebnis ist, was PC mit den wahren Unabhängigkeiten ausgäbe.")
c3.metric("Diagramme in der Klasse", "zu viele" if a.class_size is None else str(a.class_size), help="Alle azyklischen Orientierungen der ungerichteten Kanten, die keine neue v-Struktur erzeugen.")
if a.effects:
    c4.metric("Mögliche Wirkungen", f"{de(min(a.effects), 2)} bis {de(max(a.effects), 2)}", delta="enthält die Wahrheit" if a.contains else "enthält die Wahrheit nicht", delta_color="off",
              help="Für jedes Diagramm der Klasse die Anpassung an die Eltern von X; kleinster bis größter Wert (Toleranz ±0,15 für den Stichprobenfehler beim Prüfen).")
else:
    c4.metric("Mögliche Wirkungen", "—", help="Kein Diagramm der Klasse passt, oder die Klasse ist zu groß.")

left, right = st.columns(2)
with left:
    st.plotly_chart(build_true_diagram(a.parents, hidden=tuple(v for v in range(5) if v not in a.settings.obs)), width="stretch", key="true_chart")
    st.caption("Das wahre Diagramm der Welt. X (grün umrandet) ist der Faktor, Y (orange umrandet) der Servicegrad.")
with right:
    st.plotly_chart(build_learned_diagram(a), width="stretch", key="learned_chart")
    st.caption("Was PC aus den Daten baut, gefärbt nach dem Urteil gegen das wahre Diagramm.")
verdict(verdict_instance(a))
st.plotly_chart(build_effect_chart(a), width="stretch", key="effect_chart")
st.caption("Die möglichen Wirkungen des Faktors, eine je Diagramm der Klasse (in Klammern: wie viele Diagramme denselben Wert geben); orange senkrecht die Wahrheit.")
st.dataframe(style_edge_table(build_edge_table(a)), hide_index=True, width="stretch")
st.dataframe(build_class_table(a), hide_index=True, width="stretch")
st.dataframe(build_summary_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="strukturlernen.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Baukasten für Äquivalenzklassen -------------------------------------------------------------------------------------------------------

st.subheader("🧩 Äquivalenzklassen zum Bauen")
st.caption("Vier Größen A bis D, je Paar keine Kante oder ein Pfeil. Gezeigt wird, welche Diagramme dieselben Unabhängigkeiten behaupten (exakt, über alle 543 Diagramme mit vier Knoten) und was davon aus Daten bestimmbar ist: "
           "gerichtet sind nur Pfeile, die in jedem Diagramm der Klasse gleich stehen.")
for pair, key in PAIR_KEYS.items():
    st.session_state.setdefault(key, BUILDER_STATES[BUILDER_EXAMPLES["Baum mit Abzweig"].get(pair, 0)])
ex_cols = st.columns(len(BUILDER_EXAMPLES))
for col, name in zip(ex_cols, BUILDER_EXAMPLES):
    with col:
        st.button(name, width="stretch", on_click=set_builder_example, args=(name,), key=f"bld_example_{name}")
pair_cols = st.columns(3)
for k, (pair, key) in enumerate(PAIR_KEYS.items()):
    with pair_cols[k % 3]:
        i, j = pair
        labels = {BUILDER_STATES[0]: BUILDER_STATES[0], BUILDER_STATES[1]: f"{BUILDER_NAMES[i]} → {BUILDER_NAMES[j]}", BUILDER_STATES[2]: f"{BUILDER_NAMES[i]} ← {BUILDER_NAMES[j]}"}
        st.radio(f"Paar {BUILDER_NAMES[i]}–{BUILDER_NAMES[j]}", BUILDER_STATES, key=key, format_func=labels.get, horizontal=True)
states = {pair: BUILDER_STATES.index(st.session_state[key]) for pair, key in PAIR_KEYS.items()}
parents = builder_parents(states)
b = builder_result(parents)
if b["acyclic"]:
    bl, br = st.columns(2)
    with bl:
        st.plotly_chart(build_builder_diagram(P.dag_graph(parents), "Dein Diagramm"), width="stretch", key="builder_dag")
    with br:
        st.plotly_chart(build_builder_diagram(b["cpdag"], "Was aus Daten bestimmbar ist (CPDAG)"), width="stretch", key="builder_cpdag")
    m1, m2, m3 = st.columns(3)
    m1.metric("Diagramme in dieser Klasse", b["size"])
    m2.metric("v-Strukturen", len(b["vstructs"]))
    m3.metric("Ungerichtete Kanten", len(b["cpdag"][1]))
verdict(verdict_builder(b))
if b["acyclic"] and b["edges"]:
    st.dataframe(build_builder_table(b), hide_index=True, width="stretch")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie viele Daten, welches α?")
st.caption(f"{C.EXP_REPS} feste Wiederholungen je Zelle auf der Welt der Voreinstellung. Dazu eine Welt mit fast untreuer Kante Z–X (stärkere Pfeile, aber der Treue-Spielraum der Kante ist klein).")
if experiment_button("▶ Experiment 1 rechnen (etwa 20 Sekunden)", "exp1"):
    res1 = _sample()
    st.plotly_chart(build_sample_chart(res1), width="stretch", key="sample_chart")
    verdict(verdict_sample(res1))
    st.dataframe(build_sample_table(res1), hide_index=True, width="stretch")
    st.dataframe(build_unfaithful_table(res1), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Was offen bleibt")
st.caption("Links die Zufalls-DAGs: Anteil ungerichteter Kanten im CPDAG (mit den wahren Unabhängigkeiten, ohne Stichprobenfehler). Dann die Welt: Wirkungsbereich über die Klasse, ohne und mit Vorwissen, nach Stichprobe.")
if experiment_button("▶ Experiment 2 rechnen (etwa 30 Sekunden)", "exp2"):
    res2 = _open()
    st.plotly_chart(build_open_chart(res2), width="stretch", key="open_chart")
    verdict(verdict_open(res2))
    st.dataframe(build_open_table(res2), hide_index=True, width="stretch")
    st.dataframe(build_range_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Wenn die Annahmen kippen")
st.caption("Drei leise Brüche: ein direkter Pfeil, der einen Umweg genau aufhebt (Treue), eine quadratische Kante (Linearität) und eine verborgene Nachfrage. In keinem gibt PC eine Warnung aus.")
if experiment_button("▶ Experiment 3 rechnen (etwa 30 Sekunden)", "exp3"):
    res3 = _assumptions()
    st.plotly_chart(build_cancel_chart(res3), width="stretch", key="cancel_chart")
    verdict(verdict_assumptions(res3))
    st.dataframe(build_cancel_table(res3), hide_index=True, width="stretch")
    st.dataframe(build_nonlinear_table(res3), hide_index=True, width="stretch")
    st.dataframe(build_hidden_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Das Verfahren selbst")
st.caption("Der Orakel-PC über alle 543 Diagramme mit vier Knoten gegen die Aufzählung der Klassen; dazu Zufalls-DAGs mit sechs Knoten und Stichprobe, und ob die Reihenfolge der Variablen die Ausgabe ändert (ursprüngliche gegen stabile Fassung).")
if experiment_button("▶ Experiment 4 rechnen (etwa 40 Sekunden)", "exp4"):
    res4 = _procedure()
    st.plotly_chart(build_procedure_chart(res4), width="stretch", key="procedure_chart")
    verdict(verdict_procedure(res4))
    st.dataframe(build_random_table(res4), hide_index=True, width="stretch")
    st.dataframe(build_oracle_table(res4), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Treue: Unabhängigkeiten stammen nur aus dem Diagramm** | Hebt sich ein Umweg zufällig auf oder ist eine Kante fast untreu, fehlt sie, und PC setzt einen Collider, den es nicht gibt (Experimente 1 und 3). | Sensitivität und Falsifikation |
| **Linear und Gauß (Fisher-z-Test)** | Eine nichtlineare Kante ohne lineare Teilkorrelation bleibt unsichtbar, auch bei vielen Daten (Experiment 3). | nicht gebaut |
| **Alles Wichtige ist gemessen** | Eine verborgene Ursache gibt einen sauber gerichteten Graphen mit einem Wirkungsbereich, der die Wahrheit nicht enthält (Experiment 3). | Wann geht es gar nicht? |
| **Die Klasse genügt** | Ungerichtete Kanten bleiben, die Wirkung ist ein Bereich; mehr Daten ändern das nicht, Vorwissen schon (Experiment 2). | Kontinuierliches Strukturlernen; Über die Äquivalenzklasse hinaus |
| **Die Reihenfolge der Variablen ist gleich** | Das Skelett der stabilen Fassung ist unabhängig, die Orientierung nicht (Experiment 4). | nicht gebaut |
| **Kleine Graphen** | Fünf bis zehn Knoten, ein Test je Aussage ohne Mehrfachtest-Korrektur, Konflikte der Orientierung „wer zuerst kommt“. | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Lineares Gauß-Modell $V_j = \sum_i B_{ji} V_i + \varepsilon_j$ mit $\varepsilon_j \sim N(0,1)$; das Diagramm hat die Pfeile $i \to j$ mit $B_{ji} \ne 0$. Die Wirkung von $X$ auf $Y$ ist $B_{YX}$.

**Test.** Teilkorrelation $\rho_{ij\cdot S} = -P_{ij}/\sqrt{P_{ii}P_{jj}}$ mit $P$ der Inversen der Korrelationsmatrix von $(i, j, S)$; Fisher-z: $\sqrt{n - |S| - 3}\,|\operatorname{artanh} r|$ ist unter der Unabhängigkeit etwa $N(0,1)$; unabhängig, wenn der p-Wert über $\alpha$ liegt.

**Skelett.** Für Stufen $\ell = 0, 1, \dots$: Die Kante $i - j$ entfällt, wenn eine Menge $S$ von $\ell$ Nachbarn mit $i \perp j \mid S$ gefunden wird (Trennmenge). **v-Struktur:** Für nicht benachbarte $i, j$ mit gemeinsamem Nachbarn $k$ wird $i \to k \leftarrow j$ gesetzt,
wenn $k$ nicht in der Trennmenge von $i, j$ liegt. **Meek-Regeln:** R1: $a \to b - c$, $a, c$ nicht benachbart $\Rightarrow b \to c$. R2: $a \to b \to c$ und $a - c \Rightarrow a \to c$. R3: $a - b$, $a - c$, $a - d$, $c \to b$, $d \to b$, $c, d$ nicht benachbart $\Rightarrow a \to b$.

**Äquivalenz.** Zwei Diagramme haben dieselben d-Separationen genau dann, wenn sie dasselbe Skelett und dieselben v-Strukturen haben (Verma/Pearl). Der CPDAG hat die Kante $i \to j$, wenn sie in jedem Diagramm der Klasse so steht, sonst $i - j$.

**Wirkungsbereich.** Für jedes Diagramm $D$ der Klasse: $\hat\beta_D$ = Koeffizient von $X$ in der Regression von $Y$ auf $X$ und die Eltern von $X$ in $D$. Der Bereich ist $[\min_D \hat\beta_D, \max_D \hat\beta_D]$.

**Treue-Spielraum.** Je wahrer Kante $\min_S |\rho_{ij\cdot S}|$ über alle Bedingungsmengen der übrigen Größen (Bevölkerung); nahe null kann ein Test die Kante nicht von einer Trennung unterscheiden.

**Literatur.** P. Spirtes, C. Glymour, R. Scheines, *Causation, Prediction, and Search*; C. Meek, *Causal inference and causal explanation with background knowledge*, UAI 1995; T. Verma, J. Pearl, *Equivalence and synthesis of causal models*, UAI 1990;
D. Colombo, M. Maathuis, *Order-independent constraint-based causal structure learning*, JMLR 2014; M. Maathuis, M. Kalisch, P. Bühlmann, *Estimating high-dimensional intervention effects from observational data*, Annals of Statistics 2009
(alle aus dem Gedächtnis, nicht geprüft; die Gegenprobe Orakel-PC gegen Aufzählung nutzt denselben Satz von Verma/Pearl).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
