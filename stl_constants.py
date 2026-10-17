"""Konstanten: Regler-Grenzen, feste Experimente, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35

# Welt: Pfeilstärken (lineares Gauß-Modell, Einheitsrauschen). S → Z → X → Y, Z → Y, W → Y.
COEF_MIN, COEF_MAX, COEF_STEP = -1.5, 1.5, 0.02
DEFAULT_A_SZ, DEFAULT_A_ZX, DEFAULT_B_XY, DEFAULT_B_ZY, DEFAULT_B_WY = 0.8, 0.8, 0.6, 0.6, 0.6
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 10000, 100, 1000
ALPHAS = (0.001, 0.01, 0.05, 0.1)
DEFAULT_ALPHA = 0.01
VERSIONS = ("stabil", "ursprünglich")
DEFAULT_VERSION = VERSIONS[0]
KNOWLEDGE = ("keines", "Wurzeln S und W")
DEFAULT_KNOWLEDGE = KNOWLEDGE[0]
YES_NO = ("ja", "nein")
DEFAULT_OBSERVED = YES_NO[0]
FORMS = ("linear", "quadratisch")
DEFAULT_FORM = FORMS[0]
TOLERANCE = 0.15            # Spielraum für den Stichprobenfehler beim Prüfen, ob die Wahrheit im Wirkungsbereich liegt

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1030000
EXP_REPS = 300
EXP_NS, EXP_ALPHAS = (100, 300, 1000, 3000, 10000), (0.01, 0.05)                 # Experiment 1
EXP_UNFAITHFUL = dict(a_zx=0.7, b_xy=1.0, b_zy=0.8)                              # die erste, fast untreue Welt (Preset „Fast untreu“)
EXP_UNFAITHFUL_NS = (300, 1000, 3000)
EXP_OPEN_DIMS, EXP_OPEN_DEGREES, EXP_OPEN_DAGS = (6, 8, 10), (1.0, 2.0, 3.0), 200   # Experiment 2: Zufalls-DAGs (Orakel-CPDAG)
EXP_EFFECT_NS = (300, 1000, 3000)
EXP_LAMBDAS, EXP_CANCEL_NS = (0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5), (500, 5000)   # Experiment 3: Aufhebung
EXP_NONLINEAR_NS, EXP_HIDDEN_NS = (300, 1000, 5000, 20000), (1000, 5000)
EXP_RANDOM_NS, EXP_RANDOM_DAGS = (100, 1000, 10000), 200                          # Experiment 4: Zufalls-DAGs mit Stichprobe und Reihenfolge
EXP_ORDERINGS = 6

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(a_sz=DEFAULT_A_SZ, a_zx=DEFAULT_A_ZX, b_xy=DEFAULT_B_XY, b_zy=DEFAULT_B_ZY, b_wy=DEFAULT_B_WY, n=DEFAULT_N, alpha=DEFAULT_ALPHA, version=DEFAULT_VERSION, knowledge=DEFAULT_KNOWLEDGE,
                observed=DEFAULT_OBSERVED, form=DEFAULT_FORM, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Leichte Welt (Voreinstellung)": preset(),
    "Wenige Daten": preset(n=100, seed=36),
    "Fast untreu": preset(a_zx=0.7, b_xy=1.0, b_zy=0.8, alpha=0.05),
    "Mit Vorwissen": preset(knowledge=KNOWLEDGE[1]),
    "Nachfrage verborgen": preset(observed=YES_NO[1]),
    "Aufgehobener Umweg": preset(b_zy=-0.48),
    "Nichtlineare Kante": preset(form=FORMS[1]),
}
PRESET_HELP = {
    "Leichte Welt (Voreinstellung)": "Fünf Größen, kräftige Pfeile, 1 000 Standorte: PC findet die Klasse; zwei Kanten (S–Z und Z–X) bleiben ungerichtet, die Wirkung ist 0,60 oder 0,98.",
    "Wenige Daten": "Nur 100 Standorte: Mit α = 0,01 trifft PC die Klasse nur in etwa der Hälfte der Läufe. In dieser Stichprobe fällt die Kante Wetter–Servicegrad aus dem Skelett.",
    "Fast untreu": "Kräftigere Pfeile, aber die Kante Z–X hebt sich beim Bedingen auf den Servicegrad fast auf (Treue-Spielraum 0,05): Mit α = 0,05 verwirft PC sie oft.",
    "Mit Vorwissen": "Saison und Wetter sind Ursachen, nichts wirkt auf sie: Das richtet S → Z und, über die Meek-Regeln, Z → X. Aus dem Bereich wird eine Zahl.",
    "Nachfrage verborgen": "Die Nachfrage ist nicht gemessen: PC gibt einen Graphen aus (X → Y gerichtet), dessen Wirkungsbereich die Wahrheit nicht enthält. Die Ausgabe selbst warnt nicht; rot ist nur zu sehen, weil die Wahrheit hier bekannt ist.",
    "Aufgehobener Umweg": "Der direkte Pfeil Z → Y hebt den Umweg über X auf: Z und Y sind unkorreliert, PC entfernt die Kante und setzt Z → X ← Y, ein falscher Collider.",
    "Nichtlineare Kante": "Z wirkt quadratisch auf Y (symmetrisch, lineare Korrelation 0): Der Fisher-z-Test sieht die Kante nicht, auch bei vielen Daten nicht.",
}
