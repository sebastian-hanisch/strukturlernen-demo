"""Die Welt: fünf Größen eines Standortnetzes in einem linearen Gauß-Modell mit bekanntem Diagramm, Daten aus festen Strömen, wahre Kanten, Wirkung und der Treue-Spielraum.
S Saison → Z Nachfrageschwankung → X hoher Faktor → Y Servicegrad, Z → Y, W Wetter → Y. Kein Streamlit."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

import stl_constants as C
from kausal_generator import SplitMix64

S, Z, X, Y, W = range(5)
NAMES = ("S", "Z", "X", "Y", "W")
LONG = ("Saison", "Nachfrage", "Faktor", "Servicegrad", "Wetter")


@dataclass(frozen=True)
class World:
    a_sz: float = C.DEFAULT_A_SZ    # Saison → Nachfrage
    a_zx: float = C.DEFAULT_A_ZX    # Nachfrage → Faktor (der Disponent wählt nach der Nachfrage)
    b_xy: float = C.DEFAULT_B_XY    # Faktor → Servicegrad: die gesuchte Wirkung
    b_zy: float = C.DEFAULT_B_ZY    # Nachfrage → Servicegrad
    b_wy: float = C.DEFAULT_B_WY    # Wetter → Servicegrad
    quadratic_zy: bool = False       # Z → Y wirkt quadratisch (symmetrisch: lineare Korrelation 0)

    @property
    def matrix(self) -> np.ndarray:
        """B[j, i] = Koeffizient von i in j."""
        b = np.zeros((5, 5))
        b[Z, S], b[X, Z], b[Y, X], b[Y, Z], b[Y, W] = self.a_sz, self.a_zx, self.b_xy, self.b_zy, self.b_wy
        return b

    @property
    def parents(self) -> tuple:
        return parents_of_matrix(self.matrix)

    @property
    def effect(self) -> float:
        return self.b_xy

    @property
    def var_z(self) -> float:
        return self.a_sz ** 2 + 1.0


def parents_of_matrix(b: np.ndarray) -> tuple:
    return tuple(tuple(int(i) for i in np.nonzero(b[j])[0]) for j in range(b.shape[0]))


def simulate(world: World, n: int, rng: SplitMix64) -> np.ndarray:
    """n × 5 Matrix (S, Z, X, Y, W), Einheitsrauschen; die Spalten stehen im festen Strom e = rng.normal(5 n)."""
    e = rng.normal(5 * n).reshape(n, 5)
    s = e[:, S]
    z = world.a_sz * s + e[:, Z]
    x = world.a_zx * z + e[:, X]
    w = e[:, W]
    zy = (z * z - world.var_z) / world.var_z if world.quadratic_zy else z
    y = world.b_xy * x + world.b_zy * zy + world.b_wy * w + e[:, Y]
    return np.column_stack([s, z, x, y, w])


def simulate_linear(b: np.ndarray, n: int, rng: SplitMix64) -> np.ndarray:
    """Linear-Gauß für eine beliebige Koeffizientenmatrix: x = B x + e, also x = (I − B)⁻¹ e."""
    d = b.shape[0]
    e = rng.normal(n * d).reshape(n, d)
    return e @ np.linalg.inv(np.eye(d) - b).T


def population_cov(b: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.eye(b.shape[0]) - b)
    return inv @ inv.T


def partial_corr_population(cov: np.ndarray, i: int, j: int, cond=()) -> float:
    idx = [i, j] + list(cond)
    p = np.linalg.inv(cov[np.ix_(idx, idx)])
    return float(-p[0, 1] / np.sqrt(p[0, 0] * p[1, 1]))


def faithfulness_margins(b: np.ndarray) -> dict:
    """Je wahrer Kante (i → j) die kleinste Teilkorrelation über alle Bedingungsmengen aus den übrigen Größen (Bevölkerung): der Treue-Spielraum der Kante.
    Nahe null kann ein Test die Kante bei endlicher Stichprobe nicht von einer Trennung unterscheiden."""
    d = b.shape[0]
    cov = population_cov(b)
    out = {}
    for j in range(d):
        for i in range(d):
            if b[j, i] != 0:
                rest = [k for k in range(d) if k not in (i, j)]
                out[(i, j)] = min(abs(partial_corr_population(cov, i, j, cond)) for r in range(len(rest) + 1) for cond in itertools.combinations(rest, r))
    return out


def faithfulness_margin(b: np.ndarray) -> float:
    m = faithfulness_margins(b)
    return min(m.values()) if m else 1.0


def random_dag(d: int, p: float, rng: SplitMix64) -> np.ndarray:
    """Zufalls-DAG in der Knotenfolge 0..d−1: jedes Paar i < j wird mit Wahrscheinlichkeit p zur Kante i → j, Koeffizient ±U(0,5; 1,5)."""
    u = rng.uniform(3 * d * d).reshape(3, d, d)
    b = np.zeros((d, d))
    for j in range(d):
        for i in range(j):
            if u[0, j, i] < p:
                b[j, i] = (0.5 + u[1, j, i]) * (1.0 if u[2, j, i] < 0.5 else -1.0)
    return b
