"""Der PC-Algorithmus von Grund auf (numpy, math): Fisher-z-Test, Skelett (stabil oder ursprünglich), v-Strukturen, Meek-Regeln R1 bis R3, Vorwissen „Wurzeln“,
die Markov-Äquivalenzklasse durch Aufzählung und der Wirkungsbereich über die Klasse. Kein Streamlit. Knoten sind ganze Zahlen 0 bis d − 1; ein Graph ist ein Paar
(gerichtete Kanten (a, b) für a → b, ungerichtete Kanten (a, b) mit a < b)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import dsep_graph as G

MAX_UNDIRECTED_FOR_CLASS = 12          # Aufzählung der Orientierungen: höchstens 2^12


# ----------------------------------------------------------------------------- Tests

def partial_correlation(corr: np.ndarray, i: int, j: int, cond=()) -> float:
    """Teilkorrelation von i und j gegeben cond aus der Korrelationsmatrix (Präzisionsmatrix der Teilmatrix)."""
    idx = [i, j] + list(cond)
    sub = corr[np.ix_(idx, idx)]
    try:
        p = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        p = np.linalg.pinv(sub)
    r = -p[0, 1] / math.sqrt(p[0, 0] * p[1, 1])
    return float(max(min(r, 0.999999), -0.999999))


def fisher_z_pvalue(r: float, n: int, k: int) -> float:
    """Zweiseitiger p-Wert des Fisher-z-Tests für die Teilkorrelation r bei n Beobachtungen und k bedingenden Größen."""
    z = math.atanh(r)
    stat = math.sqrt(max(n - k - 3, 1)) * abs(z)
    return math.erfc(stat / math.sqrt(2))


def fisher_z_test(corr: np.ndarray, n: int, alpha: float):
    """Unabhängigkeitstest für PC: wahr, wenn der p-Wert über alpha liegt (nicht abgelehnt)."""

    def indep(i, j, cond) -> bool:
        return fisher_z_pvalue(partial_correlation(corr, i, j, cond), n, len(cond)) > alpha

    return indep


def oracle_test(parents: tuple, mapping=None):
    """Orakel: d-Separation im wahren Diagramm als Test (mapping: Knoten des Laufs → Knoten des Diagramms, sonst gleich)."""
    m = (lambda v: v) if mapping is None else (lambda v: mapping[v])
    return lambda i, j, cond: G.d_separated(parents, m(i), m(j), tuple(m(c) for c in cond))


# ----------------------------------------------------------------------------- PC

@dataclass
class PcResult:
    directed: frozenset
    undirected: frozenset
    conflicts: int = 0
    sep: dict = field(default_factory=dict)

    @property
    def graph(self) -> tuple:
        return self.directed, self.undirected


def pc(d: int, indep, stable: bool = True, order=None, roots=()) -> PcResult:
    """PC: Skelett nach wachsender Größe der Bedingungsmenge, v-Strukturen nach den Trennmengen, Vorwissen (Wurzeln orientieren ihre Kanten nach außen), Meek R1 bis R3.
    stable: die Nachbarschaften werden je Stufe eingefroren (das Skelett hängt dann nicht von der Reihenfolge ab). Konflikte der Orientierung: wer zuerst kommt, gezählt in `conflicts`."""
    nodes = list(range(d)) if order is None else list(order)
    pos = {v: k for k, v in enumerate(nodes)}
    adj = {i: set(range(d)) - {i} for i in range(d)}
    sep: dict = {}
    level = 0
    while True:
        snap = {i: set(adj[i]) for i in adj} if stable else adj
        any_test = False
        for i in nodes:
            for j in sorted(snap[i], key=pos.get):
                if j not in adj[i]:
                    continue
                cand = sorted(snap[i] - {j}, key=pos.get)
                if len(cand) < level:
                    continue
                any_test = True
                for cond in itertools.combinations(cand, level):
                    if indep(i, j, cond):
                        adj[i].discard(j)
                        adj[j].discard(i)
                        sep[(i, j)] = sep[(j, i)] = set(cond)
                        break
        if not any_test:
            break
        level += 1
    directed: set = set()
    undirected = {(i, j) for i in range(d) for j in adj[i] if i < j}
    conflicts = 0
    for k in nodes:
        for i, j in itertools.combinations(sorted(adj[k]), 2):
            if j in adj[i] or k in sep[(i, j)]:
                continue
            for a in (i, j):
                e = (min(a, k), max(a, k))
                if e in undirected:
                    undirected.discard(e)
                    directed.add((a, k))
                elif (a, k) not in directed:
                    conflicts += 1
    for r in roots:
        for e in [e for e in undirected if r in e]:
            undirected.discard(e)
            directed.add((r, e[0] if e[1] == r else e[1]))
    _meek(d, adj, directed, undirected)
    return PcResult(frozenset(directed), frozenset(undirected), conflicts, sep)


def _meek(d: int, adj: dict, directed: set, undirected: set) -> None:
    """Meek R1 (a → b – c, a und c nicht benachbart: b → c), R2 (a → b → c und a – c: a → c), R3 (a – b, a – c, a – d, c → b, d → b, c und d nicht benachbart: a → b), bis nichts mehr geschieht."""
    for _ in range(len(undirected) + 1):                    # jeder Durchgang mit Änderung richtet mindestens eine Kante, der letzte findet nichts mehr
        changed = False
        for (a, b) in list(directed):
            for c in range(d):
                e = (min(b, c), max(b, c))
                if c != a and e in undirected and c not in adj[a]:
                    undirected.discard(e)
                    directed.add((b, c))
                    changed = True
        for (a, b) in list(directed):
            for (b2, c) in list(directed):
                if b2 == b and c != a:
                    e = (min(a, c), max(a, c))
                    if e in undirected:
                        undirected.discard(e)
                        directed.add((a, c))
                        changed = True
        for e in sorted(undirected):
            for a, b in (e, e[::-1]):
                if e not in undirected:
                    break
                cs = [c for c in range(d) if c not in (a, b) and (min(a, c), max(a, c)) in undirected and (c, b) in directed]
                if any(c2 not in adj[c1] for c1, c2 in itertools.combinations(cs, 2)):
                    undirected.discard(e)
                    directed.add((a, b))
                    changed = True
        if not changed:
            break


# ----------------------------------------------------------------------------- Graphen vergleichen

def skeleton_of(g: tuple) -> frozenset:
    return frozenset(frozenset(e) for e in g[0]) | frozenset(frozenset(e) for e in g[1])


def pair_state(g: tuple, i: int, j: int) -> str:
    """Zustand des Paars (i, j) im Graphen: '-' keine Kante, 'u' ungerichtet, 'f' i → j, 'r' j → i."""
    if (min(i, j), max(i, j)) in g[1]:
        return "u"
    if (i, j) in g[0]:
        return "f"
    if (j, i) in g[0]:
        return "r"
    return "-"


def shd(g1: tuple, g2: tuple, d: int) -> int:
    """Strukturabstand: Zahl der Paare mit verschiedenem Zustand (fehlt, ungerichtet, Richtung)."""
    return sum(pair_state(g1, i, j) != pair_state(g2, i, j) for i in range(d) for j in range(i + 1, d))


def dag_graph(parents: tuple) -> tuple:
    """Das Diagramm selbst als Graph (alle Kanten gerichtet)."""
    return frozenset((p, c) for c, ps in enumerate(parents) for p in ps), frozenset()


# ----------------------------------------------------------------------------- Äquivalenzklasse durch Aufzählung

def vstructs(parents: tuple) -> frozenset:
    """Alle v-Strukturen (a, c, b), a < b: a → c ← b mit a und b nicht benachbart."""
    skel = {frozenset((c, p)) for c, ps in enumerate(parents) for p in ps}
    return frozenset((a, c, b) for c, ps in enumerate(parents) for a, b in itertools.combinations(sorted(ps), 2) if frozenset((a, b)) not in skel)


def class_members(parents: tuple) -> list:
    """Alle DAGs mit gleichem Skelett und gleichen v-Strukturen (Verma/Pearl) durch Aufzählung der Orientierungen aller Kanten; als Elterntupel."""
    n = len(parents)
    edges = sorted({tuple(sorted((c, p))) for c, ps in enumerate(parents) for p in ps})
    want = vstructs(parents)
    members = []
    for bits in range(1 << len(edges)):
        ps: list = [[] for _ in range(n)]
        for k, (a, b) in enumerate(edges):
            if bits >> k & 1:
                ps[b].append(a)
            else:
                ps[a].append(b)
        t = tuple(tuple(sorted(p)) for p in ps)
        if G._acyclic(ps) and vstructs(t) == want:
            members.append(t)
    return members


def cpdag_from_members(members: list) -> tuple:
    """CPDAG: eine Kante ist gerichtet, wenn sie in allen Diagrammen der Klasse gleich gerichtet ist, sonst ungerichtet."""
    n = len(members[0])
    edges = {tuple(sorted((c, p))) for c, ps in enumerate(members[0]) for p in ps}
    directed, undirected = set(), set()
    for a, b in edges:
        if all(a in t[b] for t in members):
            directed.add((a, b))
        elif all(b in t[a] for t in members):
            directed.add((b, a))
        else:
            undirected.add((a, b))
    return frozenset(directed), frozenset(undirected)


def cpdag_by_enumeration(parents: tuple) -> tuple:
    return cpdag_from_members(class_members(parents))


def true_cpdag(parents: tuple) -> tuple:
    """CPDAG des Diagramms über den Orakel-PC (d-Separation als Test); schnell auch für zehn Knoten."""
    return pc(len(parents), oracle_test(parents)).graph


# ----------------------------------------------------------------------------- Wirkung über die Klasse

def effect_adjusted(cov: np.ndarray, parents: tuple, x: int, y: int) -> float:
    """Koeffizient von x in der Regression y ~ x + Eltern(x) (ohne y), aus der Kovarianzmatrix: die Anpassung der Rückdoor-Formel für dieses Diagramm."""
    pa = [p for p in parents[x] if p != y]
    idx = [x] + pa
    return float(np.linalg.solve(cov[np.ix_(idx, idx)], cov[idx, y])[0])


def class_dags(g: tuple, d: int):
    """Alle azyklischen Orientierungen der ungerichteten Kanten des CPDAG, die genau die v-Strukturen der gerichteten Kanten behalten (die Diagramme der Klasse);
    None, wenn zu viele ungerichtete Kanten (mehr als MAX_UNDIRECTED_FOR_CLASS)."""
    directed, undirected = list(g[0]), sorted(g[1])
    if len(undirected) > MAX_UNDIRECTED_FOR_CLASS:
        return None
    skel = skeleton_of(g)
    want = frozenset((a, c, b) for (a, c) in directed for (b, c2) in directed if c2 == c and a < b and frozenset((a, b)) not in skel)
    base: list = [[] for _ in range(d)]
    for a, b in directed:
        base[b].append(a)
    out = []
    for bits in range(1 << len(undirected)):
        ps = [list(p) for p in base]
        for k, (a, b) in enumerate(undirected):
            if bits >> k & 1:
                ps[b].append(a)
            else:
                ps[a].append(b)
        t = tuple(tuple(sorted(p)) for p in ps)
        if G._acyclic(ps) and vstructs(t) == want:
            out.append(t)
    return out


def class_effects(g: tuple, cov: np.ndarray, x: int, y: int):
    """Mögliche Wirkungen von x auf y: die Anpassung an die Eltern von x in jedem Diagramm der Klasse (die Idee von Maathuis/Kalisch/Bühlmann, vereinfacht).
    Leere Liste, wenn kein Diagramm zur Ausgabe passt; None, wenn die Klasse zu groß ist."""
    dags = class_dags(g, cov.shape[0])
    if dags is None:
        return None
    return [effect_adjusted(cov, t, x, y) for t in dags]
