"""Auswertung: eine Analyse der Welt mit PC, die vier Experimente über feste Ströme, der Baukasten für Äquivalenzklassen und die Verdikte. Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass

import numpy as np

import dsep_graph as G
import stl_constants as C
import stl_pc as P
import stl_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


@dataclass(frozen=True)
class Settings:
    a_sz: float = C.DEFAULT_A_SZ
    a_zx: float = C.DEFAULT_A_ZX
    b_xy: float = C.DEFAULT_B_XY
    b_zy: float = C.DEFAULT_B_ZY
    b_wy: float = C.DEFAULT_B_WY
    n: int = C.DEFAULT_N
    alpha: float = C.DEFAULT_ALPHA
    version: str = C.DEFAULT_VERSION
    knowledge: str = C.DEFAULT_KNOWLEDGE
    observed: str = C.DEFAULT_OBSERVED
    form: str = C.DEFAULT_FORM
    seed: int = C.DEFAULT_SEED

    @property
    def world(self) -> W.World:
        return W.World(self.a_sz, self.a_zx, self.b_xy, self.b_zy, self.b_wy, self.form == C.FORMS[1])

    @property
    def stable(self) -> bool:
        return self.version == C.VERSIONS[0]

    @property
    def obs(self) -> tuple:
        return tuple(range(5)) if self.observed == C.YES_NO[0] else tuple(v for v in range(5) if v != W.Z)

    @property
    def roots(self) -> tuple:
        """Wurzeln des Vorwissens in Knoten der Welt (nur beobachtete)."""
        return tuple(v for v in (W.S, W.W) if v in self.obs) if self.knowledge == C.KNOWLEDGE[1] else ()


# ----------------------------------------------------------------------------- Abbildung zwischen Lauf- und Weltknoten

def _to_world(g: tuple, obs: tuple) -> tuple:
    return frozenset((obs[a], obs[b]) for a, b in g[0]), frozenset(tuple(sorted((obs[a], obs[b]))) for a, b in g[1])


def run_pc(world: W.World, data: np.ndarray, obs: tuple, alpha: float, stable: bool, roots: tuple, order=None) -> tuple:
    """PC auf den beobachteten Spalten; Rückgabe (Graph in Weltknoten, Konflikte)."""
    sub = data[:, list(obs)]
    corr = np.corrcoef(sub, rowvar=False)
    local_roots = tuple(obs.index(r) for r in roots)
    res = P.pc(len(obs), P.fisher_z_test(corr, len(sub), alpha), stable=stable, order=order, roots=local_roots)
    return _to_world(res.graph, obs), res.conflicts


def oracle_graph(parents: tuple, obs: tuple, roots: tuple = ()) -> tuple:
    """Was PC mit unendlich vielen Daten ausgäbe: der Orakel-PC (d-Separation im wahren Diagramm) auf den beobachteten Größen."""
    res = P.pc(len(obs), P.oracle_test(parents, mapping=obs), roots=tuple(obs.index(r) for r in roots))
    return _to_world(res.graph, obs)


# ----------------------------------------------------------------------------- Analyse

@dataclass
class Analysis:
    settings: Settings
    parents: tuple          # das wahre Diagramm
    dag_edges: frozenset    # wahre Kanten zwischen beobachteten Größen (a, b): a → b
    learned: tuple          # (gerichtet, ungerichtet) in Weltknoten
    oracle: tuple           # Ergebnis des Orakel-PC
    exact: bool             # gelernt = Orakel-Ergebnis
    shd_oracle: int
    conflicts: int
    edge_status: list       # (a, b, Art, Status): Art 'gerichtet' | 'ungerichtet' | 'fehlt', Status 'richtig' | 'falsch gerichtet' | 'überzählig' | 'offen' | 'fehlt'
    missing: list
    extra: list
    margins: dict           # Treue-Spielraum je wahrer Kante
    margin: float
    truth: float            # wahre Wirkung X → Y
    effects: list | None    # mögliche Wirkungen über die Diagramme der Klasse (None: Klasse zu groß), leer: keins passt
    class_size: int | None
    dags: list | None       # die Diagramme der Klasse als Elterntupel über den beobachteten Spalten (Laufknoten)
    obs: tuple              # beobachtete Weltknoten in Spaltenfolge
    contains: bool | None   # liegt die Wahrheit im Bereich (±TOLERANCE)


def edge_status(learned: tuple, dag_edges: frozenset) -> list:
    out = []
    skel = {frozenset(e) for e in dag_edges}
    for a, b in sorted(learned[0]):
        out.append((a, b, "gerichtet", "richtig" if (a, b) in dag_edges else "falsch gerichtet" if (b, a) in dag_edges else "überzählig"))
    for a, b in sorted(learned[1]):
        out.append((a, b, "ungerichtet", "offen" if frozenset((a, b)) in skel else "überzählig"))
    learned_skel = P.skeleton_of(learned)
    for a, b in sorted(dag_edges):
        if frozenset((a, b)) not in learned_skel:
            out.append((a, b, "fehlt", "fehlt"))
    return out


def analyse(s: Settings) -> Analysis:
    world = s.world
    parents = world.parents
    obs = s.obs
    data = W.simulate(world, s.n, SplitMix64(s.seed))
    learned, conflicts = run_pc(world, data, obs, s.alpha, s.stable, s.roots)
    oracle = oracle_graph(parents, obs, s.roots)
    dag_edges = frozenset((p, c) for c, ps in enumerate(parents) for p in ps if c in obs and p in obs)
    status = edge_status(learned, dag_edges)
    # Wirkungsbereich über die Klasse der gelernten Ausgabe, aus der Stichprobenkovarianz der beobachteten Spalten
    local = tuple((obs.index(a), obs.index(b)) for a, b in learned[0]), tuple((obs.index(a), obs.index(b)) for a, b in learned[1])
    local_graph = (frozenset(local[0]), frozenset((min(a, b), max(a, b)) for a, b in local[1]))
    cov = np.cov(data[:, list(obs)], rowvar=False)
    dags = P.class_dags(local_graph, len(obs))
    effects = None if dags is None else [P.effect_adjusted(cov, t, obs.index(W.X), obs.index(W.Y)) for t in dags]
    contains = None if not effects else bool(min(effects) - C.TOLERANCE <= world.effect <= max(effects) + C.TOLERANCE)
    margins = W.faithfulness_margins(world.matrix)
    return Analysis(s, parents, dag_edges, learned, oracle, learned == oracle, P.shd(learned, oracle, 5), conflicts, status, [(a, b) for a, b, k, _ in status if k == "fehlt"],
                    [(a, b) for a, b, k, st in status if st == "überzählig"], margins, min(margins.values()) if margins else 1.0, world.effect, effects, None if dags is None else len(dags), dags, obs, contains)


# ----------------------------------------------------------------------------- Experiment 1: Stichprobe × α

def _recovery(world: W.World, n: int, alpha: float, reps: int, seed: int, truth: tuple, zx_edge=None) -> dict:
    exact = skel = xy = zx_missing = 0
    shd_sum = miss = extra = 0
    t_skel = P.skeleton_of(truth)
    for r in range(reps):
        data = W.simulate(world, n, SplitMix64(seed + r))
        g, _ = run_pc(world, data, tuple(range(5)), alpha, True, ())
        exact += g == truth
        skel += P.skeleton_of(g) == t_skel
        xy += (W.X, W.Y) in g[0]
        shd_sum += P.shd(g, truth, 5)
        miss += len(t_skel - P.skeleton_of(g))
        extra += len(P.skeleton_of(g) - t_skel)
        zx_missing += frozenset((W.Z, W.X)) not in P.skeleton_of(g)
    return {"exact": exact / reps, "skeleton": skel / reps, "xy": xy / reps, "shd": shd_sum / reps, "missing": miss / reps, "extra": extra / reps, "zx_missing": zx_missing / reps}


def sample_experiment(ns=C.EXP_NS, alphas=C.EXP_ALPHAS, reps: int = C.EXP_REPS) -> dict:
    """Welt der Voreinstellung: CPDAG-Treffer, Skelett, Strukturabstand und fehlende/überzählige Kanten nach Stichprobe und α; dazu die fast untreue Welt (α = 0,05)."""
    base = W.World()
    truth = P.true_cpdag(base.parents)
    rows = []
    for i, alpha in enumerate(alphas):
        for j, n in enumerate(ns):
            rows.append({"alpha": alpha, "n": n, **_recovery(base, n, alpha, reps, C.SEED_BASE + 100000 * i + 10000 * j, truth)})
    unf = W.World(**C.EXP_UNFAITHFUL)
    unf_truth = P.true_cpdag(unf.parents)
    unfaithful = [{"n": n, **_recovery(unf, n, 0.05, reps, C.SEED_BASE + 500000 + 10000 * j, unf_truth)} for j, n in enumerate(C.EXP_UNFAITHFUL_NS)]
    return {"reps": reps, "rows": rows, "unfaithful": unfaithful, "margin": W.faithfulness_margin(base.matrix), "margin_unfaithful": W.faithfulness_margin(unf.matrix),
            "margins_unfaithful": W.faithfulness_margins(unf.matrix)}


# ----------------------------------------------------------------------------- Experiment 2: Was offen bleibt

def open_experiment(dims=C.EXP_OPEN_DIMS, degrees=C.EXP_OPEN_DEGREES, dags: int = C.EXP_OPEN_DAGS, ns=C.EXP_EFFECT_NS, reps: int = C.EXP_REPS) -> dict:
    """Zufalls-DAGs: Anteil ungerichteter Kanten im CPDAG (Orakel); die Welt: Wirkungsbereich über die Klasse ohne und mit Vorwissen (Wurzeln S und W) nach Stichprobengröße."""
    random_rows = []
    for i, d in enumerate(dims):
        for j, deg in enumerate(degrees):
            rng = SplitMix64(C.SEED_BASE + 1000000 + 100 * i + j)
            edges = undirected = full = with_edges = 0
            for _ in range(dags):
                g = P.true_cpdag(W.parents_of_matrix(W.random_dag(d, deg / (d - 1), rng)))
                e = len(g[0]) + len(g[1])
                edges += e
                undirected += len(g[1])
                with_edges += e > 0
                full += e > 0 and not g[1]
            random_rows.append({"d": d, "degree": deg, "edges": edges / dags, "undirected": undirected / max(edges, 1), "full": full / max(with_edges, 1), "with_edges": with_edges})
    world = W.World()
    parents = world.parents
    cov_pop = W.population_cov(world.matrix)
    members = P.class_members(parents)
    pop_effects = sorted({round(P.effect_adjusted(cov_pop, m, W.X, W.Y), 6) for m in members})
    knowledge = P.pc(5, P.oracle_test(parents), roots=(W.S, W.W)).graph
    know_dags = P.class_dags(knowledge, 5)
    know_effects = sorted({round(P.effect_adjusted(cov_pop, m, W.X, W.Y), 6) for m in know_dags})
    truth = P.true_cpdag(parents)
    rows = []
    for j, n in enumerate(ns):
        for mode, roots, target in (("ohne", (), truth), ("mit Vorwissen", (W.S, W.W), knowledge)):
            right = inside = 0
            lo_sum = hi_sum = width_sum = 0.0
            for r in range(reps):
                data = W.simulate(world, n, SplitMix64(C.SEED_BASE + 2000000 + 10000 * j + r))
                g, _ = run_pc(world, data, tuple(range(5)), 0.01, True, roots)
                if g != target:
                    continue
                right += 1
                es = P.class_effects(g, np.cov(data, rowvar=False), W.X, W.Y)
                lo, hi = min(es), max(es)
                lo_sum += lo
                hi_sum += hi
                width_sum += hi - lo
                inside += lo - C.TOLERANCE <= world.effect <= hi + C.TOLERANCE
            rows.append({"n": n, "mode": mode, "right": right / reps, "lo": lo_sum / max(right, 1), "hi": hi_sum / max(right, 1), "width": width_sum / max(right, 1), "inside": inside / max(right, 1)})
    return {"reps": reps, "random": random_rows, "class_size": len(members), "pop_effects": pop_effects, "know_effects": know_effects, "know_graph": knowledge, "truth_graph": truth, "rows": rows,
            "truth": world.effect}


# ----------------------------------------------------------------------------- Experiment 3: Wenn die Annahmen kippen

def cancellation_matrix(lam: float) -> np.ndarray:
    """X → M → Y (beide 1) mit direktem Pfeil X → Y = −λ · 1 · 1: bei λ = 1 heben sich Umweg und direkter Pfeil auf."""
    b = np.zeros((3, 3))
    b[1, 0], b[2, 1], b[2, 0] = 1.0, 1.0, -lam
    return b


def assumption_experiment(lambdas=C.EXP_LAMBDAS, cancel_ns=C.EXP_CANCEL_NS, nonlinear_ns=C.EXP_NONLINEAR_NS, hidden_ns=C.EXP_HIDDEN_NS, reps: int = C.EXP_REPS) -> dict:
    """Treue (Aufhebung), Linearität (quadratische Kante), Beobachtetes (verborgene Nachfrage): was PC ausgibt."""
    cancel = []
    for j, n in enumerate(cancel_ns):
        for i, lam in enumerate(lambdas):
            b = cancellation_matrix(lam)
            truth = P.true_cpdag(W.parents_of_matrix(b))
            missing = collider = right = 0
            for r in range(reps):
                data = W.simulate_linear(b, n, SplitMix64(C.SEED_BASE + 3000000 + 100000 * j + 1000 * i + r))
                corr = np.corrcoef(data, rowvar=False)
                g = P.pc(3, P.fisher_z_test(corr, n, 0.05)).graph
                missing += frozenset((0, 2)) not in P.skeleton_of(g)
                collider += (0, 1) in g[0] and (2, 1) in g[0]
                right += g == truth
            cancel.append({"n": n, "lam": lam, "missing": missing / reps, "collider": collider / reps, "right": right / reps})
    nonlinear = []
    for kind in (False, True):
        world = W.World(quadratic_zy=kind)
        truth = P.true_cpdag(world.parents)
        for j, n in enumerate(nonlinear_ns):
            missing = exact = 0
            for r in range(reps):
                data = W.simulate(world, n, SplitMix64(C.SEED_BASE + 4000000 + 100000 * int(kind) + 1000 * j + r))
                g, _ = run_pc(world, data, tuple(range(5)), 0.01, True, ())
                missing += frozenset((W.Z, W.Y)) not in P.skeleton_of(g)
                exact += g == truth
            nonlinear.append({"quadratic": kind, "n": n, "missing": missing / reps, "exact": exact / reps})
    hidden = []
    world = W.World()
    obs = (W.S, W.X, W.Y, W.W)
    for j, n in enumerate(hidden_ns):
        outputs: dict = {}
        xy = inside = 0
        lo_sum = hi_sum = 0.0
        for r in range(reps):
            data = W.simulate(world, n, SplitMix64(C.SEED_BASE + 5000000 + 10000 * j + r))
            g, _ = run_pc(world, data, obs, 0.01, True, ())
            key = tuple(sorted([f"{W.NAMES[a]}→{W.NAMES[b]}" for a, b in g[0]] + [f"{W.NAMES[a]}–{W.NAMES[b]}" for a, b in g[1]]))
            outputs[key] = outputs.get(key, 0) + 1
            xy += (W.X, W.Y) in g[0]
            local = frozenset((obs.index(a), obs.index(b)) for a, b in g[0]), frozenset((min(obs.index(a), obs.index(b)), max(obs.index(a), obs.index(b))) for a, b in g[1])
            es = P.class_effects(local, np.cov(data[:, list(obs)], rowvar=False), obs.index(W.X), obs.index(W.Y))
            if es:
                lo_sum += min(es)
                hi_sum += max(es)
                inside += min(es) - C.TOLERANCE <= world.effect <= max(es) + C.TOLERANCE
        main, count = max(outputs.items(), key=lambda kv: kv[1])
        hidden.append({"n": n, "main": main, "main_share": count / reps, "xy": xy / reps, "lo": lo_sum / reps, "hi": hi_sum / reps, "inside": inside / reps})
    return {"reps": reps, "cancel": cancel, "nonlinear": nonlinear, "hidden": hidden, "truth": world.effect}


# ----------------------------------------------------------------------------- Experiment 4: Das Verfahren selbst

def oracle_experiment() -> dict:
    """Orakel-PC über alle 543 DAGs mit vier Knoten gegen die Aufzählung der Klasse und gegen die Klassen aus den Aussagenmengen (d-Separationen)."""
    classes = G.equivalence_classes(4)
    dags = sum(len(v) for v in classes.values())
    enum_equal = pc_equal = 0
    for dag_list in classes.values():
        for t in dag_list:
            enum_equal += set(P.class_members(t)) == set(dag_list)
            pc_equal += P.true_cpdag(t) == P.cpdag_by_enumeration(t)
    sizes: dict = {}
    for v in classes.values():
        sizes[len(v)] = sizes.get(len(v), 0) + 1
    return {"dags": dags, "classes": len(classes), "enum_equal": enum_equal, "pc_equal": pc_equal, "sizes": dict(sorted(sizes.items())), "alone": sum(1 for v in classes.values() if len(v) == 1)}


def random_experiment(ns=C.EXP_RANDOM_NS, dags: int = C.EXP_RANDOM_DAGS, orderings: int = C.EXP_ORDERINGS) -> dict:
    """Zufalls-DAGs mit 6 Knoten (Grad 2) und Stichprobe: CPDAG-Treffer; Reihenfolge-Abhängigkeit des Skeletts und der Ausgabe in der ursprünglichen und der stabilen Fassung."""
    rows = []
    for j, n in enumerate(ns):
        rng = SplitMix64(C.SEED_BASE + 6000000 + j)
        exact = 0
        shd_sum = 0
        skel = {"ursprünglich": 0, "stabil": 0}
        out = {"ursprünglich": 0, "stabil": 0}
        for r in range(dags):
            b = W.random_dag(6, 2 / 5, rng)
            truth = P.true_cpdag(W.parents_of_matrix(b))
            data = W.simulate_linear(b, n, SplitMix64(C.SEED_BASE + 7000000 + 10000 * j + r))
            corr = np.corrcoef(data, rowvar=False)
            test = P.fisher_z_test(corr, n, 0.05)
            g = P.pc(6, test).graph
            exact += g == truth
            shd_sum += P.shd(g, truth, 6)
            orders = SplitMix64(C.SEED_BASE + 8000000 + 10000 * j + r).uniform(orderings * 6).reshape(orderings, 6).argsort(axis=1)
            for name, stable in (("ursprünglich", False), ("stabil", True)):
                results = [P.pc(6, test, stable=stable, order=[int(v) for v in o]).graph for o in orders]
                skel[name] += len({P.skeleton_of(g2) for g2 in results}) > 1
                out[name] += len(set(results)) > 1
        rows.append({"n": n, "exact": exact / dags, "shd": shd_sum / dags, "skel_orig": skel["ursprünglich"] / dags, "skel_stab": skel["stabil"] / dags, "out_orig": out["ursprünglich"] / dags, "out_stab": out["stabil"] / dags})
    return {"dags": dags, "orderings": orderings, "rows": rows}


def procedure_experiment() -> dict:
    return {"oracle": oracle_experiment(), "random": random_experiment()}


# ----------------------------------------------------------------------------- Baukasten für Äquivalenzklassen (vier Knoten)

BUILDER_NAMES = ("A", "B", "C", "D")
BUILDER_PAIRS = tuple(itertools.combinations(range(4), 2))
BUILDER_STATES = ("keine Kante", "→", "←")


@functools.lru_cache(maxsize=1)
def _classes4() -> dict:
    return G.equivalence_classes(4)


def builder_parents(states: dict) -> tuple:
    """states: {(i, j): 0 | 1 | 2} für i < j; 1 heißt i → j, 2 heißt j → i."""
    ps: list = [[] for _ in range(4)]
    for (i, j), s in states.items():
        if s == 1:
            ps[j].append(i)
        elif s == 2:
            ps[i].append(j)
    return tuple(tuple(sorted(p)) for p in ps)


def builder_result(parents: tuple) -> dict:
    if not G._acyclic([list(p) for p in parents]):
        return {"acyclic": False}
    members = P.class_members(parents)
    cpdag = P.cpdag_from_members(members)
    stmts = G.statement_set(parents)
    total = _classes4()
    return {"acyclic": True, "members": members, "cpdag": cpdag, "size": len(members), "same_statements": len(total[stmts]), "classes": len(total), "dags": sum(len(v) for v in total.values()),
            "vstructs": sorted(P.vstructs(parents)), "statements": sorted(stmts), "edges": len(cpdag[0]) + len(cpdag[1])}


# ----------------------------------------------------------------------------- Verdikte

def _name(a: int, b: int, arrow: str = "–") -> str:
    return f"{W.NAMES[a]}{arrow}{W.NAMES[b]}"


def _false_colliders(learned: tuple, oracle: tuple) -> list:
    """v-Strukturen a → c ← b (a, b nicht benachbart) im gelernten Graphen, die der Orakel-Graph nicht hat."""

    def vs(g):
        skel = P.skeleton_of(g)
        return {(a, c, b) for (a, c) in g[0] for (b, c2) in g[0] if c2 == c and a < b and frozenset((a, b)) not in skel}

    return sorted(vs(learned) - vs(oracle))


def effect_range_text(a: Analysis) -> str:
    if a.effects is None:
        return "Die Klasse ist zu groß für die Aufzählung (mehr als 12 ungerichtete Kanten)"
    if not a.effects:
        return "Kein Diagramm der Klasse passt zur Ausgabe (Orientierungskonflikt)"
    lo, hi = min(a.effects), max(a.effects)
    if hi - lo < 0.05:
        return f"Die Wirkung des Faktors ist eindeutig: {de(lo, 2)}"
    return f"Die Wirkung des Faktors ist ein Bereich: {de(lo, 2)} bis {de(hi, 2)} ({a.class_size} Diagramme passen)"


def verdict_instance(a: Analysis) -> tuple:
    s = a.settings
    truth = de(a.truth, 2)
    rng = effect_range_text(a)
    directed, undirected = len(a.learned[0]), len(a.learned[1])
    if s.observed != C.YES_NO[0]:
        extra = ", ".join(_name(x, y) for x, y in a.extra) or "keine"
        tail = f"Die Wahrheit {truth} liegt nicht darin; nichts in der Ausgabe warnt davor." if a.contains is False else f"Die Wahrheit {truth} liegt darin, aber nur zufällig."
        return "warning", f"Die Nachfrage Z ist nicht gemessen. PC gibt {directed + undirected} Kanten aus, überzählig davon: {extra}. {rng}. {tail}"
    if not a.exact:
        parts = []
        if a.missing:
            parts.append("fehlende Kanten: " + ", ".join(_name(x, y) for x, y in a.missing))
        if a.extra:
            parts.append("überzählige Kanten: " + ", ".join(_name(x, y) for x, y in a.extra))
        wrong = [_name(x, y, "→") for x, y, k, st in a.edge_status if st == "falsch gerichtet"]
        if wrong:
            parts.append("falsch gerichtet: " + ", ".join(wrong))
        fc = _false_colliders(a.learned, a.oracle)
        if fc:
            parts.append("ein Collider, den es nicht gibt (" + ", ".join(f"{W.NAMES[x]} → {W.NAMES[c]} ← {W.NAMES[y]}" for x, c, y in fc) + ")")
        text = f"PC findet nicht die Klasse, die mit unendlich vielen Daten herauskäme (Strukturabstand {a.shd_oracle}): " + "; ".join(parts or ["andere Orientierung"]) + "."
        weak = [e for e in a.missing if a.margins.get(e, 1.0) < 0.1]
        if s.form == C.FORMS[1] and (W.Z, W.Y) in a.missing:
            text += " Die Kante Z–Y wirkt quadratisch: Der Fisher-z-Test sieht nur lineare Zusammenhänge, mehr Daten ändern daran nichts."
        elif weak:
            text += f" Der Treue-Spielraum der wahren Kante {_name(*weak[0])} ist nur {de(a.margins[weak[0]], 3)}: Bei einer passenden Bedingungsmenge bleibt fast keine Teilkorrelation übrig."
        elif a.missing and s.n < 300:
            text += " Mit so wenigen Standorten fallen schwache Kanten aus dem Skelett."
        return "warning", text
    text = f"PC findet die Klasse: {directed} Kanten gerichtet, {undirected} ungerichtet. {rng}; die Wahrheit ist {truth}."
    if a.effects and max(a.effects) - min(a.effects) >= 0.05:
        text += " Mehr Daten verengen den Bereich nicht, Vorwissen schon."
    if s.alpha >= 0.05:
        text += f" (Mit α = {de(s.alpha, 2)} gibt es auch bei vielen Daten gelegentlich eine überzählige Kante.)"
    return "success", text


def _n(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def verdict_sample(res: dict) -> tuple:
    rows = res["rows"]
    a1 = {r["n"]: r for r in rows if r["alpha"] == 0.01}
    a5 = {r["n"]: r for r in rows if r["alpha"] == 0.05}
    n_big = max(a5)
    unf = {r["n"]: r for r in res["unfaithful"]}
    n_mid = sorted(unf)[1]
    return "info", (f"Mit α = 0,01 trifft PC den CPDAG in {pct(a1[100]['exact'])} (n = 100), {pct(a1[300]['exact'])} (n = 300) und {pct(a1[1000]['exact'])} (n = 1 000) der Läufe. Mit α = 0,05 bleiben auch bei "
                    f"n = {_n(n_big)} im Mittel {de(a5[n_big]['extra'], 2)} überzählige Kanten je Lauf: α legt die Fehlalarmrate fest, nicht die Stichprobe. In der Welt mit fast untreuer Kante (Treue-Spielraum "
                    f"{de(res['margin_unfaithful'], 2)} gegen {de(res['margin'], 2)}) verwirft PC Z–X bei n = {_n(n_mid)} in {pct(unf[n_mid]['zx_missing'])} der Läufe.")


def verdict_open(res: dict) -> tuple:
    rr = {(r["d"], r["degree"]): r for r in res["random"]}
    d_lo = min(r["d"] for r in res["random"])
    shares = [r["undirected"] for r in res["random"]]
    full = [r["full"] for r in res["random"]]
    ohne = {r["n"]: r for r in res["rows"] if r["mode"] == "ohne"}
    mit = {r["n"]: r for r in res["rows"] if r["mode"] == "mit Vorwissen"}
    ns = sorted(ohne)
    pop = res["pop_effects"]
    return "info", (f"In Zufalls-DAGs sind {pct(min(shares))} bis {pct(max(shares))} der Kanten des CPDAG ungerichtet (Grad 2, {d_lo} Knoten: {pct(rr[(d_lo, 2.0)]['undirected'])}); vollständig gerichtet sind nur "
                    f"{pct(min(full))} bis {pct(max(full))} der Graphen. In der Welt passen {res['class_size']} Diagramme; die Wirkung ist {de(pop[0], 2)} oder {de(pop[-1], 2)} (Wahrheit {de(res['truth'], 2)}). "
                    f"Der Bereich ist bei n = {ns[0]} im Mittel {de(ohne[ns[0]]['width'], 2)} breit und bei n = {_n(ns[-1])} {de(ohne[ns[-1]]['width'], 2)}: mehr Daten verengen ihn nicht. Mit Vorwissen "
                    f"(Saison und Wetter sind Wurzeln) ist die Breite {de(mit[ns[-1]]['width'], 2)}, die Wirkung {de(res['know_effects'][0], 2)}.")


def verdict_assumptions(res: dict) -> tuple:
    c = res["cancel"]
    n_lo, n_hi = min(r["n"] for r in c), max(r["n"] for r in c)
    at1 = {r["n"]: r for r in c if r["lam"] == 1.0}
    near = {r["n"]: r for r in c if r["lam"] == 0.9}
    nl = {(r["quadratic"], r["n"]): r for r in res["nonlinear"]}
    n_big = max(r["n"] for r in res["nonlinear"])
    h = res["hidden"][-1]
    return "info", (f"Hebt der direkte Pfeil den Umweg genau auf (λ = 1), fehlt die Kante in {pct(at1[n_lo]['missing'])} (n = {n_lo}) und {pct(at1[n_hi]['missing'])} (n = {_n(n_hi)}) der Läufe, und in jedem "
                    f"dieser Läufe setzt PC einen Collider, den es nicht gibt; die Zone schrumpft mit n (λ = 0,9: {pct(near[n_lo]['missing'])} gegen {pct(near[n_hi]['missing'])}). Ist die Kante Z → Y quadratisch, "
                    f"fehlt sie in {pct(nl[(True, n_big)]['missing'])} der Läufe auch bei n = {_n(n_big)} (linear: {pct(nl[(False, n_big)]['missing'])}). Ist die Nachfrage verborgen, gibt PC in "
                    f"{pct(h['main_share'], 1)} der Läufe dieselbe Ausgabe ({', '.join(h['main'])}) mit Wirkungen zwischen {de(h['lo'], 2)} und {de(h['hi'], 2)}; die Wahrheit {de(res['truth'], 2)} liegt in "
                    f"{pct(h['inside'])} der Läufe darin.")


def verdict_procedure(res: dict) -> tuple:
    o, r = res["oracle"], res["random"]["rows"]
    first, last = r[0], r[-1]
    return "info", (f"Der Orakel-PC gibt für alle {o['dags']} Diagramme mit vier Knoten genau den CPDAG der Aufzählung aus ({o['pc_equal']} von {o['dags']}); die Aufzählung der Klasse stimmt mit den {o['classes']} Klassen "
                    f"aus den Aussagenmengen überein ({o['enum_equal']} von {o['dags']}). Mit Stichprobe trifft PC bei sechs Knoten (Grad 2) den CPDAG in {pct(first['exact'])} (n = {first['n']}) bis "
                    f"{pct(last['exact'])} (n = {_n(last['n'])}). Das Skelett hängt in der ursprünglichen Fassung von der Reihenfolge der Variablen ab ({pct(first['skel_orig'])} bei n = {first['n']}, {pct(last['skel_orig'])} bei n = {_n(last['n'])}), "
                    f"in der stabilen nie; die Ausgabe hängt in beiden ab (bei n = {_n(last['n'])} ursprünglich in {pct(last['out_orig'])}, stabil in {pct(last['out_stab'])} der Datensätze).")


def verdict_builder(b: dict) -> tuple:
    if not b["acyclic"]:
        return "error", "Die Pfeile bilden einen Kreis: Das ist kein Diagramm ohne Zyklen."
    if b["edges"] == 0:
        return "info", f"Ohne Kanten sind alle Größen unabhängig; {b['same_statements']} von {b['dags']} Diagrammen haben diese Aussagen."
    und = len(b["cpdag"][1])
    if b["size"] == 1:
        return "success", f"Dieses Diagramm ist in seiner Klasse allein: Alle {b['edges']} Kanten sind aus Daten gerichtet bestimmbar ({len(b['vstructs'])} v-Struktur(en))."
    return "info", (f"{b['size']} Diagramme gehören zusammen (dieselben Aussagen über Unabhängigkeit): {und} von {b['edges']} Kanten bleiben ungerichtet, {len(b['cpdag'][0])} sind aus Daten gerichtet "
                    f"({len(b['vstructs'])} v-Struktur(en)). Unter allen {b['dags']} Diagrammen mit vier Knoten gibt es {b['classes']} Klassen.")
