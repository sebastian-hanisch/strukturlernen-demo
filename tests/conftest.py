"""Gemeinsame Fixtures: Hand-Instanzen (DEMO-PLAYBOOK §1b) und die Experimente, einmal klein (für die Struktur- und Verdikt-Tests) und einmal in voller Größe (nur für test_claims)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def corr_hand():
    """Mini-Instanz: Korrelationen r_01 = 0,5, r_02 = 0,6, r_12 = 0,8. Teilkorrelation von 0 und 1 gegeben 2: (0,5 − 0,6 · 0,8) / √((1 − 0,36)(1 − 0,64)) = 0,02 / 0,48 = 1/24."""
    return np.array([[1.0, 0.5, 0.6], [0.5, 1.0, 0.8], [0.6, 0.8, 1.0]])


@pytest.fixture(scope="session")
def small_sample():
    import stl_evaluation as ev

    return ev.sample_experiment(ns=(100, 300, 1000), alphas=(0.01, 0.05), reps=10)


@pytest.fixture(scope="session")
def small_open():
    import stl_evaluation as ev

    return ev.open_experiment(dims=(6,), degrees=(1.0, 2.0), dags=10, ns=(300, 1000), reps=10)


@pytest.fixture(scope="session")
def small_assumptions():
    import stl_evaluation as ev

    return ev.assumption_experiment(lambdas=(0.0, 0.9, 1.0), cancel_ns=(500, 5000), nonlinear_ns=(300, 1000), hidden_ns=(1000,), reps=10)


@pytest.fixture(scope="session")
def small_procedure():
    import stl_evaluation as ev

    return {"oracle": ev.oracle_experiment(), "random": ev.random_experiment(ns=(100, 1000), dags=5, orderings=3)}


@pytest.fixture(scope="session")
def full_sample():
    import stl_evaluation as ev

    return ev.sample_experiment()


@pytest.fixture(scope="session")
def full_open():
    import stl_evaluation as ev

    return ev.open_experiment()


@pytest.fixture(scope="session")
def full_assumptions():
    import stl_evaluation as ev

    return ev.assumption_experiment()


@pytest.fixture(scope="session")
def full_procedure():
    import stl_evaluation as ev

    return ev.procedure_experiment()
