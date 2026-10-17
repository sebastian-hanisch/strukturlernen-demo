"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import stl_constants as C
import stl_evaluation as ev
import stl_presets as P
import stl_world as W


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Leichte Welt (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(**p) and next(iter(C.PRESETS)) == "Leichte Welt (Voreinstellung)"
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_at_most_four_settings():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)
    base = C.PRESETS["Leichte Welt (Voreinstellung)"]
    for name, p in C.PRESETS.items():
        assert len([k for k in p if p[k] != base[k]]) <= 4, name
    assert C.PRESETS["Wenige Daten"]["n"] == 100 and C.PRESETS["Mit Vorwissen"]["knowledge"] == C.KNOWLEDGE[1] and C.PRESETS["Nachfrage verborgen"]["observed"] == C.YES_NO[1]
    assert C.PRESETS["Aufgehobener Umweg"]["b_zy"] == -0.48 and C.PRESETS["Nichtlineare Kante"]["form"] == C.FORMS[1] and C.PRESETS["Fast untreu"]["alpha"] == 0.05


def test_the_presets_show_what_their_help_says():
    by = {n: ev.analyse(ev.Settings(**p)) for n, p in C.PRESETS.items()}
    assert by["Leichte Welt (Voreinstellung)"].exact and by["Leichte Welt (Voreinstellung)"].learned[1] == frozenset({(0, 1), (1, 2)}) and by["Leichte Welt (Voreinstellung)"].class_size == 3
    assert not by["Wenige Daten"].exact and by["Wenige Daten"].missing == [(W.W, W.Y)]
    assert by["Fast untreu"].margin < 0.06 and (W.Z, W.X) in by["Fast untreu"].missing
    assert by["Mit Vorwissen"].class_size == 1 and by["Mit Vorwissen"].learned[1] == frozenset() and by["Mit Vorwissen"].exact
    assert by["Nachfrage verborgen"].contains is False and (W.X, W.Y) in by["Nachfrage verborgen"].learned[0]
    assert (W.Z, W.Y) in by["Aufgehobener Umweg"].missing and (W.Z, W.X) in by["Aufgehobener Umweg"].learned[0] and (W.Y, W.X) in by["Aufgehobener Umweg"].learned[0]
    assert (W.Z, W.Y) in by["Nichtlineare Kante"].missing and by["Nichtlineare Kante"].margin > 0.2


def test_snap_clamps_rounds_and_maps_the_choices():
    c = P.SETTING_SPECS["bzy_slider"]
    assert P.snap(9.0, c) == C.COEF_MAX and P.snap(-9.0, c) == C.COEF_MIN and P.snap(-0.479, c) == pytest.approx(-0.48) and P.snap(0.611, c) == pytest.approx(0.62)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == C.N_MIN and P.snap(10 ** 6, n) == C.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    a = P.SETTING_SPECS["alpha_slider"]
    assert P.snap(0, a) == 0.001 and P.snap(3, a) == 0.1 and P.snap(9, a) == 0.1 and P.snap(-2, a) == 0.001 and P.url_value("alpha_slider", 0.05) == 2 and P.url_value("bxy_slider", 0.5) == 0.5
    v = P.SETTING_SPECS["version_radio"]
    assert P.snap(1, v) == "ursprünglich" and P.snap(0, v) == "stabil"


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("bzy_slider") == (C.COEF_MIN, C.COEF_MAX) and P.bounds("alpha_slider") == (0, 3)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"sz", "zx", "xy", "zy", "wy", "n", "a", "v", "k", "zo", "f", "seed"} and len({s.url_param for s in P.SETTING_SPECS.values()}) == 12
    assert C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP) and C.DEFAULT_ALPHA in C.ALPHAS and C.DEFAULT_VERSION == C.VERSIONS[0] and C.DEFAULT_OBSERVED == "ja"


def test_default_coefficients_are_on_the_slider_grid():
    for v in (C.DEFAULT_A_SZ, C.DEFAULT_A_ZX, C.DEFAULT_B_XY, C.DEFAULT_B_ZY, C.DEFAULT_B_WY, -0.48, 0.7, 1.0):
        assert round((v - C.COEF_MIN) / C.COEF_STEP, 6) == pytest.approx(round((v - C.COEF_MIN) / C.COEF_STEP))


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 200 and C.EXP_NS == tuple(sorted(C.EXP_NS)) and C.EXP_ALPHAS == (0.01, 0.05) and {100, 300, 1000} <= set(C.EXP_NS)
    assert C.EXP_UNFAITHFUL_NS == tuple(sorted(C.EXP_UNFAITHFUL_NS)) and len(C.EXP_UNFAITHFUL_NS) >= 2 and 0.0 in C.EXP_LAMBDAS and 1.0 in C.EXP_LAMBDAS and 0.9 in C.EXP_LAMBDAS
    assert C.EXP_CANCEL_NS == tuple(sorted(C.EXP_CANCEL_NS)) and C.EXP_OPEN_DEGREES == (1.0, 2.0, 3.0) and 2.0 in C.EXP_OPEN_DEGREES and C.EXP_ORDERINGS >= 2 and C.TOLERANCE == 0.15
    assert C.EXP_EFFECT_NS == tuple(sorted(C.EXP_EFFECT_NS)) and C.EXP_NONLINEAR_NS == tuple(sorted(C.EXP_NONLINEAR_NS)) and C.EXP_RANDOM_NS == tuple(sorted(C.EXP_RANDOM_NS))
