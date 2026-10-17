"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import stl_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("asz_slider", "azx_slider", "bxy_slider", "bzy_slider", "bwy_slider", "n_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahre Wirkung des Faktors"] == "0,60" and m["Von PC gelernt"] == "3 gerichtet, 2 offen" and m["Diagramme in der Klasse"] == "3" and m["Mögliche Wirkungen"].count(" bis ") == 1
    assert at.slider(key="n_slider").value == C.DEFAULT_N and at.select_slider(key="alpha_slider").value == C.DEFAULT_ALPHA and at.number_input(key="seed_input").value == C.DEFAULT_SEED
    assert at.radio(key="version_radio").value == C.DEFAULT_VERSION and at.radio(key="knowledge_radio").value == C.DEFAULT_KNOWLEDGE and at.radio(key="observed_radio").value == "ja"
    assert any("PC findet die Klasse" in s.value for s in at.success)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # die Experimente laufen in eigenen Tests
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        for key, state_key in (("a_sz", "asz_slider"), ("a_zx", "azx_slider"), ("b_xy", "bxy_slider"), ("b_zy", "bzy_slider"), ("b_wy", "bwy_slider"), ("n", "n_slider")):
            assert at.slider(key=state_key).value == preset[key], (name, key)
        assert at.select_slider(key="alpha_slider").value == preset["alpha"] and at.radio(key="version_radio").value == preset["version"] and at.radio(key="knowledge_radio").value == preset["knowledge"]
        assert at.radio(key="observed_radio").value == preset["observed"] and at.radio(key="form_radio").value == preset["form"] and at.number_input(key="seed_input").value == preset["seed"]


def test_hidden_preset_shows_the_warning_and_the_graph_without_z():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Nachfrage verborgen")].click().run(timeout=TIMEOUT)
    assert any("Die Nachfrage Z ist nicht gemessen" in w.value for w in at.warning) and not at.exception
    assert _metrics(at)["Von PC gelernt"] == "3 gerichtet, 1 offen"


def test_vorwissen_preset_gives_a_single_effect():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Mit Vorwissen")].click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert m["Diagramme in der Klasse"] == "1" and m["Von PC gelernt"] == "5 gerichtet, 0 offen" and m["Mögliche Wirkungen"].startswith("0,6")


def test_choice_widgets_work():
    at = _fresh_app()
    at.radio(key="version_radio").set_value(C.VERSIONS[1]).run(timeout=TIMEOUT)
    at.radio(key="form_radio").set_value(C.FORMS[1]).run(timeout=TIMEOUT)
    at.select_slider(key="alpha_slider").set_value(0.1).run(timeout=TIMEOUT)
    assert not at.exception and at.radio(key="form_radio").value == C.FORMS[1] and at.select_slider(key="alpha_slider").value == 0.1


def test_builder_default_examples_and_toggle():
    at = _fresh_app()
    assert at.radio(key="bld_AB").value == "→" and at.radio(key="bld_AC").value == "keine Kante" and any("4 Diagramme gehören zusammen" in i.value for i in at.info)
    labels = [b.label for b in at.button]
    for name, expected in (("Collider A → B ← C", "allein"), ("Kreis (kein Diagramm)", "Kreis"), ("Dreieck", "6 Diagramme gehören zusammen"), ("Collider mit Abkömmling", "allein")):
        at.button[labels.index(name)].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert any(expected in x.value for x in list(at.success) + list(at.info) + list(at.error)), name
    at.radio(key="bld_BC").set_value("keine Kante").run(timeout=TIMEOUT)
    assert not at.exception


def test_every_experiment_runs_on_click():
    at = _fresh_app()
    assert "exp1_ready" not in at.session_state
    for key, text in (("exp1_run", "Mit α = 0,01 trifft PC den CPDAG"), ("exp2_run", "In Zufalls-DAGs sind"), ("exp3_run", "Hebt der direkte Pfeil den Umweg"), ("exp4_run", "Der Orakel-PC gibt für alle 543 Diagramme")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
        assert any(text in i.value for i in at.info), key


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "50"
    at.query_params["zy"] = "-9"
    at.query_params["a"] = "2"
    at.query_params["zo"] = "1"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="bzy_slider").value == C.COEF_MIN and at.select_slider(key="alpha_slider").value == 0.05
    assert at.radio(key="observed_radio").value == "nein" and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="bzy_slider").set_value(-0.48).run(timeout=TIMEOUT)
    assert at.query_params["zy"] in ("-0.48", ["-0.48"])
    at.radio(key="knowledge_radio").set_value(C.KNOWLEDGE[1]).run(timeout=TIMEOUT)
    assert at.query_params["k"] in ("1", ["1"])
    at.select_slider(key="alpha_slider").set_value(0.001).run(timeout=TIMEOUT)
    assert at.query_params["a"] in ("0", ["0"])
