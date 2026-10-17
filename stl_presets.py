"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet;
es gibt weder KEPT-Regler noch berechnete Grenzen. Auswahlen (α, Fassung, Vorwissen, Nachfrage beobachtet, Form von Z → Y) stehen in der Adresszeile als Index."""

import random
from dataclasses import dataclass

import streamlit as st

import stl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float
    choices: tuple | None = None


def _choice(param: str, choices: tuple, default) -> SettingSpec:
    return SettingSpec(param, int, default, 0, len(choices) - 1, 1, choices)


SETTING_SPECS = {
    "asz_slider": SettingSpec("sz", float, C.DEFAULT_A_SZ, C.COEF_MIN, C.COEF_MAX, C.COEF_STEP),
    "azx_slider": SettingSpec("zx", float, C.DEFAULT_A_ZX, C.COEF_MIN, C.COEF_MAX, C.COEF_STEP),
    "bxy_slider": SettingSpec("xy", float, C.DEFAULT_B_XY, C.COEF_MIN, C.COEF_MAX, C.COEF_STEP),
    "bzy_slider": SettingSpec("zy", float, C.DEFAULT_B_ZY, C.COEF_MIN, C.COEF_MAX, C.COEF_STEP),
    "bwy_slider": SettingSpec("wy", float, C.DEFAULT_B_WY, C.COEF_MIN, C.COEF_MAX, C.COEF_STEP),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, C.N_STEP),
    "alpha_slider": _choice("a", C.ALPHAS, C.DEFAULT_ALPHA),
    "version_radio": _choice("v", C.VERSIONS, C.DEFAULT_VERSION),
    "knowledge_radio": _choice("k", C.KNOWLEDGE, C.DEFAULT_KNOWLEDGE),
    "observed_radio": _choice("zo", C.YES_NO, C.DEFAULT_OBSERVED),
    "form_radio": _choice("f", C.FORMS, C.DEFAULT_FORM),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
}
PRESET_KEYS = {"a_sz": "asz_slider", "a_zx": "azx_slider", "b_xy": "bxy_slider", "b_zy": "bzy_slider", "b_wy": "bwy_slider", "n": "n_slider", "alpha": "alpha_slider", "version": "version_radio",
               "knowledge": "knowledge_radio", "observed": "observed_radio", "form": "form_radio", "seed": "seed_input"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen (bei einer Auswahl: auf einen gültigen Index)."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1 or spec.choices):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return spec.choices[int(value)] if spec.choices else value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def url_value(state_key: str, value):
    spec = SETTING_SPECS[state_key]
    return spec.choices.index(value) if spec.choices else value


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(url_value(state_key, value))
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
