from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at


def test_app_starts_with_all_tabs(app):
    assert [t.label for t in app.tabs] == ["Contraste", "Espace k", "Ordre de remplissage", "TSE", "Carte T2", "Importer"]


def test_preset_sets_tr_and_te(app):
    app.selectbox(key="preset").set_value("T2 (TR 2000, TE 120)").run()
    assert not app.exception
    assert (app.slider(key="tr").value, app.slider(key="te").value) == (2000, 120)


def test_every_control_runs_without_error(app):
    for slider in app.slider:
        slider.set_value(slider.max).run()
        assert not app.exception
    app.radio[0].set_value("La périphérie (hautes fréquences)").run()
    app.radio[1].set_value("centric").run()
    assert not app.exception
    for etl in (1, 32):
        app.select_slider[0].set_value(etl).run()
        assert not app.exception
        app.select_slider[1].set_value(app.select_slider[1].options[-1]).run()
        assert not app.exception
