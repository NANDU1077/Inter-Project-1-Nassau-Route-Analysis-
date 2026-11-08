"""Runs the real Streamlit script headlessly on every page and in edge states."""
import os
import sys
from pathlib import Path

os.environ["NASSAU_ML_FAST"] = "1"

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "Nassau_Candy_Distributor.csv").exists(), reason="dataset missing")

PAGES = ["A. Executive Overview", "B. Route Efficiency", "C. Geographic Map", "D. Ship Mode Comparison",
         "E. Route Drill-Down", "F. Data Quality", "G. Predictive Modelling (ML)"]


def run(page, **kw):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_relative_basis_all_pages():
    for page in PAGES:
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        at.sidebar.radio[1].set_value(at.sidebar.radio[1].options[1])
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])


def test_empty_filter_shows_warning_not_crash():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    ms = [m for m in at.sidebar.multiselect if m.label == "Factory"][0]
    ms.set_value([])
    at.run()
    assert not at.exception
    assert any("no shipments" in w.value for w in at.warning)


def test_exclude_extreme_empties_gracefully():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.checkbox[0].uncheck()
    at.run()
    assert not at.exception
    assert any("no shipments" in w.value for w in at.warning)


def test_bad_path_gives_helpful_error():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.text_input[0].set_value("C:/does/not/exist.csv")
    at.run()
    assert not at.exception


def test_ml_page_trains_and_renders():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("G. Predictive Modelling (ML)")
    at.run()
    assert not at.exception
    at.button[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("What the models show" in s.value for s in at.subheader)
    assert len(at.metric) >= 4
