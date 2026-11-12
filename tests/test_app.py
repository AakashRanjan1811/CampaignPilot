from streamlit.testing.v1 import AppTest


def test_dashboard_plan_approve_and_show_results(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    assert not at.exception
    at.button[0].click().run()  # Plan campaign
    assert not at.exception
    next(b for b in at.button if "Approve" in b.label).click().run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Messages sent", "Conversions", "Profit", "ROI on spend"} <= labels
