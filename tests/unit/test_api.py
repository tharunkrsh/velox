"""HTTP API and web UI: parity with the library, validation, limits, statelessness."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import velox.app.api as api_module
from velox.app.api import ApiLimits, create_app
from velox.app.simulation import SimulationRequest, run_simulation
from velox.ingest import load_trades
from velox.ingest.generic import read_table
from velox.presets import load_preset
from velox.report.html import asset, render_html

FIX = Path(__file__).parent.parent / "fixtures"
JOURNAL = FIX / "futures_journal.csv"
PRESET_DIR = Path(__file__).parents[2] / "src" / "velox" / "presets"

SMALL = dict(challenge_fee=49, n_paths=200, seed=11, k_min=0.5, k_max=1.5, k_step=0.5,
             haircuts=[0.5], max_eval_days=30, funded_days=20, n_samples=2)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _log_form(preset="topstep-50k", tz="America/Chicago", **extra):
    data = {"preset": preset, **extra}
    if tz is not None:
        data["tz"] = tz
    return data


def _files(path: Path = JOURNAL, **extra):
    return {"file": (path.name, path.read_bytes(), "text/csv"), **extra}


def _simulate(client, request: dict, **form):
    return client.post("/api/simulate", files=_files(),
                       data={**_log_form(**form), "request": json.dumps(request)})


# ─── metadata and static UI ──────────────────────────────────────────────────


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert {"ftmo-2step", "topstep-50k"} <= {p["id"] for p in meta["presets"]}
    topstep = next(p for p in meta["presets"] if p["id"] == "topstep-50k")
    assert (topstep["timezone"], topstep["rollover"]) == ("America/Chicago", "17:00")
    formats = {f["id"]: f for f in meta["formats"]}
    assert formats["tradovate"]["gross_pnl"] and not formats["mt5"]["gross_pnl"]
    assert formats["mt5"]["needs_timezone"] and not formats["generic"]["needs_timezone"]
    # defaults come from SimulationRequest itself, so UI and CLI cannot drift apart
    assert meta["defaults"]["n_paths"] == SimulationRequest.model_fields["n_paths"].default
    assert meta["defaults"]["haircuts"] == [0.25, 0.5]
    assert "challenge_fee" not in meta["defaults"]


def test_ui_page_and_assets(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert 'src="/assets/renderer.js"' in page.text
    # the UI draws results with the very same renderer and styles as the tearsheet
    for name in ("renderer.js", "tearsheet.css"):
        r = client.get(f"/assets/{name}")
        assert r.status_code == 200 and r.text == asset(name)
    assert client.get("/assets/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/assets/..%2Fapi.py").status_code == 404
    assert client.get("/assets/api.py").status_code == 404


# ─── inspect ─────────────────────────────────────────────────────────────────


def test_inspect_matches_library(client):
    r = client.post("/api/inspect", files=_files(), data=_log_form())
    assert r.status_code == 200, r.text
    body = r.json()
    log = load_trades(JOURNAL, timezone="America/Chicago")
    assert body["quality"]["n_trades"] == len(log)
    assert body["ruleset"]["id"] == "topstep-50k"
    assert body["trades"]["total"] == pytest.approx(float(log.pnl.sum()))


@pytest.mark.parametrize(("form", "message"), [
    (dict(tz=None), "timezone"),
    (dict(tz="Mars/Olympus"), "unknown timezone"),
    (dict(preset="nope"), "unknown preset"),
    # a request must never make the server read a file path
    (dict(preset=str(PRESET_DIR / "ftmo_2step.yaml")), "unknown preset"),
    (dict(format="mt5"), "columns not found"),
    (dict(commission_per_unit="2.5"), "already net"),
])
def test_inspect_rejects_bad_input_with_a_message(client, form, message):
    r = client.post("/api/inspect", files=_files(), data=_log_form(**form))
    assert r.status_code == 422
    assert message in r.json()["detail"]


def test_empty_and_oversized_uploads():
    small = TestClient(create_app(ApiLimits(max_upload_bytes=1024)))
    r = small.post("/api/inspect", files=_files(), data=_log_form())
    assert r.status_code == 413
    r = small.post("/api/inspect", files={"file": ("x.csv", b"", "text/csv")}, data=_log_form())
    assert r.status_code == 422 and "empty" in r.json()["detail"]


def test_custom_preset_upload(client):
    text = (PRESET_DIR / "topstep_50k.yaml").read_text(encoding="utf-8")
    custom = text.replace("id: topstep-50k", "id: my-firm", 1)
    assert custom != text
    r = client.post("/api/inspect", data=_log_form(preset="ignored"),
                    files=_files(preset_file=("firm.yaml", custom.encode(), "text/yaml")))
    assert r.status_code == 200, r.text
    assert r.json()["ruleset"]["id"] == "my-firm"

    for bad, message in [(b"id: [unclosed", "not valid YAML"), (b"- a list", "mapping"),
                         (b"id: x\nname: y", "invalid preset")]:
        r = client.post("/api/inspect", data=_log_form(),
                        files=_files(preset_file=("firm.yaml", bad, "text/yaml")))
        assert r.status_code == 422 and message in r.json()["detail"], r.text


# ─── simulate ────────────────────────────────────────────────────────────────


def test_simulate_is_byte_identical_to_the_library(client):
    r = _simulate(client, SMALL)
    assert r.status_code == 200, r.text
    expected = run_simulation(load_trades(JOURNAL, timezone="America/Chicago"),
                              load_preset("topstep-50k"), SimulationRequest(**SMALL))
    assert r.content == expected.model_dump_json().encode()
    # stateless and deterministic: the same request again gives the same bytes
    assert _simulate(client, SMALL).content == r.content


def test_simulate_balance_override(client):
    r = _simulate(client, SMALL, balance="100000")
    assert r.status_code == 200, r.text
    assert r.json()["ruleset"]["initial_balance"] == 100_000


@pytest.mark.parametrize(("change", "message"), [
    (dict(k_min=1.5, k_max=0.5), "below k_min"),
    (dict(n_paths=10), "n_paths"),
    (dict(n_paths=60_000), "capped at 50,000"),
    (dict(k_step=0.001), "size grid"),
    (dict(max_eval_days=400, funded_days=200), "capped at 500"),
    (dict(haircuts=[0.1, 0.2, 0.3, 0.4, 0.5]), "haircuts"),
    (dict(bogus=1), "bogus"),
])
def test_simulate_rejects_bad_settings(client, change, message):
    r = _simulate(client, {**SMALL, **change})
    assert r.status_code == 422
    assert message in r.json()["detail"]


def test_simulate_needs_valid_json(client):
    r = client.post("/api/simulate", files=_files(),
                    data={**_log_form(), "request": "{not json"})
    assert r.status_code == 422 and "invalid settings" in r.json()["detail"]
    r = client.post("/api/simulate", files=_files(), data=_log_form())
    assert r.status_code == 422  # request field missing


def test_busy_server_answers_429(monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = api_module.run_simulation

    def slow(*args, **kwargs):
        started.set()
        release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(api_module, "run_simulation", slow)
    app = create_app(ApiLimits(max_concurrent_simulations=1))
    first: dict = {}
    t = threading.Thread(target=lambda: first.setdefault("r", _simulate(TestClient(app), SMALL)))
    t.start()
    try:
        assert started.wait(10)
        assert _simulate(TestClient(app), SMALL).status_code == 429
    finally:
        release.set()
        t.join(30)
    assert first["r"].status_code == 200
    assert _simulate(TestClient(app), SMALL).status_code == 200  # slot released


# ─── report ──────────────────────────────────────────────────────────────────


def test_report_renders_the_posted_result(client):
    result = _simulate(client, SMALL)
    r = client.post("/api/report", content=result.content,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert r.headers["content-disposition"] == \
        'attachment; filename="velox-topstep-50k-tearsheet.html"'
    from velox.result import SimulationResult
    assert r.text == render_html(SimulationResult.model_validate_json(result.content))
    assert client.post("/api/report", json={"nope": 1}).status_code == 422


# ─── ingestion from bytes ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["futures_journal.csv", "mt5_positions_utf16.tsv",
                                  "tradovate_performance.csv"])
def test_read_table_from_bytes_matches_path(name):
    path = FIX / name
    assert read_table(path.read_bytes()).equals(read_table(path))
