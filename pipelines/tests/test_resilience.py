import duckdb
import pytest
from ondine_pipelines import lake
from ondine_pipelines.lake import with_retries
from ondine_pipelines.quality import QualityError


def test_transient_io_errors_are_retried(monkeypatch):
    monkeypatch.setattr(lake.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise duckdb.IOException("Could not connect to server")
        return "ok"

    assert with_retries(flaky, attempts=3) == "ok"
    assert calls["n"] == 3


def test_persistent_io_error_is_raised(monkeypatch):
    monkeypatch.setattr(lake.time, "sleep", lambda _s: None)
    with pytest.raises(duckdb.IOException):
        with_retries(lambda: (_ for _ in ()).throw(duckdb.IOException("down")), attempts=2)


def test_logic_errors_are_not_retried():
    calls = {"n": 0}

    def invalid():
        calls["n"] += 1
        raise QualityError("contrôle bloquant")

    with pytest.raises(QualityError):
        with_retries(invalid, attempts=5)
    assert calls["n"] == 1
