from pathlib import Path

import pandas as pd
import pytest

from spatialaug.benchmark import RunDir, latest_run, list_runs


@pytest.fixture
def tmp_runs_dir(tmp_path):
    return tmp_path / "runs"


def test_run_creates_dir(tmp_runs_dir):
    run = RunDir("test_run", base_dir=tmp_runs_dir, timestamp="2026-01-01T12-00")
    assert run.dir.exists()
    assert (run.dir / "figures").exists()
    assert "2026-01-01T12-00_test_run" in str(run.dir)


def test_invalid_name():
    with pytest.raises(ValueError, match="invalid run name"):
        RunDir("")
    with pytest.raises(ValueError, match="invalid run name"):
        RunDir("name/with/slash")


def test_save_config(tmp_runs_dir):
    run = RunDir("cfg_test", base_dir=tmp_runs_dir, timestamp="2026-01-01T12-00")
    cfg = {"target": "price", "ratio": 0.3, "methods": ["mean", "idw"]}
    path = run.save_config(cfg)
    assert path.exists()
    assert path.name == "config.json"
    loaded = run.load_config()
    assert loaded == cfg


def test_save_load_results(tmp_runs_dir):
    run = RunDir("res_test", base_dir=tmp_runs_dir, timestamp="2026-01-01T12-00")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    run.save_results(df)
    loaded = run.load_results()
    pd.testing.assert_frame_equal(df, loaded)


def test_save_text(tmp_runs_dir):
    run = RunDir("txt_test", base_dir=tmp_runs_dir, timestamp="2026-01-01T12-00")
    text = "# Test\n\nHello world"
    path = run.save_text(text)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == text


def test_figure_path(tmp_runs_dir):
    run = RunDir("fig_test", base_dir=tmp_runs_dir, timestamp="2026-01-01T12-00")
    p = run.figure_path("01_chart.png")
    assert p.parent.name == "figures"


def test_latest_run(tmp_runs_dir):
    RunDir("first", base_dir=tmp_runs_dir, timestamp="2026-01-01T10-00")
    RunDir("second", base_dir=tmp_runs_dir, timestamp="2026-01-01T15-00")
    RunDir("third", base_dir=tmp_runs_dir, timestamp="2026-01-02T08-00")
    latest = latest_run(base_dir=tmp_runs_dir)
    assert latest is not None
    assert latest.name == "third"


def test_latest_run_with_filter(tmp_runs_dir):
    RunDir("smoke_re", base_dir=tmp_runs_dir, timestamp="2026-01-01T10-00")
    RunDir("smoke_fns", base_dir=tmp_runs_dir, timestamp="2026-01-01T15-00")
    RunDir("smoke_re_v2", base_dir=tmp_runs_dir, timestamp="2026-01-02T08-00")

    latest_re = latest_run(base_dir=tmp_runs_dir, name_filter="re")
    assert latest_re is not None
    assert latest_re.name == "smoke_re_v2"

    latest_fns = latest_run(base_dir=tmp_runs_dir, name_filter="fns")
    assert latest_fns is not None
    assert latest_fns.name == "smoke_fns"


def test_latest_run_empty_returns_none(tmp_runs_dir):
    assert latest_run(base_dir=tmp_runs_dir) is None


def test_list_runs(tmp_runs_dir):
    RunDir("a", base_dir=tmp_runs_dir, timestamp="2026-01-01T10-00")
    RunDir("b", base_dir=tmp_runs_dir, timestamp="2026-01-02T10-00")
    runs = list_runs(base_dir=tmp_runs_dir)
    assert len(runs) == 2
    # Свежие сверху
    assert runs[0].name == "b"
    assert runs[1].name == "a"


def test_list_runs_empty(tmp_runs_dir):
    assert list_runs(base_dir=tmp_runs_dir) == []
