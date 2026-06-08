"""Versioned storage for experiment artifacts.

Every benchmark run lands in its own timestamped folder
``results/runs/YYYY-MM-DDTHH-MM_<name>/`` together with config,
raw results, figures and a README.

The layout doubles as a research journal: every iteration is visible
and comparable, and re-running a script never silently overwrites
prior results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BASE_DIR = Path("results/runs")


class RunDir:
    """Versioned artifact folder for a single experiment run.

    Layout:
        results/runs/
        └── 2026-06-04T15-30_smoke_re_honest_gbm/
            ├── config.json        # run parameters
            ├── results.parquet    # raw output of run_benchmark()
            ├── README.md          # human-readable interpretation
            └── figures/
                └── *.png

    Examples
    --------
    >>> run = RunDir("smoke_re_honest_gbm")
    >>> print(run.dir)
    PosixPath('results/runs/2026-06-04T15-30_smoke_re_honest_gbm')
    >>> run.save_config({"regions": ["moscow", "kazan"], "ratio": 0.3})
    >>> run.save_results(results_df)
    >>> run.save_text("# Findings\\n\\n...")
    """

    def __init__(
        self,
        name: str,
        base_dir: Path = DEFAULT_BASE_DIR,
        timestamp: str | None = None,
    ) -> None:
        
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"invalid run name: {name!r}")
        ts = timestamp or datetime.now().strftime("%Y-%m-%dT%H-%M")
        self.name = name
        self.timestamp = ts
        self.dir = base_dir / f"{ts}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "figures").mkdir(exist_ok=True)

    def save_config(self, config: dict[str, Any]) -> Path:
        """Save run parameters to config.json."""
        path = self.dir / "config.json"
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        
        return path

    def save_results(self, df: pd.DataFrame, name: str = "results") -> Path:
        """Save the results DataFrame to ``<name>.parquet`` inside the run dir."""
        path = self.dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        
        return path

    def save_text(self, text: str, filename: str = "README.md") -> Path:
        """Save arbitrary text (interpretation, notes) into the run dir."""
        path = self.dir / filename
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return path

    def figure_path(self, filename: str) -> Path:
        """Path inside the run dir where a figure should be written.

        Example: ``figure_path("01_heatmap.png")``.
        """
        return self.dir / "figures" / filename

    def load_config(self) -> dict[str, Any]:
        path = self.dir / "config.json"
        
        if not path.exists():
            return {}
        
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_results(self, name: str = "results") -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        
        return pd.read_parquet(path)

    def __str__(self) -> str:
        return str(self.dir)

    def __repr__(self) -> str:
        return f"RunDir({self.name!r}, dir={self.dir})"


def latest_run(
    base_dir: Path = DEFAULT_BASE_DIR,
    name_filter: str | None = None,
) -> RunDir | None:
    """Locate the most recent run directory.

    Parameters
    ----------
    name_filter : str, optional
        If supplied, only runs whose folder name contains
        ``name_filter`` are considered.

    Returns
    -------
    RunDir or None
        None if ``base_dir`` does not exist or has no matching runs.
    """
    if not base_dir.exists():
        return None
    candidates = sorted(
        (p for p in base_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    
    if name_filter:
        candidates = [p for p in candidates if name_filter in p.name]
    
    if not candidates:
        return None

    latest = candidates[0]
    parts = latest.name.split("_", 1)

    if len(parts) == 2:
        timestamp, name = parts
    else:
        timestamp, name = "", latest.name

    run = RunDir.__new__(RunDir)
    run.name = name
    run.timestamp = timestamp
    run.dir = latest

    return run


def list_runs(base_dir: Path = DEFAULT_BASE_DIR) -> list[RunDir]:
    """Return every run under base_dir, newest first."""
    if not base_dir.exists():
        return []
    runs = []

    for path in sorted(base_dir.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        parts = path.name.split("_", 1)
        timestamp, name = (parts[0], parts[1]) if len(parts) == 2 else ("", path.name)
        run = RunDir.__new__(RunDir)
        run.name = name
        run.timestamp = timestamp
        run.dir = path
        runs.append(run)

    return runs
