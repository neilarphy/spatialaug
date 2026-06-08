# Changelog

All notable changes to `spatialaug` are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-08

First public PyPI release.

### Imputers (`spatialaug.imputers`)
- `MeanImputer` — grouped or global mean / median baseline.
- `IDWImputer` — inverse distance weighting.
- `KNNImputer` — geographic k-nearest neighbours.
- `KrigingImputer` — Ordinary Kriging with log-transform, OLS
  detrending and an optional local mode.
- `UniversalKrigingImputer` — Universal Kriging with regional-linear
  drift, switching to KED when `feature_cols` is given. Symmetric
  `np.clip(..., -20, 25)` on both back-transform paths guards
  against PyKrige's extrapolation overflow.
- `GBMImputer` — LightGBM on (lat, lon) plus optional covariates,
  with skew-driven log-transform.
- `KrigingPriorRegression` (KpR) — Hengl variant B hybrid: kriging
  predictions become an extra feature for a GBM. Ships with a
  k-fold out-of-fold prior (`oof_folds=5` default) so the GBM
  never sees a kriging prediction that was fit on its own row.
  `cv=SpatialBlockCV(...)` is supported for spatial-aware OOF.
  `oof_folds=None` preserves the in-sample prior for benchmark
  reproducibility.
- `TabPFNImputer` — TabPFN v2 foundation-model wrapper. Weights
  are downloaded at runtime from the public GCS mirror into
  `~/.cache/tabpfn/`; the `tabpfn` runtime is an optional
  `[foundation]` extra.

### Augmenter (`spatialaug.augmenters`)
- `Augmenter` — synthetic geo-point generation in the classical
  data-augmentation sense (albumentations / SMOTE / CTGAN family).
  Four strategies (`regular_grid`, `density_fill`, `mixup`,
  `jitter`) dispatched through a dict so new strategies are a
  one-function plus one-entry addition.

### Privacy missingness (`spatialaug.privacy`)
- `ValueBasedMask` — three regulatory privacy-redaction patterns
  expressed as a module-level rules dict:
  `k_anonymity` (count below threshold), `big_business` (value
  above threshold), `concentration` (both). `kkt_col` / `bill_col`
  are fully configurable for any count + value schema.

### Cross-region transfer (`spatialaug.transfer`)
- `TransferEvaluator` — multi-seed × multi-mechanism evaluation of
  `zero_shot` vs `full_refit`, reports
  `transfer_stability = MAE(zero_shot) / MAE(full_refit)` with
  mean / std across seeds. `evaluate_matrix` supports arbitrary
  ordered (source → target) pairs.

### Metrics (`spatialaug.metrics`)
- `compute_error_suite` — MAE, RMSE, MAPE, R², bias, median error,
  heteroscedasticity ratio in a single pass.
- `by_target_quartile` — per-quartile error breakdown.
- `morans_i` + `morans_preservation_ratio` — spatial-autocorrelation
  preservation, with optional permutation p-value.
- `SpatialBlockCV` — K-fold splitter with whole-block hold-out and
  an optional buffered (Spatial+ CV) mode.

### Benchmark utilities (`spatialaug.benchmark`)
- `MissingnessSimulator` — MCAR, diffuse MNAR (cluster centres),
  focused MNAR (extreme-value pool with deterministic-to-stochastic
  knob via `focused_pool_factor`).
- `run_benchmark`, `run_benchmark_multi_seed`,
  `run_benchmark_spatial_cv` — orchestrators over an imputer
  factory map, sharing a `_fit_score_imputer` helper so the three
  runners cannot drift apart.
- `RunDir`, `latest_run`, `list_runs` — timestamped artefact
  folders for reproducible run logs.
- `datasets/fns.py` — reference dataset adapter (FNS hex-grid) with
  a documented convention for adding new loaders.

### Profiling (`spatialaug.utils`)
- `MethodProfiler` — context manager for time + peak heap via the
  standard-library `tracemalloc`. The profiler is "polite": it
  never stops an outer tracing session and resets the peak counter
  on entry so the value reported on exit reflects only its own
  block.
- `profile_fit_transform`, `profile_scaling` — one-shot wrappers
  for empirical complexity estimation.
- `utils.geo` — `KM_PER_DEGREE` constant and shared
  `km_to_deg(lat, km)` helper used across modules.

### Tooling
- ~200 unit tests covering every public class.
- `.github/workflows/ci.yml` — pytest + ruff matrix on Python 3.11
  and 3.12 (`push` and `pull_request` to `master`).
- `.github/workflows/release.yml` — tag-driven PyPI publish via
  OIDC Trusted Publisher; `verify-imports` step imports the full
  `__all__` (20 names) after installing the wheel in a clean venv.
- README rewritten for the PyPI long-description audience with a
  pip-installable Quickstart that uses synthetic data.

[0.1.0]: https://github.com/neilarphy/spatialaug/releases/tag/v0.1.0
