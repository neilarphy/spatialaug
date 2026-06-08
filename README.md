# spatialaug

[![PyPI](https://img.shields.io/pypi/v/spatialaug.svg)](https://pypi.org/project/spatialaug/)
[![Python](https://img.shields.io/pypi/pyversions/spatialaug.svg)](https://pypi.org/project/spatialaug/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A benchmark and toolkit for **geospatial imputation and augmentation**: ten
imputation methods (mean / IDW / kNN / kriging / UK / KED / GBM / KpR
hybrid / TabPFN foundation model) behind a single sklearn-style API, plus
spatial-block cross-validation, cross-region transfer evaluation,
value-based privacy masking, Moran's I preservation, and downstream
ΔF1 augmentation metrics.

## Install

```bash
pip install spatialaug
# Optional foundation-model imputer (auto-downloads the TabPFN v2 weights):
pip install "spatialaug[foundation]"
# Dev tooling (pytest, ruff, build, twine):
pip install "spatialaug[dev]"
```

Python 3.11+. Linux / macOS / Windows.

## Quickstart

```python
import numpy as np
import pandas as pd
from spatialaug import (
    UniversalKrigingImputer,    
    KrigingPriorRegression,     #
    MissingnessSimulator,
    compute_error_suite,
    TransferEvaluator,
)

# 1. Dummy data: 200 hex centroids with a feature-correlated target
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "lat": rng.uniform(55.5, 56.0, 200),
    "lon": rng.uniform(37.3, 37.9, 200),
    "kkt_count": rng.integers(1, 50, 200),
})
df["avg_bill"] = 500 + 8 * df["kkt_count"] + rng.normal(0, 50, 200)

# 2. Simulate missingness (mcar / diffuse_mnar / focused_mnar)
sim = MissingnessSimulator(mechanism="mcar", ratio=0.3, random_state=42)
df_masked, mask = sim.apply(df, target_col="avg_bill",
                            lat_col="lat", lon_col="lon")

# 3. Impute via KED (kriging with the kkt_count feature as external drift)
ked = UniversalKrigingImputer(feature_cols=["kkt_count"])
ked.fit(df_masked, lat="lat", lon="lon", target="avg_bill")
df_filled = ked.transform(df_masked)

# 4. Error suite on the held-out masked rows
errors = compute_error_suite(
    df.loc[mask, "avg_bill"].to_numpy(),
    df_filled.loc[mask, "avg_bill"].to_numpy(),
)
print(errors)  
```

Cross-region transfer evaluation (multi-seed):

```python
evaluator = TransferEvaluator(
    imputer_factory=lambda: UniversalKrigingImputer(feature_cols=["kkt_count"]),
    method_name="ked",
    lat_col="lat", lon_col="lon", target_col="avg_bill",
    seeds=[1, 2, 3],
)
results = evaluator.evaluate(
    source_df=region_a, target_df=region_b,
    source_name="A", target_name="B",
)
for r in results:
    print(f"{r.mechanism}: stability "
          f"= {r.transfer_stability_mean:.2f} ± {r.transfer_stability_std:.2f}")
```

## What's included

### Imputers (`spatialaug`)

| Class | Method |
|---|---|
| `MeanImputer` | Global / grouped mean or median baseline |
| `IDWImputer` | Inverse distance weighting |
| `KNNImputer` | Geographic k-nearest neighbours |
| `KrigingImputer` | Ordinary Kriging with log-transform + detrend |
| `UniversalKrigingImputer` | Universal Kriging (regional-linear drift, or KED via `feature_cols`) |
| `GBMImputer` | LightGBM on (lat, lon) plus optional covariates |
| `KrigingPriorRegression` | Hybrid: kriging predictions become an extra feature for GBM, with **k-fold out-of-fold prior** to avoid in-sample leak |
| `TabPFNImputer` | TabPFN v2 foundation model for tabular regression |

### Augmentation (`spatialaug.augmenters`)

`Augmenter` generates new points at new locations and predicts their
target via any built-in `Imputer` — matches the classical ML notion of
augmentation (albumentations / SMOTE / CTGAN family). Four placement
strategies: `regular_grid`, `density_fill` (sparse-region filling),
`mixup` (paired blending), `jitter` (perturbed observed coordinates).

### Privacy missingness (`spatialaug.privacy`)

`ValueBasedMask` implements three regulatory privacy-redaction patterns
that recur across jurisdictions (statistical agencies, payment
regulators, tax authorities): `k_anonymity` (count below threshold),
`big_business` (value above threshold), `concentration` (both).
Column names are fully configurable for non-retail domains.

### Cross-region transfer (`spatialaug.transfer`)

`TransferEvaluator` runs a multi-seed `zero_shot` vs `full_refit`
comparison and reports
`transfer_stability = MAE(zero_shot) / MAE(full_refit)` aggregated
across seeds and mechanisms.

### Metrics (`spatialaug.metrics`)

- `compute_error_suite` — MAE, RMSE, MAPE, R-squared, bias, median
  error, heteroscedasticity ratio.
- `by_target_quartile` — error breakdown across the four quartiles of
  the target.
- `morans_i` + `morans_preservation_ratio` — spatial-autocorrelation
  preservation under imputation.
- `SpatialBlockCV` — K-fold splitter with whole-block hold-out and
  optional buffered Spatial+ CV (Roberts et al. 2017).

### Benchmark utilities (`spatialaug.benchmark`)

- `MissingnessSimulator` — MCAR, diffuse MNAR, focused MNAR.
- `run_benchmark`, `run_benchmark_multi_seed`, `run_benchmark_spatial_cv`
  — orchestrators over an imputer factory map.

### Profiling (`spatialaug.utils`)

`MethodProfiler` is a context manager for time + peak-memory tracking
via `tracemalloc` (stdlib only, no external dependencies). It is
"polite" — it never stops an outer tracemalloc session and resets the
peak counter for accurate inner-block measurement.
`profile_fit_transform` and `profile_scaling` are one-shot wrappers
for empirical complexity estimation.

## Package layout

```
src/spatialaug/
  imputers/      Imputer interface + 8 implementations
  augmenters/    Augmenter (4 synthetic-point strategies)
  privacy/       ValueBasedMask
  transfer/      TransferEvaluator
  metrics/       SpatialBlockCV, errors, Moran's I
  benchmark/     MissingnessSimulator, runners, RunDir, dataset adapters
  utils/         MethodProfiler, profile_fit_transform, KM_PER_DEGREE
```

## Tests

The test suite runs without any private data:

```bash
pip install "spatialaug[dev]"
pytest -q
```

A small number of tests depend on the FNS reference dataset and are
auto-skipped if `data/fns/all_cities.parquet` is not present.

## Source repository

Full source code and issue tracker live at
[github.com/neilarphy/spatialaug](https://github.com/neilarphy/spatialaug).

## Acknowledgements

`spatialaug` is a thin orchestration layer over established
open-source libraries and models. The substantive contribution of
this package is the unified API, the benchmark infrastructure, and
the KpR hybrid imputer.

- **TabPFN** ([PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN))
  — transformer-based foundation model for tabular regression,
  Apache 2.0. Used as an in-context regressor in `TabPFNImputer`.
  The `Prior-Labs/TabPFN-v2-reg` weights are downloaded at runtime
  from the public GCS mirror into `~/.cache/tabpfn/`; see the
  PriorLabs repository for the full license.
- **PyKrige**
  ([GeoStat-Framework/PyKrige](https://github.com/GeoStat-Framework/PyKrige))
  — ordinary and universal kriging plus variogram fitting, BSD-3.
  Powers `KrigingImputer` and `UniversalKrigingImputer`.
- **LightGBM**
  ([microsoft/LightGBM](https://github.com/microsoft/LightGBM))
  — gradient boosting, MIT. Used in `GBMImputer` and the GBM
  refinement step of `KrigingPriorRegression`.
- **scikit-learn**, **scipy**, **pandas**, **numpy** — the standard
  scientific Python stack, BSD-3.

## References

Key references used by the implementations:

- TabPFN foundation model:
  [Hollmann et al. 2023 (ICLR)](https://arxiv.org/abs/2207.01848).
- Regression kriging (basis of KpR):
  [Hengl et al. 2007, Computers & Geosciences](https://www.sciencedirect.com/science/article/abs/pii/S0098300407001008).
- Survey:
  [Spatio-Temporal Missing Data Imputation, ACM CSUR 2026](https://dl.acm.org/doi/10.1145/3797903).
- Spatial CV:
  [Roberts et al. 2017, Ecography 40](https://onlinelibrary.wiley.com/doi/full/10.1111/ecog.02881).
- Transferability:
  [McKenzie et al. 2019, Spatial Statistics](https://www.sciencedirect.com/science/article/abs/pii/S1002016019608015).
- Variance calibration:
  [Heuvelink 2010](https://www.researchgate.net/publication/46629588).
- Multiple-testing correction:
  [Demšar 2006, JMLR](https://www.jmlr.org/papers/v7/demsar06a.html).

## License

MIT — see [LICENSE](LICENSE).
