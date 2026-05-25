# spatialaug

Фреймворк для исследования устойчивости методов интерполяции и аугментации геоданных к пропускам в признаковом пространстве и доменному сдвигу между территориями.

## Статус

В разработке. См. [plan.md](plan.md).

W1 (13–18 мая): scaffold пакета + базовые baselines.

## Установка

```bash
git clone https://github.com/neilarphy/spatialaug.git
cd spatialaug

python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

pip install -e ".[dev]"
```

## Быстрый старт

```python
import pandas as pd
from spatialaug import MeanImputer

df = pd.read_parquet("data/real_estate/labeled_final.parquet")

imp = MeanImputer(strategy="median")
df_filled = imp.fit_transform(
    df,
    lat="geo_lat",
    lon="geo_lon",
    target="price",
    group="region",
)
```

## Структура

- `src/spatialaug/imputers/` — методы имьютации (Mean, IDW, kNN, Kriging, GBM, ...)
- `src/spatialaug/augmenters/` — методы аугментации (планируется)
- `src/spatialaug/benchmark/` — бенчмарк-suite с метриками устойчивости (планируется)
- `src/spatialaug/transfer/` — оценка переноса между регионами (планируется)
- `src/spatialaug/privacy/` — эмуляция privacy-зануления (планируется)
- `tests/` — pytest-тесты

## Документы

- [plan.md](plan.md) — план работы
- [datasets.md](datasets.md) — описание собранных датасетов

## Запуск тестов

```bash
pytest
```

## Линтер

```bash
ruff check .
ruff format .
```
