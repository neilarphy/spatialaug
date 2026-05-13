# spatialaug — план работы

## 1. Контекст

Команды геоаналитики банков и ритейла решают прикладные задачи: размещение банкоматов, оценка стоимости недвижимости под кредитование, выбор локации торговой точки, скоринг районов. Геоданные систематически неполны по двум структурным причинам:

1. Коммерческие источники дают неравномерное покрытие территории.
2. Часть данных обнуляется по требованиям законодательства о персональных данных.

Из-за этого пропуски в обучающих данных не случайны — они зависят от типа источника, географии и регуляторики. Дополнительная сложность — модели, обученные в одном регионе, нужно переносить в другие регионы с иной плотностью и качеством данных.

В CV и NLP есть зрелые библиотеки аугментаций (albumentations, nlpaug). Для табличных гео-данных аналогичной зрелой библиотеки нет: классические методы аугментации не учитывают пространственную структуру, геостатистические методы локальны и фрагментированы по разным пакетам, нет стандартного интерфейса и нет общего бенчмарка.

## 2. Цели

Цель работы — построить фреймворк `spatialaug` для исследования устойчивости методов интерполяции и аугментации геоданных к пропускам в признаковом пространстве и доменному сдвигу между территориями.

Фреймворк состоит из следующих компонентов:

- **Унифицированный API** для методов интерполяции (`Imputer`) и аугментации (`Augmenter`). Вход: lat/lon + признаки. Выход: восстановленный или обогащённый датасет.
- **Модуль `benchmark`** — запускает матрицу `method × missingness × density` одной командой, считает метрики устойчивости (раздел 7), поддерживает spatial block CV.
- **Модуль `privacy`** — эмуляция privacy-зануления и проверка восстановимости.
- **Модуль `transfer`** — оценка переноса вариограмм и моделей между регионами.

Исследование устойчивости проводится через применение этого фреймворка в экспериментах E1, E2 и D_privacy (раздел 5). Результаты — численные метрики устойчивости по каждому методу и графики для отчёта.

Соответствие компонентов фреймворка официальной постановке:

| Постановка проекта | Реализация |
|---|---|
| Универсальный интерфейс к методам интерполяции/аугментации | Классы `Imputer` и `Augmenter` с API на вход lat/lon/признаки |
| Бенчмаркинг качества и стабильности в пространстве | Модуль `benchmark` + метрики устойчивости |
| Учёт privacy и других ограничений | Модуль `privacy` |
| Проверка генерализации и переноса между регионами | Модуль `transfer` + эксперимент E2 |

Артефакт: pip-устанавливаемый пакет на PyPI с документацией, тестами и воспроизводимыми ноутбуками.

## 3. Датасеты и их роли

| Файл | Содержание | Объём | Тип | Роль |
|---|---|---|---|---|
| `labeled_final.parquet` | Объявления о продаже жилой недвижимости РФ 2018–2021: цена, площадь, кухня, этаж, этажность, число комнат, тип здания + price tier (budget / mid_range / premium / luxury) | 2.28M точек, 84 региона | point, lat/lon | primary — главный для E1 и E2 |
| `bdmo_services.parquet` | Платные услуги населению по муниципалитетам (Росстат, секция 1 BDMO с 2005 года) | 1.5M записей по OKTMO | areal, муниципалитеты | E2 areal demonstration (block kriging) |
| `osm_russia_cities.parquet` | Точки интереса OpenStreetMap: банки (5.3K), банкоматы (7.6K), магазины (7.3K) | 20K POI в 5 городах | point, lat/lon | covariates для co-kriging |
| `traffic_accidents.parquet` | ДТП РФ 2015–2026 (ГИБДД через tochno.st): координаты, дата, тип ДТП, тяжесть, погода, освещение, дорожные условия | 1.6M точек | point, lat/lon | cross-domain validation (ноутбук в `notebooks/`) |
| `rfsd_sample_50k.parquet` | Бухгалтерская отчётность российских юр. лиц: 213 финансовых показателей + координаты местонахождения | 50K компаний | point, lat/lon | cross-domain validation #2 |
| `train.parquet` (Sberbank) | Сделки купли-продажи недвижимости Москвы (Kaggle): цена + 292 признака — расстояния до инфраструктуры, демография, экология, привязка к району | 30K сделок Москвы | district + 292 признака | multi-feature areal study |
| ФНС геочеки | Hex-агрегаты ФНС: KKT count + национальный референс, средний/усечённый/медианный чек, доля наличных, ordinal интенсивность чеков/выручки (1-10), топ-3 категории товаров, 24-часовой вектор активности, флаги (mall/rare/ecommerce) | 3 531 hex по 11 городам разной плотности | areal H3, multi-target | дополнительный primary с 7+ финтех-таргетами; единственный датасет с реальным privacy-MNAR |

Подробные описания каждого датасета — в `datasets.md`. RE — основа экспериментов, BDMO — areal-демонстрация, OSM — co-kriging covariates, остальные — кросс-доменная валидация в отдельных ноутбуках.

## 4. Методы

Библиотека покрывает три семейства методов с общим API.

**Базовые (baselines):**
- Регионарный mean / median imputation
- IDW (inverse distance weighting)
- Geographic kNN

**Геостатистические:**
- Ordinary Kriging (через pykrige)
- Universal Kriging (с трендом по координатам)
- Block Kriging для areal data (через pyinterpolate)

Kriging имеет асимптотическую сложность O(n³). Для регионов с числом точек >50K реализуется локальная версия (`local=True, n_neighbors=500`) — kriging считается только по k ближайшим соседям. Иначе на Москве и СПб эксперименты не сойдутся.

**ML с пространственными признаками:**
- Gradient Boosting (LightGBM / XGBoost) с lat/lon + region как фичами
- GBM с инженерными гео-фичами (расстояния до центров, плотность POI, region one-hot)
- TabPFN — foundation model для табличных данных как современный ML-baseline

**Гибриды geostat × ML:**
- KpR (Kriging prior Regression) — kriging-предикты как дополнительные фичи для GBM или TabPFN; реализуется естественно поверх нашего `KrigingImputer`

**Опционально (если время позволит):**
- DeepKriging — нейросеть с пространственными базисными функциями (Chen et al. 2024). Реализуется как baseline на одном регионе, не на всех 84.
- Bivariate DeepKriging — расширение DeepKriging на multi-target, релевантно для ФНС (7+ таргетов на hex).
- GeoAggregator — transformer-based baseline для geo-tabular.
- Co-kriging с OSM-инфраструктурой как ковариата.

## 5. Эксперименты

### E1. Imputation × Augmentation matrix

**Задача:** определить, какой метод и при какой плотности данных лучше восстанавливает пропуски, и помогает ли восстановленная информация при использовании как аугментация для downstream задачи.

**Стратегия выборки регионов:** полный прогон по всем 84 регионам RE — это 84 × 4 плотности × 3 механизма × 10 методов = 10 080 экспериментов, что нереально для Colab. Используется стратифицированная выборка:

1. **Пилот в W2** — smoke test на 3 контрастных регионах (Москва — плотный, Казань — средний, Магадан — редкий). Цель: проверить, что все методы работают и метрики считаются.
2. **Основной прогон в W3** — стратифицированная выборка 25-30 регионов по 5 квинтилям плотности (по 5-6 регионов на квинтиль). Даёт статистически валидное ранжирование методов при разумной стоимости.
3. **Расширение** (опционально, W4) — если W3 прошёл быстро, добавляем оставшиеся регионы для проверки устойчивости ранжирования.

**Протокол:**
1. На выбранных регионах варьировать долю наблюдаемых данных (10/30/50/100%).
2. Применить три механизма зануления:
   - MCAR: случайно убрать 30% точек
   - Diffuse MNAR: убрать кластеры точек (зоны)
   - Focused MNAR: убрать точки с ценой выше p90 в регионе
3. Каждый метод (5 baselines + 3 геостат + 2 ML) восстанавливает пропуски.
4. Измеряем: imputation MAE и RMSE для каждого региона.
5. Восстановленные точки используем как синтетическую аугментацию для обучения downstream-классификатора (price tier).
6. Измеряем downstream F1 на отдельном тесте и **ΔF1 = F1(augmented) − F1(original)** — явный показатель пользы или вреда аугментации.
7. Считаем **stability score** по 3 механизмам пропусков и **degradation slope** по плотностям — основные метрики устойчивости.

**Результат:** сводная фигура — kriging vs ML на разных плотностях для каждого механизма зануления (heatmap), плюс отдельная таблица `method × stability_score × degradation_slope × ΔF1`.

### E2. Transferability между регионами

**Задача:** проверить переносимость моделей между регионами с разной плотностью и качеством данных — основное прикладное требование к фреймворку по постановке проекта.

**Протокол:**
1. Обучить вариограмму и ML-модель на Москве и Санкт-Петербурге (плотные регионы).
2. Применить в четырёх разреженных регионах (Новосибирская, Казань, Воронеж, Краснодар).
3. Сравнить три стратегии адаптации:
   - Zero-shot transfer (применить как есть)
   - Partial refit (пересчитать только sill вариограммы или только bias ML-модели)
   - Full refit (полностью переобучить на целевом регионе)
4. Для kriging — измерить калибровку дисперсии (Heuvelink 2010): соответствует ли предсказанная kriging variance реальной ошибке.
5. Параллельно — повторить на BDMO areal data: обучили на Москве по OKTMO → перенесли в другие города.

**Результат:** вторая фигура — performance transfer matrix + калибровочные графики.

### D_privacy. Cluster missingness emulation

**Задача:** эмулировать privacy-зануление и проверить, что библиотека умеет с ним работать.

**Протокол:**
1. Эмулировать "вырезание зон" в RE: занулить все точки в радиусе 1–5 км от случайных центров.
2. Восстановить через block kriging и через ML с гео-фичами.
3. Измерить, при какой доле зануления (30/50/70%) методы перестают работать.

Дополнительно — на реальных hex-агрегатах ФНС: занулённые поля (null `avg_bill`/`median_bill` на hex'ах с малым `kkt_count`). Восстанавливаем эти null'ы через block kriging и оцениваем качество относительно соседних hex'ов того же tier'а.

**Результат:** ноутбук-демо в `notebooks/privacy_demo.ipynb`.

## 6. Архитектура библиотеки spatialaug

API: на вход — `lat`, `lon` и признаки, на выход — восстановленный или обогащённый датасет.

```python
import spatialaug as sa

# Imputation API
imp = sa.Imputer(method='ok_kriging')
imp.fit(df, lat='geo_lat', lon='geo_lon', target='price')
df_filled = imp.transform(df)

# Augmentation API (синтез точек для downstream)
aug = sa.Augmenter(method='kriging', n_synthetic=1000)
df_aug = aug.fit_transform(df_train, lat='geo_lat', lon='geo_lon', target='price')

# Benchmark suite
sa.benchmark.run(
    df,
    methods=['mean', 'idw', 'ok', 'uk', 'gbm', 'deepkriging'],
    mechanism='diffuse_mnar',
    cv='spatial_block',
    metrics=['mae', 'rmse', 'downstream_f1']
)

# Transferability
sa.transfer.evaluate(
    imputer=imp,
    source_df=msk,
    target_df=nsk,
    strategies=['zero_shot', 'partial_refit', 'full_refit']
)

# Privacy emulation
sim = sa.privacy.ClusterMask(radius_km=2, ratio=0.5).apply(df)
df_recovered = imp.fit_transform(sim)
```

Структура пакета:
```
spatialaug/
  imputers/      # MeanImputer, IDWImputer, KrigingImputer, GBMImputer, ...
  augmenters/    # KrigingAugmenter, ...
  benchmark/     # MissingnessSimulator, DegradationRunner
  transfer/      # TransferEvaluator, VariogramAdaptor
  privacy/       # ClusterMask, ZoneMask
  metrics/       # spatial_block_cv, calibration, downstream_f1
  utils/         # geo_kfold, distance_utils
```

GPU-зависимости опциональные: библиотека работает на CPU, GPU-ускорение включается флагом для DeepKriging и тяжёлого GBM.

## 7. Метрики оценки

| Что измеряем | Метрика |
|---|---|
| Качество восстановления | MAE, RMSE по числовым таргетам |
| Downstream полезность | F1, accuracy на price tier |
| Калибровка кригинга | соотношение predicted variance vs actual error (Heuvelink 2010) |
| Стоимость инференса | мс/запрос на одну точку, throughput на 1M точек |
| Cross-validation | spatial block CV (не random) — корректная оценка обобщения |

**Метрики устойчивости (центральные для темы работы):**

| Метрика | Формула | Интерпретация |
|---|---|---|
| Stability score | `1 − std(metric) / mean(metric)` по 3 механизмам пропусков | Чем выше, тем меньше метод "дёргается" при смене типа пропусков |
| Degradation slope | наклон линейной аппроксимации `metric(density)` | Чем более пологий, тем устойчивее метод к разреженности |
| ΔF1 (augmentation gain) | `F1(augmented) − F1(original)` | Положительный — аугментация помогает, отрицательный — вредит |
| Transfer stability | `metric(target) / metric(source)` | Доля сохранённого качества при переносе между регионами |

Бизнес-метрики (для economic analysis в W5):
- Стоимость одного полного inference прогона на регион
- Throughput при батчевой обработке 1M точек
- Memory footprint по методам
- Asymptotic complexity и точка безубыточности kriging vs ML

**Конвенции воспроизводимости:**
- Все эксперименты с фиксированным `random_state=42` (методы, CV-сплиты, бутстрап)
- Все degradation curves строятся с 95% bootstrap confidence bands по регионам — одна линия без полосы неубедительна
- Все версии библиотек зафиксированы в `pyproject.toml` с W1

## 8. Календарь — 6 недель

Подготовительный период (5–12 мая) использован для сбора датасетов (7 источников, включая парсинг ФНС-геочеков по 11 городам), описания данных в `datasets.md` и составления настоящего плана. Кодирование `spatialaug` начинается с W1.

| Неделя | Период | Цель | Артефакт к концу недели |
|---|---|---|---|
| W1 | 13–18 мая | Scaffold пакета, baselines (Mean / IDW / kNN + OK kriging), spatial block CV, фиксация версий библиотек в `pyproject.toml` | репо + 4 baselines + первый туториал-ноутбук |
| W2 | 19–25 мая | UK kriging (с `local=True` опцией), GBM, degradation curve runner, функция `stability_score`, **smoke test на 3 регионах (Москва / Казань / Магадан)** | первая degradation curve + проверенные методы на 3 контрастных регионах |
| W3 | 26 мая — 1 июня | Эксперимент E1 на стратифицированной выборке 25-30 регионов + расчёт stability score / degradation slope / ΔF1 | главная фигура + таблица устойчивости |
| W4 | 2–8 июня | Эксперимент E2 transferability + калибровка дисперсии (Heuvelink) | вторая фигура + transfer matrix |
| W5 | 9–15 июня | Privacy-demo на естественных null'ах ФНС + economic analysis + cross-domain ноутбуки | demo notebook + economics.md |
| W6 | 16–20 июня | PyPI publish, README, финал текста работы, презентация, защита | v1.0 на PyPI + защита |

Параллельные потоки:
- Описание результатов ведётся по ходу — после каждого эксперимента пара абзацев в `results/`.
- DeepKriging / Bivariate DeepKriging / GeoAggregator — опциональные методы; добавляются если W2 или W3 проходят с запасом, иначе пропускаются без потери defensibility.

## 9. Риски и митигации

| Риск | Митигация |
|---|---|
| DeepKriging обучается слишком долго на Colab | Реализовать как baseline на одном регионе, не на 84 |
| Эксперимент E1 затягивается | Возможно поджать W4, частичный результат всё равно defensible |
| Colab session limits | Чанковать длинные эксперименты, чекпоинты в gdrive |
| Методология окажется недостаточной для статьи | Это бонусная цель — для практики plan defensible сам по себе |
| Pyinterpolate / Pykrige совместимость | Зафиксировать версии в `pyproject.toml` в W1; написать `VariogramAdapter` с унифицированным API чтобы менять backend без переписывания клиентского кода |
| Compute cost — полный 84-региональный E1 невозможен на Colab | Стратифицированная выборка 25-30 регионов по 5 квинтилям плотности (см. раздел 5) |
| Kriging O(n³) ломается на регионах с >50K точек | Опция `local=True, n_neighbors=500` в `KrigingImputer` |
| ΔF1 может оказаться около нуля или отрицательным | Это валидный negative result — показать, что kriging-аугментация не помогает downstream-классификации редких ценовых сегментов |

## 10. Критерии успеха

Для защиты практики:
- Библиотека `spatialaug` опубликована на PyPI, документация и README присутствуют
- Эксперимент E1 проведён, главная фигура работы готова
- Эксперимент E2 проведён хотя бы для двух стратегий адаптации
- Бенчмарк-suite запускается на пользовательских данных одной строкой
- Текст работы с метриками, бейзлайнами, таблицами результатов

Дополнительно для статьи:
- Spatial block CV использован везде вместо random
- Калибровка дисперсии измерена (Heuvelink)
- Stability score и degradation slope рассчитаны по всем методам и визуализированы (heatmap)
- ΔF1 для аугментации измерен с разбивкой по плотности и механизму пропусков
- Multi-target evaluation на ФНС-датасете (7+ таргетов на hex)
- Cross-domain validation на минимум двух датасетах помимо RE
- Воспроизводимые скрипты для всех figures

## 11. Ключевые источники

**Обзоры:**

| Тема | Источник |
|---|---|
| Survey по spatio-temporal imputation | [Spatio-Temporal Missing Data Imputation Survey, ACM Computing Surveys 2025](https://dl.acm.org/doi/10.1145/3797903) |
| Survey: differential privacy для spatiotemporal data | [arXiv 2407.15868 (July 2024)](https://arxiv.org/abs/2407.15868) |

**Kriging и его расширения:**

| Тема | Источник |
|---|---|
| Kriging как data augmentation | [arXiv 2501.07183 (Jan 2025)](https://arxiv.org/abs/2501.07183) |
| Inductive kriging при разреженности (KITS) | [arXiv 2311.02565 (Nov 2023)](https://arxiv.org/abs/2311.02565) |
| DeepKriging — нейросеть со spatial basis functions | [Statistica Sinica 2024](https://www3.stat.sinica.edu.tw/statistica/J34N1/J34N113/J34N113.html) |
| Bivariate DeepKriging + LMC для multi-target | [arXiv 2307.08038 (Nov 2025)](https://arxiv.org/abs/2307.08038) |
| Feature-Free Regression Kriging | [arXiv 2507.07382 (Jul 2025)](https://arxiv.org/abs/2507.07382) |
| KpR — Kriging prior Regression с TabPFN | [arXiv 2509.09408 (Sept 2025)](https://arxiv.org/abs/2509.09408) |
| Калибровка кригинговой дисперсии (Heuvelink) | [ResearchGate / Spatial Statistics](https://www.researchgate.net/publication/46629588) |

**ML / DL для геоданных:**

| Тема | Источник |
|---|---|
| GeoAggregator — transformer для geo-tabular | [arXiv 2507.17977 (Jul 2025)](https://arxiv.org/abs/2507.17977) |
| Hex2vec — embeddings для H3 hex'ов | [arXiv 2111.00970 (2021)](https://arxiv.org/abs/2111.00970) |
| Rethinking GNNs and Missing Features (related work) | [arXiv 2601.04855 (Jan 2026)](https://arxiv.org/abs/2601.04855) |

**Transferability и domain adaptation:**

| Тема | Источник |
|---|---|
| Unsupervised DA Regression через Gram matrix alignment | [arXiv 2411.06917 (Nov 2024)](https://arxiv.org/abs/2411.06917) |

**Методология валидации:**

| Тема | Источник |
|---|---|
| Spatial+ Cross-Validation | [Remote Sensing 2023](https://www.sciencedirect.com/science/article/pii/S1569843223001887) |
