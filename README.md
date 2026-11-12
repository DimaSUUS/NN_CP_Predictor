# Time Series Change Point Detection

Conservative foundation with offline CP detection, evaluation, and a
causal online detector. **Ruptures-rs is no longer a hard dependency.**

## Setup

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
pip install torch --index-url https://download.pytorch.org/whl/cpu

The exact dependency versions are pinned for reproducibility. If the current Python/OS does not have a compatible wheel for `ruptures-rs`, install dependencies in the target environment and verify its API there. The core has a dynamic-programming fallback for `l2` and `normal` models, but it is slower than PELT.

## CLI

```bash
python main.py collect
python main.py process
python main.py detect                        # default backend: fallback
python main.py detect --backend rustures     # only if rustures installed
python main.py evaluate --tolerance 5
python main.py report
python main.py check
python main.py inspect --series synth_mu_shift
python main.py stream --series synth_mu_shift --limit 20   # simulator only
```

`stream` is explicitly a stream simulator; it does not yet perform incremental detection.

## Important limitations

- `build_pipeline()` and the default `mark_outliers()` are offline/retrospective. Their STL decomposition and centered/global robust statistics must not be reused as causal features for forecasting.
- A `stl_period` of `0` disables STL by setting `trend=value`, `seasonal=0`, `resid=0`. The outlier flags then use the observed `value` column; the `resid` plot itself remains zero and should not be interpreted as an anomaly signal.
- Multi-series CSV validation checks uniqueness of `(series_id, timestamp)`, not timestamp globally. This is required for CBR and yfinance exports where instruments share observation dates.
- Processing prefers the new `synth_*.csv` files and ignores duplicate legacy `synthetic_*.csv` files for the same series. A per-run manifest at `data/reports/collection_status.json` records failed/empty downloads; stale processed outputs for unavailable market series are removed, while raw CSVs are preserved. `--allow-stale` is an explicit override.
- Configured financial instruments (FX, equities, crypto) are analysed in log-return space for CP detection; this targets changes in return drift/variance rather than mistaking trending price levels for repeated breaks. Generic level series use detrended levels for mean shifts and first differences for variance shifts. The synthetic random-walk benchmark uses log returns too. Its drift reversal was strengthened because the former effect size was not reliably detectable at the configured noise level.
- The `cluster_outliers()` long-run rule is a heuristic, not a CP detector. The final CP list is produced by the segmentation detector.
- The DP fallback is intended as a correctness-oriented fallback, not a speed-equivalent implementation of PELT.
- The checked-in requirements pin a proposed baseline, but this audit environment does not have `ruptures_rs`, `yfinance`, or `holidays` installed. The 20 tests below exercise the fallback detector and offline pipeline; external download behavior and the installed `ruptures-rs` API still require a run in the target environment.
- Real financial series have no ground-truth change points in this project, so their precision/recall cannot be inferred from the current reports. Calibrate penalties on synthetic/semi-synthetic benchmarks and use stability checks before interpreting real-series detections.
- `stream` remains a simulator, not an online detector. Offline STL and centered statistics are not causal and must not be used directly as forecasting features.


## Единые метрики выбросов и baseline

- `evaluation.py` — единое место для метрик change-point detection и счётчиков выбросов.
- `is_outlier` сохраняет все исходные флаги робастного детектора. Кластерная эвристика не переписывает этот столбец: короткие группы попадают в `is_isolated_outlier`, длинные группы отмечаются в `cp_candidate`. Поэтому счётчик в логе и отчёте теперь имеет одно определение.
- `cp_candidate` — эвристический кандидат, а не подтверждённая точка разладки. Окончательный список CP выдаёт сегментационный детектор.
- Явный выбор алгоритма доступен через `--backend auto|fallback|ruptures_rs` для `detect` и `evaluate`.

### Повторяемый baseline

Запускается без сети и не зависит от текущих CSV:

```bash
python -m unittest discover -s tests -v
python baseline.py --backend fallback
python calibrate.py --backend fallback
```

Команды сохраняют `data/reports/baseline_fallback.json` и `data/reports/baseline_ruptures_rs.json`. Второй запуск требует установленного `ruptures-rs`; если пакет отсутствует или API несовместим, команда должна завершиться явной ошибкой, а не молча переключиться на fallback. Сохраняются версии библиотек, параметры, seed-ы, результаты по четырём синтетическим сценариям и SHA-256 основных исходников. Файлы baseline нужно сохранить вместе с конфигурацией в момент принятия фундамента.

В каждом сравнении проверяйте:
1. mean shift и variance shift должны обнаруживаться в пределах допуска;
2. random walk должен давать одну точку в пределах допуска;
3. контрольный ряд без разладки не должен создавать ложные точки;
4. если результаты fallback и PELT различаются, не подгоняйте ожидания автоматически — сначала выясните, почему меняется сегментация.

Реальные финансовые ряды без экспертной разметки не используйте как источник precision/recall. На них сравнивайте устойчивость результата, экономический/статистический смысл и чувствительность к параметрам.


