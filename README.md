# Time Series Change Point Detection

Учебный проект: сбор рыночных данных, офлайн-детекция change points (CP),
каузальный онлайн-детектор и supervised-модель для предсказания времени
до следующей CP. Разработан без жёсткой зависимости от `ruptures` /
`ruptures-rs`: используется собственный dynamic-programming fallback.

## Что умеет

- **Сбор**: CBR XML (USD/EUR/CNY к рублю), опционально yfinance,
  массовая генерация синтетических рядов.
- **Обработка**: STL-разложение, robust outliers, causal-признаки.
- **Детекция CP**: два прохода (mean shift + variance shift),
  DP-fallback по умолчанию, `rustures`/`ruptures_rs` как опция.
- **Оценка**: one-to-one precision/recall/F1 с допуском, mean offset.
- **Онлайн**: `OnlineDetector` — причинный (только прошлое) детектор.
- **Экспорт датасета**: `build_nn_dataset.py` — окна + метки для NN.
- **Обучение**: `train_nn.py` (TCN + focal loss) — предсказание τ до CP.

## Требования

- Python 3.10–3.12
- ~2 GB свободного диска (базовый проект + данные + чекпойнты)
- 8 GB RAM достаточно (датасет для NN читается через memmap)

## Установка

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Для обучения NN дополнительно:
# CPU (~200 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121