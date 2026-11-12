"""Конфигурация проекта: пути, параметры рядов и детекторов."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVENTS_DIR = DATA_DIR / "events"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
GROUND_TRUTH_DIR = REPORTS_DIR / "ground_truth"
DETECTIONS_DIR = REPORTS_DIR / "detections"

MANUAL_EVENTS = EVENTS_DIR / "events.csv"
CALENDAR_EVENTS = EVENTS_DIR / "calendar_events.csv"
GROUND_TRUTH_CSV = RAW_DIR / "synthetic_ground_truth.csv"


def ensure_dirs() -> None:
    """Создать каталоги по требованию, а не при импорте модуля."""
    for path in (RAW_DIR, PROCESSED_DIR, EVENTS_DIR, REPORTS_DIR,
                 PLOTS_DIR, GROUND_TRUTH_DIR, DETECTIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)


SERIES_CONFIG = {
    "USD_RUB": {"kind": "fx", "stl_period": 5, "detection_space": "log_returns", "country": "RU", "weekends": False, "paydays": True, "tax_days": True},
    "EUR_RUB": {"kind": "fx", "stl_period": 5, "detection_space": "log_returns", "country": "RU", "weekends": False, "paydays": True, "tax_days": True},
    "CNY_RUB": {"kind": "fx", "stl_period": 5, "detection_space": "log_returns", "country": "RU", "weekends": False, "paydays": True, "tax_days": True},
    "SPY": {"kind": "equity", "stl_period": 5, "detection_space": "log_returns", "country": "NYSE", "weekends": False, "paydays": False, "tax_days": False},
    "AAPL": {"kind": "equity", "stl_period": 5, "detection_space": "log_returns", "country": "NYSE", "weekends": False, "paydays": False, "tax_days": False},
    "BTC-USD": {"kind": "crypto", "stl_period": 7, "detection_space": "log_returns", "country": None, "weekends": True, "paydays": False, "tax_days": False},
    "synth_mu_shift": {"kind": "synthetic", "stl_period": 0, "country": None, "weekends": False, "paydays": False, "tax_days": False},
    "synth_sigma_shift": {"kind": "synthetic", "stl_period": 0, "country": None, "weekends": False, "paydays": False, "tax_days": False},
    "synth_random_walk": {"kind": "synthetic", "stl_period": 0, "detection_space": "log_returns", "country": None, "weekends": False, "paydays": False, "tax_days": False},
}

DEFAULT_CONFIG = {
    "kind": "unknown", "stl_period": 7, "country": None,
    "weekends": False, "paydays": False, "tax_days": False,
    # Stage 2: cluster heuristic off by default. Final CP list comes from detect_for_series.
    "cluster_outliers": False,
}

# --- Offline outlier detection ---
OUTLIER_WINDOW = 60
OUTLIER_Z = 6.0
CP_CLUSTER_MIN = 5

# --- CP detection penalties (calibrated on synthetic baseline, F1=1.0) ---
# The exact values were validated by `calibrate.py` on the 4 synthetic scenarios.
# CP_PENALTY_TREND is reserved (the simplified pipeline no longer runs a "trend"
# pass); it is kept in the baseline JSON schema for backward compatibility.
CP_PENALTY_MEAN = 10.0
CP_PENALTY_TREND = 50.0
CP_PENALTY_VAR = 30.0
CP_MIN_SIZE = 20
CP_DEDUP_TOL = 10
CP_EVAL_TOLERANCE = 5

# --- Feature flags (Stage 2) ---
# Auto-generated calendar events add noise while calibrating the detector.
USE_CALENDAR_EVENTS = False
GENERATE_CALENDAR_EVENTS = False


def get_series_config(series_id: str) -> dict:
    """Возвращает независимую копию конфигурации серии."""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(SERIES_CONFIG.get(str(series_id), {}))
    return cfg