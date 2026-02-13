# Config loader - reads settings from config.cfg
import configparser
from pathlib import Path

_config = configparser.ConfigParser()
_config_path = Path(__file__).parent.parent / "config.cfg"
_config.read(_config_path)

# Dask settings
DASK_THRESHOLD = _config.getint("dask", "threshold_mb", fallback=500) * 1024 * 1024
DASK_SAMPLE_SIZE = _config.getint("dask", "sample_size", fallback=50000)

# Training settings
MAX_TRAIN_ROWS = _config.getint("training", "max_train_rows", fallback=1000000)

# Processing settings
CORRELATION_THRESHOLD = _config.getint("processing", "correlation_threshold", fallback=95) / 100
MISSING_THRESHOLD = _config.getint("processing", "missing_threshold", fallback=50) / 100
