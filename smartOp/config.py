import configparser
from pathlib import Path

_cfg = configparser.ConfigParser()
_cfg.read(Path(__file__).parent.parent / "config.cfg")
_g = _cfg.getint

# dask
DASK_THRESHOLD = _g("dask", "threshold_mb", fallback=500) * 1024 * 1024
DASK_SAMPLE_SIZE = _g("dask", "sample_size", fallback=50000)
# training
MAX_TRAIN_ROWS = _g("training", "max_train_rows", fallback=1000000)
TREE_SAMPLE_CAP = _g("training", "tree_sample_cap", fallback=200000)
# processing
CORRELATION_THRESHOLD = _g("processing", "correlation_threshold", fallback=95) / 100
MISSING_THRESHOLD = _g("processing", "missing_threshold", fallback=50) / 100
MAX_LABEL_CARDINALITY = _g("processing", "max_label_cardinality", fallback=10000)
LARGE_DF_THRESHOLD = _g("processing", "large_df_threshold", fallback=100000)
# loader
SQL_ROWS_LARGE = _g("loader", "sql_rows_large", fallback=100000)
SQL_ROWS_MEDIUM = _g("loader", "sql_rows_medium", fallback=250000)
SQL_ROWS_SMALL = _g("loader", "sql_rows_small", fallback=500000)
SQL_SIZE_LARGE_MB = _g("loader", "sql_size_large_mb", fallback=2048)
SQL_SIZE_MEDIUM_MB = _g("loader", "sql_size_medium_mb", fallback=512)
SQL_CHUNK_SIZE = _g("loader", "sql_chunk_size", fallback=50000)
# optimizer
DTYPE_SAMPLE_THRESHOLD = _g("optimizer", "dtype_sample_threshold", fallback=200000)
DTYPE_SAMPLE_SIZE = _g("optimizer", "dtype_sample_size", fallback=50000)
