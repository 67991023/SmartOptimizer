import numpy as np
from .config import DTYPE_SAMPLE_THRESHOLD, DTYPE_SAMPLE_SIZE

def is_dask_dataframe(df) -> bool:
    return hasattr(df, 'compute') and hasattr(df, 'npartitions')

def get_memory_usage_mb(df) -> float:
    if is_dask_dataframe(df): return 0.0
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def optimize_dtypes(df):
    n = len(df)
    use_sample = n > DTYPE_SAMPLE_THRESHOLD
    sample = df.sample(n=DTYPE_SAMPLE_SIZE, random_state=0) if use_sample else df
    for col in df.columns:
        t = df[col].dtype
        if t == 'object':
            ratio = sample[col].nunique() / len(sample)
            if ratio < 0.5: df[col] = df[col].astype('category')
        elif t == 'float64': df[col] = df[col].astype('float32')
        elif t == 'int64':
            lo, hi = df[col].min(), df[col].max()
            if lo >= 0:
                df[col] = df[col].astype('uint8' if hi < 255 else 'uint16' if hi < 65535 else 'uint32')
            else:
                df[col] = df[col].astype('int8' if lo > -128 and hi < 127 else 'int16' if lo > -32768 and hi < 32767 else 'int32')
    return df

def finalize_df(df, size_mb: float, **extra) -> tuple:
    if is_dask_dataframe(df):
        meta = {"file_size_mb": round(size_mb, 2), "using_dask": True,
                "original_rows": df.npartitions * 100_000, "partitions": df.npartitions,
                "memory_mb": "lazy (out-of-core)"}
        meta.update(extra)
        return df, meta
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    meta = {"file_size_mb": round(size_mb, 2), "using_dask": False,
            "original_rows": len(df), "memory_mb": round(mem_after, 2),
            "memory_saved_pct": round((1 - mem_after / mem_before) * 100, 1) if mem_before > 0 else 0}
    meta.update(extra)
    return df, meta

def sanitize_value(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if not np.isnan(v) else None
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (bytes, bytearray)): return '<binary>'
    return v

def sanitize_preview(df, n: int = 5) -> list:
    try:
        preview = df.head(n)
        preview = preview.compute() if hasattr(preview, 'compute') else preview.copy()
    except Exception:
        import pandas as pd
        preview = pd.DataFrame({c: ['...'] for c in df.columns}, index=[0])
    for col in preview.columns:
        if preview[col].dtype == object:
            preview[col] = preview[col].apply(lambda x: '<binary>' if isinstance(x, (bytes, bytearray)) else (str(x) if x is not None else ''))
        elif hasattr(preview[col], 'cat'): preview[col] = preview[col].astype(str)
    return [{k: sanitize_value(v) for k, v in row.items()} for row in preview.fillna('').to_dict(orient='records')]
