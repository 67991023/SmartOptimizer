# Shared utilities
import numpy as np

def is_dask_dataframe(df) -> bool:
    return hasattr(df, 'compute') and hasattr(df, 'npartitions')

def sanitize_value(v):
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    elif isinstance(v, (np.floating, np.float64, np.float32)):
        return float(v) if not np.isnan(v) else None
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, (bytes, bytearray)):
        return '<binary data>'
    return v

def sanitize_preview(df, num_rows: int = 5) -> list:
    if is_dask_dataframe(df):
        preview_df = df.head(num_rows)
        if hasattr(preview_df, 'compute'):
            preview_df = preview_df.compute()
    else:
        preview_df = df.head(num_rows).copy()
    
    for col in preview_df.columns:
        if preview_df[col].dtype == object:
            preview_df[col] = preview_df[col].apply(
                lambda x: '<binary data>' if isinstance(x, (bytes, bytearray)) 
                else (str(x) if x is not None else '')
            )
        elif hasattr(preview_df[col], 'cat'):
            preview_df[col] = preview_df[col].astype(str)
    
    preview_df = preview_df.fillna('')
    preview = preview_df.to_dict(orient="records")
    return [{k: sanitize_value(v) for k, v in row.items()} for row in preview]

def build_metadata(file_size_mb: float, using_dask: bool, original_rows: int, 
                   memory_mb: float, memory_saved_pct: float, **extra) -> dict:
    meta = {
        "file_size_mb": round(file_size_mb, 2),
        "using_dask": using_dask,
        "original_rows": original_rows,
        "memory_mb": round(memory_mb, 2) if memory_mb else None,
        "memory_saved_pct": round(memory_saved_pct, 1) if memory_saved_pct else None
    }
    meta.update(extra)
    return meta
