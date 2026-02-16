# Dask Quick Reference for Team

## What is Dask?
Dask is a library that lets us process files larger than available RAM. It works like pandas but loads data in chunks ("lazy evaluation").

## When Does It Activate?
Only for files **>1GiB** (configurable in `config.cfg`). For normal files, standard pandas is used - Dask won't be activated at all.

## Key Difference: Lazy vs Eager

```python
# Pandas (eager) - runs immediately
result = df['column'].mean()  # Returns: 42.5

# Dask (lazy) - builds a task graph, doesn't run yet
result = df['column'].mean()  # Returns: Delayed('mean-abc123')

# To get the actual value from Dask:
result = df['column'].mean().compute()  # Returns: 42.5
```

## How to Check Which Type You Have

```python
from smartOp.utils import is_dask_dataframe

if is_dask_dataframe(df):
    # Dask path - may need .compute() for final values
    preview = df.head(5).compute()
else:
    # # Normal pandas
    preview = df.head(5)
```

## Operations That Work the Same

| Operation | Pandas | Dask |
|-----------|--------|------|
| `df.head(n)` | ✅ | ✅ |
| `df['col']` | ✅ | ✅ |
| `df.fillna(value)` | ✅ | ✅ |
| `df.drop(columns=[...])` | ✅ | ✅ |
| `df[df['col'] > 5]` | ✅ | ✅ |

## Operations That Need `.compute()`

| Operation | Dask Behavior |
|-----------|--------------|
| `len(df)` | Returns lazy - use `.compute()` or check `df.npartitions` |
| `df.mean()` | Returns Delayed - call `.compute()` |
| `df.to_dict()` | Must compute first |
| `df.nunique()` | Returns Delayed |

## Code Pattern Used in This Project

```python
def my_function(df):
    is_dask = is_dask_dataframe(df)
    
    if is_dask:
        return _my_function_dask(df)   # Dask-optimized path
    else:
        return _my_function_pandas(df)  # Standard pandas path
```

## Config Settings (`config.cfg`)

```ini
[dask]
threshold_mb = 500    # Files above this use Dask
sample_size = 50000   # Rows to sample for analysis
```

Increase `threshold_mb` if you want Dask to kick in less often.

## Files with Dask Logic

- `loader.py` - Loads large CSVs with Dask
- `cleaner.py` - Has `_clean_dask()` for large file cleaning
- `encoder.py` - Has `_transform_dask()` for large file encoding
- `scanner.py` - Samples Dask DataFrames for analysis
- `trainer.py` - Converts Dask to pandas (with sampling) before training

## TL;DR

1. Check with `is_dask_dataframe(df)`
2. If Dask, call `.compute()` when you need actual values
3. Most operations work identically to pandas
