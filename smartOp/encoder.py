from .utils import is_dask_dataframe as _is_dask
from .config import MAX_LABEL_CARDINALITY, LARGE_DF_THRESHOLD
import numpy as np

class FeatureEncoder:
    def transform(self, df, report: dict):
        is_dask = _is_dask(df)
        recs = report.get("recommendations", {})
        large = (not is_dask) and len(df) > LARGE_DF_THRESHOLD

        label_cols, onehot_cols = [], []
        for key, method in recs.items():
            if not key.startswith("encode_"): continue
            col = key.replace("encode_", "")
            if col not in df.columns: continue
            if method == "onehot" and not large: onehot_cols.append(col)
            else: label_cols.append(col)

        if is_dask:
            all_enc = onehot_cols + label_cols
            if all_enc:
                df = df.drop(columns=[c for c in all_enc if c in df.columns], errors='ignore')
        else:
            import pandas as pd
            if onehot_cols: df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
            if large:
                drop = [c for c in label_cols
                        if (len(df[c].cat.categories) if hasattr(df[c], 'cat')
                            else df[c].nunique()) > MAX_LABEL_CARDINALITY]
                if drop:
                    df = df.drop(columns=drop)
                    label_cols = [c for c in label_cols if c not in drop]
            for col in label_cols:
                df[col] = df[col].cat.codes if hasattr(df[col], 'cat') else pd.factorize(df[col])[0]

        scale_cols = [k.replace("scale_", "") for k in recs if k.startswith("scale_")]
        scale_cols = [c for c in scale_cols if c in df.columns]

        if not is_dask and scale_cols:
            for col in scale_cols:
                method = recs.get(f"scale_{col}", "standard")
                arr = df[col].to_numpy(dtype='float64')
                if method == "robust":
                    center, scale = np.nanmedian(arr), np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
                elif method == "minmax":
                    center, scale = np.nanmin(arr), np.nanmax(arr) - np.nanmin(arr)
                else:
                    center, scale = np.nanmean(arr), np.nanstd(arr)
                if scale != 0: df[col] = (arr - center) / scale
        elif is_dask and scale_cols:
            sample = df.head(10000)
            if hasattr(sample, 'compute'): sample = sample.compute()
            for col in scale_cols:
                method = recs.get(f"scale_{col}", "standard")
                data = sample[col].dropna()
                if data.empty: continue
                try:
                    if method == "robust": center, scale = data.median(), data.quantile(0.75) - data.quantile(0.25)
                    elif method == "minmax": center, scale = data.min(), data.max() - data.min()
                    else: center, scale = data.mean(), data.std()
                    if scale != 0: df[col] = (df[col] - center) / scale
                except: continue

        return df