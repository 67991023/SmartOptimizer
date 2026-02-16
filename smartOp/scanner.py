from .utils import is_dask_dataframe as _is_dask
from .config import DASK_SAMPLE_SIZE, CORRELATION_THRESHOLD, OUTLIER_RATIO_THRESHOLD
import numpy as np

class AdvancedDataScanner:
    def analyze(self, df) -> dict:
        is_dask = _is_dask(df)
        if is_dask:
            sample = df.head(DASK_SAMPLE_SIZE)
            if hasattr(sample, 'compute'): sample = sample.compute()
            row_count = df.npartitions * 100000
        else:
            row_count = len(df)
            sample = df.sample(n=min(DASK_SAMPLE_SIZE, row_count), random_state=42) if row_count > DASK_SAMPLE_SIZE else df

        num_cols = list(sample.select_dtypes(include=[np.number]).columns)
        report = {"rows": row_count, "columns": len(sample.columns), "using_dask": is_dask,
                  "missing_analysis": {}, "outlier_analysis": {}, "correlation_alert": [],
                  "recommendations": {}, "sampled": row_count > DASK_SAMPLE_SIZE}

        dups = sample.duplicated().sum()
        if dups > 0: report["recommendations"]["drop_duplicates"] = int(dups)

        for col in sample.columns:
            miss = sample[col].isnull().mean()
            if miss > 0:
                report["missing_analysis"][col] = round(float(miss) * 100, 2)

            if col in num_cols:
                Q1, Q3 = sample[col].quantile(0.25), sample[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                n_sample = len(sample)
                outliers = int(((sample[col] < lo) | (sample[col] > hi)).sum())
                ratio = outliers / n_sample if n_sample > 0 else 0
                if outliers > 0 and ratio <= OUTLIER_RATIO_THRESHOLD:
                    report["outlier_analysis"][col] = {"count": outliers, "pct": round(ratio * 100, 2), "method": "IQR", "bounds": [float(lo), float(hi)]}
                    report["recommendations"][f"scale_{col}"] = "robust"
                else:
                    report["recommendations"][f"scale_{col}"] = "standard"
            else:
                report["recommendations"][f"encode_{col}"] = "onehot" if sample[col].nunique() <= 10 else "label"

        if len(num_cols) > 1:
            corr = sample[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            details, to_drop = [], []
            for c in upper.columns:
                for idx, val in upper[c].items():
                    if val > CORRELATION_THRESHOLD:
                        details.append({"col1": idx, "col2": c, "correlation": round(float(val), 4)})
                        if c not in to_drop: to_drop.append(c)
            if to_drop:
                report["correlation_alert"] = to_drop
                report["correlation_details"] = details

        return report
