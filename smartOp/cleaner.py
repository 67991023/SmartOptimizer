from .utils import is_dask_dataframe as _is_dask
from .config import DASK_SAMPLE_SIZE, MISSING_THRESHOLD
import numpy as np

class SmartCleaner:
    def __init__(self, logger=None):
        self.logger, self.operations = logger, []

    def _log(self, t, msg):
        self.operations.append(f"[{t}] {msg}")
        if self.logger: getattr(self.logger, t, self.logger.info)(msg)

    def _block(self, t, msg, items):
        self.operations.append(f"[{t}] {msg}")
        for c in items: self.operations.append(f"[tree] {c}")
        if self.logger and hasattr(self.logger, 'block'): self.logger.block(t, msg, items)

    def clean_data(self, df, report: dict):
        is_dask = _is_dask(df)
        self.operations = []
        recs = report.get("recommendations", {})

        if is_dask:
            sample = df.head(DASK_SAMPLE_SIZE)
            if hasattr(sample, 'compute'): sample = sample.compute()
        else:
            bin_cols = []
            for c in df.columns:
                if df[c].dtype != object: continue
                idx = df[c].first_valid_index()
                if idx is not None and isinstance(df[c].at[idx], (bytes, bytearray)):
                    bin_cols.append(c)
            if bin_cols:
                self._log("write", f"Dropping {len(bin_cols)} binary columns: {bin_cols}")
                df = df.drop(columns=bin_cols)
            sample = df.sample(n=min(DASK_SAMPLE_SIZE, len(df)), random_state=42) if len(df) > DASK_SAMPLE_SIZE else df

        initial_cols = len(df.columns)
        n_sample = len(sample)

        id_cols = [c for c in df.columns if sample[c].nunique() == n_sample]
        if id_cols:
            df = df.drop(columns=id_cols)
            if is_dask: sample = sample.drop(columns=id_cols)
            self._block("write", f"Dropped {len(id_cols)} ID columns:", id_cols)

        if not is_dask:
            cols_before = set(df.columns)
            df = df.dropna(axis=1, thresh=int(len(df) * (1 - MISSING_THRESHOLD)))
            high_miss = list(cols_before - set(df.columns))
        else:
            high_miss = [c for c in df.columns if sample[c].isnull().mean() > MISSING_THRESHOLD]
            if high_miss: df = df.drop(columns=high_miss)
        if high_miss:
            self._block("write", f"Dropped {len(high_miss)} columns >50% missing:", high_miss)

        if not is_dask and "drop_duplicates" in recs:
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0: self._log("write", f"Removed {removed} duplicate rows")

        imputed = []
        for key, method in recs.items():
            if not key.startswith("impute_"): continue
            col = key.replace("impute_", "")
            if col not in df.columns: continue
            src = sample if is_dask and col in sample.columns else df
            if method == "mean": val = src[col].mean()
            elif method == "median": val = src[col].median()
            elif method == "mode":
                m = src[col].mode()
                val = m[0] if not m.empty else None
            else: val = None
            if val is not None:
                df[col] = df[col].fillna(val)
                imputed.append(f"{col} → {method}")
        if imputed: self._block("write", f"Imputed {len(imputed)} columns:", imputed)

        outlier_info, outlier_items = report.get("outlier_analysis", {}), []
        mask = np.ones(len(df), dtype=bool) if not is_dask else None
        for col, info in outlier_info.items():
            if col not in df.columns: continue
            lo, hi = info["bounds"]
            if is_dask:
                df = df[(df[col] >= lo) & (df[col] <= hi)]
            else:
                mask &= (df[col] >= lo) & (df[col] <= hi)
            outlier_items.append(f"{col}: [{lo:.2f}, {hi:.2f}]")
        if not is_dask and mask is not None and not mask.all():
            df = df[mask]
        if outlier_items: self._block("write", f"Filtered outliers in {len(outlier_items)} columns:", outlier_items)

        drop_cols = report.get("correlation_alert", [])
        corr_details = report.get("correlation_details", [])
        actual = [c for c in drop_cols if c in df.columns]
        if actual:
            df = df.drop(columns=actual)
            items = []
            for col in actual:
                match = next((d for d in corr_details if d['col2'] == col), None)
                items.append(f"{col} ← {match['correlation']*100:.1f}% with {match['col1']}" if match else col)
            self._block("write", f"Dropped {len(actual)} correlated columns:", items)

        self._log("checkpoint", f"Clean complete: {initial_cols}→{len(df.columns)} cols")
        return df
