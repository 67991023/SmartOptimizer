from .utils import is_dask_dataframe as _is_dask
from .config import DASK_SAMPLE_SIZE, MISSING_THRESHOLD

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
            self._block("write", f"Dropped {len(high_miss)} columns >{MISSING_THRESHOLD*100:.0f}% missing:", high_miss)

        if not is_dask and "drop_duplicates" in recs:
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0: self._log("write", f"Removed {removed} duplicate rows")

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
