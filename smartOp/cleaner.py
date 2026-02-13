# Data Cleaner

from .utils import is_dask_dataframe as _is_dask

def _compute_if_dask(value):
    if hasattr(value, 'compute'):
        return value.compute()
    return value

class SmartCleaner:
    def __init__(self, logger=None):
        self.logger = logger
        self.operations = []
    
    def _log(self, log_type: str, message: str):
        self.operations.append(f"[{log_type}] {message}")
        if self.logger:
            getattr(self.logger, log_type, self.logger.info)(message)
        else:
            print(f"[{log_type}] {message}")
    
    def _block(self, log_type: str, parent_message: str, children: list):
        self.operations.append(f"[{log_type}] {parent_message}")
        for child in children:
            self.operations.append(f"[tree] {child}")
        
        if self.logger and hasattr(self.logger, 'block'):
            self.logger.block(log_type, parent_message, children)
        else:
            print(f"[{log_type}] {parent_message}")
            for child in children:
                print(f"  └─ {child}")
    
    def _drop_binary_columns(self, df):
        binary_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(1)
                if len(sample) > 0:
                    val = sample.iloc[0]
                    if isinstance(val, (bytes, bytearray)):
                        binary_cols.append(col)
        
        if binary_cols:
            self._log("write", f"Dropping {len(binary_cols)} binary columns: {binary_cols}")
            df = df.drop(columns=binary_cols)
        return df
    
    def clean_data(self, df, report: dict):
        is_dask = _is_dask(df)
        self.operations = []
        
        if not is_dask:
            df = self._drop_binary_columns(df)
        
        if is_dask:
            return self._clean_dask(df, report)
        else:
            return self._clean_pandas(df, report)
    
    def _clean_pandas(self, df, report: dict):
        df_clean = df.copy()
        recs = report.get("recommendations", {})
        initial_rows = len(df_clean)
        initial_cols = len(df_clean.columns)

        dropped_id_cols = []
        for col in list(df_clean.columns):
            if df_clean[col].nunique() == len(df_clean):
                dropped_id_cols.append(col)
                df_clean.drop(columns=[col], inplace=True)
        
        if dropped_id_cols:
            self._block("write", f"Dropped {len(dropped_id_cols)} ID columns (unique value per row):", dropped_id_cols)

        cols_before = set(df_clean.columns)
        limit = len(df_clean) * 0.5
        df_clean.dropna(axis=1, thresh=int(limit), inplace=True)
        dropped_missing = list(cols_before - set(df_clean.columns))
        if dropped_missing:
            self._block("write", f"Dropped {len(dropped_missing)} columns with >50% missing:", dropped_missing)

        if "drop_duplicates" in recs:
            rows_before = len(df_clean)
            df_clean.drop_duplicates(inplace=True)
            dups_removed = rows_before - len(df_clean)
            if dups_removed > 0:
                self._log("write", f"Removed {dups_removed} duplicate rows")
        
        imputed_cols = []
        for key, method in recs.items():
            if key.startswith("impute_"):
                col = key.replace("impute_", "")
                if col not in df_clean.columns:
                    continue

                if method == "mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    imputed_cols.append(f"{col} → filled with mean")
                elif method == "median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    imputed_cols.append(f"{col} → filled with median")
                elif method == "mode":
                    if not df_clean[col].mode().empty:
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                        imputed_cols.append(f"{col} → filled with mode")
        
        if imputed_cols:
            self._block("write", f"Imputed {len(imputed_cols)} columns with missing values:", imputed_cols)
        
        outlier_info = report.get("outlier_analysis", {})
        rows_before_outliers = len(df_clean)
        outlier_details = []
        for col, info in outlier_info.items():
            if col in df_clean.columns:
                lower, upper = info["bounds"]
                before = len(df_clean)
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
                removed = before - len(df_clean)
                if removed > 0:
                    outlier_details.append(f"{col}: {removed} rows outside [{lower:.2f}, {upper:.2f}]")
        
        outliers_removed = rows_before_outliers - len(df_clean)
        if outliers_removed > 0:
            self._block("write", f"Removed {outliers_removed} rows with outliers:", outlier_details)

        drop_cols = report.get("correlation_alert", [])
        corr_details = report.get("correlation_details", [])
        if drop_cols:
            actual_dropped = [c for c in drop_cols if c in df_clean.columns]
            if actual_dropped:
                df_clean.drop(columns=actual_dropped, errors='ignore', inplace=True)
                corr_items = []
                for col in actual_dropped:
                    for detail in corr_details:
                        if detail['col2'] == col:
                            corr_items.append(f"{col} ← {detail['correlation']*100:.1f}% correlated with {detail['col1']}")
                            break
                    else:
                        corr_items.append(f"{col}")
                self._block("write", f"Dropped {len(actual_dropped)} highly correlated columns:", corr_items)
        
        final_rows = len(df_clean)
        final_cols = len(df_clean.columns)
        self._log("checkpoint", f"Cleaning complete: {initial_rows}→{final_rows} rows, {initial_cols}→{final_cols} columns")

        return df_clean

    def _clean_dask(self, df, report: dict):
        recs = report.get("recommendations", {})
        
        sample = df.head(50000)
        if hasattr(sample, 'compute'):
            sample = sample.compute()
        
        cols_to_drop = []
        for col in df.columns:
            if sample[col].nunique() == len(sample):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self._block("write", f"Dropped {len(cols_to_drop)} likely ID columns:", cols_to_drop)
            df = df.drop(columns=cols_to_drop)
            sample = sample.drop(columns=cols_to_drop)
        
        dropped_missing = []
        for col in list(df.columns):
            missing_pct = sample[col].isnull().mean()
            if missing_pct > 0.5:
                dropped_missing.append(f"{col}: {missing_pct*100:.1f}% missing")
                df = df.drop(columns=[col])
        
        if dropped_missing:
            self._block("write", f"Dropped {len(dropped_missing)} columns with >50% missing:", dropped_missing)
        
        fill_values = {}
        impute_details = []
        for key, method in recs.items():
            if key.startswith("impute_"):
                col = key.replace("impute_", "")
                if col not in df.columns or col not in sample.columns:
                    continue
                
                if method == "mean":
                    fill_values[col] = sample[col].mean()
                    impute_details.append(f"{col} → filled with mean")
                elif method == "median":
                    fill_values[col] = sample[col].median()
                    impute_details.append(f"{col} → filled with median")
                elif method == "mode":
                    mode_series = sample[col].mode()
                    if not mode_series.empty:
                        fill_values[col] = mode_series[0]
                        impute_details.append(f"{col} → filled with mode")
        
        for col, fill_val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
        
        if impute_details:
            self._block("write", f"Imputed {len(impute_details)} columns:", impute_details)
        
        outlier_info = report.get("outlier_analysis", {})
        outlier_items = []
        for col, info in outlier_info.items():
            if col in df.columns:
                lower, upper = info["bounds"]
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                outlier_items.append(f"{col}: kept values in [{lower:.2f}, {upper:.2f}]")
        
        if outlier_items:
            self._block("write", f"Filtered outliers in {len(outlier_items)} columns:", outlier_items)
        
        drop_cols = report.get("correlation_alert", [])
        corr_details = report.get("correlation_details", [])
        if drop_cols:
            existing_cols = [c for c in drop_cols if c in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
                corr_items = []
                for col in existing_cols:
                    for detail in corr_details:
                        if detail['col2'] == col:
                            corr_items.append(f"{col} ← {detail['correlation']*100:.1f}% correlated with {detail['col1']}")
                            break
                    else:
                        corr_items.append(col)
                self._block("write", f"Dropped {len(existing_cols)} highly correlated columns:", corr_items)
        
        self._log("checkpoint", f"Dask cleaning complete: {len(df.columns)} columns")
        return df
