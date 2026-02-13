# Data Scanner

from .utils import is_dask_dataframe as _is_dask
from .config import DASK_SAMPLE_SIZE


class AdvancedDataScanner:
    def analyze(self, df) -> dict:
        import numpy as np
        
        is_dask = _is_dask(df)
        
        # For Dask: sample once and analyze the sample (FAST)
        if is_dask:
            # Get estimated row count from partitions (don't compute len())
            estimated_rows = df.npartitions * 100000  # rough estimate
            
            # Sample to pandas for fast analysis
            sample_df = df.head(DASK_SAMPLE_SIZE)
            if hasattr(sample_df, 'compute'):
                sample_df = sample_df.compute()
            
            return self._analyze_pandas(sample_df, estimated_rows, is_dask=True)
        else:
            return self._analyze_pandas(df, len(df), is_dask=False)
    
    def _analyze_pandas(self, df, row_count: int, is_dask: bool) -> dict:
        """Core analysis on pandas DataFrame"""
        import numpy as np
        
        report = {
            "rows": row_count,
            "columns": len(df.columns),
            "missing_analysis": {},
            "outlier_analysis": {},
            "correlation_alert": [],
            "recommendations": {},
            "using_dask": is_dask,
            "sampled": is_dask  # Indicate if results are from sample
        }

        # Check Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            report["recommendations"]["drop_duplicates"] = int(dup_count)

        # Get numeric columns
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        
        for col in df.columns:
            # --- A. Missing Value Analysis ---
            missing_pct = df[col].isnull().mean()
            
            if missing_pct > 0:
                report["missing_analysis"][col] = round(float(missing_pct) * 100, 2)
                
                if col in numeric_cols:
                    skew = df[col].skew()
                    if abs(skew) > 1.0:
                        report["recommendations"][f"impute_{col}"] = "median"
                    else:
                        report["recommendations"][f"impute_{col}"] = "mean"
                else:
                    report["recommendations"][f"impute_{col}"] = "mode"

            # --- B. Outlier Analysis (IQR Method) ---
            if col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    report["outlier_analysis"][col] = {
                        "count": int(outlier_count),
                        "method": "IQR",
                        "bounds": [float(lower), float(upper)]
                    }
                    report["recommendations"][f"scale_{col}"] = "robust"
                else:
                    report["recommendations"][f"scale_{col}"] = "standard"

            # --- C. Cardinality (Encoding) ---
            if col not in numeric_cols:
                unique_count = df[col].nunique()
                
                if unique_count <= 10:
                    report["recommendations"][f"encode_{col}"] = "onehot"
                else:
                    report["recommendations"][f"encode_{col}"] = "label"

        # Correlation Check - Find columns that move together (multicollinearity)
        # Correlation > 0.95 means ~95% of variance is shared - redundant for ML
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Store which columns correlate with what and how much
            corr_details = []
            to_drop = []
            for column in upper.columns:
                for idx, val in upper[column].items():
                    if val > 0.95:
                        corr_details.append({
                            "col1": idx,
                            "col2": column,
                            "correlation": round(float(val), 4)
                        })
                        if column not in to_drop:
                            to_drop.append(column)
            
            if to_drop:
                report["correlation_alert"] = to_drop
                report["correlation_details"] = corr_details

        return report
