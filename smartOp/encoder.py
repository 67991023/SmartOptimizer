# Feature Encoder - Handles encoding and scaling

from .utils import is_dask_dataframe as _is_dask

class FeatureEncoder:
    def transform(self, df, report: dict):
        is_dask = _is_dask(df)
        
        if is_dask:
            return self._transform_dask(df, report)
        else:
            return self._transform_pandas(df, report)
    
    def _transform_pandas(self, df, report: dict):
        """Transform pandas DataFrame"""
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
        
        df_encoded = df.copy()
        recs = report.get("recommendations", {})

        # 1. Encoding
        for key, method in recs.items():
            if key.startswith("encode_"):
                col = key.replace("encode_", "")
                if col not in df_encoded.columns:
                    continue

                if method == "onehot":
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
                elif method == "label":
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        # 2. Scaling
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            rec_key = f"scale_{col}"
            method = recs.get(rec_key, "standard")

            scaler = None
            if method == "robust":
                scaler = RobustScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            if scaler:
                data = df_encoded[col].values.reshape(-1, 1)
                df_encoded[col] = scaler.fit_transform(data)

        return df_encoded
    
    def _transform_dask(self, df, report: dict):
        import numpy as np
        import dask.dataframe as dd
        
        recs = report.get("recommendations", {})
        
        # Get actual current columns (convert to set for fast lookup)
        current_cols = set(df.columns)
        
        # Collect columns to encode
        cols_to_encode = []
        for key, method in recs.items():
            if key.startswith("encode_"):
                col = key.replace("encode_", "")
                if col in current_cols:
                    cols_to_encode.append(col)
        
        # Batch categorize all columns at once (much faster)
        if cols_to_encode:
            try:
                # Convert to string first, then categorize all at once
                for col in cols_to_encode:
                    df[col] = df[col].astype(str)
                
                df = df.categorize(columns=cols_to_encode)
                
                # Now convert to codes
                for col in cols_to_encode:
                    df[col] = df[col].cat.codes
            except Exception as e:
                print(f"Warning: Could not encode columns: {e}")
                # Fall back to dropping these columns
                for col in cols_to_encode:
                    if col in df.columns:
                        try:
                            df = df.drop(columns=[col])
                        except:
                            pass
        
        # 2. Scaling - sample-based for speed
        # Get numeric columns from actual current schema
        try:
            sample = df.head(10000)
            if hasattr(sample, 'compute'):
                sample = sample.compute()
            numeric_cols = list(sample.select_dtypes(include=[np.number]).columns)
        except:
            numeric_cols = []
        
        # Compute stats from sample, apply lazily
        for col in numeric_cols:
            if col not in current_cols:
                continue
                
            rec_key = f"scale_{col}"
            method = recs.get(rec_key, "standard")
            
            try:
                col_sample = sample[col].dropna()
                if len(col_sample) == 0:
                    continue
                    
                if method == "robust":
                    median = col_sample.median()
                    q1 = col_sample.quantile(0.25)
                    q3 = col_sample.quantile(0.75)
                    iqr = q3 - q1
                    if iqr != 0:
                        df[col] = (df[col] - median) / iqr
                elif method == "minmax":
                    min_val = col_sample.min()
                    max_val = col_sample.max()
                    range_val = max_val - min_val
                    if range_val != 0:
                        df[col] = (df[col] - min_val) / range_val
                else:
                    mean = col_sample.mean()
                    std = col_sample.std()
                    if std != 0:
                        df[col] = (df[col] - mean) / std
            except Exception as e:
                print(f"Warning: Could not scale column {col}: {e}")
                continue
        
        return df