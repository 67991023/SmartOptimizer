from .utils import is_dask_dataframe as _is_dask
from .config import MAX_TRAIN_ROWS, TREE_SAMPLE_CAP, NAN_FILL_LABEL

class ModelTrainer:
    def train(self, df, target: str, model_type: str):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
        import numpy as np

        if target not in df.columns: return {"error": f"Target column '{target}' not found"}, None

        note = None
        if _is_dask(df):
            total = len(df)
            if total > MAX_TRAIN_ROWS:
                df_train = df.sample(frac=MAX_TRAIN_ROWS / total).compute()
                note = f"Sampled {len(df_train):,} of {total:,} rows"
            else:
                df_train = df.compute()
        else:
            cap = TREE_SAMPLE_CAP if "random_forest" in model_type else MAX_TRAIN_ROWS
            if len(df) > cap:
                df_train = df.sample(n=cap, random_state=42).copy()
                note = f"Sampled {cap:,} of {len(df):,} rows"
            else:
                df_train = df.copy()

        for col in df_train.columns:
            if hasattr(df_train[col], 'cat'):
                df_train[col] = df_train[col].cat.codes

        X, y = df_train.drop(columns=[target]), df_train[target]

        non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_num: X = X.drop(columns=non_num)
        if X.empty: return {"error": "No numeric features available for training"}, None

        # Fill NaN in training copy (doesn't touch session data)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        if y.isna().any():
            y = y.fillna(y.median() if np.issubdtype(y.dtype, np.number) else NAN_FILL_LABEL)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {"linear_regression": (lambda: LinearRegression(), True),
                  "random_forest_reg": (lambda: RandomForestRegressor(n_estimators=100), True),
                  "logistic_regression": (lambda: LogisticRegression(max_iter=1000), False),
                  "random_forest_clf": (lambda: RandomForestClassifier(n_estimators=100), False)}
        if model_type not in models: return {"error": "Unsupported model type"}, None

        make_model, is_reg = models[model_type]
        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if is_reg:
            metrics = {"mse": float(mean_squared_error(y_test, y_pred)),
                       "r2_score": float(r2_score(y_test, y_pred)), "type": "Regression"}
        else:
            metrics = {"accuracy": float(accuracy_score(y_test, y_pred)), "type": "Classification"}
        if note: metrics["note"] = note

        return {"status": "success", "metrics": metrics}, model
