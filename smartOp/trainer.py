# Model Trainer

from .utils import is_dask_dataframe as _is_dask
from .config import MAX_TRAIN_ROWS


class ModelTrainer:
    def train(self, df, target: str, model_type: str):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
        
        is_dask = _is_dask(df)
        
        if target not in df.columns:
            return {"error": f"Target column '{target}' not found"}, None
        
        # Convert dask to pandas (with sampling if needed)
        if is_dask:
            total_rows = len(df)
            if total_rows > MAX_TRAIN_ROWS:
                # Sample for training
                frac = MAX_TRAIN_ROWS / total_rows
                df_train = df.sample(frac=frac).compute()
                sampled = True
                sample_info = f"Sampled {len(df_train):,} of {total_rows:,} rows for training"
            else:
                df_train = df.compute()
                sampled = False
                sample_info = None
        else:
            df_train = df
            sampled = False
            sample_info = None

        X = df_train.drop(columns=[target])
        y = df_train[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        is_regression = True

        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "random_forest_reg":
            model = RandomForestRegressor(n_estimators=100)
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
            is_regression = False
        elif model_type == "random_forest_clf":
            model = RandomForestClassifier(n_estimators=100)
            is_regression = False
        else:
            return {"error": "Unsupported model type"}, None

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = {}
        if is_regression:
            metrics["mse"] = float(mean_squared_error(y_test, y_pred))
            metrics["r2_score"] = float(r2_score(y_test, y_pred))
            metrics["type"] = "Regression"
        else:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["type"] = "Classification"
        
        if sampled:
            metrics["note"] = sample_info

        return {"status": "success", "metrics": metrics}, model
