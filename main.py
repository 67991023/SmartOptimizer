import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from scipy import stats
import io
import copy
import uuid
from fastapi.middleware.cors import CORSMiddleware

# ==============================================================================
# 🧠 PART 1: LOGIC LAYER (CORE CLASSES)
# ==============================================================================

class AdvancedDataScanner:
    """
    🔍 นักสืบ: สแกนข้อมูลเพื่อหาความผิดปกติ (Diagnosis)
    """
    def analyze(self, df: pd.DataFrame) -> dict:
        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_analysis": {},
            "outlier_analysis": {},
            "correlation_alert": [],
            "recommendations": {}
        }

        # 1. Check Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            report["recommendations"]["drop_duplicates"] = int(dup_count)

        # 2. Analyze Columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in df.columns:
            # --- A. Missing Value Analysis ---
            missing_pct = df[col].isnull().mean()
            if missing_pct > 0:
                report["missing_analysis"][col] = round(missing_pct * 100, 2)
                
                # Logic การเลือกวิธีเติมค่า (Imputation Strategy)
                if col in numeric_cols:
                    skew = df[col].skew()
                    if abs(skew) > 1.0: # ข้อมูลเบ้มาก
                        report["recommendations"][f"impute_{col}"] = "knn" 
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
                count = len(outliers)
                
                if count > 0:
                    report["outlier_analysis"][col] = {
                        "count": count,
                        "method": "IQR",
                        "bounds": [float(lower), float(upper)]
                    }
                    # แนะนำ Robust Scaler ถ้ามี Outlier เยอะ
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

        # 3. Correlation Check (Multi-collinearity)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            if to_drop:
                report["correlation_alert"] = to_drop

        return report

class SmartCleaner:
    """
    Cleaner: Prepares data by handling IDs, missing values, duplicates, and outliers.
    """
    def clean_data(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        df_clean = df.copy()
        recs = report.get("recommendations", {})

        # 1. Drop ID Columns & High Cardinality
        for col in df_clean.columns:
            if df_clean[col].nunique() == len(df_clean):
                print(f"Dropping ID column: {col}")
                df_clean.drop(columns=[col], inplace=True)
                continue

        # 2. Drop Columns with > 50% Missing Values
        limit = len(df_clean) * 0.5
        df_clean.dropna(axis=1, thresh=limit, inplace=True)

        # 3. Imputation (Specific from Report)
        for key, method in recs.items():
            if key.startswith("impute_"):
                col = key.replace("impute_", "")
                if col not in df_clean.columns: continue

                if method == "mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif method == "mode":
                    if not df_clean[col].mode().empty:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                elif method == "knn":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # 4. Final Sweep: Fill ANY remaining NaNs
        
        # 4.1 Fill remaining NUMERIC columns with Median
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
            df_clean[num_cols] = df_clean[num_cols].fillna(0)

        # 4.2 Fill remaining OBJECT/CATEGORY columns with 'Unknown'
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                df_clean[col] = df_clean[col].astype(str).replace('nan', 'Unknown')
                df_clean[col] = df_clean[col].fillna("Unknown")

        # 5. Outlier Handling
        outlier_info = report.get("outlier_analysis", {})
        for col, info in outlier_info.items():
            if col in df_clean.columns:
                lower, upper = info["bounds"]
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        # 6. Final check
        if df_clean.isnull().sum().sum() > 0:
            print(f"Warning: {df_clean.isnull().sum().sum()} NaNs remaining. Dropping rows.")
            df_clean.dropna(inplace=True)

        return df_clean

class FeatureEncoder:
    def transform(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        df_encoded = df.copy()
        recs = report.get("recommendations", {})

        # 1. Manual Encoding
        for key, method in recs.items():
            if key.startswith("encode_"):
                col = key.replace("encode_", "")
                if col not in df_encoded.columns: continue

                if method == "onehot":
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True, dtype=int)
                elif method == "label":
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        # 2. Scaling
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            rec_key = f"scale_{col}"
            method = recs.get(rec_key, "standard")
            scaler = None
            if method == "robust": scaler = RobustScaler()
            elif method == "minmax": scaler = MinMaxScaler()
            else: scaler = StandardScaler()
            
            if scaler:
                try:
                    data = df_encoded[col].values.reshape(-1, 1)
                    df_encoded[col] = scaler.fit_transform(data)
                except Exception as e:
                    pass

        # 3. Auto-Encoding
        remaining_obj_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        if len(remaining_obj_cols) > 0:
            print(f"Auto-Encoding remaining {len(remaining_obj_cols)} columns...")
            for col in remaining_obj_cols:
                try:
                    df_encoded[col] = df_encoded[col].astype('category').cat.codes
                except:
                    df_encoded.drop(columns=[col], inplace=True)
        
        return df_encoded

class ModelTrainer:
    """
    Trainer: Splits data and trains the model.
    Handles regression rounding ONLY during evaluation.
    """
    def train(self, df: pd.DataFrame, target: str, model_type: str):
        if target not in df.columns:
            return {"error": f"Target column '{target}' not found"}, None

        # 1. Clean Target: Drop rows where target is NaN
        df_ready = df.dropna(subset=[target])
        
        X = df_ready.drop(columns=[target])
        y = df_ready[target]

        # 2. Fix Target Type for Classifiers
        # Classifiers need integers (0, 1), not floats (0.0, 1.0)
        if "clf" in model_type or "logistic" in model_type:
            y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        is_regression = False

        if model_type == "linear_regression":
            model = LinearRegression()
            is_regression = True
        elif model_type == "random_forest_reg":
            model = RandomForestRegressor(n_estimators=20, random_state=42)
            is_regression = True
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
            is_regression = False
        elif model_type == "random_forest_clf":
            model = RandomForestClassifier(n_estimators=20, random_state=42)
            is_regression = False
        else:
            return {"error": "Unsupported model type"}, None

        # Train
        try:
            print(f"Training {model_type}...")
            model.fit(X_train, y_train)
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}, None
            
        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = {}
        
        if is_regression:
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["r2_score"] = r2_score(y_test, y_pred)
            metrics["type"] = "Regression"
            
            # --- Round ONLY for Accuracy check ---
            # Round predictions to nearest integer (0 or 1)
            y_pred_rounded = np.round(y_pred).astype(int)
            # Ensure y_test is integer for comparison
            accuracy = accuracy_score(y_test.astype(int), y_pred_rounded)
            metrics["rounded_accuracy"] = accuracy
            print(f"Converted Regression to Class: Accuracy = {accuracy:.4f}")

        else:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["type"] = "Classification"

        return {"status": "success", "metrics": metrics}, model

"""
--- Example ---
if __name__ == "__main__":
    print("--- Phase 0: Reading ---")
    try:
        df = pd.read_csv("train.csv", nrows=5000, low_memory=False) 
        df = df.sample(n=500, random_state=42)
        print("Data Loaded.")
    except FileNotFoundError:
        print("Error: train.csv not found.")
        exit()

    print("--- Phase 0.5: Start ---")
    
    cleaning_report = {
        "recommendations": {
            "impute_RtpStateBitfield": "mode",
            "impute_AVProductsInstalled": "mean",
            "encode_ProductName": "onehot",
            "encode_EngineVersion": "label",
            "encode_AppVersion": "label",
            "encode_CityIdentifier": "label",
            "scale_Census_TotalPhysicalRAM": "standard"
        },
        "outlier_analysis": {
            "Census_TotalPhysicalRAM": {"bounds": [0, 16384]} 
        }
    }

    print("--- Phase 1: Cleaning ---")
    cleaner = SmartCleaner()
    df_clean = cleaner.clean_data(df, cleaning_report)
    print(f"Original Shape: {df.shape} -> Clean Shape: {df_clean.shape}")

    print("\n--- Phase 2: Feature Engineering ---")
    encoder = FeatureEncoder()
    df_encoded = encoder.transform(df_clean, cleaning_report)
    print(f"Encoded Shape: {df_encoded.shape}")

    non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"Warning: Non-numeric columns remaining: {non_numeric}")
    else:
        print("All columns converted to numeric.")

    print("\n--- Phase 3: Model Training ---")
    trainer = ModelTrainer()
    target_col = "HasDetections"
    model_type = "random_forest_clf" 

    if target_col in df_encoded.columns:
        result, model = trainer.train(df_encoded, target=target_col, model_type=model_type)
        
        if "status" in result and result["status"] == "success":
            print(f"Training Success!")
            print(f"Model: {model_type}")
            print(f"Metrics: {result['metrics']}")
        else:
            print(f"Training Failed: {result.get('error')}")
    else:
        print(f"Error: Target column '{target_col}' missing after cleaning!")
"""
# ==============================================================================
# 🗄️ PART 2: STATE MANAGER
# ==============================================================================
class SessionState:
    def __init__(self):
        self.history = [] 
        self.current_step = -1
        self.latest_report = {}
        self.trained_model = None  # เก็บโมเดลที่เทรนเสร็จแล้ว

    def push(self, df, report=None):
        self.history = self.history[:self.current_step + 1]
        self.history.append(df.copy())
        self.current_step += 1
        if report:
            self.latest_report = report

    def get_current(self):
        if self.current_step >= 0:
            return self.history[self.current_step]
        return None

    def undo(self):
        if self.current_step > 0:
            self.current_step -= 1
            return self.history[self.current_step]
        return None

sessions: Dict[str, SessionState] = {}

# ==============================================================================
# 🚀 PART 3: SERVICE LAYER (API)
# ==============================================================================

app = FastAPI(title="Smart Data Optimizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    model_type: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        session_id = str(uuid.uuid4())
        state = SessionState()
        state.push(df)
        sessions[session_id] = state
        
        return {
            "message": "File uploaded successfully",
            "session_id": session_id,
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan/{session_id}")
def scan_data(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[session_id]
    df = state.get_current()
    scanner = AdvancedDataScanner()
    report = scanner.analyze(df)
    state.latest_report = report
    return report

@app.post("/auto-clean/{session_id}")
def auto_clean(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    df = state.get_current()
    report = state.latest_report
    
    if not report:
        raise HTTPException(status_code=400, detail="Please scan data first")

    cleaner = SmartCleaner()
    df_clean = cleaner.clean_data(df, report)
    
    encoder = FeatureEncoder()
    df_final = encoder.transform(df_clean, report)
    
    state.push(df_final)
    
    return {
        "message": "Data cleaned and transformed",
        "columns": list(df_final.columns),
        "preview": df_final.head(5).to_dict(orient="records")
    }

@app.post("/undo/{session_id}")
def undo_action(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[session_id]
    df_prev = state.undo()
    if df_prev is None:
        return {"message": "Nothing to undo"}
    return {"message": "Undo successful", "preview": df_prev.head(5).to_dict(orient="records")}

@app.post("/train")
def train_model(req: TrainRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[req.session_id]
    df = state.get_current()
    
    trainer = ModelTrainer()
    try:
        result, model = trainer.train(df, req.target_column, req.model_type)
        if model:
            state.trained_model = model # Save model to session
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{session_id}")
async def predict_new_data(session_id: str, file: UploadFile = File(...)):
    """
    🔮 ฟังก์ชันทำนายผล: รับไฟล์ Test CSV -> ใช้ Model ที่เทรนไว้ -> ส่งคืนผลลัพธ์
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    if not state.trained_model:
        raise HTTPException(status_code=400, detail="Model not trained yet!")

    # 1. Load Test Data
    content = await file.read()
    try:
        df_test = pd.read_csv(io.BytesIO(content))
    except:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    # 2. Preprocess Test Data 
    # (หมายเหตุ: ในระบบจริงต้องใช้ Scaler/Encoder ตัวเดิมจากตอน Train
    # แต่ใน MVP นี้เราจะใช้ Logic เดิมไปก่อนเพื่อให้รันผ่าน)
    cleaner = SmartCleaner()
    encoder = FeatureEncoder()
    
    # ใช้ Report เดิมจากตอน Train เพื่อให้ Clean แบบเดียวกัน
    df_test_clean = cleaner.clean_data(df_test, state.latest_report)
    df_test_final = encoder.transform(df_test_clean, state.latest_report)

    # 3. Align Columns (กันเหนียว: เติมคอลัมน์ที่ขาดให้ครบ 0)
    # หา Feature ที่ Model ต้องการ (ไม่รวม Target)
    # ใน MVP นี้เราสมมติว่า User ส่ง column มาครบตรงกับตอนเทรน
    
    # 4. Predict
    try:
        # ตรวจสอบว่ามี column แปลกปลอมไหม
        model_features = state.trained_model.feature_names_in_ if hasattr(state.trained_model, 'feature_names_in_') else df_test_final.columns
        
        # เลือกเฉพาะ column ที่ model รู้จัก
        X_test = df_test_final[model_features]
        
        predictions = state.trained_model.predict(X_test)
        
        # 5. Return Result
        results = df_test.copy()
        results["predicted_result"] = predictions
        
        return {
            "message": "Prediction successful",
            "predictions": results[["predicted_result"]].head(10).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}. Columns might not match.")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)