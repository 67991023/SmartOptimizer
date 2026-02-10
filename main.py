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
    🧹 หมอ: รักษาข้อมูล (อัปเกรดเพื่อรับมือ Microsoft Malware Dataset)
    """
    def clean_data(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        df_clean = df.copy()
        recs = report.get("recommendations", {})

        # 1. Drop ID Columns & High Cardinality (แก้ปัญหา MachineIdentifier)
        # ถ้าจำนวน unique เท่ากับจำนวนแถว แสดงว่าเป็น ID -> ลบทิ้ง
        for col in df_clean.columns:
            if df_clean[col].nunique() == len(df_clean):
                print(f"Dropping ID column: {col}")
                df_clean.drop(columns=[col], inplace=True)
                continue # ลบแล้วข้ามไป

        # 2. Drop Columns with > 50% Missing Values (แก้ปัญหา PuaMode)
        limit = len(df_clean) * 0.5
        df_clean.dropna(axis=1, thresh=limit, inplace=True)

        # 3. Drop Duplicates
        if "drop_duplicates" in recs:
            df_clean.drop_duplicates(inplace=True)
        
        # 4. Imputation (เติมค่าว่างส่วนที่เหลือ)
        knn_cols = []
        for key, method in recs.items():
            if key.startswith("impute_"):
                col = key.replace("impute_", "")
                if col not in df_clean.columns: continue

                if method == "mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif method == "mode":
                    if not df_clean[col].mode().empty:
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif method == "knn":
                    # สำหรับ Dataset ใหญ่ๆ KNN ช้ามาก เปลี่ยนเป็น Median ดีกว่าเพื่อความไว
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # 5. Outlier Handling (Removing Rows)
        outlier_info = report.get("outlier_analysis", {})
        for col, info in outlier_info.items():
            if col in df_clean.columns:
                lower, upper = info["bounds"]
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        # 6. Correlation Drop
        drop_cols = report.get("correlation_alert", [])
        if drop_cols:
            df_clean.drop(columns=drop_cols, errors='ignore', inplace=True)

        return df_clean

class FeatureEncoder:
    """
    ✨ สไตลิสต์: แปลงข้อมูลให้ Model เข้าใจ (Encoding & Scaling)
    """
    def transform(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        df_encoded = df.copy()
        recs = report.get("recommendations", {})

        # 1. Encoding
        for key, method in recs.items():
            if key.startswith("encode_"):
                col = key.replace("encode_", "")
                if col not in df_encoded.columns: continue

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

class ModelTrainer:
    """
    🤖 โค้ช: สร้างและเทรนโมเดล
    """
    def train(self, df: pd.DataFrame, target: str, model_type: str):
        if target not in df.columns:
            return {"error": f"Target column '{target}' not found"}, None

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        is_regression = True

        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "random_forest_reg":
            model = RandomForestRegressor(n_estimators=100)
        elif model_type == "logistic_regression":
            model = LogisticRegression()
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
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["r2_score"] = r2_score(y_test, y_pred)
            metrics["type"] = "Regression"
        else:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["type"] = "Classification"

        return {"status": "success", "metrics": metrics}, model

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