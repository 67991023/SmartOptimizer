# FastAPI Backend

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from .state import SessionState, sessions
from .logger import get_logger, compute_data_hash

BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"

app = FastAPI(title="Smart Data Optimizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    model_type: str

@app.get("/")
async def serve_frontend():
    return FileResponse(UI_DIR / "index.html")

@app.get("/styles.css")
async def serve_css():
    return FileResponse(UI_DIR / "styles.css")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    from .loader import (
        load_csv_smart, load_json_smart, load_xml_smart, 
        load_sql_smart, load_excel_smart, load_parquet_smart
    )
    from .utils import is_dask_dataframe, sanitize_preview, sanitize_value
    
    try:
        content = await file.read()
        session_id = str(uuid.uuid4())
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df, metadata = load_csv_smart(content, session_id=session_id)
        elif filename.endswith('.json'):
            df, metadata = load_json_smart(content)
        elif filename.endswith('.xml'):
            df, metadata = load_xml_smart(content)
        elif filename.endswith(('.db', '.sqlite', '.sqlite3', '.sql')):
            df, metadata = load_sql_smart(content, session_id=session_id)
        elif filename.endswith(('.xlsx', '.xls')):
            df, metadata = load_excel_smart(content)
        elif filename.endswith('.parquet'):
            df, metadata = load_parquet_smart(content)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: CSV, JSON, XML, SQLite (.db/.sqlite/.sql), Excel (.xlsx/.xls), Parquet"
            )
        
        state = SessionState()
        state.push(df)
        state.is_dask = metadata.get("using_dask", False)
        sessions[session_id] = state
        
        db_hash = compute_data_hash(df)
        logger = get_logger(session_id, db_hash)
        logger.checkpoint(f"File uploaded: {file.filename}")
        logger.read(f"Loaded {metadata.get('original_rows', 'unknown')} rows, {len(df.columns)} columns")
        if metadata.get('converted_from'):
            logger.info(f"Converted from {metadata['converted_from']} format")
        
        preview = sanitize_preview(df)
        sanitized_metadata = {k: sanitize_value(v) for k, v in metadata.items()}
        
        response = {
            "message": "File uploaded successfully",
            "session_id": session_id,
            "preview": preview,
            "stats": sanitized_metadata
        }
        
        if metadata.get("converted_from"):
            response["info"] = f"Converted from {metadata['converted_from'].upper()} format"
            if metadata.get("table_used"):
                response["info"] += f" (table: {metadata['table_used']})"
            if metadata.get("sheet_used"):
                response["info"] += f" (sheet: {metadata['sheet_used']})"
        
        if metadata.get("using_dask"):
            response["info"] = f"Large file ({metadata['file_size_mb']:.0f}MB) - Using Dask out-of-core processing. Full {metadata['original_rows']:,} rows available."
        
        logger.checkpoint("Upload complete - session ready")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan/{session_id}")
def scan_data(session_id: str):
    from .scanner import AdvancedDataScanner
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    df = state.get_current()
    
    logger = get_logger(session_id)
    logger.info("Starting data scan for anomalies")
    
    scanner = AdvancedDataScanner()
    report = scanner.analyze(df)
    state.latest_report = report
    
    missing_analysis = report.get('missing_analysis', {})
    outlier_analysis = report.get('outlier_analysis', {})
    
    if missing_analysis:
        missing_items = [f"{col}: {pct}% missing" 
                        for col, pct in sorted(missing_analysis.items(), key=lambda x: -x[1])[:10]]
        if len(missing_analysis) > 10:
            missing_items.append(f"... and {len(missing_analysis) - 10} more columns")
        logger.block("info", f"Missing values found in {len(missing_analysis)} columns:", missing_items)
    
    if outlier_analysis:
        outlier_items = [f"{col}: {info['count']} outliers (bounds: {info['bounds'][0]:.2f} to {info['bounds'][1]:.2f})"
                        for col, info in list(outlier_analysis.items())[:5]]
        logger.block("info", f"Outliers found in {len(outlier_analysis)} columns:", outlier_items)
    
    corr_details = report.get('correlation_details', [])
    if corr_details:
        corr_items = [f"{detail['col1']} <-> {detail['col2']}: {detail['correlation']*100:.1f}% correlated"
                     for detail in corr_details]
        corr_items.append("Note: Correlated columns share variance - redundant for ML")
        logger.block("warning", "Highly correlated columns (will be dropped):", corr_items)
    
    logger.checkpoint(f"Scan complete: {len(missing_analysis)} cols missing, {len(outlier_analysis)} cols with outliers, {len(corr_details)} correlation pairs")
    
    return report

@app.post("/auto-clean/{session_id}")
def auto_clean(session_id: str):
    from .cleaner import SmartCleaner
    from .encoder import FeatureEncoder
    from .utils import sanitize_preview
    import traceback
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    df = state.get_current()
    report = state.latest_report
    
    if not report:
        raise HTTPException(status_code=400, detail="Please scan data first")
    
    logger = get_logger(session_id)
    logger.info("Starting auto-clean pipeline")

    try:
        cleaner = SmartCleaner(logger=logger)
        df_clean = cleaner.clean_data(df, report)
        
        encoder = FeatureEncoder()
        df_final = encoder.transform(df_clean, report)
        logger.write(f"Encoded features: {len(df_final.columns)} columns after transformation")
        
        state.push(df_final)
        logger.update_hash(df_final)
        
        preview = sanitize_preview(df_final)
        
        logger.checkpoint(f"Auto-clean complete: {len(df_final.columns)} columns")
        return {
            "message": "Data cleaned and transformed",
            "columns": list(df_final.columns),
            "preview": preview
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Auto-clean failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/undo/{session_id}")
def undo_action(session_id: str):
    from .utils import sanitize_preview
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    logger = get_logger(session_id)
    
    df_prev = state.undo()
    if df_prev is None:
        logger.info("Undo attempted but nothing to undo")
        return {"message": "Nothing to undo"}
    
    logger.write("Undo: reverted to previous state")
    logger.update_hash(df_prev)
    
    preview = sanitize_preview(df_prev)
    
    return {"message": "Undo successful", "preview": preview}

@app.post("/train")
def train_model(req: TrainRequest):
    from .trainer import ModelTrainer
    
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[req.session_id]
    df = state.get_current()
    logger = get_logger(req.session_id)
    
    logger.info(f"Training {req.model_type} model on target: {req.target_column}")
    
    trainer = ModelTrainer()
    try:
        result, model = trainer.train(df, req.target_column, req.model_type)
        if model:
            state.trained_model = model
            metrics = result.get('metrics', {})
            score = metrics.get('accuracy') or metrics.get('r2_score') or 'N/A'
            logger.checkpoint(f"Model trained: {req.model_type}, score={score}")
        return result
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{session_id}")
async def predict_new_data(session_id: str, file: UploadFile = File(...)):
    import pandas as pd
    import io
    from .cleaner import SmartCleaner
    from .encoder import FeatureEncoder
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    if not state.trained_model:
        raise HTTPException(status_code=400, detail="Model not trained yet!")
    
    logger = get_logger(session_id)
    logger.info("Starting prediction on new data")

    content = await file.read()
    try:
        df_test = pd.read_csv(io.BytesIO(content))
        logger.read(f"Loaded test data: {len(df_test)} rows")
    except:
        logger.error("Invalid CSV file for prediction")
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    cleaner = SmartCleaner(logger=logger)
    encoder = FeatureEncoder()
    
    df_test_clean = cleaner.clean_data(df_test, state.latest_report)
    df_test_final = encoder.transform(df_test_clean, state.latest_report)

    try:
        model_features = state.trained_model.feature_names_in_ if hasattr(state.trained_model, 'feature_names_in_') else df_test_final.columns
        X_test = df_test_final[model_features]
        predictions = state.trained_model.predict(X_test)
        
        results = df_test.copy()
        results["predicted_result"] = predictions
        
        logger.checkpoint(f"Prediction complete: {len(predictions)} rows predicted")
        
        return {
            "message": "Prediction successful",
            "predictions": results[["predicted_result"]].head(10).to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}. Columns might not match.")

@app.get("/export/{session_id}")
async def export_data(session_id: str):
    from fastapi.responses import StreamingResponse
    from .utils import is_dask_dataframe
    import io
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[session_id]
    df = state.get_current()
    
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")
    
    if is_dask_dataframe(df):
        df = df.compute()
    
    logger = get_logger(session_id)
    logger.checkpoint(f"Exporting data: {len(df)} rows")
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cleaned_data_{session_id[:8]}.csv"}
    )
