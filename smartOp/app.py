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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    model_type: str

@app.get("/")
async def serve_frontend(): return FileResponse(UI_DIR / "index.html")

@app.get("/styles.css")
async def serve_css(): return FileResponse(UI_DIR / "styles.css")

def _get(session_id):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    from .loader import load_csv_smart, load_json_smart, load_xml_smart, load_sql_smart, load_excel_smart, load_parquet_smart, TEMP_DIR
    from .utils import sanitize_preview, sanitize_value
    import asyncio
    try:
        session_id, fn = str(uuid.uuid4()), file.filename.lower()

        import gc
        for _sid in list(sessions.keys()):
            try: del sessions[_sid]
            except: pass
        sessions.clear()
        gc.collect()

        db_exts = ('.db', '.sqlite', '.sqlite3', '.sql')
        is_db = any(fn.endswith(e) for e in db_exts)
        suffix = next((e for e in db_exts if fn.endswith(e)), None) or '.' + fn.rsplit('.', 1)[-1] if '.' in fn else '.bin'
        temp_path = TEMP_DIR / f"{session_id}{suffix}"
        size = 0
        with open(temp_path, 'wb') as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)
                size += len(chunk)

        def _load():
            if is_db: return load_sql_smart(temp_path, size, session_id)
            if fn.endswith('.csv'): return load_csv_smart(temp_path, size, session_id)
            if fn.endswith('.parquet'): return load_parquet_smart(temp_path, size)
            if size > 2 * 1024 * 1024 * 1024:
                raise ValueError("Files over 2 GB only supported for CSV, SQLite, and Parquet")
            content = temp_path.read_bytes()
            try:
                if fn.endswith('.json'): return load_json_smart(content)
                if fn.endswith('.xml'): return load_xml_smart(content)
                if any(fn.endswith(e) for e in ('.xlsx', '.xls')): return load_excel_smart(content)
                raise ValueError("Unsupported format. Supported: CSV, JSON, XML, SQLite, Excel, Parquet")
            finally:
                del content

        df, meta = await asyncio.to_thread(_load)
        if not meta.get("using_dask") and temp_path.exists():
            try: temp_path.unlink()
            except: pass

        import gc
        state = SessionState()
        state.push(df)
        state.is_dask = meta.get("using_dask", False)
        sessions[session_id] = state
        gc.collect()

        logger = get_logger(session_id, compute_data_hash(df))
        logger.checkpoint(f"File uploaded: {file.filename}")
        logger.read(f"Loaded {meta.get('original_rows', '?')} rows, {len(df.columns)} columns")

        resp = {"message": "File uploaded successfully", "session_id": session_id,
                "preview": sanitize_preview(df), "stats": {k: sanitize_value(v) for k, v in meta.items()}}

        info_parts = []
        if meta.get("converted_from"):
            info_parts.append(f"Converted from {meta['converted_from'].upper()}")
            if meta.get("table_used"): info_parts.append(f"table: {meta['table_used']}")
            if meta.get("sheet_used"): info_parts.append(f"sheet: {meta['sheet_used']}")
        if meta.get("using_dask"):
            resp["info"] = f"Large file ({meta['file_size_mb']:.0f}MB) - Dask out-of-core. {meta['original_rows']:,} rows."
        elif info_parts:
            resp["info"] = " | ".join(info_parts)

        logger.checkpoint("Upload complete")
        return resp
    except HTTPException: raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan/{session_id}")
def scan_data(session_id: str):
    from .scanner import AdvancedDataScanner
    state = _get(session_id)
    logger = get_logger(session_id)
    logger.info("Starting data scan")

    report = AdvancedDataScanner().analyze(state.get_current())
    state.latest_report = report

    missing = report.get('missing_analysis', {})
    outliers = report.get('outlier_analysis', {})
    if missing:
        items = [f"{c}: {p}% missing" for c, p in sorted(missing.items(), key=lambda x: -x[1])[:10]]
        if len(missing) > 10: items.append(f"... and {len(missing)-10} more")
        logger.block("info", f"Missing values in {len(missing)} columns:", items)
    if outliers:
        logger.block("info", f"Outliers in {len(outliers)} columns:",
                     [f"{c}: {i['count']} outliers [{i['bounds'][0]:.2f}, {i['bounds'][1]:.2f}]" for c, i in list(outliers.items())[:5]])
    corr = report.get('correlation_details', [])
    if corr:
        logger.block("warning", "Highly correlated columns:",
                     [f"{d['col1']} <-> {d['col2']}: {d['correlation']*100:.1f}%" for d in corr])
    logger.checkpoint(f"Scan complete: {len(missing)} missing, {len(outliers)} outliers, {len(corr)} correlations")
    return report

@app.post("/auto-clean/{session_id}")
def auto_clean(session_id: str):
    from .cleaner import SmartCleaner
    from .encoder import FeatureEncoder
    from .utils import sanitize_preview
    state, logger = _get(session_id), get_logger(session_id)
    if not state.latest_report: raise HTTPException(status_code=400, detail="Scan data first")
    logger.info("Starting auto-clean pipeline")
    try:
        df = SmartCleaner(logger=logger).clean_data(state.get_current(), state.latest_report)
        df = FeatureEncoder().transform(df, state.latest_report)
        logger.write(f"Encoded: {len(df.columns)} columns")
        state.push(df)
        logger.update_hash(df)
        logger.checkpoint(f"Auto-clean complete: {len(df.columns)} columns")
        # Skip preview for Dask — head() on a complex lazy graph hangs
        preview = [] if state.is_dask else sanitize_preview(df)
        return {"message": "Data cleaned and transformed", "columns": list(df.columns), "preview": preview}
    except Exception as e:
        import traceback; traceback.print_exc()
        logger.error(f"Auto-clean failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/undo/{session_id}")
def undo_action(session_id: str):
    from .utils import sanitize_preview
    state, logger = _get(session_id), get_logger(session_id)
    df_prev = state.undo()
    if df_prev is None:
        logger.info("Nothing to undo")
        return {"message": "Nothing to undo"}
    logger.write("Undo: reverted to previous state")
    logger.update_hash(df_prev)
    return {"message": "Undo successful", "preview": sanitize_preview(df_prev)}

@app.post("/train")
def train_model(req: TrainRequest):
    from .trainer import ModelTrainer
    state, logger = _get(req.session_id), get_logger(req.session_id)
    logger.info(f"Training {req.model_type} on target: {req.target_column}")
    try:
        result, model = ModelTrainer().train(state.get_current(), req.target_column, req.model_type)
        if model:
            state.trained_model = model
            m = result.get('metrics', {})
            logger.checkpoint(f"Model trained: {req.model_type}, score={m.get('accuracy') or m.get('r2_score', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{session_id}")
async def predict_new_data(session_id: str, file: UploadFile = File(...)):
    import pandas as pd, io
    from .cleaner import SmartCleaner
    from .encoder import FeatureEncoder
    state, logger = _get(session_id), get_logger(session_id)
    if not state.trained_model: raise HTTPException(status_code=400, detail="Model not trained yet!")
    content = await file.read()
    try: df_test = pd.read_csv(io.BytesIO(content))
    except: raise HTTPException(status_code=400, detail="Invalid CSV file")
    logger.read(f"Loaded test data: {len(df_test)} rows")
    df_test = SmartCleaner(logger=logger).clean_data(df_test, state.latest_report)
    df_test = FeatureEncoder().transform(df_test, state.latest_report)
    try:
        feats = state.trained_model.feature_names_in_ if hasattr(state.trained_model, 'feature_names_in_') else df_test.columns
        preds = state.trained_model.predict(df_test[feats])
        results = df_test.copy()
        results["predicted_result"] = preds
        logger.checkpoint(f"Prediction complete: {len(preds)} rows")
        return {"message": "Prediction successful", "predictions": results[["predicted_result"]].head(10).to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}. Columns might not match.")

@app.get("/export/{session_id}")
async def export_data(session_id: str):
    from fastapi.responses import StreamingResponse
    from .utils import is_dask_dataframe
    import io
    state = _get(session_id)
    df = state.get_current()
    if df is None: raise HTTPException(status_code=400, detail="No data to export")
    if is_dask_dataframe(df): df = df.compute()
    get_logger(session_id).checkpoint(f"Exporting: {len(df)} rows")
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=cleaned_{session_id[:8]}.csv"})
