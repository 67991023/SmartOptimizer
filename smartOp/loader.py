import io, tempfile
import pandas as pd
from pathlib import Path
from typing import Tuple
from .utils import finalize_df
from .config import (DASK_THRESHOLD, SQL_ROWS_LARGE, SQL_ROWS_MEDIUM, SQL_ROWS_SMALL,
                     SQL_SIZE_LARGE_MB, SQL_SIZE_MEDIUM_MB, SQL_CHUNK_SIZE)

TEMP_DIR = Path(tempfile.gettempdir()) / "smartoptimizer"
TEMP_DIR.mkdir(exist_ok=True)

def load_csv_smart(file_path, file_size: int = 0, session_id: str = "default") -> Tuple:
    from pathlib import Path
    file_path = Path(file_path)
    size_mb = file_size / (1024 * 1024) if file_size else file_path.stat().st_size / (1024 * 1024)

    if file_size > DASK_THRESHOLD:
        import dask.dataframe as dd
        sample = pd.read_csv(file_path, nrows=1000, low_memory=False)
        dtypes = {c: ('float64' if pd.api.types.is_numeric_dtype(t) else 'object') for c, t in sample.dtypes.items()}
        del sample
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.readline()
                sb = 0
                for i, line in enumerate(f):
                    if i >= 1000: break
                    sb += len(line)
                row_count = int(file_size / (sb / 1000)) if sb else None
        except: row_count = None
        df = dd.read_csv(file_path, assume_missing=True, dtype=dtypes)
        return finalize_df(df, size_mb, original_rows=row_count or df.npartitions * 50000, temp_file=str(file_path))

    return finalize_df(pd.read_csv(file_path, low_memory=False), size_mb)

def load_json_smart(content: bytes) -> Tuple:
    return finalize_df(pd.read_json(io.BytesIO(content)), len(content) / (1024 * 1024))

def load_xml_smart(content: bytes) -> Tuple:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(content)
    children = list(root)
    if not children: raise ValueError("Empty XML")
    records = []
    for rec in root.findall(f".//{children[0].tag}"):
        row = dict(rec.attrib)
        for ch in rec:
            if list(ch):
                for sub in ch: row[f"{ch.tag}_{sub.tag}"] = sub.text
            else: row[ch.tag] = ch.text
        records.append(row)
    if not records: raise ValueError("No records parsed from XML")
    return finalize_df(pd.DataFrame(records), len(content) / (1024 * 1024), converted_from="xml")

def load_sql_smart(db_path, file_size: int = 0, session_id: str = "default") -> Tuple:
    import sqlite3, gc, time
    from pathlib import Path
    temp_db = Path(db_path)
    size_mb = file_size / (1024 * 1024) if file_size else temp_db.stat().st_size / (1024 * 1024)
    conn = sqlite3.connect(str(temp_db))
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        valid = []
        for name, sql in rows:
            if sql and "VIRTUAL TABLE" in sql.upper(): continue
            try:
                count = cursor.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
                valid.append((name, count))
            except: continue
        if not valid: raise ValueError("No readable tables")
        table = max(valid, key=lambda x: x[1])[0]
        row_count = max(valid, key=lambda x: x[1])[1]

        if size_mb > SQL_SIZE_LARGE_MB:     MAX_ROWS = SQL_ROWS_LARGE
        elif size_mb > SQL_SIZE_MEDIUM_MB:   MAX_ROWS = SQL_ROWS_MEDIUM
        else:                                MAX_ROWS = SQL_ROWS_SMALL

        sampled = row_count > MAX_ROWS
        limit = min(row_count, MAX_ROWS)
        query = f"SELECT * FROM [{table}] LIMIT {limit}"

        if limit <= SQL_ROWS_LARGE:
            df = pd.read_sql(query, conn)
        else:
            chunks = list(pd.read_sql(query, conn, chunksize=SQL_CHUNK_SIZE))
            df = pd.concat(chunks, ignore_index=True)
            del chunks
        gc.collect()

        extra = dict(converted_from="sqlite", table_used=table,
                     tables_found=[t[0] for t in valid])
        if sampled:
            extra["note"] = f"Sampled {limit:,} of {row_count:,} rows (file: {size_mb:.0f}MB)"
            extra["original_rows"] = row_count
    finally:
        if conn: conn.close()
        gc.collect()
        if temp_db.exists():
            for _ in range(3):
                try: temp_db.unlink(); break
                except (PermissionError, FileNotFoundError): time.sleep(0.1)
    return finalize_df(df, size_mb, **extra)

def load_excel_smart(content: bytes) -> Tuple:
    xls = pd.ExcelFile(io.BytesIO(content))
    sheets = xls.sheet_names
    if len(sheets) == 1:
        sheet = sheets[0]
    else:
        sheet = max(sheets, key=lambda s: len(pd.read_excel(xls, sheet_name=s, nrows=100)))
    return finalize_df(pd.read_excel(xls, sheet_name=sheet), len(content) / (1024 * 1024),
                       converted_from="excel", sheet_used=sheet, sheets_found=sheets)

def load_parquet_smart(file_path, file_size: int = 0) -> Tuple:
    from pathlib import Path
    file_path = Path(file_path)
    size_mb = file_size / (1024 * 1024) if file_size else file_path.stat().st_size / (1024 * 1024)
    return finalize_df(pd.read_parquet(file_path), size_mb, converted_from="parquet")

def cleanup_temp_files(session_id: str):
    for f in TEMP_DIR.glob(f"{session_id}*"):
        try: f.unlink()
        except: pass
