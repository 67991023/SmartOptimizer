# Data Loader with Dask support

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
from .utils import is_dask_dataframe, build_metadata
from .config import DASK_THRESHOLD

TEMP_DIR = Path(tempfile.gettempdir()) / "smartoptimizer"
TEMP_DIR.mkdir(exist_ok=True)

def optimize_dtypes(df):
    import numpy as np
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
        
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
        
        elif col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype('uint8')
                elif c_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif c_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype('int32')
    
    return df

def get_memory_usage_mb(df) -> float:
    return df.memory_usage(deep=True).sum() / (1024 * 1024)

def to_pandas(df, sample_size: Optional[int] = None):
    if is_dask_dataframe(df):
        if sample_size:
            total_rows = len(df)
            frac = min(1.0, sample_size / total_rows)
            return df.sample(frac=frac).compute()
        else:
            return df.compute()
    return df

def load_csv_smart(content: bytes, session_id: str = "default") -> Tuple[any, dict]:
    import pandas as pd
    
    file_size = len(content)
    file_size_mb = file_size / (1024 * 1024)
    
    metadata = {
        "file_size_mb": round(file_size_mb, 2),
        "using_dask": False,
        "original_rows": None,
        "memory_mb": None,
        "memory_saved_pct": None
    }
    
    if file_size > DASK_THRESHOLD:
        import dask.dataframe as dd
        
        temp_file = TEMP_DIR / f"{session_id}.csv"
        temp_file.write_bytes(content)
        
        sample_df = pd.read_csv(temp_file, nrows=10000, low_memory=False)
        
        dtype_dict = {}
        for col in sample_df.columns:
            col_dtype = sample_df[col].dtype
            if col_dtype == 'object' or col_dtype.name == 'object':
                dtype_dict[col] = 'object'
            elif 'int' in str(col_dtype):
                dtype_dict[col] = 'float64'
            elif 'float' in str(col_dtype):
                dtype_dict[col] = 'float64'
            else:
                dtype_dict[col] = 'object'
        
        df = dd.read_csv(temp_file, assume_missing=True, dtype=dtype_dict)
        
        try:
            with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline()
                sample_bytes = len(header)
                for i, line in enumerate(f):
                    if i >= 1000:
                        break
                    sample_bytes += len(line)
                avg_bytes_per_row = sample_bytes / (i + 2) if i > 0 else 100
                row_count = int(file_size / avg_bytes_per_row)
        except:
            row_count = df.npartitions * 100000
        
        metadata["using_dask"] = True
        metadata["original_rows"] = row_count
        metadata["partitions"] = df.npartitions
        metadata["memory_mb"] = "lazy (out-of-core)"
        metadata["temp_file"] = str(temp_file)
        
        return df, metadata
    
    df = pd.read_csv(io.BytesIO(content))
    metadata["original_rows"] = len(df)
    
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    
    metadata["memory_mb"] = round(mem_after, 2)
    metadata["memory_saved_pct"] = round((1 - mem_after / mem_before) * 100, 1) if mem_before > 0 else 0
    
    return df, metadata

def load_json_smart(content: bytes) -> Tuple[any, dict]:
    import pandas as pd
    
    file_size_mb = len(content) / (1024 * 1024)
    
    df = pd.read_json(io.BytesIO(content))
    
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    
    mem_saved = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
    metadata = build_metadata(file_size_mb, False, len(df), mem_after, mem_saved)
    
    return df, metadata

def load_xml_smart(content: bytes) -> Tuple[any, dict]:
    import pandas as pd
    import xml.etree.ElementTree as ET
    
    file_size_mb = len(content) / (1024 * 1024)
    
    try:
        root = ET.fromstring(content)
        records = []
        children = list(root)
        if children:
            record_tag = children[0].tag
            for record in root.findall(f".//{record_tag}"):
                row = {}
                row.update(record.attrib)
                for child in record:
                    if len(list(child)) > 0:
                        for subchild in child:
                            row[f"{child.tag}_{subchild.tag}"] = subchild.text
                    else:
                        row[child.tag] = child.text
                records.append(row)
        
        if not records:
            raise ValueError("Could not parse XML structure. No records found.")
        
        df = pd.DataFrame(records)
        
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")
    
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    
    mem_saved = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
    metadata = build_metadata(file_size_mb, False, len(df), mem_after, mem_saved, converted_from="xml")
    
    return df, metadata


def load_sql_smart(content: bytes, session_id: str = "default") -> Tuple[any, dict]:
    import pandas as pd
    import sqlite3
    import gc
    
    file_size_mb = len(content) / (1024 * 1024)
    
    temp_db = TEMP_DIR / f"{session_id}.db"
    temp_db.write_bytes(content)
    
    conn = None
    cursor = None
    df = None
    table_used = None
    tables = []
    
    try:
        conn = sqlite3.connect(str(temp_db))
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
            AND name NOT LIKE '%_node'
            AND name NOT LIKE '%_rowid'
            AND name NOT LIKE '%_parent'
            AND sql NOT LIKE '%VIRTUAL%'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            raise ValueError("No readable tables found in SQLite database")
        
        largest_table = None
        largest_count = 0
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                count = cursor.fetchone()[0]
                if count > largest_count:
                    largest_count = count
                    largest_table = table
            except sqlite3.OperationalError:
                continue
        
        if not largest_table:
            raise ValueError("No readable tables found in SQLite database")
        
        try:
            df = pd.read_sql(f"SELECT * FROM [{largest_table}]", conn)
            table_used = largest_table
        except Exception as e:
            raise ValueError(f"Could not read table '{largest_table}': {e}")
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        
        gc.collect()
        
        import time
        for _ in range(3):
            try:
                if temp_db.exists():
                    temp_db.unlink()
                break
            except PermissionError:
                time.sleep(0.1)
    
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    
    mem_saved = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
    metadata = build_metadata(file_size_mb, False, len(df), mem_after, mem_saved,
                              converted_from="sqlite", table_used=table_used, tables_found=tables)
    
    return df, metadata

def load_excel_smart(content: bytes) -> Tuple[any, dict]:
    import pandas as pd
    
    file_size_mb = len(content) / (1024 * 1024)
    
    excel_file = pd.ExcelFile(io.BytesIO(content))
    sheet_names = excel_file.sheet_names
    
    if len(sheet_names) == 1:
        df = pd.read_excel(excel_file, sheet_name=sheet_names[0])
        sheet_used = sheet_names[0]
    else:
        largest_sheet = None
        largest_count = 0
        for sheet in sheet_names:
            temp_df = pd.read_excel(excel_file, sheet_name=sheet)
            if len(temp_df) > largest_count:
                largest_count = len(temp_df)
                largest_sheet = sheet
        
        df = pd.read_excel(excel_file, sheet_name=largest_sheet)
        sheet_used = largest_sheet
    
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    
    mem_saved = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
    metadata = build_metadata(file_size_mb, False, len(df), mem_after, mem_saved,
                              converted_from="excel", sheet_used=sheet_used, sheets_found=sheet_names)
    
    return df, metadata

def load_parquet_smart(content: bytes) -> Tuple[any, dict]:
    import pandas as pd
    
    file_size_mb = len(content) / (1024 * 1024)
    
    df = pd.read_parquet(io.BytesIO(content))
    
    mem_before = get_memory_usage_mb(df)
    df = optimize_dtypes(df)
    mem_after = get_memory_usage_mb(df)
    
    mem_saved = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
    metadata = build_metadata(file_size_mb, False, len(df), mem_after, mem_saved, converted_from="parquet")
    
    return df, metadata

def cleanup_temp_files(session_id: str):
    temp_file = TEMP_DIR / f"{session_id}.csv"
    if temp_file.exists():
        temp_file.unlink()
    temp_db = TEMP_DIR / f"{session_id}.db"
    if temp_db.exists():
        temp_db.unlink()
