# Session Logger - CSV logging per session
import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

LogType = Literal["error", "read", "write", "checkpoint", "info", "warning", "tree"]

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class SessionLogger:
    def __init__(self, session_id: str, db_hash: Optional[str] = None):
        self.session_id = session_id
        self.db_hash = db_hash or "no_data"
        self.start_time = datetime.now()
        
        # Create log file with timestamp name
        timestamp = self.start_time.strftime("%Y-%m-%d-%H-%M-%S")
        self.log_file = LOGS_DIR / f"{timestamp}_{session_id[:8]}.csv"
        
        # Initialize CSV with headers
        self._init_log_file()
    
    def _init_log_file(self):
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "db_hash", "type", "message"])
    
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    def log(self, log_type: LogType, message: str):
        if log_type == "tree":
            timestamp, db_hash = "-", "-"
        else:
            timestamp, db_hash = self._get_timestamp(), self.db_hash
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, db_hash, log_type, message])
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def update_hash(self, data) -> str:
        try:
            if hasattr(data, 'shape'):
                # DataFrame - hash shape and column names
                hash_input = f"{data.shape}_{list(data.columns)}"
            else:
                hash_input = str(type(data))
            
            self.db_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        except Exception:
            self.db_hash = "hash_error"
        
        return self.db_hash
    
    def error(self, message: str): self.log("error", message)
    def read(self, message: str): self.log("read", message)
    def write(self, message: str): self.log("write", message)
    def checkpoint(self, message: str): self.log("checkpoint", message)
    def info(self, message: str): self.log("info", message)
    def warning(self, message: str): self.log("warning", message)
    def tree(self, message: str): self.log("tree", message)
    
    def block(self, log_type: LogType, parent_message: str, children: list[str]):
        self.log(log_type, parent_message)
        for child in children:
            self.tree(child)


_loggers: dict[str, SessionLogger] = {}


def get_logger(session_id: str, db_hash: Optional[str] = None) -> SessionLogger:
    if session_id not in _loggers:
        _loggers[session_id] = SessionLogger(session_id, db_hash)
    return _loggers[session_id]


def compute_data_hash(df) -> str:
    try:
        # For dask, get shape lazily
        if hasattr(df, 'compute'):
            shape = (len(df), len(df.columns))
            cols = list(df.columns)
        else:
            shape = df.shape
            cols = list(df.columns)
        
        # Include a sample of data for uniqueness
        sample_str = ""
        try:
            head = df.head(3)
            if hasattr(head, 'compute'):
                head = head.compute()
            sample_str = head.to_string()[:200]
        except Exception:
            pass
        
        hash_input = f"{shape}_{cols}_{sample_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    except Exception:
        return "hash_error"


def cleanup_logger(session_id: str):
    if session_id in _loggers:
        del _loggers[session_id]
