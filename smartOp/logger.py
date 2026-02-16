import csv, hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

LogType = Literal["error", "read", "write", "checkpoint", "info", "warning", "tree"]
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

class SessionLogger:
    def __init__(self, session_id: str, db_hash: Optional[str] = None):
        self.session_id, self.db_hash = session_id, db_hash or "no_data"
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_file = LOGS_DIR / f"{ts}_{session_id[:8]}.csv"
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["timestamp", "db_hash", "type", "message"])

    def log(self, log_type: LogType, message: str):
        ts, h = ("-", "-") if log_type == "tree" else (datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), self.db_hash)
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([ts, h, log_type, message])
        except Exception as e: print(f"Log write error: {e}")

    def update_hash(self, data) -> str:
        try:
            cols = list(data.columns) if hasattr(data, 'columns') else []
            ncols = len(cols)
            # Avoid .shape on Dask (triggers compute for row count)
            nrows = len(data) if not hasattr(data, 'npartitions') else f"dask_{data.npartitions}p"
            h = f"({nrows},{ncols})_{cols}"
            self.db_hash = hashlib.md5(h.encode()).hexdigest()[:12]
        except: self.db_hash = "hash_error"
        return self.db_hash

    def error(self, msg): self.log("error", msg)
    def read(self, msg): self.log("read", msg)
    def write(self, msg): self.log("write", msg)
    def checkpoint(self, msg): self.log("checkpoint", msg)
    def info(self, msg): self.log("info", msg)
    def warning(self, msg): self.log("warning", msg)
    def tree(self, msg): self.log("tree", msg)

    def block(self, log_type: LogType, parent: str, children: list[str]):
        self.log(log_type, parent)
        for c in children: self.tree(c)

_loggers: dict[str, SessionLogger] = {}

def get_logger(session_id: str, db_hash: Optional[str] = None) -> SessionLogger:
    if session_id not in _loggers: _loggers[session_id] = SessionLogger(session_id, db_hash)
    return _loggers[session_id]

def compute_data_hash(df) -> str:
    try:
        shape = (len(df), len(df.columns)) if not hasattr(df, 'compute') else (df.npartitions, len(df.columns))
        head = df.head(3)
        if hasattr(head, 'compute'): head = head.compute()
        sample = head.to_string()[:200]
        return hashlib.md5(f"{shape}_{list(df.columns)}_{sample}".encode()).hexdigest()[:12]
    except: return "hash_error"

def cleanup_logger(session_id: str):
    _loggers.pop(session_id, None)
