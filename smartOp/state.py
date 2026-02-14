from typing import Dict
from .utils import is_dask_dataframe as _is_dask

class SessionState:
    def __init__(self):
        self.history, self.current_step = [], -1
        self.latest_report, self.trained_model, self.is_dask = {}, None, False

    def push(self, df, report=None):
        import gc
        # Replace entire history — only keep the latest state
        self.history = [df]
        self.current_step = 0
        if report: self.latest_report = report
        gc.collect()

    def get_current(self):
        return self.history[self.current_step] if self.current_step >= 0 else None

    def undo(self):
        return None  # undo disabled — only 1 state kept to save memory

sessions: Dict[str, SessionState] = {}
