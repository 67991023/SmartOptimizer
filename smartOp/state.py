# State management for sessions - lightweight module

from typing import Dict, Any, Optional
from .utils import is_dask_dataframe as _is_dask

class SessionState:
    def __init__(self):
        self.history = [] 
        self.current_step = -1
        self.latest_report = {}
        self.trained_model = None
        self.is_dask = False  # Track if using dask

    def push(self, df, report=None):
        self.history = self.history[:self.current_step + 1]
        # For dask, don't copy (it's lazy anyway)
        if _is_dask(df):
            self.history.append(df)
        else:
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

# Global sessions store
sessions: Dict[str, SessionState] = {}
