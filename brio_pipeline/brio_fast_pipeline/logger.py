"""
Automatic timestamped logging for all fast pipeline scripts.

Usage (add two lines at the top of any entry-point script):

    from logger import setup_logging
    setup_logging("infer_sample114")   # creates logs/YYYYMMDD_HHMMSS_infer_sample114.log

After setup_logging() is called, all print() output and stderr go to both
the terminal and the log file simultaneously.
"""
import sys
import os
from pathlib import Path
from datetime import datetime


class _Tee:
    """Writes to both the original stream and a file."""
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh     = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    # Proxy everything else (isatty, fileno, etc.)
    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging(label: str = "run") -> Path:
    """
    Redirect stdout and stderr to a timestamped log file while keeping
    terminal output intact.

    Args:
        label : short description appended to the filename, e.g. "infer_sample114"

    Returns the Path of the log file created.
    """
    from config import LOGS_DIR
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = LOGS_DIR / f"{timestamp}_{label}.log"

    fh = open(log_path, "w", buffering=1, encoding="utf-8")

    # Write a header so it's clear when and how the script was invoked
    header = (
        f"# Log: {log_path.name}\n"
        f"# Started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"# Command: {' '.join(sys.argv)}\n"
        f"# PID: {os.getpid()}\n"
        + "-" * 60 + "\n"
    )
    fh.write(header)

    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)

    # Print after redirect so the path appears in the log too
    print(f"[Log] Writing to {log_path}")
    return log_path
