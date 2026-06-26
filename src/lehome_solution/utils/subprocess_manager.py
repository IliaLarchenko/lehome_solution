"""Subprocess helpers for running training/eval scripts."""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def run_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path | str | None = None,
    label: str = "",
    tail_patterns: list[str] | None = None,
) -> int:
    """Run subprocess with output to log file + background tail of key lines."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    patterns = tail_patterns or []

    log_fh = open(log_path, "w")
    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None,
            stdout=log_fh, stderr=subprocess.STDOUT,
            env=env,
        )

        stop_event = threading.Event()

        def _tail():
            try:
                with open(log_path, "r") as tf:
                    while not stop_event.is_set():
                        line = tf.readline()
                        if line:
                            stripped = line.rstrip()
                            if any(p in stripped for p in patterns):
                                logger.info("[%s] %s", label, stripped)
                        else:
                            stop_event.wait(0.5)
            except Exception:
                pass

        if patterns:
            tail_thread = threading.Thread(target=_tail, daemon=True)
            tail_thread.start()

        proc.wait()
        stop_event.set()
    finally:
        log_fh.close()

    return proc.returncode


def run_subprocess_passthrough(
    cmd: list[str],
    *,
    cwd: Path | str | None = None,
    label: str = "",
    log_path: Path | None = None,
) -> int:
    """Run subprocess with inherited stdio (needed for Isaac Sim Vulkan init).

    If log_path is provided, stdout/stderr are tee'd to the log file while
    still being printed to the terminal.
    """
    logger.info("[%s] Running: %s", label, " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    if log_path is None:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_fh = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        def _tee():
            """Read subprocess output line-by-line, write to both terminal and log."""
            try:
                for raw_line in proc.stdout:
                    line = raw_line.decode("utf-8", errors="replace")
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_fh.write(line)
                    log_fh.flush()
            except Exception:
                pass

        tee_thread = threading.Thread(target=_tee, daemon=True)
        tee_thread.start()
        proc.wait()
        tee_thread.join(timeout=5)
    finally:
        log_fh.close()

    return proc.returncode
