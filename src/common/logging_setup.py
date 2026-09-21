"""Extended logging setup for the new Steps 2-6 modules.

Same pattern as src/baseline/logging_setup.py (a plain stdlib logger writing to a file under
outputs/, not just stdout) — kept as a separate module rather than editing the ported one, so
the new work can evolve its logging independently. The one extension: each new module gets its
own log file under outputs/logs/ by default (outputs/logs/language.log,
outputs/logs/graphs.log, ...) instead of everything sharing a single outputs/pipeline.log, since
Steps 2-6 involve enough independent long-running jobs (embedding extraction, graph
construction, multi-seed training sweeps) that per-module logs are easier to tail during a run.
"""

import logging
import os

_DEFAULT_LOG_DIR = "outputs/logs"


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Get a logger that writes to both a file and stdout.

    `name` is typically __name__, matching src/baseline/logging_setup.py's convention.
    `log_file` defaults to outputs/logs/<top-level-package>.log, e.g. a logger created as
    get_logger("src.language.embeddings") writes to outputs/logs/language.log.
    """
    if log_file is None:
        parts = name.split(".")
        module_label = parts[1] if len(parts) > 1 and parts[0] == "src" else parts[0]
        log_file = os.path.join(_DEFAULT_LOG_DIR, f"{module_label}.log")

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Re-calling get_logger(same name) (e.g. across test modules) would otherwise attach
        # duplicate handlers and double-log every message.
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
