import logging
import os
from pathlib import Path


def setup_logger(log_file="run_example.log"):
    log_dir = Path(os.environ.get("STRIDERSPD_LOG_DIR", "outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("striderspd")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_dir / log_file, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"))
        logger.addHandler(handler)
    return logger
