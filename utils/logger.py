"""Logging utilities."""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "omniverifier", log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "omniverifier") -> logging.Logger:
    """Get existing logger."""
    return logging.getLogger(name)
