"""Logging configuration for Knowl."""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        level = os.environ.get("KNOWL_LOG_LEVEL", "WARNING").upper()
        handler.setLevel(getattr(logging, level, logging.WARNING))
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(handler.level)
    return logger
