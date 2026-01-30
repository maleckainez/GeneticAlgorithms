"""Public logging API for GA experiments."""

from logging import Logger, LoggerAdapter

from .generation_log import log_generation
from .runtime import from_config, from_config_and_layout, from_layout, initialize

LoggerType = Logger | LoggerAdapter | None

__all__ = [
    "initialize",
    "from_config_and_layout",
    "from_config",
    "from_layout",
    "log_generation",
    "LoggerType",
]
