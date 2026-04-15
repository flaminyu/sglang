"""Logger utility for sglang_simulator."""

import logging
import sys


class SimulatorLogger(logging.Logger):
    """Custom logger for sglang simulator."""

    def __init__(self, name: str):
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        
        if not self.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)


_loggers = {}


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name. If None, returns a default logger.
        
    Returns:
        Logger instance.
    """
    if name is None:
        name = "sgl_simulator"
    
    if name not in _loggers:
        _loggers[name] = SimulatorLogger(name)
    
    return _loggers[name]
