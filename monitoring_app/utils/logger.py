"""
Structured logging module for the monitoring application.
Provides console and file logging with JSON formatting.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# Global loggers dictionary
_loggers: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "monitoring",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console output
        file: Enable file output
        json_format: Use JSON formatting for files
        max_bytes: Maximum log file size
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    global _loggers
    
    # Return existing logger if already set up
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if sys.stdout.isatty():
            # Use colored output for terminal
            console_format = ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # Plain output for non-terminal
            console_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "monitoring") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create with default settings
    return setup_logger(name)


class LogContext:
    """Context manager for adding extra data to log records."""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            **kwargs: Extra data to add to log records
        """
        self.logger = logger
        self.extra_data = kwargs
        self._old_factory = None
    
    def __enter__(self):
        """Enter context."""
        self._old_factory = logging.getLogRecordFactory()
        extra_data = self.extra_data
        
        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.extra_data = extra_data
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        logging.setLogRecordFactory(self._old_factory)
        return False


def log_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    level: int = logging.INFO,
    **kwargs
) -> None:
    """
    Log a structured event.
    
    Args:
        logger: Logger instance
        event_type: Type of event (e.g., 'anomaly', 'audio', 'alert')
        message: Log message
        level: Logging level
        **kwargs: Additional event data
    """
    with LogContext(logger, event_type=event_type, **kwargs):
        logger.log(level, message)


def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    unit: str = "",
    **kwargs
) -> None:
    """
    Log a metric value.
    
    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        **kwargs: Additional metric data
    """
    message = f"METRIC | {metric_name}={value}{unit}"
    with LogContext(logger, metric=metric_name, value=value, unit=unit, **kwargs):
        logger.info(message)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    success: bool = True,
    **kwargs
) -> None:
    """
    Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        success: Whether the operation succeeded
        **kwargs: Additional performance data
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"PERF | {operation} | {status} | {duration_ms:.2f}ms"
    
    level = logging.INFO if success else logging.WARNING
    with LogContext(logger, operation=operation, duration_ms=duration_ms, 
                    success=success, **kwargs):
        logger.log(level, message)
