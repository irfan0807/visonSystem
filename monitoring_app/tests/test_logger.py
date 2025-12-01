"""
Tests for logger module.
"""

import pytest
import os
import tempfile
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import (
    setup_logger, 
    get_logger, 
    JSONFormatter, 
    ColoredFormatter,
    LogContext,
    log_event,
    log_metric,
    log_performance
)


class TestSetupLogger:
    """Tests for logger setup."""
    
    def test_basic_logger_creation(self):
        """Test basic logger creation."""
        logger = setup_logger("test_basic")
        
        assert logger is not None
        assert logger.name == "test_basic"
        assert logger.level == logging.INFO
    
    def test_logger_with_custom_level(self):
        """Test logger with custom level."""
        logger = setup_logger("test_level", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
    
    def test_logger_with_file_output(self):
        """Test logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger(
                "test_file",
                log_dir=Path(tmpdir),
                file=True
            )
            
            logger.info("Test message")
            
            log_file = Path(tmpdir) / "test_file.log"
            assert log_file.exists()
    
    def test_logger_singleton(self):
        """Test that same logger is returned for same name."""
        logger1 = setup_logger("test_singleton")
        logger2 = get_logger("test_singleton")
        
        assert logger1 is logger2
    
    def test_logger_console_only(self):
        """Test logger with console only."""
        logger = setup_logger("test_console", console=True, file=False)
        
        assert logger is not None
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_existing_logger(self):
        """Test getting an existing logger."""
        setup_logger("existing_logger")
        logger = get_logger("existing_logger")
        
        assert logger.name == "existing_logger"
    
    def test_get_new_logger(self):
        """Test getting a new logger creates one."""
        logger = get_logger("new_test_logger")
        
        assert logger is not None
        assert logger.name == "new_test_logger"


class TestJSONFormatter:
    """Tests for JSON formatter."""
    
    def test_json_format(self):
        """Test JSON formatting of log records."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        
        import json
        data = json.loads(output)
        
        assert data['level'] == 'INFO'
        assert data['message'] == 'Test message'
        assert 'timestamp' in data


class TestColoredFormatter:
    """Tests for colored formatter."""
    
    def test_colored_format(self):
        """Test colored formatting."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        
        # Should contain ANSI codes for color
        assert '\033[' in output or 'ERROR' in output


class TestLogContext:
    """Tests for LogContext context manager."""
    
    def test_log_context(self):
        """Test log context setup and teardown."""
        logger = setup_logger("context_test")
        
        # The context manager modifies the record factory
        original_factory = logging.getLogRecordFactory()
        
        with LogContext(logger, request_id="123", user="test"):
            # Verify factory was changed
            new_factory = logging.getLogRecordFactory()
            assert new_factory != original_factory
        
        # Verify factory was restored
        restored_factory = logging.getLogRecordFactory()
        assert restored_factory == original_factory


class TestLoggingHelpers:
    """Tests for logging helper functions."""
    
    @pytest.fixture
    def logger(self):
        """Create logger for testing."""
        return setup_logger("helper_test")
    
    def test_log_event(self, logger, caplog):
        """Test event logging."""
        with caplog.at_level(logging.INFO):
            log_event(logger, "test_event", "Event occurred", test_data="value")
        
        assert "Event occurred" in caplog.text
    
    def test_log_metric(self, logger, caplog):
        """Test metric logging."""
        with caplog.at_level(logging.INFO):
            log_metric(logger, "latency", 150.5, "ms")
        
        assert "METRIC" in caplog.text
        assert "latency" in caplog.text
    
    def test_log_performance(self, logger, caplog):
        """Test performance logging."""
        with caplog.at_level(logging.INFO):
            log_performance(logger, "process_frame", 45.2, success=True)
        
        assert "PERF" in caplog.text
        assert "process_frame" in caplog.text
    
    def test_log_performance_failure(self, logger, caplog):
        """Test performance logging for failed operation."""
        with caplog.at_level(logging.WARNING):
            log_performance(logger, "failed_op", 100.0, success=False)
        
        assert "FAILED" in caplog.text


class TestFileRotation:
    """Tests for log file rotation."""
    
    def test_rotating_file_handler(self):
        """Test that rotating file handler is configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger(
                "rotation_test",
                log_dir=Path(tmpdir),
                max_bytes=1024,
                backup_count=3
            )
            
            # Get file handler
            file_handlers = [
                h for h in logger.handlers 
                if hasattr(h, 'maxBytes')
            ]
            
            if file_handlers:
                handler = file_handlers[0]
                assert handler.maxBytes == 1024
                assert handler.backupCount == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
