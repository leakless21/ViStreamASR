"""
ViStreamASR Logging Module

This module provides centralized logging configuration using Loguru
with support for console and file output, colored formatting, and log rotation.
"""

import sys
import logging
import inspect
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect them to Loguru.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging_module = filename == logging.__file__
            is_frozen_importlib = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging_module or is_frozen_importlib):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def intercept_standard_logging() -> None:
    """
    Intercept standard library logging and redirect it to Loguru.
    """
    # Remove existing handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Add Loguru's intercept handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def setup_logging(settings: Dict[str, Any]) -> None:
    """
    Configure Loguru logging based on provided settings.
    
    Args:
        settings: Dictionary containing logging configuration
    """
    # Remove default handler to start fresh
    logger.remove()

    # Intercept standard library logging
    intercept_standard_logging()
    
    # Get logging settings from the dictionary
    logging_config = settings.get('logging', {})
    
    # Configure console sink with color support
    console_format = logging_config.get('format', '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')
    logger.add(
        sys.stderr,
        level=logging_config.get('console_log_level', 'INFO'),
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Configure file sink if file_path is specified
    file_path = logging_config.get('file_path')
    if file_path:
        # Ensure log directory exists
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File logging without color codes
        file_format = logging_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}')
        # Remove color tags from file format
        file_format = (
            file_format.replace('<green>', '')
            .replace('</green>', '')
            .replace('<level>', '')
            .replace('</level>', '')
            .replace('<cyan>', '')
            .replace('</cyan>', '')
            .replace('<red>', '')
            .replace('</red>', '')
            .replace('<yellow>', '')
            .replace('</yellow>', '')
            .replace('<blue>', '')
            .replace('</blue>', '')
            .replace('<magenta>', '')
            .replace('</magenta>', '')
        )
        
        logger.add(
            file_path,
            level=logging_config.get('file_log_level', 'INFO'),
            format=file_format,
            rotation=logging_config.get('rotation', '10 MB'),
            retention=logging_config.get('retention', '7 days'),
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

        # Configure JSON file sink if log_to_json is True and file_path is specified
        log_to_json = logging_config.get('log_to_json', False)
        if log_to_json and file_path:
            json_file_path = Path(file_path).with_suffix('.json.log')
            logger.add(
                json_file_path,
                level=logging_config.get('file_log_level', 'INFO'),
                serialize=True,
                rotation=logging_config.get('rotation', '10 MB'),
                retention=logging_config.get('retention', '7 days'),
                compression="zip",
                encoding="utf-8",
                enqueue=True,
                backtrace=True,
                diagnose=True
            )
    
    # Set global logger level based on console_log_level
    logger.level(logging_config.get('console_log_level', 'INFO'))


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name. If None, uses 'vistreamasr'.
        
    Returns:
        Loguru logger instance
    """
    if name is None:
        name = 'vistreamasr'
    
    return logger.bind(name=name)


# Convenience functions for different log levels
def debug(message: str, **kwargs) -> None:
    """Log a debug message."""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs) -> None:
    """Log an info message."""
    logger.info(message, **kwargs)


def warning(message: str, **kwargs) -> None:
    """Log a warning message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs) -> None:
    """Log an error message."""
    logger.error(message, **kwargs)


def critical(message: str, **kwargs) -> None:
    """Log a critical message."""
    logger.critical(message, **kwargs)


# Symbol-based logging functions to maintain compatibility with existing code
def log_with_symbol(symbol: str, message: str, level: str = "info") -> None:
    """
    Log a message with a symbol prefix using appropriate log level.
    
    Args:
        symbol: Symbol to prefix the message (e.g., '🎤', '🔧')
        message: Message to log
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
    """
    color_map = {
        '🎤': 'cyan',
        '🔧': 'blue', 
        '✅': 'green',
        '❌': 'red',
        '⚠️': 'yellow',
        '📥': 'blue',
        '🔄': 'blue',
        '🚀': 'magenta',
        '📏': 'yellow',
        '📊': 'cyan',
        '📈': 'green',
        '📝': 'yellow',
        '🧹': 'blue',
        '⏰': 'yellow',
        '🏁': 'green',
        '⏭️': 'blue',
        '📁': 'cyan',
        '🎵': 'magenta',
        '📖': 'blue',
        '🏠': 'cyan',
        '🧠': 'magenta',
        '🔊': 'cyan',
        '⏹️': 'red',
        '💾': 'green',
        '⏱️': 'yellow',
        '📡': 'blue',
        '🌐': 'blue',
        '🔍': 'yellow',
        '📋': 'cyan',
        '🎯': 'red',
        '🛡️': 'green',
        '⚡': 'yellow',
        '🌟': 'yellow',
        '🎨': 'magenta',
        '🔑': 'yellow',
        '📦': 'cyan',
        '🚨': 'red',
        '📢': 'blue',
        '💡': 'yellow',
        '🎪': 'magenta',
        '🎭': 'magenta',
        '🎨': 'magenta',
        '🎯': 'red',
        '🏆': 'yellow',
        '🎊': 'magenta',
        '🎉': 'magenta',
        '🎈': 'magenta',
        '🎁': 'magenta',
        '🎀': 'magenta',
    }
    
    color = color_map.get(symbol, 'white')
    
    # Use the appropriate logging function based on level
    log_func = getattr(logger, level.lower(), logger.info)
    
    # Log with color formatting
    logger.bind(symbol=symbol).opt(colors=True).info(
        f"<{color}>{symbol}</{color}> {message}"
    )


# Initialize logging with default settings
def initialize_logging(config_path: Optional[Path] = None):
    """
    Initialize logging with settings from configuration file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        The loaded settings, or a default ViStreamASRSettings object if config fails to load.
    """
    try:
        from .config import get_settings, ViStreamASRSettings
        settings = get_settings(config_path)
        
        # Convert settings to dictionary for setup_logging
        settings_dict = {
            'logging': {
                'console_log_level': settings.logging.console_log_level,
                'file_log_level': settings.logging.file_log_level,
                'format': settings.logging.format_string,
                'file_path': settings.logging.file_path or "",
                'rotation': settings.logging.rotation,
                'retention': settings.logging.retention,
                'log_to_json': settings.logging.log_to_json
            }
        }
        
        setup_logging(settings_dict)
        
        # Log initialization
        logger.info("Logging system initialized")
        logger.debug(f"Console log level: {settings.logging.console_log_level}")
        logger.debug(f"File log level: {settings.logging.file_log_level}")
        if settings.logging.file_path:
            logger.info(f"Logging to file: {settings.logging.file_path}")
        
        return settings
    except ImportError:
        # Fallback to default settings if config module is not available
        # We need to import ViStreamASRSettings here to create a default instance.
        # However, if .config import failed, importing ViStreamASRSettings directly might also fail.
        # As a robust fallback, we'll try to import ViStreamASRSettings directly.
        try:
            from .config import ViStreamASRSettings
            default_settings = ViStreamASRSettings()
        except ImportError:
            # If direct import also fails, it means there's a larger issue with the config module.
            # We should raise an error or handle it more gracefully.
            # For now, we'll let it propagate, as this indicates a broken environment.
            raise RuntimeError(
                "Failed to import configuration module. "
                "Please ensure 'src.vistreamasr.config' is accessible."
            )

        default_settings_dict = {
            'logging': {
                'console_log_level': default_settings.logging.console_log_level,
                'file_log_level': default_settings.logging.file_log_level,
                'format': default_settings.logging.format_string,
                'file_path': default_settings.logging.file_path or "",
                'rotation': default_settings.logging.rotation,
                'retention': default_settings.logging.retention,
                'log_to_json': default_settings.logging.log_to_json
            }
        }
        setup_logging(default_settings_dict)
        logger.info("Logging system initialized with default settings")
        return default_settings