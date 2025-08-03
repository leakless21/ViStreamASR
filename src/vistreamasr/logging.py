"""
ViStreamASR Logging Module

This module provides centralized logging configuration using Loguru
with support for console and file output, colored formatting, and log rotation.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger


def setup_logging(settings: Dict[str, Any]) -> None:
    """
    Configure Loguru logging based on provided settings.
    
    Args:
        settings: Dictionary containing logging configuration
    """
    # Remove default handler to start fresh
    logger.remove()
    
    # Get logging settings from the dictionary
    logging_config = settings.get('logging', {})
    
    # Configure console sink with color support
    console_format = logging_config.get('format', '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')
    logger.add(
        sys.stderr,
        level=logging_config.get('level', 'INFO'),
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
            level=logging_config.get('level', 'INFO'),
            format=file_format,
            rotation=logging_config.get('rotation', '10 MB'),
            retention=logging_config.get('retention', '7 days'),
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
    
    # Set global logger level
    logger.level(logging_config.get('level', 'INFO'))


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
        The loaded settings
    """
    try:
        from .config import get_settings
        settings = get_settings(config_path)
        
        # Convert settings to dictionary for setup_logging
        settings_dict = {
            'logging': {
                'level': settings.logging.level,
                'format': settings.logging.format_string,
                'file_path': settings.logging.file_path or "",
                'rotation': settings.logging.rotation,
                'retention': settings.logging.retention
            }
        }
        
        setup_logging(settings_dict)
        
        # Log initialization
        logger.info("Logging system initialized")
        logger.debug(f"Log level: {settings.logging.level}")
        if settings.logging.file_path:
            logger.info(f"Logging to file: {settings.logging.file_path}")
        
        return settings
    except ImportError:
        # Fallback to default settings if config module is not available
        default_settings = {
            'logging': {
                'level': 'INFO',
                'format': '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
                'file_path': "",
                'rotation': '10 MB',
                'retention': '7 days'
            }
        }
        setup_logging(default_settings)
        logger.info("Logging system initialized with default settings")
        return default_settings