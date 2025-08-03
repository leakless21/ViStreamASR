"""
ViStreamASR - Vietnamese Streaming Automatic Speech Recognition Library

A simple and efficient library for real-time Vietnamese speech recognition with Silero-VAD integration.
"""

__version__ = "0.1.4"
__author__ = "ViStreamASR Team"

# Initialize empty exports that will be populated after successful imports
StreamingASR = None
ASREngine = None
VADProcessor = None
VADASRCoordinator = None
ViStreamASRSettings = None
get_settings = None
ModelConfig = None
VADConfig = None
LoggingConfig = None
setup_logging = None
get_logger = None
initialize_logging = None
log_with_symbol = None

# Try to import all modules
try:
    from .streaming import StreamingASR
    from .core import ASREngine
    from .vad import VADProcessor, VADASRCoordinator
    from .config import ViStreamASRSettings, get_settings, ModelConfig, VADConfig, LoggingConfig
    from .logging import setup_logging, get_logger, initialize_logging, log_with_symbol
except ImportError as e:
    # Log the error but don't crash - allow partial imports
    import sys
    print(f"Warning: Some imports failed in ViStreamASR: {e}", file=sys.stderr)
    # Continue with partial imports

__all__ = [
    "StreamingASR", 
    "ASREngine", 
    "VADProcessor", 
    "VADASRCoordinator",
    "ViStreamASRSettings",
    "get_settings",
    "ModelConfig",
    "VADConfig", 
    "LoggingConfig",
    "setup_logging",
    "get_logger",
    "initialize_logging",
    "log_with_symbol"
]