"""
ViStreamASR - Vietnamese Streaming Automatic Speech Recognition Library

A simple and efficient library for real-time Vietnamese speech recognition.
"""

__version__ = "0.1.3"
__author__ = "ViStreamASR Team"

# Handle imports for both installed package and development mode
try:
    from .streaming import StreamingASR
    from .core import ASREngine
except ImportError:
    # Fallback for development mode
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from streaming import StreamingASR
    from core import ASREngine

__all__ = ["StreamingASR", "ASREngine"] 