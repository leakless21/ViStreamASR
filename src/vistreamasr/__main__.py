#!/usr/bin/env python3
"""
ViStreamASR Main Entry Point

This module provides the main entry point for the ViStreamASR package.
It allows the package to be executed as a module using `python -m vistreamasr`.
"""

from .cli import main

if __name__ == "__main__":
    main()