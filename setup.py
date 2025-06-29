#!/usr/bin/env python3
"""
Setup script for ViStreamASR package.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the README file
this_directory = Path(__file__).parent
long_description = ""
if (this_directory / "README.md").exists():
    long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from package without executing imports
def get_version():
    version_file = Path("src/__init__.py")
    version_text = version_file.read_text()
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', version_text, re.MULTILINE)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def get_author():
    version_file = Path("src/__init__.py")
    version_text = version_file.read_text()
    author_match = re.search(r'^__author__ = [\'"]([^\'"]*)[\'"]', version_text, re.MULTILINE)
    if author_match:
        return author_match.group(1)
    return "ViStreamASR Team"

setup(
    name="ViStreamASR",
    version=get_version(),
    author=get_author(),
    author_email="nguyenvulebinh@gmail.com",
    description="Vietnamese Streaming Automatic Speech Recognition Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nguyenvulebinh/ViStreamASR",
    package_dir={"ViStreamASR": "src"},
    packages=["ViStreamASR"],
    py_modules=[],  # Don't use py_modules when using packages
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "flashlight-text",
        "librosa",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "vistream-asr=ViStreamASR.cli:cli_main",
            "vstreamasr=ViStreamASR.cli:cli_main",
        ],
    },
    keywords=[
        "speech recognition",
        "asr",
        "vietnamese",
        "streaming",
        "real-time",
        "audio processing",
        "machine learning",
        "deep learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/nguyenvulebinh/ViStreamASR/issues",
        "Source": "https://github.com/nguyenvulebinh/ViStreamASR",
        "Documentation": "https://github.com/nguyenvulebinh/ViStreamASR/blob/main/README.md",
    },
) 