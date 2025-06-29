# Multi-Platform Support for ViStreamASR

This document explains how ViStreamASR supports multiple operating systems and architectures, and how to build and distribute the library across different platforms.

## Supported Platforms

### Operating Systems
- **Linux** (Ubuntu, CentOS, RHEL, etc.)
  - x86_64 architecture
  - ARM64 architecture (experimental)
- **macOS** 
  - Intel x86_64 (macOS 10.14+)
  - Apple Silicon ARM64 (macOS 11.0+)
- **Windows**
  - x86_64 architecture (Windows 10+)

### Python Versions
- Python 3.8+
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

## Build Strategy

### 1. Modern Python Packaging

We use `pyproject.toml` with setuptools backend for modern, standardized packaging:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
```

### 2. Cross-Platform Dependencies

Platform-specific dependencies are handled automatically:

- **PyTorch/TorchAudio**: Platform-specific wheels from PyPI
- **Audio libraries**: System-specific audio codecs and libraries
- **Binary dependencies**: Managed through pip where possible

### 3. Automated CI/CD

GitHub Actions workflow tests and builds on:
- Ubuntu (latest)
- Windows (latest) 
- macOS (latest, both Intel and ARM64)

## Building Locally

### Quick Build

```bash
# Install build tools
pip install build wheel

# Build both source and wheel distributions
python -m build

# Results in dist/ directory
ls dist/
# ViStreamASR-0.1.0.tar.gz (source)
# ViStreamASR-0.1.0-py3-none-any.whl (wheel)
```

### Using Build Script

We provide a comprehensive build script:

```bash
# Run full build and test process
python scripts/build_multiplatform.py

# Clean only
python scripts/build_multiplatform.py --clean-only
```

The build script:
- ✅ Detects system information
- ✅ Checks build dependencies
- ✅ Runs basic functionality tests
- ✅ Builds source and wheel distributions
- ✅ Tests installation from wheel
- ✅ Provides detailed feedback

## Platform-Specific Considerations

### Linux
- **System dependencies**: `libsndfile1`, `ffmpeg`
- **Installation**: `sudo apt-get install libsndfile1 ffmpeg`
- **GPU support**: CUDA-enabled PyTorch for NVIDIA GPUs

### macOS
- **System dependencies**: `libsndfile`, `ffmpeg`  
- **Installation**: `brew install libsndfile ffmpeg`
- **Apple Silicon**: Native ARM64 support with PyTorch 2.5+
- **GPU support**: Metal Performance Shaders (MPS) backend

### Windows
- **System dependencies**: Handled by pip packages
- **Audio codecs**: Included in Python packages
- **GPU support**: CUDA-enabled PyTorch for NVIDIA GPUs

## CI/CD Workflow

Our GitHub Actions workflow (`build-and-test.yml`) provides:

### 1. Multi-Platform Testing
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.8", "3.9", "3.10", "3.11"]
```

### 2. System Dependency Installation
- Linux: `apt-get install libsndfile1 ffmpeg`
- macOS: `brew install libsndfile ffmpeg`
- Windows: Managed by pip

### 3. Wheel Building
- Builds wheels for each platform
- Tests installation and basic functionality
- Uploads artifacts for distribution

### 4. Automated Publishing
- Publishes to PyPI on release
- Supports multiple platform wheels

## Distribution

### PyPI Distribution
```bash
# Users install with simple command
pip install ViStreamASR

# Works on all supported platforms
```

### From Source
```bash
# Clone repository
git clone https://github.com/nguyenvulebinh/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pip install -r requirements.txt

# Option 1: Direct usage
python test_library.py

# Option 2: Development install
pip install -e .
```

## Testing Multi-Platform Support

### Local Testing
```bash
# Test on current platform
python test_library.py

# Build and test wheel
python scripts/build_multiplatform.py
```

### CI Testing
- Push to GitHub triggers automatic testing
- Tests run on all supported platforms
- Results visible in GitHub Actions

## Performance Expectations

### CPU Performance (No GPU)
- **RTF**: 0.3-0.4x on systems with 5GB+ RAM and 2+ cores
- **Latency**: ~640ms with default chunk size
- **Memory**: ~5GB for model and processing

### GPU Performance  
- **RTF**: 0.1-0.2x with CUDA/MPS acceleration
- **Latency**: ~320ms with optimized settings
- **Memory**: ~3GB GPU memory

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation
```bash
# Force reinstall platform-specific PyTorch
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Audio Dependencies (Linux)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libsndfile1-dev ffmpeg

# CentOS/RHEL
sudo yum install libsndfile-devel ffmpeg
```

#### 3. Permission Issues (macOS)
```bash
# If Homebrew has permission issues
sudo chown -R $(whoami) /usr/local/var/homebrew
```

#### 4. Windows Audio Codecs
```bash
# Install additional audio support
pip install soundfile librosa[soundfile]
```

### Platform-Specific Tips

#### Linux
- Use virtual environments to avoid system package conflicts
- Consider using conda for complex dependency management

#### macOS
- Ensure Xcode command line tools are installed
- Use Homebrew for system dependencies

#### Windows
- Use Windows Subsystem for Linux (WSL) for Linux-compatible builds
- Consider using conda environments for dependency management

## Contributing

When contributing cross-platform changes:

1. **Test locally** on your platform first
2. **Use the build script** to verify builds work
3. **Submit PRs** to trigger CI testing
4. **Check all platforms** pass before merging

## Future Enhancements

- **ARM64 Linux**: Experimental support for ARM64 Linux systems
- **Docker containers**: Pre-built containers for consistent environments  
- **WebAssembly**: Browser-based inference support
- **Mobile platforms**: iOS/Android support via PyTorch Mobile 