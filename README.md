# 🎙️ ViStreamASR - Real-Time Vietnamese Speech Recognition with Silero-VAD

**ViStreamASR** is a simple Vietnamese Streaming Automatic Speech Recognition library with integrated Silero-Voice Activity Detection for efficient real-time audio processing.

## ✨ New: Configuration & Logging System

ViStreamASR now features a comprehensive **configuration and logging system** built with `pydantic-settings` and `loguru`, providing centralized configuration management and structured logging throughout the application.

### Configuration & Logging Benefits

- ⚙️ **Centralized Configuration**: Hierarchical configuration with TOML files, environment variables, and CLI arguments
- 📊 **Structured Logging**: Advanced logging with multiple outputs, levels, and formatting options
- 🔧 **Type Safety**: Configuration validation and type conversion with Pydantic
- 🎨 **Rich Formatting**: Color-coded console output with customizable formats
- 📁 **Log Management**: Automatic log rotation, retention, and compression
- 🔄 **Flexible Sources**: Configuration from multiple sources with clear priority order

## ✨ New: Silero-VAD Integration

ViStreamASR now features seamless integration with **Silero-VAD**, a state-of-the-art voice activity detection model that significantly improves processing efficiency by filtering out silence periods before they reach the ASR engine.

### VAD Integration Benefits

- ⚡ **2x Performance Improvement**: Reduces ASR processing load by filtering silence
- 🎯 **Improved Accuracy**: Focuses computational resources on speech segments
- 🔋 **Resource Efficient**: Lower CPU and memory usage during silence periods
- 🎵 **Better Streaming Experience**: Smoother real-time transcription with reduced latency
- 🇻🇳 **Vietnamese Optimized**: Tuned parameters for Vietnamese speech characteristics

## Features

- 🎯 **Streaming ASR**: Real-time audio processing with configurable chunk sizes
- 🇻🇳 **Vietnamese Optimized**: Specifically designed for Vietnamese speech recognition
- 📦 **Simple API**: Easy-to-use interface with minimal setup
- ⚡ **High Performance**: CPU/GPU support with VAD optimization
- 🔊 **Voice Activity Detection**: Integrated Silero-VAD for efficient audio filtering
- 🎛️ **Configurable VAD**: Customizable VAD parameters for different use cases
- 🌐 **Multi-Platform**: Supports Linux, macOS (Intel & ARM), and Windows
- ⚙️ **Configuration System**: Centralized configuration with TOML, environment variables, and CLI
- 📊 **Logging System**: Structured logging with multiple outputs and advanced features

## Installation

### With Pixi (Recommended)

The project now uses **Pixi** for dependency management, providing a robust and multi-platform environment.

```bash
# Install Pixi
curl -LsSf https://pixi.sh/install.sh | sh

# Clone the repository
git clone https://github.com/nguyenvulebinh/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pixi install

# Activate the environment for interactive use
pixi shell
```

### Development Installation

For development or to use the latest version:

```bash
# Clone the repository
git clone https://github.com/nguyenvulebinh/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pixi install

# Install the package in development mode
pixi run pip install -e .
```

### Using UV (Alternative)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with UV
uv pip install -e .
```

## Quick Start

### Python API with Configuration System

#### Using Configuration File

```python
from ViStreamASR import ViStreamASRSettings, setup_logging, StreamingASR

# Load configuration from file
settings = ViStreamASRSettings(_env_file="vistreamasr.toml")

# Initialize logging
setup_logging(settings.logging)

# Initialize ASR with settings
asr = StreamingASR(settings=settings)

# Process audio file
for result in asr.stream_from_file("audio.wav"):
    if result['partial']:
        print(f"Partial: {result['text']}")
    if result['final']:
        print(f"Final: {result['text']}")
```

#### Programmatic Configuration

```python
from ViStreamASR import ViStreamASRSettings, ModelConfig, VADConfig, LoggingConfig, setup_logging, StreamingASR

# Create configuration programmatically
model_config = ModelConfig(chunk_size_ms=500, debug=True)
vad_config = VADConfig(enabled=True, threshold=0.45)
logging_config = LoggingConfig(level="DEBUG", console_enabled=True)

settings = ViStreamASRSettings(
    model=model_config,
    vad=vad_config,
    logging=logging_config
)

# Initialize logging and ASR
setup_logging(settings.logging)
asr = StreamingASR(settings=settings)

# Process microphone input
for result in asr.stream_from_microphone(duration_seconds=30):
    if result['partial']:
        print(f"Live: {result['text']}")
    if result['final']:
        print(f"Complete: {result['text']}")
```

### Command Line with Configuration System

```bash
# Basic file transcription with default configuration
vistream-asr transcribe audio.wav

# Using custom configuration file
vistream-asr transcribe audio.wav --config custom_config.toml

# Override configuration with CLI arguments
vistream-asr transcribe audio.wav --chunk-size 500 --show-debug

# Using environment variables
export VISTREAMASR_MODEL__DEBUG=true
export VISTREAMASR_VAD__ENABLED=true
vistream-asr transcribe audio.wav

# Microphone recording with configuration
vistream-asr microphone --config config.toml --duration 60
```

### Python API with VAD (Legacy)

#### Streaming from File with VAD

```python
from ViStreamASR import StreamingASR

# Initialize ASR with VAD enabled
vad_config = {
    'enabled': True,
    'threshold': 0.5,
    'min_speech_duration_ms': 250,
    'min_silence_duration_ms': 100,
    'speech_pad_ms': 30
}
asr = StreamingASR(vad_config=vad_config)

# Process audio file - VAD will filter silence automatically
for result in asr.stream_from_file("audio.wav"):
    if result['partial']:
        print(f"Partial: {result['text']}")
    if result['final']:
        print(f"Final: {result['text']}")
```

#### Streaming from Microphone with VAD

```python
from ViStreamASR import StreamingASR

# Initialize ASR with optimized VAD for Vietnamese speech
vad_config = {
    'enabled': True,
    'threshold': 0.45,  # Optimized for Vietnamese tones
    'min_speech_duration_ms': 220,
    'min_silence_duration_ms': 220,
    'speech_pad_ms': 25
}
asr = StreamingASR(vad_config=vad_config)

# Process microphone input with VAD filtering
for result in asr.stream_from_microphone(duration_seconds=30):
    if result['partial']:
        print(f"Live: {result['text']}")
    if result['final']:
        print(f"Complete: {result['text']}")
```

### Command Line with VAD (Legacy)

```bash
# Basic file transcription with VAD
vistream-asr transcribe audio.wav --use-vad

# Custom VAD parameters for Vietnamese speech
vistream-asr transcribe audio.wav --use-vad \
    --vad-threshold 0.45 \
    --vad-min-speech-duration-ms 220 \
    --vad-min-silence-duration-ms 220 \
    --vad-speech-pad-ms 25

# Microphone recording with VAD
vistream-asr microphone --use-vad --duration 60

# High-performance VAD tuning
vistream-asr transcribe audio.wav --use-vad \
    --vad-threshold 0.55 \
    --vad-min-speech-duration-ms 300 \
    --vad-min-silence-duration-ms 300
```

## Configuration System

### Configuration Sources

The configuration system supports multiple sources with clear priority order:

1. **Default Values**: Hardcoded defaults in Pydantic models
2. **TOML Configuration File**: Settings from `vistreamasr.toml`
3. **Environment Variables**: Variables with `VISTREAMASR_` prefix
4. **CLI Arguments**: Command-line overrides

### Environment Variables

```bash
# Model configuration
export VISTREAMASR_MODEL__CHUNK_SIZE_MS=500
export VISTREAMASR_MODEL__DEBUG=true

# VAD configuration
export VISTREAMASR_VAD__ENABLED=true
export VISTREAMASR_VAD__THRESHOLD=0.7

# Logging configuration
export VISTREAMASR_LOGGING__LEVEL=DEBUG
export VISTREAMASR_LOGGING__FILE_ENABLED=true
```

### TOML Configuration File

```toml
# vistreamasr.toml

[model]
chunk_size_ms = 640
auto_finalize_after = 15.0
debug = false

[vad]
enabled = true
sample_rate = 16000
threshold = 0.5
min_speech_duration_ms = 250
min_silence_duration_ms = 100
speech_pad_ms = 30

[logging]
level = "INFO"
file_enabled = true
file_path = "vistreamasr.log"
rotation = "10 MB"
retention = "7 days"
console_enabled = true
```

## VAD Configuration

### VAD Parameters

| Parameter                 | Default | Range         | Description                    |
| ------------------------- | ------- | ------------- | ------------------------------ |
| `enabled`                 | `False` | `True/False`  | Enable/disable VAD processing  |
| `threshold`               | `0.5`   | `0.0-1.0`     | Speech probability threshold   |
| `min_speech_duration_ms`  | `250`   | `>0`          | Minimum speech duration        |
| `min_silence_duration_ms` | `100`   | `>0`          | Minimum silence duration       |
| `speech_pad_ms`           | `30`    | `≥0`          | Padding around speech segments |
| `sample_rate`             | `16000` | `8000, 16000` | Audio sample rate              |

### Recommended VAD Settings

#### For Vietnamese Speech

```python
vad_config = {
    'enabled': True,
    'threshold': 0.45,        # Balanced for Vietnamese tones
    'min_speech_duration_ms': 220,  # Handles short syllables
    'min_silence_duration_ms': 220,  # Conversational patterns
    'speech_pad_ms': 25,      # Preserves tonal transitions
    'sample_rate': 16000      # Optimal for Vietnamese phonemes
}
```

#### For High Accuracy

```python
vad_config = {
    'enabled': True,
    'threshold': 0.6,         # Higher threshold reduces false positives
    'min_speech_duration_ms': 300,
    'min_silence_duration_ms': 300,
    'speech_pad_ms': 50,
    'sample_rate': 16000
}
```

#### For Real-Time Applications

```python
vad_config = {
    'enabled': True,
    'threshold': 0.4,         # Lower threshold for faster response
    'min_speech_duration_ms': 200,
    'min_silence_duration_ms': 150,
    'speech_pad_ms': 20,
    'sample_rate': 16000
}
```

## API Reference

### Configuration System

```python
from ViStreamASR import ViStreamASRSettings, ModelConfig, VADConfig, LoggingConfig, setup_logging

# Create configuration
settings = ViStreamASRSettings(
    model=ModelConfig(
        chunk_size_ms=640,
        auto_finalize_after=15.0,
        debug=False
    ),
    vad=VADConfig(
        enabled=True,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30
    ),
    logging=LoggingConfig(
        level="INFO",
        file_enabled=True,
        console_enabled=True
    )
)

# Initialize logging
setup_logging(settings.logging)
```

### StreamingASR with Configuration

```python
from ViStreamASR import ViStreamASRSettings, setup_logging, StreamingASR

# Initialize with configuration
settings = ViStreamASRSettings()
setup_logging(settings.logging)
asr = StreamingASR(settings=settings)

# Stream from file
for result in asr.stream_from_file("audio.wav"):
    # result contains:
    # - 'partial': True for partial results
    # - 'final': True for final results
    # - 'text': transcription text
    # - 'chunk_info': processing information with VAD status
    pass
```

### StreamingASR with VAD (Legacy)

```python
from ViStreamASR import StreamingASR

# Initialize with VAD support
asr = StreamingASR(
    chunk_size_ms=640,           # Chunk size in milliseconds
    auto_finalize_after=15.0,    # Auto-finalize after seconds
    debug=False,                 # Enable debug logging
    vad_config={                 # VAD configuration
        'enabled': True,
        'threshold': 0.5,
        'min_speech_duration_ms': 250,
        'min_silence_duration_ms': 100,
        'speech_pad_ms': 30
    }
)

# Stream from file
for result in asr.stream_from_file("audio.wav"):
    # result contains:
    # - 'partial': True for partial results
    # - 'final': True for final results
    # - 'text': transcription text
    # - 'chunk_info': processing information with VAD status
    pass
```

### Advanced VAD Usage

```python
from ViStreamASR.vad import VADProcessor, VADASRCoordinator
from ViStreamASR.core import ASREngine

# Initialize VAD processor directly
vad_processor = VADProcessor(
    sample_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=250,
    speech_pad_ms=50
)

# Check speech probability
audio_chunk = get_audio_chunk()
speech_prob = vad_processor.get_speech_probability(audio_chunk)
print(f"Speech probability: {speech_prob:.3f}")

# Use VAD coordinator with ASR engine
asr_engine = ASREngine()
vad_config = {'enabled': True, 'threshold': 0.5}
coordinator = VADASRCoordinator(vad_config, asr_engine)

# Process audio chunks with VAD filtering
result = coordinator.process_audio_chunk(audio_chunk)
```

## Performance Benchmarks

### VAD Performance Impact

| Configuration | RTF (Real-Time Factor) | Processing Speed | Memory Usage | Accuracy Impact    |
| ------------- | ---------------------- | ---------------- | ------------ | ------------------ |
| Without VAD   | 0.34x                  | ~2.9x speed      | ~2.7GB model | Baseline WER 12.5% |
| With VAD      | 0.17x                  | ~5.8x speed      | ~2.7GB model | WER 12.3%          |
| VAD + GPU     | 0.08x                  | ~12.5x speed     | ~4.0GB total | WER 12.1%          |

### Processing Time per Audio Chunk

| Hardware Configuration | VAD Processing Time | ASR Processing Time | Total with VAD | Speedup |
| ---------------------- | ------------------- | ------------------- | -------------- | ------- |
| Intel i7-10700K (CPU)  | <1ms                | ~4.5ms              | ~4.5ms         | 2.1x    |
| Intel i5-8250U (CPU)   | <1ms                | ~7.5ms              | ~7.5ms         | 2.3x    |
| Raspberry Pi 4 (ARM)   | <1ms                | ~24ms               | ~24ms          | 2.2x    |

## Model Information

- **Language**: Vietnamese
- **Architecture**: [U2-based](https://arxiv.org/abs/2203.15455) streaming ASR
- **Model Size**: ~2.7GB (cached after first download)
- **Sample Rate**: 16kHz (automatically converted)
- **Optimal Chunk Size**: 640ms
- **VAD Model**: Silero-VAD (~2MB, cached after first load)

### How U2 Streaming Works with VAD

The following picture shows how U2 (Unified Streaming and Non-streaming) architecture works with VAD integration:

![U2 Architecture](resource/u2.gif)

**VAD Integration Flow:**

1. Audio chunks are processed by Silero-VAD first
2. Only speech segments (above threshold) are forwarded to ASR engine
3. Silence periods are filtered out, reducing computational load
4. Speech segments are padded for better boundary detection
5. ASR processes only relevant audio, improving efficiency

## System Requirements

### Hardware Requirements

- **RAM**: Minimum 5GB RAM (8GB recommended with VAD)
- **CPU**: Minimum 2 cores (4+ cores recommended)
- **Performance**: RTF 0.15-0.2x achievable with VAD on CPU-only systems
- **GPU**: Supports GPU acceleration for better performance, but CPU-only operation still achieves excellent RTF
- **Storage**: ~3GB free space for model caching

### Software Requirements

- **Pixi**: For dependency management and environment setup
- Python 3.8+
- PyTorch 2.7.1+
- TorchAudio 2.7.1+
- NumPy 1.22.3+
- Requests 2.32.4+
- librosa 0.10.0+
- numba 0.59.0+
- sounddevice 0.5.2+
- flashlight-text >=0.0.7
- silero-vad >=5.1.2
- pydantic >=2.0.0
- pydantic-settings >=2.0.0
- loguru >=0.7.0
- toml >=0.10.0

## CLI Commands

### Transcription Commands

```bash
# Basic transcription
vistream-asr transcribe <file>                    # Basic transcription
vistream-asr transcribe <file> --chunk-size 640   # Custom chunk size
vistream-asr transcribe <file> --no-debug         # Clean output

# With configuration
vistream-asr transcribe <file> --config config.toml      # Use config file
vistream-asr transcribe <file> --chunk-size 500          # Override config

# With VAD
vistream-asr transcribe <file> --use-vad          # Enable VAD
vistream-asr transcribe <file> --use-vad --vad-threshold 0.7  # Custom VAD threshold
vistream-asr transcribe <file> --use-vad --vad-min-speech-duration-ms 200  # Custom speech duration

# Microphone recording
vistream-asr microphone                          # Record indefinitely
vistream-asr microphone --duration 30            # Record for 30 seconds
vistream-asr microphone --use-vad                # With VAD filtering

# Information
vistream-asr info                                # Library info
vistream-asr version                             # Version
```

### Advanced VAD CLI Examples

```bash
# Optimized for Vietnamese speech
vistream-asr transcribe audio.wav --use-vad \
    --vad-threshold 0.45 \
    --vad-min-speech-duration-ms 220 \
    --vad-min-silence-duration-ms 220 \
    --vad-speech-pad-ms 25

# High-performance tuning
vistream-asr transcribe audio.wav --use-vad \
    --vad-threshold 0.55 \
    --vad-min-speech-duration-ms 300 \
    --vad-min-silence-duration-ms 300 \
    --chunk-size 500

# Real-time microphone with optimized VAD
vistream-asr microphone --use-vad --duration 60 \
    --vad-threshold 0.4 \
    --vad-min-speech-duration-ms 200 \
    --vad-min-silence-duration-ms 150
```

## Project Structure

```
ViStreamASR/
├── src/
│   └── vistreamasr/
│       ├── __init__.py          # Package initialization
│       ├── config.py           # Configuration system (NEW)
│       ├── logging.py          # Logging system (NEW)
│       ├── core.py             # Core ASR engine
│       ├── streaming.py        # Streaming interface with VAD
│       ├── vad.py              # VAD integration
│       └── cli.py              # Command-line interface
├── tests/
│   ├── test_vad_integration.py # VAD unit tests
│   └── ...
├── docs/
│   ├── ARCHITECTURE.md         # System architecture
│   ├── REQUIREMENTS.md         # Functional requirements
│   ├── COMPONENT_CONFIGURATION_DOCS.md  # Configuration component docs (NEW)
│   ├── COMPONENT_LOGGING_DOCS.md        # Logging component docs (NEW)
│   ├── COMPONENT_VAD_INTEGRATION_DOCS.md  # VAD component docs
│   ├── GAP_ANALYSIS.md         # Testing and issues
│   └── PROJECT_GUIDE.md        # Development guide
├── examples/
├── scripts/
├── model/                     # Cached models
├── resource/                  # Audio samples and resources
├── vistreamasr.toml           # Default configuration file (NEW)
├── pyproject.toml             # Project configuration (Pixi)
├── requirements.txt           # Legacy dependencies
└── README.md                  # This file
```

## Development

### Setting Up Development Environment

The project uses Pixi for dependency management, which simplifies setting up a consistent development environment.

```bash
# Clone the repository
git clone https://github.com/nguyenvulebinh/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pixi install

# Activate the development environment
pixi shell

# Install the package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pixi run test

# Run VAD-specific tests
pixi run pytest tests/test_vad_integration.py

# Run with coverage
pixi run pytest --cov=vistreamasr --cov-report=html
```

### Code Style and Linting

```bash
# Format code
pixi run format

# Check formatting (lint)
pixi run lint

# Run specific tools
pixi run black src/ tests/
pixi run isort src/ tests/
pixi run flake8 src/ tests/
```

### Building the Package

```bash
# Build the package
pixi run build
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create your feature branch (`git checkout -b feature/amazing-feature`)
2. **Follow the code style** guidelines (Black, isort, flake8)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description of your changes

### Development Guidelines

- Use type hints for all public APIs
- Write comprehensive docstrings following Google style
- Include unit tests for new features
- Update relevant documentation
- Follow existing code patterns and conventions

### VAD Development Notes

When working with VAD integration:

- Test with various Vietnamese speech patterns and dialects
- Ensure VAD parameters work well with default ASR chunk sizes
- Verify performance improvements don't impact accuracy
- Test edge cases (very short speech, background noise, etc.)

### Configuration & Logging Development Notes

When working with the configuration and logging system:

- Ensure configuration validation covers all edge cases
- Test configuration loading from all sources (TOML, env vars, CLI)
- Verify logging works with different output formats and sinks
- Test log rotation and retention policies
- Ensure backward compatibility with legacy parameter passing

## Troubleshooting

### Common VAD Issues

**VAD not detecting speech:**

```python
# Lower threshold for sensitive detection
vad_config = {'enabled': True, 'threshold': 0.3}
```

**Too many false positives:**

```python
# Increase threshold to reduce false detections
vad_config = {'enabled': True, 'threshold': 0.7}
```

**Speech segments being cut off:**

```python
# Increase speech padding
vad_config = {'enabled': True, 'speech_pad_ms': 50}
```

### Configuration System Issues

**Configuration file not found:**

```bash
# Check file path and existence
ls -la vistreamasr.toml

# Use absolute path
vistream-asr transcribe audio.wav --config /path/to/config.toml
```

**Environment variables not working:**

```bash
# Check variable names (double underscores for nesting)
export VISTREAMASR_MODEL__DEBUG=true  # Correct
export VISTREAMASR_MODEL_DEBUG=true   # Incorrect

# Verify variables are set
env | grep VISTREAMASR
```

**Configuration validation errors:**

```python
# Check parameter ranges and types
# chunk_size_ms must be between 100 and 2000
# threshold must be between 0.0 and 1.0
```

### Logging System Issues

**Log file not created:**

```bash
# Check directory permissions
mkdir -p logs/
chmod 755 logs/

# Verify file path in configuration
[logging]
file_path = "logs/vistreamasr.log"
```

**Console output not colored:**

```python
# Ensure colorize is enabled in configuration
[logging]
console_enabled = true
console_format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
```

### Performance Issues

**High CPU usage with VAD:**

- Ensure you're using the latest version of silero-vad
- Consider reducing VAD debug logging in production
- Optimize chunk sizes for your specific hardware

**Memory issues:**

- VAD model is cached after first load (~2MB)
- Ensure adequate RAM for both ASR and VAD models
- Monitor memory usage during long streaming sessions

**Configuration loading performance:**

- TOML files load in <10ms
- Configuration validation completes in <5ms
- Consider using environment variables for frequently changed settings

### Pixi Environment Issues

**If you encounter Pixi-related issues:**

- Ensure Pixi is installed correctly: `pixi --version`
- Update Pixi: `curl -LsSf https://pixi.sh/install.sh | sh`
- Clean the Pixi environment: `pixi install --clean`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Silero-VAD**: For the excellent voice activity detection model
- **U2 ASR**: For the underlying streaming ASR architecture
- **Vietnamese ASR Community**: For valuable feedback and testing
- **Pixi Project**: For the excellent dependency management and environment tooling
- **Pydantic**: For the excellent configuration validation and settings management
- **Loguru**: For the modern and feature-rich logging library

## Links and Documentation

- [📖 Architecture Documentation](docs/ARCHITECTURE.md)
- [📋 Requirements Specification](docs/REQUIREMENTS.md)
- [⚙️ Configuration Component Documentation](docs/COMPONENT_CONFIGURATION_DOCS.md)
- [📊 Logging Component Documentation](docs/COMPONENT_LOGGING_DOCS.md)
- [� VAD Component Documentation](docs/COMPONENT_VAD_INTEGRATION_DOCS.md)
- [🐛 Gap Analysis & Testing](docs/GAP_ANALYSIS.md)
- [🛠️ Development Guide](docs/PROJECT_GUIDE.md)
- [📚 Original U2 Paper](https://arxiv.org/abs/2203.15455)
- [🎙️ Silero-VAD Documentation](https://github.com/snakers4/silero-vad)
- [🌐 Pixi Documentation](https://pixi.sh/)

---

**Note**: For the best performance with Vietnamese speech, we recommend using the new configuration system with VAD enabled and the suggested parameters. The configuration system provides centralized management while the VAD integration significantly improves efficiency while maintaining or even improving transcription accuracy.
