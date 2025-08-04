# 🎙️ ViStreamASR - Real-Time Vietnamese Speech Recognition with Silero-VAD

**ViStreamASR** is a simple Vietnamese Streaming Automatic Speech Recognition library with integrated Silero-Voice Activity Detection for efficient real-time audio processing.

## Features

- **Streaming ASR**: Real-time audio processing with configurable chunk sizes
- **VAD Integration**: Silero-VAD for efficient audio filtering
- **Configuration System**: Centralized configuration with TOML, environment variables, and CLI
- **Logging System**: Structured logging with multiple outputs and advanced features
- **Multi-Platform**: Supports Linux, macOS (Intel & ARM), and Windows

## Installation

### With Pixi (Recommended)

```bash
# Install Pixi
curl -LsSf https://pixi.sh/install.sh | sh

# Clone the repository
git clone https://github.com/leakless21/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pixi install

# Activate the environment for interactive use
pixi shell
```

### Using UV (Alternative)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .[dev]
```

## Quick Start

### File Transcription

```bash
vistream-asr transcribe audio.wav
```

```python
from vistreamasr import StreamingASR, ViStreamASRSettings

settings = ViStreamASRSettings()
asr = StreamingASR(settings=settings)

for result in asr.stream_from_file("audio.wav"):
    if result['final']:
        print(f"Final: {result['text']}")
```

### Microphone Transcription

```bash
vistream-asr microphone --duration 30
```

```python
from vistreamasr import StreamingASR, ViStreamASRSettings

settings = ViStreamASRSettings()
asr = StreamingASR(settings=settings)

for result in asr.stream_from_microphone(duration_seconds=30):
    if result['final']:
        print(f"Complete: {result['text']}")
```

## Configuration

ViStreamASR uses a hierarchical configuration system. See [`docs/COMPONENT_CONFIGURATION_DOCS.md`](docs/COMPONENT_CONFIGURATION_DOCS.md) for details.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidelines.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Component Overview](docs/COMPONENT_OVERVIEW.md)
- [CLI Interface](docs/COMPONENT_CLI_INTERFACE_DOCS.md)
- [Configuration System](docs/COMPONENT_CONFIGURATION_DOCS.md)
- [Core Processing](docs/COMPONENT_CORE_PROCESSING_DOCS.md)
- [Logging System](docs/COMPONENT_LOGGING_DOCS.md)
- [Streaming Interface](docs/COMPONENT_STREAMING_INTERFACE_DOCS.md)
- [VAD Integration](docs/COMPONENT_VAD_INTEGRATION_DOCS.md)
- [Gap Analysis](docs/GAP_ANALYSIS.md)
- [Requirements](docs/REQUIREMENTS.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
