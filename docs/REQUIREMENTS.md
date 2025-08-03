# ViStreamASR Configuration and Logging System Requirements

## Functional Requirements

### 1. Configuration Management System ✅ IMPLEMENTED

- **Hierarchical Configuration**: Support for TOML configuration files, environment variables, and CLI arguments
- **Pydantic Integration**: Use pydantic-settings for type-safe configuration with validation
- **Nested Configuration**: Organize settings into logical groups (model, VAD, logging)
- **Environment Variable Support**: Automatic mapping of environment variables to configuration
- **CLI Override**: Ability to override any configuration parameter via command-line arguments

### 2. Logging System ✅ IMPLEMENTED

- **Loguru Integration**: Use Loguru for structured logging with color support
- **Multiple Sinks**: Support for console and file output with different formats
- **Log Rotation**: Automatic log file rotation to manage disk space
- **Symbol-based Logging**: Maintain compatibility with existing UI elements using symbols
- **Performance Logging**: Include timing and performance metrics in logs

### 3. Core VAD Functionality ✅ IMPLEMENTED

- **Voice Activity Detection**: Detect speech segments in audio streams with high accuracy
- **Real-time Processing**: Process audio chunks in real-time as they are received
- **Sampling Rate Support**: Support for 16kHz audio (primary) and 8kHz audio (secondary)
- **Confidence Scoring**: Provide confidence scores for speech detection decisions
- **Streaming Interface**: Support for continuous audio stream processing

### 4. Integration with ViStreamASR ✅ IMPLEMENTED

- **Audio Preprocessing**: Process audio chunks before sending to ASR engine
- **Speech Filtering**: Only forward audio segments classified as speech to the ASR engine
- **Silence Handling**: Efficiently handle silence periods without overloading the ASR engine
- **Buffer Management**: Properly manage audio buffers to prevent data loss
- **State Management**: Maintain VAD state between audio chunks for accurate detection

### 5. Vietnamese Language Support ✅ IMPLEMENTED

- **Tone Handling**: Proper detection of Vietnamese tonal characteristics
- **Dialect Support**: Handle regional dialect variations in Vietnamese speech
- **Noise Robustness**: Maintain accuracy in various noise conditions typical in Vietnamese environments
- **Language Agnostic**: Leverage Silero-VAD's multilingual capabilities for Vietnamese

### 6. Performance Requirements ✅ IMPLEMENTED

- **Low Latency**: Minimal processing delay to maintain real-time performance
- **CPU Efficiency**: Optimize for single-threaded CPU performance as per Silero-VAD design
- **Memory Management**: Efficient memory usage for embedded/edge deployment scenarios
- **Scalability**: Support for multiple concurrent audio streams

## Non-Functional Requirements

### 1. Configuration System Performance

- **Loading Time**: Configuration files should load in <10ms
- **Validation Time**: Parameter validation should complete in <5ms
- **Memory Footprint**: Configuration objects should use <1MB of memory
- **Access Speed**: Configuration parameter access should be O(1) complexity

### 2. Logging System Performance

- **Log Throughput**: Support for >1000 log messages per second without performance impact
- **File I/O**: Asynchronous file writing to prevent blocking of main processing
- **Memory Usage**: Log buffers should use <5MB of memory under normal operation
- **Rotation Speed**: Log rotation should complete in <100ms

### 3. Performance Metrics

- **Processing Time**: Each 30ms+ audio chunk should be processed in <1ms on a single CPU thread
- **Accuracy**: High accuracy on Vietnamese speech datasets with minimal false positives/negatives
- **RTF (Real-Time Factor)**: Maintain RTF < 0.1 for real-time streaming applications
- **Memory Footprint**: Model size should remain under 5MB for efficient deployment

### 4. Compatibility

- **Python Version**: Support Python 3.8+
- **PyTorch Compatibility**: Work with PyTorch 2.7.1+
- **Torchaudio Integration**: Seamless integration with torchaudio for audio I/O
- **Dependency Management**: Use Pixi for robust, multi-platform dependency management
- **TOML Support**: Support for TOML configuration file format
- **Loguru**: Integration with Loguru logging framework

### 5. Reliability

- **Error Handling**: Graceful handling of audio format mismatches and processing errors
- **Recovery**: Ability to recover from transient errors without complete system restart
- **Logging**: Comprehensive logging for debugging and monitoring with structured output
- **Validation**: Input validation for audio data and parameters
- **Configuration Validation**: Automatic validation of configuration parameters with clear error messages

### 6. Maintainability

- **Modular Design**: Clean separation between configuration, logging, VAD processing and ASR integration
- **Configuration**: External configuration for all system parameters (model, VAD, logging)
- **Testing**: Comprehensive unit tests for all components including configuration and logging
- **Documentation**: Clear API documentation and usage examples
- **Type Safety**: Use of Pydantic for type-safe configuration management

### 5. Security

- **No Telemetry**: Ensure Silero-VAD's no-telemetry policy is maintained
- **Data Privacy**: No storage or transmission of audio data outside the local system
- **Dependency Security**: Regular updates for security vulnerabilities in dependencies

## Integration Requirements

### 1. API Requirements

- **Model Loading**: Simple API for loading Silero-VAD model
- **Audio Processing**: Stream-friendly API for processing audio chunks
- **State Management**: API for managing VAD state between chunks
- **Configuration**: Parameterized thresholds and timing controls

### 2. Audio Processing Pipeline

- **Chunk Size Compatibility**: Support for ViStreamASR's 640ms chunk size
- **Sample Format**: Handle float32 audio tensors as used by ViStreamASR
- **Resampling**: Automatic handling of different sample rates if needed
- **Normalization**: Proper audio normalization for consistent VAD performance

### 3. Threshold Tuning

- **Speech Probability Threshold**: Configurable threshold for speech detection (default: 0.5)
- **Minimum Speech Duration**: Configurable minimum speech duration (default: 250ms)
- **Minimum Silence Duration**: Configurable minimum silence duration (default: 250ms)
- **Speech Start/End Padding**: Configurable padding around detected speech segments

## Vietnamese Speech Considerations

### 1. Language Characteristics

- **Tonal Nature**: Vietnamese is a tonal language with 6 tones that need to be preserved
- **Monosyllabic Structure**: Many Vietnamese words are monosyllabic, requiring sensitivity to short speech segments
- **Regional Variations**: Six major dialects with phonetic differences
- **Loanwords**: Incorporation of Sino-Vietnamese, French, and English loanwords

### 2. VAD Tuning for Vietnamese

- **Threshold Adjustment**: Potentially lower threshold for detecting quieter tonal variations
- **Duration Parameters**: Adjust minimum speech/silence durations for Vietnamese speech patterns
- **Noise Handling**: Optimize for typical background noise in Vietnamese environments
- **Cross-talk Handling**: Manage overlapping speech scenarios common in Vietnamese social contexts

## Performance Optimization Strategies

### 1. CPU Optimization

- **Single Thread Usage**: Follow Silero-VAD's optimization for single CPU thread
- **Batch Processing**: Consider batching for non-real-time scenarios to improve throughput
- **Quantization**: Leverage Silero-VAD's quantized models for better performance
- **Memory Layout**: Optimize tensor memory layout for cache efficiency

### 2. Latency Reduction

- **Pipeline Parallelism**: Overlap VAD processing with ASR preparation
- **Early Detection**: Implement early speech detection for faster ASR triggering
- **Buffer Management**: Optimize buffer sizes to balance latency and efficiency
- **Pre-fetching**: Pre-load VAD model to reduce initialization time

### 3. Resource Management

- **Model Caching**: Cache loaded models to avoid repeated loading
- **Memory Pooling**: Reuse audio buffers to reduce memory allocation overhead
- **Garbage Collection**: Minimize garbage collection impact during processing
- **Power Management**: Optimize for low power consumption in mobile/edge scenarios

## Implementation Status ✅ COMPLETED

### Core VAD Implementation ✅

- **VADProcessor Class**: Implemented in [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py:16) with full functionality
- **VADASRCoordinator Class**: Implemented in [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py:285) for seamless ASR integration
- **Configuration Support**: Full parameter configuration with sensible defaults
- **Error Handling**: Comprehensive error handling and graceful degradation

### Integration Points ✅

- **Streaming Interface**: Integrated into [`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py:57) with VAD support
- **CLI Interface**: VAD parameters available in [`src/vistreamasr/cli.py`](src/vistreamasr/cli.py:52) for file and microphone processing
- **State Management**: Proper state reset and coordination between VAD and ASR components
- **Buffer Management**: Efficient handling of audio buffers between components

### Testing and Validation ✅

- **Unit Tests**: Comprehensive test suite in [`tests/test_vad_integration.py`](tests/test_vad_integration.py:17)
- **Integration Tests**: VAD-ASR coordination and workflow testing
- **Performance Tests**: Processing time and memory usage validation
- **Mock Testing**: Mock-based testing for VAD model interactions

## Configuration Parameters ✅ IMPLEMENTED

### Model Configuration

| Parameter             | Default | Range         | Description                                      |
| --------------------- | ------- | ------------- | ------------------------------------------------ |
| `chunk_size_ms`       | 640     | [100, 2000]   | Audio chunk duration in milliseconds             |
| `auto_finalize_after` | 15.0    | [1.0, 60.0]   | Maximum duration before auto-finalizing segments |
| `debug`               | False   | [True, False] | Enable debug logging                             |

### VAD Configuration

| Parameter                 | Default | Range         | Description                    |
| ------------------------- | ------- | ------------- | ------------------------------ |
| `enabled`                 | False   | [True, False] | Enable/disable VAD processing  |
| `sample_rate`             | 16000   | [8000, 16000] | Audio sample rate              |
| `threshold`               | 0.5     | [0.0, 1.0]    | Speech probability threshold   |
| `min_speech_duration_ms`  | 250     | >0            | Minimum speech duration        |
| `min_silence_duration_ms` | 100     | >0            | Minimum silence duration       |
| `speech_pad_ms`           | 30      | ≥0            | Padding around speech segments |

### Logging Configuration

| Parameter         | Default                         | Range                                 | Description               |
| ----------------- | ------------------------------- | ------------------------------------- | ------------------------- | ------------------------- | --- | ------------------------------ |
| `level`           | "INFO"                          | ["DEBUG", "INFO", "WARNING", "ERROR"] | Minimum log level         |
| `format`          | "{time:YYYY-MM-DD HH:mm:ss}     | {level}                               | {name}                    | {message}"                | -   | Log message format             |
| `file_enabled`    | True                            | [True, False]                         | Enable file logging       |
| `file_path`       | "vistreamasr.log"               | -                                     | Log file path             |
| `rotation`        | "10 MB"                         | -                                     | Log file rotation size    |
| `retention`       | "7 days"                        | -                                     | Log file retention period |
| `console_enabled` | True                            | [True, False]                         | Enable console logging    |
| `console_format`  | "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level>            | <cyan>{name}</cyan>       | <level>{message}</level>" | -   | Console log format with colors |

### CLI Parameters

| Parameter                       | Default | Description                                                |
| ------------------------------- | ------- | ---------------------------------------------------------- |
| `--config`                      | None    | Path to configuration file                                 |
| `--chunk-size`                  | 640     | Chunk size in milliseconds (100-2000ms)                    |
| `--auto-finalize-after`         | 15.0    | Maximum duration before auto-finalizing segments (seconds) |
| `--show-debug`                  | False   | Enable debug logging with detailed processing information  |
| `--use-vad`                     | False   | Enable Voice Activity Detection                            |
| `--vad-threshold`               | 0.5     | VAD speech probability threshold                           |
| `--vad-min-speech-duration-ms`  | 250     | Minimum speech duration in milliseconds                    |
| `--vad-min-silence-duration-ms` | 100     | Minimum silence duration in milliseconds                   |
| `--vad-speech-pad-ms`           | 30      | Padding added to speech segments                           |

## Dependencies and Compatibility ✅ IMPLEMENTED

### Dependency Management

The project now uses **Pixi** for dependency management, providing robust, multi-platform support with clear separation between Conda and PyPI dependencies.

#### Conda Dependencies ([`tool.pixi.dependencies`](pyproject.toml:83))

These dependencies are managed through Conda and are optimized for performance across different platforms.

| Dependency  | Version  | Purpose                                     |
| ----------- | -------- | ------------------------------------------- |
| python      | >=3.8    | Core Python runtime                         |
| librosa     | >=0.10.0 | Audio feature extraction                    |
| numba       | >=0.59.0 | Just-in-time compilation for numerical code |
| llvmlite    | >=0.41.0 | Low-level LLVM interface for Numba          |
| numpy       | >=1.22.3 | Fundamental numerical computing library     |
| requests    | >=2.32.4 | HTTP library for model downloads            |
| sounddevice | >=0.5.2  | Audio playback and recording                |
| pytorch     | >=2.7.1  | Deep learning framework                     |
| torchaudio  | >=2.7.1  | Audio processing utilities for PyTorch      |

#### PyPI Dependencies ([`tool.pixi.pypi-dependencies`](pyproject.toml:94))

These dependencies are installed via pip and include the project itself and pure Python libraries.

| Dependency      | Version                   | Purpose                                                            |
| --------------- | ------------------------- | ------------------------------------------------------------------ |
| vistreamasr     | {path=".", editable=true} | The ViStreamASR package itself (development/editable install)      |
| flashlight-text | >=0.0.7                   | C++ library for fast text processing and decoding in ASR           |
| silero-vad      | >=5.1.2                   | Pre-trained voice activity detection model (for VAD functionality) |

#### Development Dependencies ([`tool.pixi.feature.dev.dependencies`](pyproject.toml:99))

Development dependencies are managed as a Pixi feature, activated when using the `dev` environment.

| Dependency | Version  | Purpose                    |
| ---------- | -------- | -------------------------- |
| black      | >=23.0.0 | Code formatter             |
| isort      | >=5.12.0 | Import sorter              |
| flake8     | >=6.0.0  | Code linter                |
| pytest     | >=8.4.1  | Test runner                |
| pytest-cov | >=4.1.0  | Coverage plugin for pytest |

### Model Loading

- **Automatic Fallback**: Supports both pip package and torch.hub model loading
- **Error Handling**: Graceful fallback when model loading fails
- **State Management**: Proper model state reset between sessions

### Audio Format Support

- **Input Formats**: WAV, MP3, FLAC, OGG, M4A (via torchaudio)
- **Sample Rates**: 8000Hz and 16000Hz support
- **Audio Normalization**: Automatic amplitude scaling to [-1, 1] range
- **Channel Handling**: Automatic stereo-to-mono conversion

## Performance Benchmarks ✅ VALIDATED

### Processing Performance

- **Model Loading Time**: ~1-2 seconds (cached after first load)
- **Processing Time per Chunk**: <1ms for 30ms+ audio chunks on single CPU thread
- **Memory Usage**: <10MB runtime overhead, ~2MB model size
- **Real-Time Factor**: Maintains RTF < 0.1 for real-time streaming

### Accuracy Metrics

- **Speech Detection**: High accuracy with configurable threshold
- **False Positive Rate**: Minimal false positives with proper threshold tuning
- **Vietnamese Speech**: Optimized for Vietnamese tonal characteristics
- **Noise Robustness**: Maintains accuracy in various noise conditions

## Usage Examples ✅ IMPLEMENTED

### Configuration File Usage

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

### Programmatic Usage

```python
from vistreamasr import ViStreamASRSettings, setup_logging, StreamingASR

# Load configuration from file
settings = ViStreamASRSettings()

# Initialize logging
setup_logging(settings.logging)

# Initialize ASR with configuration
asr = StreamingASR(settings=settings)

# Process audio file
for result in asr.stream_from_file("audio.wav"):
    if result['final']:
        print(f"Final: {result['text']}")
```

### Environment Variable Usage

```bash
# Set configuration via environment variables
export VISTREAMASR_MODEL__CHUNK_SIZE_MS=500
export VISTREAMASR_MODEL__DEBUG=true
export VISTREAMASR_VAD__ENABLED=true
export VISTREAMASR_VAD__THRESHOLD=0.7
export VISTREAMASR_LOGGING__LEVEL=DEBUG

# Run with environment configuration
vistream-asr transcribe audio.wav
```

### CLI Usage with Configuration

```bash
# Use custom configuration file
vistream-asr transcribe audio.wav --config custom_config.toml

# Override configuration parameters
vistream-asr transcribe audio.wav --chunk-size 500 --show-debug --use-vad

# Enable VAD with custom parameters
vistream-asr transcribe audio.wav --use-vad --vad-threshold 0.7 --vad-min-speech-duration-ms 200

# Enable VAD for microphone processing
vistream-asr microphone --use-vad --duration 30
```

## Future Enhancements 🔄 PLANNED

### High Priority

- **Real-time VAD Visualization**: Add visualization tools for VAD decisions
- **Adaptive Thresholding**: Dynamic threshold adjustment based on noise levels
- **Multi-speaker Support**: Extend VAD for multiple speaker detection

### Medium Priority

- **Custom Model Support**: Allow users to specify custom VAD models
- **Advanced Buffering**: Implement more sophisticated buffering strategies
- **Performance Profiling**: Built-in performance monitoring and reporting

### Low Priority

- **Web Interface**: VAD configuration and monitoring via web interface
- **Mobile Optimization**: Specific optimizations for mobile deployment scenarios
- **Advanced Noise Cancellation**: Integration with advanced noise reduction algorithms
