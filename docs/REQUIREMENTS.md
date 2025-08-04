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

- **Audio Preprocessing**: Process audio chunks before sending to the ASR engine
- **Speech Filtering**: Only forward audio segments classified as speech to the ASR engine
- **Silence Handling**: Efficiently handle silence periods without overloading the ASR engine
- **Buffer Management**: Properly manage audio buffers to prevent data loss
- **State Management**: Maintain VAD state between audio chunks for accurate detection

### 5. Streaming Interface Refactoring ✅ IMPLEMENTED

- **Facade Pattern**: StreamingASR acts as a facade for specialized streamers
- **FileStreamer**: Dedicated component for file-based audio streaming
- **MicrophoneStreamer**: Dedicated component for microphone audio streaming
- **VAD Integration**: Seamless VAD integration through helper methods
- **Configuration Management**: Centralized configuration for all streaming components

### 6. Core Processing Refactoring ✅ IMPLEMENTED

- **ASRState Class**: Separated state management from engine logic
- **Helper Functions**: Extracted \_pad_tensor_list for tensor operations
- **Named Constants**: Replaced magic numbers with named constants for better maintainability
- **Private Method Refactoring**: Broke down process_audio_chunk into smaller, focused methods
- **Improved Error Handling**: Enhanced validation and error reporting throughout processing

### 7. CLI Interface Enhancements ✅ IMPLEMENTED

- **Text Formatting**: Added \_wrap_and_print_text helper for consistent output formatting
- **Configuration Support**: Full integration with hierarchical configuration system
- **VAD Integration**: Support for VAD configuration through CLI arguments
- **Progress Tracking**: Enhanced progress indicators and status reporting
- **Error Handling**: Improved error reporting and recovery mechanisms

### 8. Performance and Optimization ✅ IMPLEMENTED

- **Real-time Processing**: Maintain real-time performance (RTF < 0.4x) on CPU
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Memory Management**: Efficient memory usage for large audio files
- **Model Caching**: Automatic model caching to avoid repeated downloads
- **Chunk Processing**: Optimized chunk-based processing for low latency

### 9. Code Quality and Maintainability ✅ IMPLEMENTED

- **Modular Design**: Clear separation of concerns across components
- **Helper Functions**: Eliminated code duplication through extracted helper functions
- **Named Constants**: Improved code readability with meaningful constant names
- **Documentation**: Comprehensive documentation aligned with code structure
- **Testing**: Comprehensive test coverage for all components

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
- **Accuracy**: High accuracy on speech datasets with minimal false positives/negatives
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
- **Code Organization**: Well-organized codebase with clear responsibilities and interfaces

### 7. Security

- **No Telemetry**: Ensure Silero-VAD's no-telemetry policy is maintained
- **Data Privacy**: No storage or transmission of audio data outside the local system
- **Dependency Security**: Regular updates for security vulnerabilities in dependencies

## Integration Requirements

### 1. API Requirements

- **Model Loading**: Simple API for loading Silero-VAD model
- **Audio Processing**: Stream-friendly API for processing audio chunks
- **State Management**: API for managing VAD state between chunks
- **Configuration**: Parameterized thresholds and timing controls
- **Streaming Interface**: Unified API for file and microphone streaming
- **VAD Integration**: Optional VAD processing with seamless fallback

### 2. Audio Processing Pipeline

- **Chunk Size Compatibility**: Support for ViStreamASR's 640ms chunk size
- **Sample Format**: Handle float32 audio tensors as used by ViStreamASR
- **Resampling**: Automatic handling of different sample rates if needed
- **Normalization**: Proper audio normalization for consistent VAD performance
- **Format Support**: Support for WAV, MP3, FLAC, OGG, M4A and more via torchaudio

### 3. Threshold Tuning

- **Speech Probability Threshold**: Configurable threshold for speech detection (default: 0.5)
- **Minimum Speech Duration**: Configurable minimum speech duration (default: 250ms)
- **Minimum Silence Duration**: Configurable minimum silence duration (default: 250ms)
- **Speech Start/End Padding**: Configurable padding around detected speech segments

### 4. Configuration Integration

- **Hierarchical Settings**: Support for nested configuration structure
- **Environment Variables**: Override with `VISTREAMASR_` prefix support
- **CLI Arguments**: Full override capability for all parameters
- **Validation**: Automatic validation with clear error messages
- **Default Values**: Sensible defaults for all parameters

## Performance Optimization Strategies

### 1. CPU Optimization

- **Single Thread Usage**: Follow Silero-VAD's optimization for single CPU thread
- **Batch Processing**: Consider batching for non-real-time scenarios to improve throughput
- **Quantization**: Leverage Silero-VAD's quantized models for better performance
- **Memory Layout**: Optimize tensor memory layout for cache efficiency
- **Modular Processing**: Leverage refactored helper functions for better performance analysis

### 2. Latency Reduction

- **Pipeline Parallelism**: Overlap VAD processing with ASR preparation
- **Early Detection**: Implement early speech detection for faster ASR triggering
- **Buffer Management**: Optimize buffer sizes to balance latency and efficiency
- **Pre-fetching**: Pre-load VAD model to reduce initialization time
- **Streaming Architecture**: Use optimized streaming components for lower latency

### 3. Resource Management

- **Model Caching**: Cache loaded models to avoid repeated loading
- **Memory Pooling**: Reuse audio buffers to reduce memory allocation overhead
- **Garbage Collection**: Minimize garbage collection impact during processing
- **Power Management**: Optimize for low power consumption in mobile/edge scenarios
- **State Management**: Efficient state handling to reduce memory overhead

## Implementation Status ✅ COMPLETED

### Core VAD Implementation ✅

- **VADProcessor Class**: Implemented in [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py) with full functionality
- **VADASRCoordinator Class**: Implemented in [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py) for seamless ASR integration
- **Configuration Support**: Full parameter configuration with sensible defaults
- **Error Handling**: Comprehensive error handling and graceful degradation

### Streaming Interface Refactoring ✅

- **StreamingASR Class**: Refactored as facade in [`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py)
- **FileStreamer Class**: New dedicated file streaming component in [`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py)
- **MicrophoneStreamer Class**: New dedicated microphone streaming component in [`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py)
- **VAD Integration**: Seamless VAD integration through helper methods
- **Configuration Support**: Full configuration integration with validation

### Core Processing Refactoring ✅

- **ASREngine Class**: Refactored with improved separation of concerns in [`src/vistreamasr/core.py`](src/vistreamasr/core.py)
- **ASRState Class**: New state management component in [`src/vistreamasr/core.py`](src/vistreamasr/core.py)
- **Helper Functions**: Extracted `_pad_tensor_list` in [`src/vistreamasr/core.py`](src/vistreamasr/core.py)
- **Named Constants**: Introduced `FINAL_CHUNK_PADDING_SAMPLES` and `MINIMUM_CHUNK_SIZE_SAMPLES` in [`src/vistreamasr/core.py`](src/vistreamasr/core.py)
- **Private Methods**: Refactored `process_audio_chunk` into smaller, focused methods

### CLI Interface Enhancements ✅

- **CLI Functions**: Enhanced file and microphone processing in [`src/vistreamasr/cli.py`](src/vistreamasr/cli.py)
- **Helper Function**: Added `_wrap_and_print_text` in [`src/vistreamasr/cli.py`](src/vistreamasr/cli.py)
- **Configuration Support**: Full integration with hierarchical configuration
- **VAD Integration**: Support for VAD configuration through CLI
- **Error Handling**: Improved error reporting and recovery

### Integration Points ✅

- **Streaming Interface**: Integrated VAD support in [`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py)
- **CLI Interface**: Full configuration integration in [`src/vistreamasr/cli.py`](src/vistreamasr/cli.py)
- **Core Processing**: Enhanced state management in [`src/vistreamasr/core.py`](src/vistreamasr/core.py)
- **Configuration System**: Package-level exports in [`src/vistreamasr/__init__.py`](src/vistreamasr/__init__.py)

### Testing and Validation ✅

- **Unit Tests**: Comprehensive test suite in [`tests/test_vad_integration.py`](tests/test_vad_integration.py)
- **Integration Tests**: VAD-ASR coordination and workflow testing
- **Performance Tests**: Processing time and memory usage validation
- **Mock Testing**: Mock-based testing for VAD model interactions
- **Refactoring Validation**: Tests for new helper functions and constants

## Configuration Parameters ✅ IMPLEMENTED

### Model Configuration

| Parameter             | Default | Range         | Description                                      |
| --------------------- | ------- | ------------- | ------------------------------------------------ |
| `name`       | "whisper"     | -   | Name of the ASR model to use             |
| `chunk_size_ms`       | 640     | >0   | Audio chunk duration in milliseconds             |
| `stride_ms`       | 320     | >0   | Stride in milliseconds between chunks             |
| `auto_finalize_after` | 15.0    | [0.5, 60.0]   | Maximum duration before auto-finalizing segments |

### VAD Configuration

| Parameter                 | Default | Range         | Description                    |
| ------------------------- | ------- | ------------- | ------------------------------ |
| `enabled`                 | True   | [True, False] | Enable/disable VAD processing  |
| `aggressiveness`             | 3   | [0, 3] | VAD aggressiveness level (0-3)              |
| `frame_size_ms`               | 30     | >0    | Frame size in milliseconds for VAD processing   |
| `min_silence_duration_ms` | 500     | >0            | Minimum silence duration       |
| `speech_pad_ms`           | 100      | >0            | Padding around speech segments |
| `sample_rate`           | 16000      | -            | Audio sample rate for VAD processing |

### Logging Configuration

| Parameter         | Default                         | Range                                 | Description               |
| ----------------- | ------------------------------- | ------------------------------------- | ------------------------- | ------------------------- | --- | ------------------------------ |
| `file_log_level`           | "INFO"                          | ["DEBUG", "INFO", "WARNING", "ERROR"] | Minimum log level for file output         |
| `console_log_level`           | "INFO"                          | ["DEBUG", "INFO", "WARNING", "ERROR"] | Minimum log level for console output         |
| `rotation`          | None     | -                | Log file rotation size    |
| `retention`    | None                            | -                         | Log file retention period       |
| `file_path`       | None               | -                                     | Log file path             |
| `format_string`        | "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"                         | -                                     | Log message format    |
| `enable_colors` | True                            | [True, False]                         | Enable colored output for console logs       |
| `log_to_json` | False                            | [True, False]                         | Enable JSON file logging. |

### CLI Parameters

| Parameter                       | Default | Description                                                |
| ------------------------------- | ------- | ---------------------------------------------------------- |
| `--config`                      | None    | Path to configuration file                                 |
| `--model.chunk-size-ms`                  | 640     | Chunk size in milliseconds (100-2000ms)                    |
| `--model.auto-finalize-after`         | 15.0    | Maximum duration before auto-finalizing segments (seconds) |
| `--vad.enabled`                     | False   | Enable Voice Activity Detection                            |
| `--vad.aggressiveness`               | 3     | VAD speech probability threshold                           |
| `--vad.min-speech-duration-ms`  | 250     | Minimum speech duration in milliseconds                    |
| `--vad.min_silence_duration_ms` | 100     | Minimum silence duration in milliseconds                   |
| `--vad.speech_pad_ms`           | 30      | Padding added to speech segments                           |

## Future Enhancements 🔄 PLANNED

### High Priority

- **Real-time VAD Visualization**: Add visualization tools for VAD decisions
- **Adaptive Thresholding**: Dynamic threshold adjustment based on noise levels
- **Multi-speaker Support**: Extend VAD for multiple speaker detection
- **Enhanced Error Handling**: More sophisticated error recovery mechanisms

### Medium Priority

- **Custom Model Support**: Allow users to specify custom VAD models
- **Advanced Buffering**: Implement more sophisticated buffering strategies
- **Performance Profiling**: Built-in performance monitoring and reporting
- **Streaming Enhancements**: More sophisticated streaming strategies and optimizations

### Low Priority

- **Web Interface**: VAD configuration and monitoring via web interface
- **Mobile Optimization**: Specific optimizations for mobile deployment scenarios
- **Advanced Noise Cancellation**: Integration with advanced noise reduction algorithms
- **Plugin Architecture**: Support for plugins extending functionality