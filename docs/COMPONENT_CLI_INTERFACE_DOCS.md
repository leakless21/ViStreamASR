# CLI Interface Component Documentation

## Overview

The CLI Interface component provides command-line functionality for the ViStreamASR system, handling both file transcription and microphone streaming with support for Voice Activity Detection (VAD) and hierarchical configuration management.

## Component Responsibilities

- **Command-line argument parsing and validation**
- **Configuration file loading and merging**
- **Audio file transcription with progress tracking**
- **Real-time microphone streaming**
- **VAD integration for speech detection**
- **Text formatting and output presentation**
- **Error handling and user feedback**

## Related Classes and Files

### Primary Files

- **[`src/vistreamasr/cli.py`](src/vistreamasr/cli.py)** - Main CLI interface implementation
- **[`src/vistreamasr/config.py`](src/vistreamasr/config.py)** - Configuration management
- **[`src/vistreamasr/__init__.py`](src/vistreamasr/__init__.py)** - CLI function exports

### Key Classes and Functions

| Function/Class         | Location                                                         | Purpose                          |
| ---------------------- | ---------------------------------------------------------------- | -------------------------------- |
| `transcribe_file_streaming`      | [`src/vistreamasr/cli.py:70`](src/vistreamasr/cli.py:70)         | Main file transcription function |
| `transcribe_microphone_streaming`   | [`src/vistreamasr/cli.py:157`](src/vistreamasr/cli.py:157)       | Microphone streaming processing  |
| `_wrap_and_print_text` | [`src/vistreamasr/cli.py:45`](src/vistreamasr/cli.py:45)         | Text formatting helper function  |
| `setup_logging`        | [`src/vistreamasr/logging.py:60`](src/vistreamasr/logging.py:60) | Logging configuration            |

## Detailed Implementation

### Core Functions

#### `transcribe_file_streaming` Function

**Location**: [`src/vistreamasr/cli.py:70`](src/vistreamasr/cli.py:70)

**Purpose**: Main function for transcribing audio files with comprehensive configuration support and VAD integration.

**Parameters**:

- `audio_file` (str): Path to the audio file to transcribe
- `settings` (ViStreamASRSettings): Configuration object

**Key Features**:

- **Hierarchical Configuration**: Supports TOML files, environment variables, and CLI arguments
- **VAD Integration**: Optional Voice Activity Detection for improved accuracy
- **Progress Tracking**: Real-time progress indicators with timing information
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Debug Mode**: Detailed logging for troubleshooting

**Implementation Details**:

```python
def transcribe_file_streaming(
    audio_file: str,
    settings: ViStreamASRSettings,
) -> str:
    """Transcribe an audio file using ViStreamASR with VAD support."""
    # Setup logging with configuration
    setup_logging(settings.logging.model_dump())

    # Initialize ASR with full configuration
    asr = StreamingASR(settings=settings)

    # Process with progress tracking
    start_time = time.time()
    total_chunks = 0

    for result in asr.stream_from_file(audio_file):
        if result['final']:
            _wrap_and_print_text(f"Final: {result['text']}")

    # Performance summary
    processing_time = time.time() - start_time
    _wrap_and_print_text(
        f"Processing completed in {processing_time:.2f}s "
    )

    # Return final transcription
    return asr.engine.state.current_transcription
```

#### `transcribe_microphone_streaming` Function

**Location**: [`src/vistreamasr/cli.py:157`](src/vistreamasr/cli.py:157)

**Purpose**: Handle real-time microphone streaming with VAD support and configuration management.

**Parameters**:

- `duration_seconds` (float): Duration to record from microphone
- `settings` (ViStreamASRSettings): Configuration object

**Key Features**:

- **Real-time Processing**: Low-latency microphone streaming
- **VAD Integration**: Optional Voice Activity Detection for speech filtering
- **Duration Control**: Precise recording duration management
- **Error Handling**: Graceful handling of microphone and audio errors
- **Debug Information**: Detailed logging for troubleshooting

**Implementation Details**:

```python
def transcribe_microphone_streaming(
    duration_seconds: float,
    settings: ViStreamASRSettings,
) -> None:
    """Process audio from microphone with real-time streaming."""
    # Setup logging
    setup_logging(settings.logging.model_dump())

    # Initialize ASR with configuration
    asr = StreamingASR(settings=settings)

    # Record from microphone
    _wrap_and_print_text(f"Recording from microphone for {duration_seconds}s...")

    try:
        for result in asr.stream_from_microphone(duration_seconds):
            if result['final']:
                _wrap_and_print_text(f"Final: {result['text']}")
    except KeyboardInterrupt:
        _wrap_and_print_text("Recording stopped by user")
    except Exception as e:
        _wrap_and_print_text(f"Error during microphone processing: {e}")
        raise
    finally:
        _wrap_and_print_text("Microphone processing completed")
```

### Helper Functions

#### `_wrap_and_print_text` Helper Function

**Location**: [`src/vistreamasr/cli.py:45`](src/vistreamasr/cli.py:45)

**Purpose**: Enhanced text formatting and printing with consistent styling and progress tracking.

**Parameters**:

- `text` (str): Text to format and print
- `width` (int): The maximum width of each line (default: 80)

**Key Features**:

- **Consistent Formatting**: Standardized text output format
- **Progress Tracking**: Integration with progress indicators
- **Escape Sequences**: Control over carriage return and line feed behavior
- **User Experience**: Clean, readable output format

**Implementation Details**:

```python
def _wrap_and_print_text(text: str, width: int = 80) -> None:
    """Wrap and print text with consistent formatting.

    Args:
        text: Text to format and print
        width: The maximum width of each line (default: 80)
    """
    # ... implementation ...
```

## Configuration Integration

### CLI Parameter Support

The CLI interface supports comprehensive configuration through multiple sources:

| Parameter                     | Environment Variable                       | CLI Argument                    | Description                      |
| ----------------------------- | ------------------------------------------ | ------------------------------- | -------------------------------- |
| `model.chunk_size_ms`               | `VISTREAMASR_MODEL__CHUNK_SIZE_MS`         | `--model.chunk-size-ms`                  | Audio chunk duration             |
| `model.auto_finalize_after`         | `VISTREAMASR_MODEL__AUTO_FINALIZE_AFTER`   | `--model.auto-finalize-after`         | Auto-finalization timeout        |
| `vad.enabled`                 | `VISTREAMASR_VAD__ENABLED`                 | `--vad.enabled`                     | Enable VAD processing            |
| `vad.aggressiveness`               | `VISTREAMASR_VAD__AGGRESSIVENESS`               | `--vad.aggressiveness`               | VAD speech probability threshold |
| `vad.min_speech_duration_ms`  | `VISTREAMASR_VAD__MIN_SPEECH_DURATION_MS`  | `--vad.min-speech-duration-ms`  | Minimum speech duration          |
| `vad.min_silence_duration_ms` | `VISTREAMASR_VAD__MIN_SILENCE_DURATION_MS` | `--vad.min-silence-duration-ms` | Minimum silence duration         |
| `vad.speech_pad_ms`           | `VISTREAMASR_VAD__SPEECH_PAD_MS`           | `--vad.speech-pad-ms`           | Speech segment padding           |

### Configuration Priority

The configuration system follows this priority order (highest to lowest):

1. **CLI Arguments** - Direct command-line overrides
2. **Environment Variables** - System-wide configuration
3. **Configuration File** - TOML file settings
4. **Default Values** - Built-in sensible defaults

**Example Configuration**:

```toml
# vistreamasr.toml
[model]
chunk_size_ms = 640
auto_finalize_after = 15.0

[vad]
enabled = true
aggressiveness = 3
min_speech_duration_ms = 250
min_silence_duration_ms = 500
speech_pad_ms = 100

[logging]
console_log_level = "INFO"
file_log_level = "DEBUG"
file_path = "vistreamasr.log"
```

## VAD Integration

### CLI VAD Support

The CLI interface provides comprehensive VAD configuration:

- **Toggle VAD**: `--vad.enabled` flag to enable/disable Voice Activity Detection
- **Threshold Configuration**: `--vad.aggressiveness` (0-3) for speech detection sensitivity
- **Duration Settings**: Configurable minimum speech and silence durations
- **Padding Control**: Configurable padding around detected speech segments

### VAD Workflow Integration

1. **Configuration Loading**: VAD parameters loaded from hierarchical configuration
2. **Streamer Initialization**: VAD processor created based on configuration
3. **Audio Processing**: Audio chunks passed through VAD before ASR processing
4. **Speech Filtering**: Only speech segments forwarded to ASR engine
5. **Progress Reporting**: VAD decisions included in debug output when enabled

**Example VAD Usage**:

```bash
# Enable VAD with default parameters
vistream-asr transcribe audio.wav --vad.enabled

# Custom VAD parameters
vistream-asr transcribe audio.wav --vad.enabled --vad.aggressiveness 2 --vad.min-speech-duration-ms 200

# VAD with microphone streaming
vistream-asr microphone --vad.enabled --duration 30
```

## Error Handling

### Comprehensive Error Management

The CLI interface includes robust error handling:

- **File Validation**: Checks for file existence and format compatibility
- **Configuration Validation**: Automatic parameter validation with clear error messages
- **Audio Format Handling**: Graceful handling of unsupported audio formats
- **Microphone Errors**: Proper handling of microphone access and recording issues
- **Network Errors**: Handling of model download and network connectivity issues
- **Graceful Degradation**: Fallback mechanisms when VAD or other components fail

### User-Friendly Error Messages

Error messages are designed to be helpful and actionable:

```python
# File not found error
"Error: Audio file 'nonexistent.wav' not found"

# Configuration validation error
"Error: vad.aggressiveness must be between 0 and 3, got 4"

# Microphone access error
"Error: Unable to access microphone. Please check permissions."

# Model loading error
"Error: Failed to load ASR model. Please check your internet connection."
```

## Performance and Progress Tracking

### Progress Indicators

The CLI interface provides real-time progress information:

- **Chunk Processing**: Shows current chunk number and processing speed
- **Real-Time Factor**: Displays RTF (Real-Time Factor) for performance monitoring
- **Timing Information**: Processing time and total duration
- **Memory Usage**: Optional memory usage display in debug mode

**Example Progress Output**:

```
[ViStreamASR] Processing chunk 1... 
[ViStreamASR] Final: This is the transcribed text
[ViStreamASR] Processing completed in 2.34s
```

### Performance Metrics

Key performance metrics tracked and displayed:

- **Real-Time Factor (RTF)**: Ratio of processing time to audio duration
- **Throughput**: Number of chunks processed per second
- **Latency**: Time from audio input to text output
- **Memory Usage**: RAM and GPU memory consumption (debug mode)

## Usage Examples

### Basic File Transcription

```bash
# Simple file transcription
vistream-asr transcribe audio.wav
```

### Configuration Usage

```bash
# Using configuration file
vistream-asr transcribe audio.wav --config vistreamasr.toml

# Overriding configuration parameters
vistream-asr transcribe audio.wav --model.chunk-size-ms 500
```

### VAD Integration

```bash
# Enable VAD with default parameters
vistream-asr transcribe audio.wav --vad.enabled

# Custom VAD configuration
vistream-asr transcribe audio.wav --vad.enabled --vad.aggressiveness 2

# VAD with microphone streaming
vistream-asr microphone --vad.enabled --duration 30
```

### Environment Variable Usage

```bash
# Set configuration via environment variables
export VISTREAMASR_MODEL__CHUNK_SIZE_MS=500
export VISTREAMASR_VAD__ENABLED=true
export VISTREAMASR_LOGGING__CONSOLE_LOG_LEVEL=DEBUG

# Run with environment configuration
vistream-asr transcribe audio.wav
```

## Integration with Other Components

### Streaming Interface Integration

The CLI component integrates seamlessly with the streaming interface:

- **StreamingASR**: Uses StreamingASR as the main transcription engine
- **Configuration**: Passes configuration settings to streaming components
- **VAD Processing**: Integrates VAD through the streaming interface
- **State Management**: Leverages ASRState for transcription state tracking

### Configuration System Integration

- **ViStreamASRSettings**: Uses the centralized configuration object
- **Validation**: Inherits parameter validation from configuration system
- **Logging**: Integrates with structured logging system
- **Environment Variables**: Supports environment variable overrides

### Core Processing Integration

- **ASREngine**: Uses ASREngine for actual speech recognition processing
- **Model Management**: Leverages model loading and caching mechanisms
- **Audio Processing**: Integrates with audio preprocessing pipeline
- **State Management**: Uses ASRState for maintaining transcription context

## API Reference

### Public Functions

#### `transcribe_file_streaming(audio_file, settings)`

Transcribe an audio file using ViStreamASR with VAD support.

**Parameters**:

- `audio_file` (str): Path to the audio file to transcribe
- `settings` (ViStreamASRSettings): Configuration object

**Returns**:

- `str`: Final transcription text

**Raises**:

- `ValueError`: If file doesn't exist or is invalid
- `RuntimeError`: If ASR processing fails

#### `transcribe_microphone_streaming(duration_seconds, settings)`

Process audio from microphone with real-time streaming.

**Parameters**:

- `duration_seconds` (float): Duration to record from microphone
- `settings` (ViStreamASRSettings): Configuration object

**Raises**:

- `ValueError`: If duration is invalid
- `RuntimeError`: If microphone access fails
- `KeyboardInterrupt`: If user stops recording

### Helper Functions

#### `_wrap_and_print_text(text, width=80)`

Wrap and print text with consistent formatting.

**Parameters**:

- `text` (str): Text to format and print
- `width` (int): The maximum width of each line (default: 80)

**Returns**:

- None

## Testing and Validation

### Unit Tests

The CLI interface includes comprehensive unit tests:

- **Configuration Loading**: Tests for hierarchical configuration parsing
- **Parameter Validation**: Tests for parameter validation and error handling
- **File Processing**: Tests for file transcription functionality
- **VAD Integration**: Tests for VAD configuration and processing
- **Error Handling**: Tests for error scenarios and graceful degradation

### Integration Tests

Integration tests cover:

- **End-to-End Workflow**: Complete file transcription workflow
- **Configuration Integration**: Integration with configuration system
- **VAD Integration**: Integration with VAD processing
- **Streaming Interface**: Integration with streaming components
- **Logging Integration**: Integration with structured logging

### Performance Tests

Performance validation includes:

- **Processing Speed**: Measurement of transcription throughput
- **Memory Usage**: Monitoring of memory consumption during processing
- **Real-Time Performance**: Validation of real-time processing capabilities
- **VAD Overhead**: Measurement of VAD processing impact on performance

## Future Enhancements

### Planned Improvements

- **Enhanced Progress Indicators**: More sophisticated progress visualization
- **Batch Processing**: Support for processing multiple files in batch
- **Advanced Configuration**: More granular configuration options
- **Plugin System**: Support for CLI plugins and extensions
- **Output Formatting**: Multiple output format options (JSON, CSV, etc.)

### Potential Extensions

- **Web Interface**: CLI integration with web-based monitoring
- **Mobile Support**: CLI functionality for mobile deployment
- **Cloud Integration**: Integration with cloud storage and processing
- **Advanced Analytics**: Performance analytics and reporting
- **Custom Models**: Support for custom ASR and VAD models