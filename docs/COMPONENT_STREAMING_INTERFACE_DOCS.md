# Streaming Interface Component Documentation

## Overview

The Streaming Interface component provides a unified facade for audio streaming operations in ViStreamASR. Following the refactoring, it now implements the Facade Pattern, delegating specific streaming operations to specialized components: FileStreamer for file-based audio streaming and MicrophoneStreamer for real-time microphone input. This architecture improves modularity, maintainability, and extensibility.

## Component Responsibilities

- **Facade Pattern**: Provide a unified interface for different streaming operations
- **File Streaming**: Handle audio file streaming with VAD integration
- **Microphone Streaming**: Handle real-time microphone input with VAD integration
- **VAD Integration**: Seamless Voice Activity Detection support
- **Configuration Management**: Centralized configuration for all streaming components
- **State Management**: Maintain streaming state and context
- **Error Handling**: Robust error handling for streaming operations

## Related Classes and Files

### Primary Files

- **[`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py)** - Streaming interface implementation
- **[`src/vistreamasr/__init__.py`](src/vistreamasr/__init__.py)** - Streaming function exports

### Key Classes and Functions

| Class/Function           | Location                                                               | Purpose                                    |
| ------------------------ | ---------------------------------------------------------------------- | ------------------------------------------ |
| `StreamingASR`           | [`src/vistreamasr/streaming.py:339`](src/vistreamasr/streaming.py:339) | Main facade class for streaming operations |
| `FileStreamer`           | [`src/vistreamasr/streaming.py:31`](src/vistreamasr/streaming.py:31)   | Specialized file-based audio streaming     |
| `MicrophoneStreamer`     | [`src/vistreamasr/streaming.py:231`](src/vistreamasr/streaming.py:231) | Specialized microphone audio streaming     |

## Detailed Implementation

### Streaming Interface Architecture

The streaming interface has been refactored to implement the **Facade Pattern**, where:

- **StreamingASR** acts as the main facade providing a unified interface
- **FileStreamer** handles file-based audio streaming operations
- **MicrophoneStreamer** handles real-time microphone streaming operations
- **VAD Integration** is provided through helper methods in the facade

This separation of concerns allows for:

- **Specialized Components**: Each streamer type can be optimized for its specific use case
- **Easier Testing**: Individual components can be tested in isolation
- **Better Maintainability**: Changes to one streamer type don't affect others
- **Extensibility**: New streamer types can be easily added

### Main Facade Class

#### `StreamingASR` Class

**Location**: [`src/vistreamasr/streaming.py:339`](src/vistreamasr/streaming.py:339)

**Purpose**: Main facade class providing unified interface for streaming operations.

**Key Features**:

- **Facade Pattern**: Unified interface for different streaming operations
- **Lazy Initialization**: Components initialized only when needed
- **Configuration Management**: Centralized configuration for all components
- **VAD Integration**: Seamless Voice Activity Detection support
- **State Management**: Maintain streaming state and context
- **Error Handling**: Comprehensive error handling and recovery

**Initialization**:

```python
def __init__(self, settings: Optional[ViStreamASRSettings] = None, **kwargs):
    """Initialize the StreamingASR facade.

    Args:
        settings: Configuration object for all components
    """
    # ... implementation ...
```

### Specialized Streamer Classes

#### `FileStreamer` Class

**Location**: [`src/vistreamasr/streaming.py:31`](src/vistreamasr/streaming.py:31)

**Purpose**: Specialized component for file-based audio streaming operations.

**Key Features**:

- **File Processing**: Efficient processing of audio files
- **Chunk Management**: Intelligent chunk size management
- **Progress Tracking**: Built-in progress indicators
- **Error Handling**: Graceful handling of file access and format errors
- **VAD Integration**: Optional Voice Activity Detection support
- **Memory Management**: Efficient memory usage for large files

**Core Methods**:

##### `stream` Method

**Location**: [`src/vistreamasr/streaming.py:125`](src/vistreamasr/streaming.py:125)

**Purpose**: Process an audio file and yield transcription results.

**Parameters**:

- `file_path` (str): Path to the audio file
- `chunk_size_ms` (Optional[int]): Override chunk size for this session
- `process_vad_chunk_func` (Optional[callable]): Function to process chunks with VAD

#### `MicrophoneStreamer` Class

**Location**: [`src/vistreamasr/streaming.py:231`](src/vistreamasr/streaming.py:231)

**Purpose**: Specialized component for real-time microphone audio streaming.

**Key Features**:

- **Real-time Processing**: Low-latency audio streaming
- **Microphone Management**: Proper microphone access and cleanup
- **Buffer Management**: Efficient audio buffer management
- **Error Handling**: Graceful handling of microphone errors
- **VAD Integration**: Optional Voice Activity Detection support
- **Performance Monitoring**: Real-time performance metrics

**Core Methods**:

##### `stream` Method

**Location**: [`src/vistreamasr/streaming.py:245`](src/vistreamasr/streaming.py:245)

**Purpose**: Process the microphone stream and yield transcription results.

**Parameters**:

- `duration_seconds` (Optional[float]): Duration to record (None for indefinite)
- `process_vad_chunk_func` (Optional[callable]): Callback for each chunk

### VAD Integration Helper Methods

The streaming interface includes helper methods for VAD integration:

#### `_process_vad_chunk` Method

**Location**: [`src/vistreamasr/streaming.py:468`](src/vistreamasr/streaming.py:468)

**Purpose**: Process audio chunks with VAD filtering.

## Configuration Integration

### Streaming Configuration

The streaming interface integrates with the hierarchical configuration system:

| Parameter                     | Default | Description                                       |
| ----------------------------- | ------- | ------------------------------------------------- |
| `model.chunk_size_ms`               | 640     | Audio chunk duration in milliseconds              |
| `model.auto_finalize_after`         | 15.0    | Maximum segment duration before auto-finalization |
| `vad.enabled`                 | True   | Enable/disable VAD processing                     |
| `vad.aggressiveness`             | 3   | VAD aggressiveness level (0-3)              |
| `vad.min_silence_duration_ms` | 500     | Minimum silence duration for VAD                  |
| `vad.speech_pad_ms`           | 100      | Padding around speech segments                    |

### Configuration Usage

```python
from vistreamasr.streaming import StreamingASR
from vistreamasr.config import ViStreamASRSettings

# Load configuration
settings = ViStreamASRSettings()

# Configure streaming
settings.model.chunk_size_ms = 640
settings.model.auto_finalize_after = 15.0

# Configure VAD
settings.vad.enabled = True
settings.vad.aggressiveness = 2

# Initialize streaming facade
streaming_asr = StreamingASR(settings=settings)

# Stream from file
for result in streaming_asr.stream_from_file("audio.wav"):
    if result['final']:
        print(f"Final: {result['text']}")
```

## Performance Optimization

### Real-Time Processing

The streaming interface is optimized for real-time performance:

- **Lazy Initialization**: Components initialized only when needed
- **Efficient Memory Management**: Proper cleanup of resources
- **Low-Latency Processing**: Optimized for real-time microphone streaming

### Performance Monitoring

Key performance metrics tracked:

- **Chunk Processing Time**: Time to process each audio chunk
- **Real-Time Factor (RTF)**: Ratio of processing time to audio duration

## Error Handling

### Streaming Error Management

Comprehensive error handling for streaming operations:

- **File Access Errors**: Graceful handling of file access and format errors
- **Microphone Errors**: Proper handling of microphone access and recording issues
- **Model Errors**: Fallback mechanisms when model loading fails
- **VAD Errors**: Graceful degradation when VAD processing fails

## Testing and Validation

### Unit Tests

Comprehensive unit test coverage:

- **StreamingASR Tests**: Facade functionality and component management
- **FileStreamer Tests**: File processing and chunk management
- **MicrophoneStreamer Tests**: Microphone streaming and real-time processing
- **VAD Integration Tests**: VAD processing and filtering

### Integration Tests

Integration test coverage:

- **End-to-End Streaming**: Complete streaming workflows
- **Component Integration**: Integration between facade and specialized components
- **VAD Integration**: Integration with VAD processing

## Usage Examples

### Basic File Streaming

```python
from vistreamasr.streaming import StreamingASR
from vistreamasr.config import ViStreamASRSettings

# Load configuration
settings = ViStreamASRSettings()

# Initialize streaming facade
streaming_asr = StreamingASR(settings=settings)

# Stream from file
for result in streaming_asr.stream_from_file("audio.wav"):
    if result['final']:
        print(f"Final: {result['text']}")
```

### Real-time Microphone Streaming

```python
from vistreamasr.streaming import StreamingASR
from vistreamasr.config import ViStreamASRSettings

# Load configuration
settings = ViStreamASRSettings()

# Initialize streaming facade
streaming_asr = StreamingASR(settings=settings)

# Stream from microphone
for result in streaming_asr.stream_from_microphone(duration_seconds=30):
    if result['final']:
        print(f"Final: {result['text']}")
```

## API Reference

### Public Classes

#### `StreamingASR` Class

**Location**: [`src/vistreamasr/streaming.py:339`](src/vistreamasr/streaming.py:339)

**Purpose**: Main facade class for streaming operations.

**Methods**:

- `__init__(settings, **kwargs)`: Initialize with configuration
- `stream_from_file(file_path, chunk_size_ms=None)`: Stream from audio file
- `stream_from_microphone(duration_seconds=None)`: Stream from microphone

#### `FileStreamer` Class

**Location**: [`src/vistreamasr/streaming.py:31`](src/vistreamasr/streaming.py:31)

**Purpose**: Specialized component for file-based audio streaming.

**Methods**:

- `stream(file_path, chunk_size_ms=None, process_vad_chunk_func=None)`: Process audio file

#### `MicrophoneStreamer` Class

**Location**: [`src/vistreamasr/streaming.py:231`](src/vistreamasr/streaming.py:231)

**Purpose**: Specialized component for real-time microphone streaming.

**Methods**:

- `stream(duration_seconds=None, process_vad_chunk_func=None)`: Process microphone stream

## Future Enhancements

### Planned Improvements

- **Enhanced Streaming**: More sophisticated streaming strategies and optimizations
- **Advanced VAD Integration**: Improved VAD algorithms and integration
- **Multi-speaker Support**: Extended support for multiple speaker detection