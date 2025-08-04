# Core Processing Component Documentation

## Overview

The Core Processing component handles the fundamental speech recognition functionality in ViStreamASR, including model management, audio processing, state management, and transcription logic. This component has been refactored to improve modularity, maintainability, and performance.

## Component Responsibilities

- **Model Management**: Loading, caching, and managing speech recognition models
- **Audio Processing**: Handling audio chunk processing and tensor operations
- **State Management**: Maintaining transcription state and context
- **Speech Recognition**: Core speech-to-text conversion functionality
- **Performance Optimization**: Ensuring real-time processing capabilities
- **Error Handling**: Robust error handling and recovery mechanisms

## Related Classes and Files

### Primary Files

- **[`src/vistreamasr/core.py`](src/vistreamasr/core.py)** - Core processing implementation
- **[`src/vistreamasr/__init__.py`](src/vistreamasr/__init__.py)** - Core function exports

### Key Classes and Functions

| Class/Function                | Location                                                     | Purpose                                |
| ----------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| `ASRState`                    | [`src/vistreamasr/core.py:440`](src/vistreamasr/core.py:440) | State management class                 |
| `ASREngine`                   | [`src/vistreamasr/core.py:459`](src/vistreamasr/core.py:459) | Main speech recognition engine         |
| `_pad_tensor_list`            | [`src/vistreamasr/core.py:67`](src/vistreamasr/core.py:67)   | Tensor padding helper function         |
| `FINAL_CHUNK_PADDING_SAMPLES` | [`src/vistreamasr/core.py:35`](src/vistreamasr/core.py:35)   | Named constant for final chunk padding |
| `MINIMUM_CHUNK_SIZE_SAMPLES`  | [`src/vistreamasr/core.py:36`](src/vistreamasr/core.py:36)   | Named constant for minimum chunk size  |
| `load_models`                 | [`src/vistreamasr/core.py:115`](src/vistreamasr/core.py:115)     | Model loading utility function         |

## Detailed Implementation

### Named Constants

The core processing component uses named constants to improve code readability and maintainability:

#### `FINAL_CHUNK_PADDING_SAMPLES`

**Location**: [`src/vistreamasr/core.py:35`](src/vistreamasr/core.py:35)

**Purpose**: Defines the number of samples to pad the final audio chunk with.

**Value**: `2000` samples (equivalent to 125ms at 16kHz)

**Usage**:

```python
# Pad final chunk to ensure proper processing
final_padding = FINAL_CHUNK_PADDING_SAMPLES
padded_audio = audio[-final_padding:] if len(audio) >= final_padding else audio
```

#### `MINIMUM_CHUNK_SIZE_SAMPLES`

**Location**: [`src/vistreamasr/core.py:36`](src/vistreamasr/core.py:36)

**Purpose**: Defines the minimum number of samples required for meaningful ASR processing.

**Value**: `320` samples (equivalent to 20ms at 16kHz)

**Usage**:

```python
# Ensure minimum chunk size for processing
if len(audio_tensor) < MINIMUM_CHUNK_SIZE_SAMPLES:
    # Skip processing or pad as needed
    return None
```

### Helper Functions

#### `_pad_tensor_list` Helper Function

**Location**: [`src/vistreamasr/core.py:67`](src/vistreamasr/core.py:67)

**Purpose**: Helper function to pad a list of tensors to ensure consistent batch processing.

**Parameters**:

- `tensor_list` (List[torch.Tensor]): List of input tensors
- `padding_value` (float): Value to use for padding (default: 0.0)

**Key Features**:

- **Batch Consistency**: Ensures all tensors in the batch have the same length
- **Efficient Padding**: Uses PyTorch's F.pad for efficient tensor operations
- **Flexible Handling**: Handles both single tensors and lists of tensors

**Implementation Details**:

```python
def _pad_tensor_list(
    tensor_list: List[torch.Tensor],
    padding_value: float = 0.0
) -> torch.Tensor:
    """Pad a list of tensors to ensure consistent batch processing.

    Args:
        tensor_list: List of input tensors to pad
        padding_value: Value to use for padding (default: 0.0)

    Returns:
        Padded tensor with consistent dimensions
    """
    # ... implementation ...
```

### State Management

#### `ASRState` Class

**Location**: [`src/vistreamasr/core.py:440`](src/vistreamasr/core.py:440)

**Purpose**: Manages the state of the ASR processing, including transcription context and performance metrics.

**Key Features**:

- **Transcription State**: Tracks current and previous transcriptions
- **Performance Metrics**: Monitors processing time and real-time factor
- **Context Management**: Maintains context for continuous transcription
- **State Persistence**: Supports state saving and loading for long-form transcription

**Class Attributes**:

- `current_transcription` (str): Current transcription text
- `previous_transcription` (str): Previous final transcription
- `processing_start_time` (float): Start time of current processing session
- `total_processing_time` (float): Total time spent processing
- `total_audio_duration` (float): Total duration of audio processed
- `chunk_count` (int): Number of chunks processed

**Implementation Details**:

```python
class ASRState:
    """State management class for ASR processing."""

    def __init__(self):
        self.current_transcription = ""
        self.previous_transcription = ""
        self.processing_start_time = None
        self.total_processing_time = 0.0
        self.total_audio_duration = 0.0
        self.chunk_count = 0

    # ... methods ...
```

### Core Processing Engine

#### `ASREngine` Class

**Location**: [`src/vistreamasr/core.py:459`](src/vistreamasr/core.py:459)

**Purpose**: Main speech recognition engine that handles audio processing, model inference, and transcription generation.

**Key Features**:

- **Modular Design**: Separated concerns through private helper methods
- **State Management**: Integrated ASRState for tracking processing state
- **Performance Optimization**: Efficient processing with real-time factor monitoring
- **Error Handling**: Comprehensive error handling and recovery
- **Configuration Support**: Full integration with hierarchical configuration

**Class Attributes**:

- `state` (ASRState): State management object
- `models` (Dict): Loaded models (encoder, decoder, decoder_with_past)
- `tokenizer` (Tokenizer): Text tokenizer for encoding/decoding
- `device` (torch.device): Computation device (CPU/GPU)
- `chunk_size_ms` (int): Audio chunk size in milliseconds
- `max_duration_before_forced_finalization` (float): Maximum segment duration
- `debug_mode` (bool): Debug logging enable flag

**Refactored Methods**:

##### `__init__` Method

**Location**: [`src/vistreamasr/core.py:462`](src/vistreamasr/core.py:462)

**Purpose**: Initialize the ASR engine with configuration parameters.

**Parameters**:

- `chunk_size_ms` (int): Audio chunk size in milliseconds
- `max_duration_before_forced_finalization` (float): Maximum segment duration
- `debug_mode` (bool): Enable debug logging

**Implementation**:

```python
def __init__(
    self,
    chunk_size_ms: int = 640,
    max_duration_before_forced_finalization: float = 15.0,
    debug_mode: bool = False,
):
    """Initialize the ASR engine."""
    # ... implementation ...
```

##### `initialize_models` Method

**Location**: [`src/vistreamasr/core.py:489`](src/vistreamasr/core.py:489)

**Purpose**: Load and initialize the speech recognition models and tokenizer.

**Key Features**:

- **Lazy Loading**: Models are loaded only when needed
- **Fallback Mechanisms**: Multiple loading strategies for robustness
- **Device Management**: Automatic device placement
- **Error Handling**: Graceful handling of model loading failures

**Implementation**:

```python
def initialize_models(self) -> None:
    """Initialize the speech recognition models and tokenizer."""
    # ... implementation ...
```

##### `process_audio_chunk` Method (Refactored)

**Location**: [`src/vistreamasr/core.py:518`](src/vistreamasr/core.py:518)

**Purpose**: Process a single audio chunk through the complete pipeline.

**Key Features**:

- **Modular Processing**: Broken down into smaller, focused helper methods
- **State Management**: Integrated with ASRState for tracking
- **Performance Monitoring**: Real-time factor calculation
- **Error Handling**: Graceful handling of processing errors

**Implementation**:

```python
def process_audio_chunk(
    self,
    audio_data: torch.Tensor,
    sample_rate: int = 16000,
    is_last: bool = False
) -> Dict[str, Any]:
    """Process a single audio chunk and return the transcription result."""
    # ... implementation ...
```

### Model Loading Utilities

#### `load_models` Function

**Location**: [`src/vistreamasr/core.py:115`](src/vistreamasr/core.py:115)

**Purpose**: Utility function to load speech recognition models with fallback mechanisms.

**Parameters**:

- `debug_mode` (bool): Enable debug logging

**Key Features**:

- **Multiple Loading Strategies**: Supports both pip package and torch.hub loading
- **Fallback Mechanisms**: Graceful fallback when primary loading fails
- **Device Management**: Automatic device placement
- **Error Handling**: Comprehensive error handling with informative messages

**Implementation Details**:

```python
def load_models(
    debug_mode: bool = False
) -> Tuple[Dict[str, Any], Any]:
    """Load speech recognition models with fallback mechanisms."""
    # ... implementation details from the actual code ...
```

## Configuration Integration

### Core Processing Configuration

The core processing component integrates with the hierarchical configuration system:

| Parameter                                 | Default | Description                                       |
| ----------------------------------------- | ------- | ------------------------------------------------- |
| `chunk_size_ms`                           | 640     | Audio chunk duration in milliseconds              |
| `max_duration_before_forced_finalization` | 15.0    | Maximum segment duration before auto-finalization |
| `debug_mode`                              | False   | Enable debug logging                              |

### Configuration Usage

```python
from vistreamasr.core import ASREngine
from vistreamasr.config import ViStreamASRSettings

# Load configuration
settings = ViStreamASRSettings()

# Initialize engine with configuration
engine = ASREngine(
    chunk_size_ms=settings.model.chunk_size_ms,
    max_duration_before_forced_finalization=settings.model.auto_finalize_after,
    debug_mode=settings.model.debug
)

# Initialize models
engine.initialize_models()

# Process audio chunks
for audio_chunk in audio_stream:
    result = engine.process_audio_chunk(audio_chunk)
    if result['final']:
        print(f"Final: {result['text']}")
```

## Performance Optimization

### Real-Time Processing

The core processing component is optimized for real-time performance:

- **Efficient Tensor Operations**: Uses PyTorch optimized operations
- **Lazy Initialization**: Models loaded only when needed
- **State Management**: Efficient state tracking with minimal overhead
- **Memory Management**: Proper cleanup and resource management

### Performance Monitoring

Key performance metrics tracked:

- **Real-Time Factor (RTF)**: Ratio of processing time to audio duration
- **Processing Rate**: Chunks processed per second
- **Memory Usage**: GPU/CPU memory consumption
- **Latency**: End-to-end processing latency

### Optimization Strategies

- **Batch Processing**: Uses tensor batching for improved throughput
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Model Caching**: Models cached after first load
- **Efficient Padding**: Optimized tensor padding operations

## Error Handling

### Input Validation

Comprehensive input validation ensures robust operation:

- **Audio Data Validation**: Checks tensor dimensions and values
- **Sample Rate Validation**: Ensures supported sample rates
- **Model State Validation**: Verifies model initialization
- **Configuration Validation**: Validates configuration parameters

### Error Recovery

Graceful error handling mechanisms:

- **Model Loading Failures**: Fallback to alternative loading methods
- **Audio Processing Errors**: Skip problematic chunks and continue
- **Memory Errors**: Automatic fallback to CPU if GPU memory insufficient
- **Configuration Errors**: Use sensible defaults when configuration invalid

## Testing and Validation

### Unit Tests

Comprehensive unit test coverage:

- **ASRState Tests**: State management and serialization
- **ASREngine Tests**: Core processing functionality
- **Helper Function Tests**: `_pad_tensor_list` and utility functions
- **Constant Validation**: Named constant correctness
- **Error Handling Tests**: Error scenario handling

### Integration Tests

Integration test coverage:

- **Model Loading Integration**: End-to-end model loading workflow
- **Audio Processing Pipeline**: Complete audio processing pipeline
- **State Management Integration**: State tracking across processing sessions
- **Performance Monitoring**: Real-time factor and processing rate validation

### Performance Tests

Performance validation:

- **Real-Time Performance**: RTF < 0.1 for real-time applications
- **Throughput Testing**: Maximum processing rate validation
- **Memory Usage**: Memory consumption under various load conditions
- **Latency Testing**: End-to-end processing latency measurement

## Usage Examples

### Basic Usage

```python
from vistreamasr.core import ASREngine, load_models

# Initialize engine
engine = ASREngine(
    chunk_size_ms=640,
    max_duration_before_forced_finalization=15.0,
    debug_mode=True
)

# Load models
engine.initialize_models()

# Process audio chunk
audio_tensor = torch.randn(10240)  # 640ms at 16kHz
result = engine.process_audio_chunk(audio_tensor)
print(f"Transcription: {result['text']} (RTF: {result['rtf']:.2f})")
```

### Advanced Usage with State Management

```python
from vistreamasr.core import ASREngine

# Initialize engine
engine = ASREngine(debug_mode=True)

# Load models
engine.initialize_models()

# Process multiple chunks
audio_chunks = [torch.randn(10240) for _ in range(5)]

for i, chunk in enumerate(audio_chunks):
    result = engine.process_audio_chunk(chunk, is_last=(i == len(audio_chunks) - 1))
    print(f"Chunk {i+1}: {result['text']} (Final: {result['final']})")

# Get performance summary
print(f"Overall RTF: {engine.get_asr_rtf():.2f}")
print(f"Processing rate: {engine.get_processing_rate():.2f} chunks/s")
```

## API Reference

### Public Classes

#### `ASRState` Class

**Location**: [`src/vistreamasr/core.py:440`](src/vistreamasr/core.py:440)

**Purpose**: Manages the state of ASR processing.

**Methods**:

- `__init__()`: Initialize ASR state components

#### `ASREngine` Class

**Location**: [`src/vistreamasr/core.py:459`](src/vistreamasr/core.py:459)

**Purpose**: Main speech recognition engine.

**Methods**:

- `__init__(chunk_size_ms, max_duration_before_forced_finalization, debug_mode)`: Initialize engine
- `initialize_models()`: Load and initialize models
- `process_audio_chunk(audio_data, sample_rate, is_last=False)`: Process audio chunk
- `get_asr_rtf()`: Get current real-time factor
- `reset_state()`: Reset processing state

**Properties**:

- `state`: Current ASRState object
- `chunk_size_ms`: Audio chunk size
- `debug_mode`: Debug logging status

### Public Functions

#### `load_models(debug_mode)`

**Location**: [`src/vistreamasr/core.py:115`](src/vistreamasr/core.py:115)

**Purpose**: Load speech recognition models.

**Parameters**:

- `debug_mode` (bool): Enable debug logging

**Returns**:

- `Tuple[Any, Any, Any]`: acoustic_model, ngram_lm, beam_search

### Helper Functions

#### `_pad_tensor_list(tensor_list, padding_value=0.0)`

**Location**: [`src/vistreamasr/core.py:67`](src/vistreamasr/core.py:67)

**Purpose**: Pad list of tensors for consistent batch processing.

**Parameters**:

- `tensor_list` (List[torch.Tensor]): Input tensors
- `padding_value` (float): Padding value (default: 0.0)

**Returns**:

- `torch.Tensor`: Padded tensor stack

### Named Constants

#### `FINAL_CHUNK_PADDING_SAMPLES`

**Location**: [`src/vistreamasr/core.py:35`](src/vistreamasr/core.py:35)

**Value**: `2000`

**Purpose**: Number of samples to pad final audio chunk.

#### `MINIMUM_CHUNK_SIZE_SAMPLES`

**Location**: [`src/vistreamasr/core.py:36`](src/vistreamasr/core.py:36)

**Value**: `320`

**Purpose**: Minimum samples required for meaningful ASR processing.

## Future Enhancements

### Planned Improvements

- **Advanced Optimization**: Further optimization for low-latency processing
- **Model Selection**: Support for multiple ASR model architectures
- **Quantization**: Support for quantized models for improved performance
- **Streaming Improvements**: Enhanced streaming capabilities and state management
- **Multi-language Support**: Extended support for additional languages

### Potential Extensions

- **Custom Models**: Support for custom-trained ASR models
- **Advanced Features**: Punctuation capitalization, speaker diarization
- **Integration**: Enhanced integration with other ASR components
- **Monitoring**: Built-in performance monitoring and alerting
- **Deployment**: Optimized deployment strategies for various environments