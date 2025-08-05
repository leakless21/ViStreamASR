# ViStreamASR Architecture with U2 Backbone, Configuration, and Logging System

## Current Status

> **Note:** This document describes the target architecture for ViStreamASR, which includes both currently implemented features and planned features. The goal is to provide a clear vision for the project's development.
>
> - **Implemented:**
>
>   - Core ASR processing pipeline ([`core.py`](src/vistreamasr/core.py), [`streaming.py`](src/vistreamasr/streaming.py)).
>   - Basic VAD (Voice Activity Detection) integration.
>   - Web interface for real-time transcription.
>   - FFmpeg for audio capture and conversion.
>
> - **Planned:**
>   - Fully modular and pluggable ASR backends.
>   - Advanced speaker diarization.
>   - Comprehensive configuration options.

## 1. Overview 🔄 PLANNED

This document describes the architecture of the ViStreamASR system. The system is designed to be a comprehensive, real-time speech-to-text solution with a modular architecture that supports multiple backends, speaker diarization, and a web-based interface.

The core of the system is a **pluggable ASR backend**, allowing users to choose the most suitable speech recognition engine for their needs. The architecture is built on `asyncio` to ensure high performance and scalability, making it capable of handling multiple concurrent clients in real-time.

Key architectural features include:

- **Modular ASR Backend**: A pluggable architecture that allows for easy integration of different ASR engines (e.g., `faster-whisper`, `whisper-timestamped`).
- **Speaker Diarization**: Integrated speaker diarization to identify and label different speakers in a conversation.
- **Web Interface**: A `FastAPI` web server with a WebSocket interface for real-time communication with a web-based frontend.
- **Asynchronous Processing**: The entire system is built on `asyncio` for concurrent, non-blocking processing of audio, transcription, and diarization.
- **Decoupled Components**: The architecture is composed of decoupled components, including an `AudioProcessor` for each client and a singleton `TranscriptionEngine` to manage ASR models.

This document provides a detailed overview of the system's architecture, components, data flow, and interfaces.

## 2. Architecture Diagram 🔄 PLANNED

```mermaid
graph TD
    subgraph "User Interface"
        UI[Web Frontend: HTML/JS]
    end

    subgraph "Web Server Layer"
        WS[FastAPI Application]
        WSE[WebSocket Endpoint]
    end

    subgraph "Application Layer"
        AP[AudioProcessor per Client]
        TE[TranscriptionEngine Singleton]
    end

    subgraph "ASR Backend Layer"
        subgraph "Pluggable Backends"
            FW[FasterWhisperBackend]
            WT[WhisperTimestampedBackend]
            OAI[OpenAIWhisperBackend]
        end
        ASR_Base[ASRBackend Base Class]
    end

    subgraph "Diarization Layer"
        Diarization[SpeakerDiarization: diart/pyannote.audio]
    end

    subgraph "Core Services"
        VAD[VADProcessor: silero-vad]
        Config[Configuration Management]
        Logging[Structured Logging]
    end

    UI -- "WebSocket" --> WSE
    WSE -- "Audio Stream" --> AP
    AP -- "Transcription/Diarization" --> WSE
    WSE -- "Results" --> UI

    AP -- "Uses" --> VAD
    AP -- "Uses" --> Diarization
    AP -- "Calls" --> TE

    TE -- "Delegates to" --> ASR_Base
    ASR_Base <|-- FW
    ASR_Base <|-- WT
    ASR_Base <|-- OAI

    AP -- "Configured by" --> Config
    TE -- "Configured by" --> Config
    VAD -- "Configured by" --> Config
    Diarization -- "Configured by" --> Config
    Logging -- "Used by all components"

    style TE fill:#e6ffe6,stroke:#333,stroke-width:2px
    style AP fill:#e6f3ff,stroke:#333,stroke-width:1px
    style ASR_Base fill:#fff0e6,stroke:#333,stroke-width:2px
```

## 3. Component Architecture 🔄 PLANNED

### 3.1 Web Server Layer

The Web Server Layer is responsible for handling client connections and real-time communication.

#### FastAPI Application

- **Responsibility**: Serves the web frontend and manages the WebSocket lifecycle.
- **Key Features**:
  - Serves the main HTML/JS frontend.
  - Handles HTTP requests.
  - Manages WebSocket connections.

#### WebSocket Endpoint

- **Responsibility**: Handles real-time, bidirectional communication with clients.
- **Key Features**:
  - Accepts incoming WebSocket connections.
  - Receives audio streams from clients.
  - Sends real-time transcription and diarization results back to clients.

### 3.2 Application Layer

The Application Layer contains the core logic for processing audio and managing transcription tasks.

#### AudioProcessor

- **Responsibility**: Manages the audio processing pipeline for a single client. An instance of `AudioProcessor` is created for each client connection.
- **Key Features**:
  - Receives audio chunks from the WebSocket.
  - Uses the `VADProcessor` to detect speech.
  - Sends speech segments to the `TranscriptionEngine`.
  - Integrates results from the `SpeakerDiarization` component.
  - Runs in its own `asyncio` task to handle concurrent processing.

#### TranscriptionEngine

- **Responsibility**: Manages the ASR models and performs transcription. This is a singleton component to ensure that models are loaded only once.
- **Key Features**:
  - Loads and manages the selected ASR backend.
  - Provides a unified interface for transcription.
  - Optimized to handle requests from multiple `AudioProcessor` instances.

### 3.3 ASR Backend Layer

The ASR Backend Layer provides a pluggable interface for different speech recognition engines.

#### ASRBackend (Base Class)

- **Responsibility**: Defines the common interface that all ASR backends must implement.
- **Key Features**:
  - Abstract methods for transcription.
  - A standardized way to load and configure models.

#### Pluggable Backends

- **FasterWhisperBackend**: An implementation of the `ASRBackend` interface that uses `faster-whisper`.
- **WhisperTimestampedBackend**: An implementation that uses `whisper-timestamped` for more accurate timestamps.
- **OpenAIWhisperBackend**: An implementation that uses the official OpenAI Whisper API.

### 3.4 Diarization Layer

The Diarization Layer is responsible for identifying different speakers in the audio.

#### SpeakerDiarization

- **Responsibility**: Performs speaker diarization on the audio stream.
- **Key Features**:
  - Uses a library like `diart` or `pyannote.audio`.
  - Identifies speaker segments and assigns a unique ID to each speaker.
  - Integrates with the `AudioProcessor` to provide speaker labels for the transcription.

### 3.5 Core Services

The Core Services layer provides essential functionalities that are used across the entire system.

#### VADProcessor

- **Responsibility**: Detects voice activity in the audio stream.
- **Key Features**:
  - Uses `silero-vad` for efficient and accurate voice activity detection.
  - Filters out non-speech segments to reduce unnecessary processing.

#### Configuration Management

- **Responsibility**: Manages all configuration for the system.
- **Key Features**:
  - Uses a hierarchical configuration system (TOML, environment variables, CLI).
  - Provides type-safe configuration using `pydantic-settings`.

#### Structured Logging

- **Responsibility**: Provides structured, configurable logging.
- **Key Features**:
  - Uses `Loguru` for flexible and powerful logging.
  - Supports multiple output sinks (console, file) with different formats.

## 4. Data Flow Architecture 🔄 PLANNED

### 4.1 Real-time Transcription Data Flow

```mermaid
sequenceDiagram
    participant C as Client (Web Frontend)
    participant WS as WebSocket Endpoint
    participant AP as AudioProcessor
    participant VAD as VADProcessor
    participant SD as SpeakerDiarization
    participant TE as TranscriptionEngine
    participant ASR as ASRBackend

    C->>+WS: WebSocket Connection
    WS->>+AP: Create AudioProcessor instance
    loop Audio Streaming
        C->>WS: Send Audio Chunk
        WS->>AP: Receive Audio Chunk
        AP->>VAD: Process for Speech
        alt Speech Detected
            AP->>SD: Process for Speaker ID
            AP->>TE: Transcribe Segment
            TE->>ASR: Get Transcription
            ASR-->>TE: Return Transcription
            TE-->>AP: Return Transcription
            AP-->>WS: Send Result (Transcription + Speaker)
            WS-->>C: Display Result
        end
    end
    C->>-WS: Close Connection
    WS->>-AP: Terminate AudioProcessor
```

## 5. Frontend Design

The new frontend design is inspired by the `WhisperLiveKit` reference and aims to provide a modern, user-friendly interface for real-time transcription.

### 5.1 Layout and Styling

- **Layout**: The layout will be centered, with a prominent record button at the top, followed by the transcription display area. The design will be responsive and adapt to different screen sizes.
- **Styling**: The UI will use a clean, modern aesthetic with a light color scheme. The record button will feature a circular design with a recording icon that animates when active. The transcription text will be displayed in a clear, legible font with speaker labels.

### 5.2 Interactive Elements

- **Record Button**: An animated record button will serve as the primary user control. When clicked, it will transition to a "recording" state, with a visual indicator (e.g., a waveform) to show that audio is being captured.
- **Waveform Display**: A real-time waveform will be displayed next to the record button to provide visual feedback that audio is being detected.
- **Transcription Display**: The transcription will be displayed in a dedicated area, with each new segment appearing in real-time. Speaker labels will be used to differentiate between speakers.

### 5.3 JavaScript Modifications

- **Real-time UI Updates**: The `client.js` file will be updated to handle real-time UI updates, including the waveform display and transcription rendering.
- **Event Handling**: The JavaScript will manage user interactions, such as clicking the record button, and will communicate with the backend via WebSockets.
- **State Management**: The client-side script will manage the recording state and update the UI accordingly.

## 6. Asynchronous Processing 🔄 PLANNED

The entire system is designed around `asyncio` to ensure high performance and scalability.

- **Concurrent Clients**: The `FastAPI` server can handle many concurrent WebSocket connections, creating a separate `AudioProcessor` task for each one.
- **Non-blocking I/O**: All I/O operations, including receiving audio from the client and sending results back, are non-blocking.
- **Parallel Processing**: VAD, diarization, and transcription can be performed concurrently, reducing the overall latency.

## 7. Performance Considerations ✅ IMPLEMENTED

### 7.1 Processing Performance

| Metric                  | Target                       | Implementation Status |
| ----------------------- | ---------------------------- | --------------------- |
| **VAD Processing Time** | <1ms per 30ms+ chunk         | ✅ Achieved           |
| **Model Loading Time**  | ~1-2 seconds                 | ✅ Implemented        |
| **Memory Usage**        | <10MB runtime                | ✅ Optimized          |
| **CPU Usage**           | Single-threaded optimization | ✅ Implemented        |

U2-oriented streaming emphasizes bounded look-ahead, incremental emissions, and timely finalization. The configuration parameters (chunk_size_ms, auto_finalize_after) are tuned to deliver low end-to-end latency while maintaining transcript quality consistent with streaming constraints. See the comparative notes in [`docs/COMPARISON_ANALYSIS.md`](docs/COMPARISON_ANALYSIS.md) for context on design trade-offs versus Whisper-based systems.

### 7.2 Optimization Strategies

#### CPU Optimization

- **Single-threaded Processing**: Follows Silero-VAD design optimization
- **Model Caching**: Cached model loading to avoid repeated initialization
- **Efficient Buffering**: Minimal memory allocations during processing
- **Early Rejection**: Fast rejection of silence chunks
- **Modular Processing**: Separated concerns for better performance analysis

#### Memory Optimization

- **Tensor Reuse**: Efficient tensor memory layout
- **Buffer Management**: Optimized speech segment buffering
- **Garbage Collection**: Minimal GC impact during processing
- **Model Size**: ~2MB model footprint for efficient deployment
- **State Management**: Centralized state reduces memory overhead

## 8. Error Handling and Reliability ✅ IMPLEMENTED

### 8.1 Error Handling Strategy

- **Model Loading Errors**: Graceful fallback and warning messages
- **Audio Processing Errors**: Individual chunk error handling
- **State Synchronization**: Proper state reset on errors
- **Configuration Validation**: Parameter validation with clear error messages
- **VAD Integration Errors**: Graceful degradation when VAD fails

### 8.2 Recovery Mechanisms

- **Transient Error Recovery**: Continue processing after temporary errors
- **State Reset**: Complete state reset on session boundaries
- **Graceful Degradation**: Disable VAD if critical errors occur
- **Logging**: Comprehensive error logging for debugging
- **Chunk Processing**: Skip problematic chunks while maintaining stream

## 9. Configuration Management ✅ IMPLEMENTED

### 9.1 Configuration Hierarchy

1.  **Default Values**: Sensible defaults defined in Pydantic models
2.  **TOML Configuration Files**: Structured configuration in `vistreamasr.toml`
3.  **Environment Variables**: Override with `VISTREAMASR_` prefix
4.  **CLI Arguments**: Final override layer for runtime configuration

### 9.2 Configuration Loading Process

1.  **File Loading**: Load TOML configuration file if specified
2.  **Environment Variables**: Override with environment variables
3.  **CLI Arguments**: Apply command-line argument overrides
4.  **Validation**: Validate all parameters with Pydantic models
5.  **Type Conversion**: Automatic type conversion and validation

### 9.3 Parameter Validation

- **Range Validation**: Ensures parameters are within valid ranges
- **Type Validation**: Validates parameter types and formats using Pydantic
- **Dependency Validation**: Ensures parameter combinations are valid
- **Error Reporting**: Clear error messages for invalid configurations
- **Default Value Handling**: Automatic application of default values

### 9.4 Environment Variable Mapping

- **Prefix Support**: Environment variables use `VISTREAMASR_` prefix
- **Nested Structure**: Double underscores (`__`) for nested configuration
- **Type Conversion**: Automatic conversion from string to target types
- **Case Insensitivity**: Environment variables are case-insensitive

## 10. Testing and Validation ✅ IMPLEMENTED

### 10.1 Testing Architecture

- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end VAD-ASR workflow testing
- **Performance Tests**: Processing time and memory usage validation
- **Error Handling Tests**: Graceful error handling and recovery
- **Refactoring Validation**: Tests for new helper functions and constants

### 10.2 Test Coverage Areas

- **VAD Processing**: Speech/silence detection accuracy
- **State Management**: State transitions and persistence
- **Integration Coordination**: VAD-ASR component interaction
- **CLI Integration**: Command-line parameter handling
- **Error Scenarios**: Error handling and recovery mechanisms
- **Helper Functions**: Testing of new \_wrap_and_print_text and \_pad_tensor_list
- **Constants Validation**: Testing with named constants

## 11. Deployment Considerations ✅ IMPLEMENTED

### 11.1 System Requirements

- **Python 3.8+**: Runtime environment requirement
- **Pixi Environment**: Dependency management and execution environment
- **PyTorch 2.7.1+**: Deep learning framework dependency
- **CPU Support**: AVX/AVX2/AVX-512 for optimal performance
- **Memory**: Minimum 4GB RAM for smooth operation

### 11.2 Deployment Scenarios

- **Development Environment**: Full debug logging and testing using `pixi run`
- **Production Environment**: Optimized performance with minimal logging
- **Edge Deployment**: Lightweight configuration for resource-constrained environments
- **Cloud Deployment**: Scalable multi-instance processing

### 11.3 Dependency Management

The project now uses **Pixi** for dependency management, providing robust, multi-platform support.

#### Pixi Environment Setup

- **Installation**: Install Pixi via `curl -LsSf https://pixi.sh/install.sh | sh`
- **Project Installation**: Use `pixi install` to set up the environment
- **Dependency Resolution**: Automatic resolution of Conda and PyPI dependencies
- **Environment Management**: Isolated environments for different project needs

#### Pixi Features

- **Multi-platform Support**: Defined in [`tool.pixi.workspace.platforms`](pyproject.toml:81), supporting `linux-64`, `osx-64`, `osx-arm64`, and `win-64`.
- **Tasks**: Common tasks like `test`, `lint`, `format`, and `build` are defined in [`tool.pixi.tasks`](pyproject.toml:110) and can be run with `pixi run <task>`.
- **Development Environment**: Activate the development environment with `pixi run --dev <command>` or use `pix shell` for an interactive shell.

#### Environments

- **Default Environment**: Standard execution environment with core dependencies.
- **Development Environment**: Activated via `pixi run -e dev <command>`, includes development dependencies from the `dev` feature.

## 12. Future Enhancements 🔄 PLANNED

### 12.1 Performance Optimizations

- **Model Quantization**: Quantized models for better performance
- **Batch Processing**: Batch processing for non-real-time scenarios
- **GPU Acceleration**: GPU support for VAD processing (planned)
- **Adaptive Thresholding**: Dynamic threshold adjustment

### 12.2 Feature Enhancements

- **Multi-speaker VAD**: Extended support for multiple speakers
- **Real-time Visualization**: VAD decision visualization tools
- **Advanced Noise Cancellation**: Integration with noise reduction
- **Custom Model Support**: User-provided VAD model support
- **Enhanced Streaming**: More sophisticated streaming strategies

### 12.3 Monitoring and Analytics

- **Performance Metrics**: Real-time performance monitoring
- **Usage Statistics**: VAD usage and effectiveness tracking
- **Error Analytics**: Advanced error reporting and analysis
- **Resource Monitoring**: CPU, memory, and usage monitoring

## 13. Code Quality and Maintainability ✅ IMPLEMENTED

### 13.1 Refactoring Improvements

The recent refactoring has significantly improved code quality:

#### Extracted Helper Functions

- **\_wrap_and_print_text**: Eliminated duplicated text formatting logic in CLI
- **\_pad_tensor_list**: Centralized tensor padding operations for better maintainability
- **Private processing methods**: Improved separation of concerns in ASREngine

#### Introduction of Named Constants

- **FINAL_CHUNK_PADDING_SAMPLES**: Replaced magic number 2000
- **MINIMUM_CHUNK_SIZE_SAMPLES**: Replaced magic number 320
- Improved code readability and easier configuration

#### Enhanced Class Separation

- **ASRState**: Separated state management from engine logic
- **FileStreamer/MicrophoneStreamer**: Split streaming responsibilities
- **StreamingASR**: Reduced to a clean facade interface

### 13.2 Code Organization Benefits

- **Maintainability**: Smaller, focused functions are easier to understand and modify
- **Testability**: Individual components can be tested in isolation
- **Debugging**: Better error reporting and state tracking
- **Extensibility**: New features can be added with minimal impact on existing code
- **Documentation**: Clear separation makes documentation more accurate

### 13.3 Documentation Alignment

All documentation has been updated to reflect the new architecture:

- **Architecture Diagram**: Updated to show all new components and relationships
- **Component Interfaces**: Documented all new classes and methods
- **Data Flow**: Updated to reflect new processing patterns
- **Integration Points**: Documented new interaction patterns

## 14. Migration and Compatibility ✅ IMPLEMENTED

### 14.1 Backward Compatibility

The refactoring maintains backward compatibility through:

- **Legacy Parameter Support**: StreamingASR still accepts direct parameters
- **Configuration Fallback**: Graceful handling of both old and new configuration styles
- **API Consistency**: Public interfaces remain unchanged where possible
- **Default Values**: Sensible defaults ensure existing code continues to work

### 14.2 Migration Path

For users upgrading from previous versions:

1.  **No Breaking Changes**: Existing code should continue to work unchanged
2.  **Recommended Migration**: Move to configuration-based approach for better control
3.  **Gradual Adoption**: Can adopt new features incrementally
4.  **Configuration Benefits**: New users should start with configuration-based approach

## 15. Summary ✅ IMPLEMENTED

The ViStreamASR architecture has been successfully refactored to improve maintainability, testability, and code organization while maintaining full backward compatibility. It adopts a U2 backbone for the ASR pipeline, enabling low-latency streaming with incremental decoding and controlled finalization.

Key improvements include:

- **Modular Design**: Clear separation of concerns across components
- **Helper Functions**: Eliminated code duplication and improved maintainability
- **Named Constants**: Replaced magic numbers for better readability
- **State Management**: Centralized state handling for better consistency
- **U2-based Streaming**: Unified streaming/non-streaming design for real-time performance
- **Streaming Architecture**: Facade pattern with specialized streamer classes
- **Enhanced Documentation**: Comprehensive documentation aligned with new architecture

The system provides high-performance, low-latency streaming ASR with predictable resource usage and configurable latency/accuracy trade-offs, consistent with the analysis in [`docs/COMPARISON_ANALYSIS.md`](docs/COMPARISON_ANALYSIS.md).

## Related Files and Components

### Core Implementation Files

- **[`src/vistreamasr/config.py`](src/vistreamasr/config.py)**: Configuration management
- **[`src/vistreamasr/logging.py`](src/vistreamasr/logging.py)**: Logging system
- **[`src/vistreamasr/core.py`](src/vistreamasr/core.py)**: Core ASR engine and processing
- **[`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py)**: Streaming interface and facade
- **[`src/vistreamasr/cli.py`](src/vistreamasr/cli.py)**: Command-line interface
- **[`src/vistreamasr/vad.py`](src/vistreamasr/vad.py)**: Voice activity detection

### Documentation Files

- **[`ARCHITECTURE.md`](ARCHITECTURE.md)**: This architecture document (U2 backbone clarified)
- **[`COMPARISON_ANALYSIS.md`](COMPARISON_ANALYSIS.md)**: Comparison of U2-based ViStreamASR vs Whisper-based systems
- **[`REQUIREMENTS.md`](REQUIREMENTS.md)**: System requirements and specifications
- **[`COMPONENT_OVERVIEW.md`](COMPONENT_OVERVIEW.md)**: Component overview and relationships
- **[`COMPONENT_CORE_PROCESSING_DOCS.md`](COMPONENT_CORE_PROCESSING_DOCS.md)**: Core processing documentation
- **[`COMPONENT_STREAMING_INTERFACE_DOCS.md`](COMPONENT_STREAMING_INTERFACE_DOCS.md)**: Streaming interface documentation
- **[`COMPONENT_CLI_INTERFACE_DOCS.md`](COMPONENT_CLI_INTERFACE_DOCS.md)**: CLI interface documentation
