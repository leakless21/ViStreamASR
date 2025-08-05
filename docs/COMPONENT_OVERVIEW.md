# ViStreamASR Component Overview

## System Architecture Overview 🔄 PLANNED

ViStreamASR is a modular, extensible Vietnamese Automatic Speech Recognition system designed with a focus on real-time performance and developer flexibility. The system architecture is based on Domain-Driven Design principles, with a clear separation of concerns that allows for independent development, testing, and deployment of each component.

### Architecture Principles

The system is built on several key architectural principles:

- **Modular ASR Backend**: A pluggable architecture that allows for easy integration of different ASR engines.
- **Asynchronous Processing**: The entire system is built on `asyncio` for concurrent, non-blocking processing.
- **Decoupled Components**: The architecture is composed of decoupled components, including an `AudioProcessor` for each client and a singleton `TranscriptionEngine`.
- **Configuration-Driven**: All components are configured through a centralized, hierarchical configuration system.
- **Error Resilience**: The system is designed to be resilient to errors, with graceful degradation and recovery mechanisms.

### Component Architecture

The system is organized into the following main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    ViStreamASR System                        │
├─────────────────────────────────────────────────────────────┤
│  Web Server Component                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  FastAPI Application │ WebSocket Endpoint                │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Application Layer Component                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  AudioProcessor (per client)                            │ │
│  │  TranscriptionEngine (singleton)                        │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ASR Backend Component                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ASRBackend (Base Class)                                │ │
│  │  ├─ FasterWhisperBackend                               │ │
│  │  └─ WhisperTimestampedBackend                          │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Diarization Component                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SpeakerDiarization (diart/pyannote.audio)              │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Core Services                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  VADProcessor      │ Configuration      │ Logging       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 0. Web Interface Component

Primary Responsibility: Provide an in-browser interface for live audio capture, WebSocket streaming to the ASR backend, waveform visualization, and real-time transcript rendering.

Documentation: [`COMPONENT_WEB_INTERFACE_DOCS.md`](docs/COMPONENT_WEB_INTERFACE_DOCS.md)

Related Files: [`index.html`](src/vistreamasr/web/static/index.html), [`style.css`](src/vistreamasr/web/static/style.css), [`client.js`](src/vistreamasr/web/static/client.js)

### 1. Web Server Component

**Primary Responsibility**: Handle client connections and real-time communication.

**Key Functions**:

- **FastAPI Application**: Serves the web frontend and manages the WebSocket lifecycle.
- **WebSocket Endpoint**: Handles real-time, bidirectional communication with clients.

### 2. Application Layer Component

**Primary Responsibility**: Contain the core logic for processing audio and managing transcription tasks.

**Key Functions**:

- **AudioProcessor**: Manages the audio processing pipeline for a single client.
- **TranscriptionEngine**: Manages the ASR models and performs transcription (singleton).

### 3. ASR Backend Component

**Primary Responsibility**: Provide a pluggable interface for different speech recognition engines.

**Key Functions**:

- **ASRBackend (Base Class)**: Defines the common interface for all ASR backends.
- **Pluggable Backends**: Concrete implementations for different ASR engines (e.g., `faster-whisper`).

### 4. Diarization Component

**Primary Responsibility**: Identify different speakers in the audio.

**Key Functions**:

- **SpeakerDiarization**: Performs speaker diarization using a library like `diart` or `pyannote.audio`.

### 5. Core Services

**Primary Responsibility**: Provide essential functionalities that are used across the entire system.

**Key Functions**:

- **VADProcessor**: Detects voice activity in the audio stream.
- **Configuration Management**: Manages all configuration for the system.
- **Structured Logging**: Provides structured, configurable logging.

## Data Flow Architecture

### Real-time Transcription Data Flow

1.  **Client Connection**: A client connects to the `FastAPI` server via a WebSocket.
2.  **Audio Streaming**: The client streams audio to the server.
3.  **Audio Processing**: For each client, an `AudioProcessor` instance is created, which:
    - Uses the `VADProcessor` to detect speech.
    - Sends speech segments to the `TranscriptionEngine`.
    - Uses the `SpeakerDiarization` component to identify the speaker.
4.  **Transcription**: The `TranscriptionEngine` uses the configured ASR backend to transcribe the audio.
5.  **Results**: The transcription and speaker information are sent back to the client in real-time.

## Integration Points

### Configuration Integration

All components are configured through the centralized `ViStreamASRSettings` object, which allows for easy management of all system parameters.

### Asynchronous Integration

The use of `asyncio` allows for seamless integration of all components, enabling concurrent processing of audio, transcription, and diarization.

## Performance Characteristics

- **Real-Time Factor**: The system is designed to maintain a low Real-Time Factor (RTF) for real-time applications.
- **Scalability**: The asynchronous architecture allows the system to scale to handle many concurrent clients.
- **Latency**: The use of VAD and efficient processing pipelines minimizes end-to-end latency.

## Error Handling and Recovery

The system includes robust error handling and recovery mechanisms, including graceful degradation, automatic retries, and clear user feedback.

## Testing Strategy

The system has a comprehensive testing strategy, including unit tests for each component, integration tests for end-to-end workflows, and performance tests to validate real-time capabilities.

## Deployment Considerations

The system can be deployed as a standalone application, a server, or in the cloud. The configuration system allows for easy adaptation to different deployment environments.

## Future Roadmap

The modular architecture of the system provides a solid foundation for future enhancements, including:

- Advanced punctuation and speaker change detection.
- Support for more ASR backends and diarization libraries.
- A plugin system for custom components.
- Enhanced monitoring and analytics.

## Conclusion

The ViStreamASR system is a comprehensive, real-time speech-to-text solution with a modular, asynchronous architecture. It provides a flexible and extensible platform for a wide range of speech recognition applications.
