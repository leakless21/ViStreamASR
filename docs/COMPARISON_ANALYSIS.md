# Comparison Analysis: ViStreamASR vs. WhisperLiveKit

This document provides a detailed comparison between our current project, `ViStreamASR`, and the reference program, `WhisperLiveKit`. The goal is to identify potential improvements, new features, and architectural changes that can be incorporated into `ViStreamASR`.

## 1. High-Level Overview

### WhisperLiveKit

`WhisperLiveKit` is a real-time, fully local speech-to-text solution with speaker diarization. It is built on `WhisperStreaming` and `SimulStreaming`, providing a ready-to-use backend and a simple, customizable frontend.

**Key Features:**

- Real-time transcription
- Speaker diarization
- Multi-user support
- Multiple backend support (`faster-whisper`, `simulstreaming`, etc.)
- Web interface (via `FastAPI` and WebSockets)

### ViStreamASR

`ViStreamASR` is a streaming ASR library with a focus on Vietnamese, integrating `silero-vad` for voice activity detection.

**Key Features:**

- Streaming ASR
- VAD integration (`silero-vad`)
- Command-line interface

## 2. Detailed Comparison

| Feature/Aspect          | WhisperLiveKit                                                                                                                              | ViStreamASR                                                                              | Analysis & Recommendation                                                                                                                                                                                                                                                           |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core ASR Engine**     | Pluggable backends (`faster-whisper`, `simulstreaming`, OpenAI Whisper, etc.)                                                               | Appears to have a more tightly coupled implementation.                                   | **Recommendation:** Refactor `ViStreamASR` to support multiple ASR backends. This would allow users to choose the best engine for their needs (e.g., `faster-whisper` for speed, `whisper` for accuracy).                                                                           |
| **Speaker Diarization** | Integrated with `diart`.                                                                                                                    | Not present.                                                                             | **Recommendation:** Integrate a diarization library like `diart` or `pyannote.audio`. This would be a major new feature, enabling `ViStreamASR` to identify different speakers in a conversation.                                                                                   |
| **Web Interface**       | `FastAPI` server with a WebSocket for real-time communication. Provides an HTML/JS frontend.                                                | No built-in web interface.                                                               | **Recommendation:** Implement a `FastAPI` or similar web server with a WebSocket interface. This would make `ViStreamASR` much more accessible and easier to integrate into web applications.                                                                                       |
| **Architecture**        | Decoupled architecture with a `TranscriptionEngine` singleton and an `AudioProcessor` per client. Uses `asyncio` for concurrent processing. | The architecture is less clear from the file structure alone, but seems more monolithic. | **Recommendation:** Adopt a more modular, asynchronous architecture similar to `WhisperLiveKit`. This would improve scalability and maintainability. The use of a singleton for the transcription engine is a good pattern to follow to avoid reloading models for each connection. |
| **Dependencies**        | `fastapi`, `websockets`, `faster-whisper`, `diart`.                                                                                         | `silero-vad`, `flashlight-text`.                                                         | **Recommendation:** Add dependencies like `fastapi`, `websockets`, and `diart` to support the recommended new features.                                                                                                                                                             |
| **Configuration**       | Extensive command-line arguments for configuration.                                                                                         | Configuration seems to be handled through a `config.py` file.                            | **Recommendation:** While a config file is good, adding command-line arguments (perhaps using a library like `click` or `argparse`) would provide more flexibility for users.                                                                                                       |

## 3. Proposed Changes for ViStreamASR

Based on the comparison, here is a list of concrete recommendations for improving `ViStreamASR`:

1.  **Modularize the ASR Backend:**

    - **Benefit:** Allows users to switch between different Whisper implementations.
    - **Changes:**
      - Create a base `ASRBackend` class with a common interface.
      - Implement different backend classes (e.g., `FasterWhisperBackend`, `WhisperTimestampedBackend`).
      - Update the core processing logic to use the selected backend.

2.  **Integrate Speaker Diarization:**

    - **Benefit:** Adds the ability to identify who is speaking.
    - **Changes:**
      - Add `diart` as a dependency.
      - Create a `Diarization` class that wraps `diart`.
      - Integrate the diarization results with the transcription, similar to how `WhisperLiveKit` does it.

3.  **Implement a Web Server and Frontend:**

    - **Benefit:** Provides a user-friendly way to interact with the ASR system.
    - **Changes:**
      - Add `fastapi` and `websockets` as dependencies.
      - Create a `FastAPI` application that serves a simple HTML/JS frontend.
      - Implement a WebSocket endpoint that accepts audio streams and returns real-time transcriptions.

4.  **Refactor for Asynchronous Processing:**
    - **Benefit:** Improves performance and scalability.
    - **Changes:**
      - Use `asyncio` to handle audio processing, transcription, and diarization concurrently.
      - Adopt a similar architecture to `WhisperLiveKit`'s `AudioProcessor` to manage these tasks.

By implementing these changes, `ViStreamASR` can evolve from a focused ASR library into a more comprehensive and versatile real-time speech-to-text solution.
