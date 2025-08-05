# Component: FFmpeg Manager

**Component Name:** `ffmpeg_manager`

## 1. Introduction

The FFmpeg Manager is a critical component responsible for handling all interactions with the FFmpeg library. It provides a simplified and robust interface for capturing, processing, and converting audio streams from various sources, making them suitable for the ASR engine.

This component is essential for ensuring that the audio data fed into the ASR pipeline is in the correct format (e.g., 16-bit, 16kHz, mono PCM audio), which is a mandatory requirement for most speech recognition models.

**Source Code:**

- [`src/vistreamasr/ffmpeg_manager.py`](src/vistreamasr/ffmpeg_manager.py:1)

## 2. Responsibilities

The primary responsibilities of the FFmpeg Manager are:

- **Audio Capture:** Capture audio from a specified input source (e.g., microphone, system audio).
- **Audio Conversion:** Convert the captured audio into a standardized raw PCM format.
- **Error Handling:** Manage and report errors that may occur during FFmpeg processes.
- **Process Management:** Handle the lifecycle of FFmpeg subprocesses, ensuring they are started and stopped correctly.

## 3. Interfaces

The FFmpeg Manager interacts with the following components:

- **Streaming Interface (`streaming.py`):** The Streaming Interface uses the FFmpeg Manager to obtain the audio stream that is fed into the ASR engine.
- **Configuration (`config.py`):** The manager is configured via the main application configuration, which specifies the audio source and other parameters.

## 4. Technical Requirements

| Requirement | Description                                                                                            | Status      |
| ----------- | ------------------------------------------------------------------------------------------------------ | ----------- |
| **FFM-001** | The component **must** be able to capture audio from a system microphone.                              | Implemented |
| **FFM-002** | The component **must** convert the captured audio to 16kHz, 16-bit mono PCM format.                    | Implemented |
| **FFM-003** | The component **must** handle FFmpeg process errors gracefully and provide diagnostic information.     | Implemented |
| **FFM-004** | The component **should** support capturing audio from other sources, such as files or network streams. | Planned     |

## 5. Known Issues and Gaps

- **High-Priority Issue:** As noted in [`GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md:23), there is an open issue related to the FFmpeg process not being correctly terminated on exit, which can lead to resource leaks. This is a high-priority bug that needs to be addressed.
