# Gap Analysis

This document tracks bugs and missing features.

## Resolved Issues

### [GAP-001] Missing `web` task in `pixi.toml`

- **Description:** The `pixi run web` command was failing with the error `web: command not found` because the `web` task was not defined in the `[tasks]` section of the [`pixi.toml`](pixi.toml:1) file.
- **Root Cause:** The `web` task, which should execute `python -m vistreamasr web`, was missing from the project's task configuration.
- **Resolution:** Added the `web = "python -m vistreamasr web"` task definition to the `[tasks]` section in [`pixi.toml`](pixi.toml:10).
- **Date Resolved:** 2025-08-05

### [GAP-002] Missing `debug` attribute in `ModelConfig`

- **Description:** The application failed to start with an `AttributeError: 'ModelConfig' object has no attribute 'debug'`. This error occurred during the FastAPI server's `lifespan` startup event when the `StreamingASR` service attempted to access `settings.model.debug`.
- **Root Cause:** The `ModelConfig` class in [`src/vistreamasr/config.py`](src/vistreamasr/config.py:12) was missing the `debug` attribute, which was being accessed in the `StreamingASR` class's `__init__` method in [`src/vistreamasr/streaming.py`](src/vistreamasr/streaming.py:369).
- **Resolution:** Added a `debug: bool` field with a default value of `False` and an appropriate description to the `ModelConfig` class in [`src/vistreamasr/config.py`](src/vistreamasr/config.py:40).
- **Date Resolved:** 2025-08-05

## Open Issues

### [GAP-003] FFmpeg Process Transport Closure Issue

- **Description:** The FFmpeg process starts successfully but its stdin transport immediately closes, causing continuous "WriteUnixTransport closed=True" errors when trying to write audio data. This prevents audio streaming functionality from working.
- **Root Cause:** Two potential causes identified:
  1. **Process Health Validation Gap**: The code doesn't verify that the FFmpeg process remains alive and healthy after creation
  2. **Race Condition**: No synchronization mechanism ensures FFmpeg is ready to receive input before write operations begin
- **Symptoms:**
  - FFmpeg starts with "FFmpeg started." log message
  - Immediate "Cannot read, FFmpeg state: FFmpegState.STARTING" warning
  - Continuous "Error writing to FFmpeg: unable to perform operation on <WriteUnixTransport closed=True...>" errors
- **Current Status:** Diagnostic logging added to [`src/vistreamasr/ffmpeg_manager.py`](src/vistreamasr/ffmpeg_manager.py:49) to validate the hypothesis
- **Test Plan:**
  1. Run `pixi run web` and connect via WebSocket
  2. Observe enhanced diagnostic logs to identify exact failure point
  3. Check if FFmpeg process terminates immediately or if stdin closes prematurely
- **Priority:** HIGH - Blocks core audio streaming functionality
- **Date Identified:** 2025-08-05
