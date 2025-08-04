# Comparison Analysis: ViStreamASR vs WhisperLiveKit

This document provides a structured comparison between ViStreamASR and WhisperLiveKit across defaults, Voice Activity Detection (VAD), advanced streaming features, and practical implications for accuracy, latency, and resource usage.

References:

- ViStreamASR configuration: [`src/vistreamasr/config.py`](src/vistreamasr/config.py)
- ViStreamASR VAD: [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py)
- WhisperLiveKit overview: [`WhisperLiveKit/README.md`](WhisperLiveKit/README.md)

## 1. High-Level Default Settings Comparison

| Feature             | ViStreamASR ([`src/vistreamasr/config.py`](src/vistreamasr/config.py)) | WhisperLiveKit ([`README.md`](WhisperLiveKit/README.md))         | Notes                                                                                                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ASR Model           | U2 Backbone                                                            | Whisper (`tiny`)                                                 | `ViStreamASR` is built on a U2 (Unified Streaming and Non-streaming) architecture, which is fundamentally different from `WhisperLiveKit`'s use of OpenAI's Whisper model. U2 models are designed for low-latency streaming ASR. |
| Language            | "en"                                                                   | "en"                                                             | Both applications default to English for transcription.                                                                                                                                                                          |
| Backend/Device      | device="cuda", compute_type="float16"                                  | faster-whisper (CPU by default)                                  | Our application is optimized for GPU (cuda) out-of-the-box, using float16 for improved performance. WhisperLiveKit defaults to CPU and requires manual configuration for GPU usage.                                              |
| VAD Model           | Enabled (models/silero_vad.onnx)                                       | VAD is enabled, but the specific model is not Silero by default. | Both projects utilize Voice Activity Detection. Ours specifies the Silero VAD model path directly in the configuration.                                                                                                          |
| Max Speech Duration | 10.0 seconds                                                           | 30.0 seconds (--audio-max-len)                                   | WhisperLiveKit allows for a longer audio buffer by default when using its SimulStreaming backend.                                                                                                                                |
| Diarization         | Not configured by default                                              | False                                                            | Both have speaker diarization disabled by default.                                                                                                                                                                               |
| Server Host         | Not specified in config (typically localhost)                          | localhost                                                        | Both are set up for local hosting by default.                                                                                                                                                                                    |
| Server Port         | Not specified in config (typically 8000)                               | 8000                                                             | Both use the same default port.                                                                                                                                                                                                  |

## 2. In-Depth VAD and Advanced Feature Comparison

| Feature               | ViStreamASR              | WhisperLiveKit           | Analysis & Implications                                                                                                                                          |
| :-------------------- | :----------------------- | :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VAD Threshold         | 0.5                      | 0.5                      | Aligned. Both projects use the same default speech probability threshold.                                                                                        |
| Min Silence Duration  | 500 ms                   | 500 ms                   | Aligned. The configuration has been updated to ensure both projects use the same value.                                                                          |
| Speech Padding        | 100 ms                   | 100 ms                   | Aligned. The configuration has been updated to match.                                                                                                            |
| VAD Hysteresis        | Not implemented          | 0.15 (for end detection) | Feature Gap. WhisperLiveKit uses hysteresis to prevent premature speech end detection, making it more robust to short pauses. ViStreamASR lacks this feature.    |
| Punctuation Splitting | Not implemented          | False (by default)       | Feature Gap. WhisperLiveKit can use punctuation to create more natural transcript boundaries. ViStreamASR does not have this capability.                         |
| Confidence Validation | Not implemented          | False (by default)       | Feature Gap. WhisperLiveKit can use confidence scores to validate tokens faster, a feature ViStreamASR lacks.                                                    |
| Min Chunk Size        | Fixed model chunk/stride | 0.5s (time-based gate)   | Different Approaches. ViStreamASR processes audio in fixed-size chunks, while WhisperLiveKit waits for a minimum duration of audio, which can be more efficient. |
| Buffer Trimming       | Not implemented          | segment strategy, 15s    | Feature Gap. WhisperLiveKit has a strategy for trimming the audio buffer to manage memory and context, which is absent in ViStreamASR.                           |

Notes and pointers:

- ViStreamASR VAD parameters are set via configuration and applied in the VAD integration logic. See [`src/vistreamasr/config.py`](src/vistreamasr/config.py) and [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py) for the runtime behavior.
- WhisperLiveKit's advanced streaming features (e.g., hysteresis, min chunk time-gating, buffer trimming) are exposed through its SimulStreaming components. Refer to [`WhisperLiveKit/whisperlivekit/simul_whisper/config.py`](WhisperLiveKit/whisperlivekit/simul_whisper/config.py) and [`WhisperLiveKit/whisperlivekit/simul_whisper/eow_detection.py`](WhisperLiveKit/whisperlivekit/simul_whisper/eow_detection.py) for details.

## 3. Analysis and Recommendations

### Explanation of the min_silence_duration_ms Discrepancy

Root cause: A local default within the VAD processing component applied a hardcoded fallback for min_silence_duration_ms that differed from the central configuration exposed in the application-level config ([`src/vistreamasr/config.py`](src/vistreamasr/config.py)). As a result, even if the global configuration was set correctly, the VADProcessor's internal default could override it at runtime, causing inconsistent behavior.

Resolution: The VAD implementation has been updated to consistently source min_silence_duration_ms from the central configuration and pass it through end-to-end to the VAD execution path. This ensures that the effective value matches the configured value, aligning ViStreamASR with the intended 500 ms default and bringing it in line with WhisperLiveKit's default.

Key takeaways:

- Centralized configuration must be the single source of truth for all runtime components.
- Avoid hardcoded local defaults that can diverge from config and cause hidden discrepancies.
- Unit coverage should include assertions validating that configured VAD parameters propagate to runtime. See tests in [`tests/test_vad_integration.py`](tests/test_vad_integration.py).

### Suggestions for Feature Parity

To close the functional gaps and approach feature parity with WhisperLiveKit, the following enhancements are recommended for ViStreamASR:

1. Implement VAD hysteresis for robust end-of-speech detection

- Description: Add a hysteresis mechanism (e.g., a secondary lower threshold or a sustained-silence time window) to avoid premature cutoffs from brief pauses.
- Expected benefits: Reduced false segment breaks, smoother phrase boundaries, improved transcript readability.
- References for design: See WhisperLiveKit's end-of-word/speech logic in [`whisperlivekit/simul_whisper/eow_detection.py`](WhisperLiveKit/whisperlivekit/simul_whisper/eow_detection.py).

2. Add punctuation-based splitting

- Description: Post-process partial transcripts to split on punctuation (., !, ?) with a configurable toggle.
- Expected benefits: More natural segmentation for downstream consumers (captioning, UI updates, analytics).
- Defaults: Keep disabled by default to match the requested baseline; expose a configuration flag.

3. Implement confidence validation for token-level gating

- Description: Maintain a rolling confidence threshold for incremental tokens to decide whether to emit or delay low-confidence tokens until more context arrives.
- Expected benefits: Fewer corrections/overwrites in streaming UIs; more stable partials.

4. Consider time-based audio chunking (min chunk size gate)

- Description: Add a time-based minimum chunk duration (e.g., 0.5s) before feeding audio to the model when latency constraints allow.
- Expected benefits: Fewer model invocations and better batching efficiency, potentially reducing compute overhead without harming responsiveness in many cases.
- Implementation: Make this configurable with a lower bound for ultra-low-latency modes.

5. Add a buffer trimming strategy

- Description: Introduce a configurable audio and token buffer trimming strategy (e.g., segment-based with a default of 15s) to prevent unbounded growth and to preserve relevant context.
- Expected benefits: Stable memory usage, predictable latency, and avoidance of long-context degradation.

Roadmap integration and testing:

- Expose each feature behind config flags in [`src/vistreamasr/config.py`](src/vistreamasr/config.py).
- Extend VAD integration details and interfaces in [`src/vistreamasr/vad.py`](src/vistreamasr/vad.py) and the core pipeline in [`src/vistreamasr/core.py`](src/vistreamasr/core.py).
- Add unit tests in `tests/` mirroring WhisperLiveKit's expected behaviors, including hysteresis thresholds, punctuation toggles, confidence gate edge cases, and buffer trimming limits.
