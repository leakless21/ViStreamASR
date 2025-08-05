# Web Interface Component

Component name: Web Interface
Location: [`src/vistreamasr/web/static`](src/vistreamasr/web/static)

Purpose
The Web Interface provides an in-browser, zero-install client for live audio capture, streaming to the backend via WebSocket, real-time waveform visualization, and incremental transcript rendering (partial and final). It is designed for manual testing, demos, and low-friction operator workflows.

Key features

- Microphone recording using MediaRecorder with chunked sending every 100 ms
- WebSocket signaling and streaming to the ASR server endpoint
- Real-time partial transcript display and finalized transcript entries
- Timeline-based transcript view with a timestamp for each final segment (speaker labels removed)
- Canvas-based audio waveform visualization from the live microphone stream
- Responsive UI with a central record button and visual recording state

Related files

- [`index.html`](src/vistreamasr/web/static/index.html)
- [`style.css`](src/vistreamasr/web/static/style.css)
- [`client.js`](src/vistreamasr/web/static/client.js)

HTML structure overview ([`index.html`](src/vistreamasr/web/static/index.html))
The page establishes a minimal, focused layout:

- Title/header:
  - h1: ViStreamASR Web Interface
- Controls region:
  - div#controls: container for input and visualization
    - div#waveformContainer: container framing the canvas
      - canvas#waveform: rendering target for the live waveform
    - button#recordButton: circular primary control that toggles recording. The button displays a microphone icon (🎤) when idle and a stop icon (⏹️) when recording. It includes aria-labels for accessibility, initially set to “Start Recording” and toggled to “Stop Recording” while recording.
- Status region:
  - div#status: shows connection and recording status, e.g., “Status: Idle”, “Connecting to WebSocket…”
- Transcript region:
  - div#transcriptContainer: scrollable container holding:
    - dynamically added .transcript-entry (finalized utterances)
    - div#partialTranscript: continuously updated with partial results
- Script bundle:
  - script src="client.js": client logic binding UI and audio/WebSocket

Styling and animations ([`style.css`](src/vistreamasr/web/static/style.css))
Layout and typography

- Global:
  - body uses a system-ui font stack and centers column layout with generous spacing, light background (#f8f9fa), and dark text.
  - h1 styled with stronger weight and spacing for prominence.
- Controls:
  - #controls arranged vertically with gap for consistent spacing.
- Waveform:
  - #waveformContainer: responsive card with fixed height, border, and rounded corners; contains the canvas.
  - #waveform: takes full container width/height.
- Record button:
  - #recordButton: circular (80×80), primary blue background (#007bff), hover scaling, subtle shadow, disabled state styling.
  - .recording state switches to danger red (#dc3545) and applies a pulse animation.
- Pulse animation:
  - @keyframes pulse controls a ring-like expanding box shadow to indicate active recording.
- Status:
  - #status styled as subdued informational text.
- Transcript:
  - #transcriptContainer: card-like panel with border, rounded corners, shadow, min/max height with vertical scroll.
  - .transcript-entry: neutral card with left blue accent and padding.
  - .transcript-timestamp: small, muted timestamp displayed above each text segment.
  - .transcript-text: main text content for the finalized segment.
  - #partialTranscript: muted color, italicized, separated by a top border for clarity.

Client-side logic ([`client.js`](src/vistreamasr/web/static/client.js))
Key modules and state

- DOM references:
  - recordButton, statusDiv, transcriptContainer, partialTranscriptDiv, waveformCanvas (+ 2D context)
- Audio/recording:
  - MediaRecorder instance and a ring buffer of audioChunks
  - AudioContext with AnalyserNode + ScriptProcessorNode for waveform sampling
- WebSocket:
  - Connection to ws://{hostname}:{port}/asr
- UI state:
  - isRecording boolean

Initialization

- Canvas resizing:
  - resizeCanvas() sets canvas width/height from offset sizes; runs on load and window resize to maintain crisp rendering.
- WebSocket connect:
  - connectWebSocket():
    - onopen: enables the record button and updates status.
    - onmessage: parses JSON payloads; handles:
      - type === "partial": update partialTranscriptDiv with data.text
      - type === "final": if non-empty, add a finalized transcript entry with a timestamp and clear partial
    - onclose/onerror: disables record button, updates status, and if recording, stops capture.
  - Page load hook:
    - Shows “Connecting to WebSocket…”, disables record button, then calls connectWebSocket().

Recording workflow

- Start:
  - startRecording():
    - Requests microphone via navigator.mediaDevices.getUserMedia({ audio: true }).
    - Sets up AudioContext:
      - createAnalyser(), createMediaStreamSource(stream), createScriptProcessor(2048, 1, 1).
      - Configures analyser (smoothingTimeConstant=0.8, fftSize=1024).
      - Hooks onaudioprocess to drawWaveform() when isRecording is true.
    - Creates MediaRecorder(stream) and starts with a 100 ms timeslice for chunking.
    - ondataavailable:
      - For each chunk, send event.data binary to server if WebSocket is OPEN.
    - onstop:
      - Clears audioChunks and closes AudioContext to release resources.
    - UI updates:
      - isRecording = true, button text → “Stop Recording”, add .recording class, status “Recording…”, kick off drawWaveform().
    - Error handling:
      - If microphone access fails, updates status and alerts user.
- Stop:
  - stopRecording():
    - Stops MediaRecorder, stops tracks, resets UI (text → “Start Recording”, remove .recording), clears waveform via cancelAnimationFrame and canvas clear.

Waveform visualization

- drawWaveform():
  - Uses analyser.getByteTimeDomainData() on a Uint8Array of frequencyBinCount size.
  - Clears canvas and draws a 2px #007bff path across the center line using a computed sliceWidth, mapping amplitudes around the vertical midpoint.
  - Scheduled with requestAnimationFrame while recording.

Transcript rendering

- addTranscriptEntry(text, timestamp):
  - Creates .transcript-entry with:
    - .transcript-timestamp showing a localized time derived from the numeric timestamp (seconds) if provided; otherwise displays “Just now”.
    - .transcript-text containing the final text.
  - Inserts before partialTranscriptDiv and auto-scrolls to the bottom.
- Partial vs final:
  - partialTranscriptDiv shows non-committed interim hypotheses.
  - Final messages are appended as discrete entries with a visible timestamp and clear the partial line.

WebSocket message format

- Expected inbound shapes from server:
  - Partial:
    - {"type": "partial", "text": "&lt;string&gt;"}
  - Final:
    - {"type": "final", "text": "&lt;string&gt;", "timestamp": &lt;number (seconds)&gt; | null}
- Outbound audio:
  - Raw binary audio chunks from MediaRecorder (Blob). The exact MIME type commonly defaults to audio/webm; servers should be prepared to demux/convert.

User interactions

- Primary control is the Record button:
  - Toggles start/stopRecording().
  - Disabled until WebSocket is connected.
  - Visual pulsing and red color indicate active capture.

Error handling and resilience

- Graceful handling of:
  - WebSocket failures (disables recording; stops if active).
  - Microphone permission denial with user alert.
- AudioContext cleanly closed on stop to free resources.
- onbeforeunload closes WebSocket and stops any active recording.

Customization and extension points

- WebSocket path and host:
  - Constructed from location host/port to “/asr”; adjust if server mapping changes.
- Timeline and timestamp formatting:
  - Timestamps are rendered via new Date(timestamp * 1000).toLocaleTimeString(); alter formatting for absolute times, relative elapsed times, or include date if needed.
- MediaRecorder codec/timeslice:
  - Adjust start(timeslice) or specify mimeType if the backend prefers a particular container/codec.
- Waveform styling:
  - Modify stroke color/width and background fill for different aesthetics.
- Accessibility:
  - Add ARIA labels and keyboard bindings as needed.

Dependencies

- Native Web APIs only (no external JS libs):
  - MediaRecorder, Web Audio API (AudioContext/AnalyserNode), WebSocket, Canvas 2D.

Security and privacy considerations

- Microphone access:
  - Requires HTTPS in browsers for production and explicit user permission.
- Data handling:
  - Audio chunks sent to backend in near-real-time; ensure transport security and server-side validation.

Operational notes

- The web static assets are intended to be served by the FastAPI server component alongside a WebSocket endpoint at /asr.
- The interface assumes that upon connecting, the WebSocket is ready to accept raw audio chunks and will return well-formed JSON messages for transcription updates.

Change log summary (timeline redesign)

- Replaced speaker-based alternating labels with a timeline-based view.
- Each final transcript entry now displays a timestamp instead of a speaker label.
- Updated styles to emphasize timestamp and neutral entry cards.
- Updated client logic to parse {"type":"final","text", "timestamp"} and render localized time strings.

File index

- HTML: [`index.html`](src/vistreamasr/web/static/index.html)
- Styles: [`style.css`](src/vistreamasr/web/static/style.css)
- Client logic: [`client.js`](src/vistreamasr/web/static/client.js)
# Web Interface Component

Component name: Web Interface
Location: [`src/vistreamasr/web/static`](src/vistreamasr/web/static)

Purpose
The Web Interface provides an in-browser, zero-install client for live audio capture, streaming to the backend via WebSocket, real-time waveform visualization, and incremental transcript rendering (partial and final). It is designed for manual testing, demos, and low-friction operator workflows.

Key features

- Microphone recording using MediaRecorder with chunked sending every 100 ms
- WebSocket signaling and streaming to the ASR server endpoint
- Real-time partial transcript display and finalized transcript entries
- Simple alternating speaker labeling for demo purposes
- Canvas-based audio waveform visualization from the live microphone stream
- Responsive UI with a central record button and visual recording state

Related files

- [`index.html`](src/vistreamasr/web/static/index.html)
- [`style.css`](src/vistreamasr/web/static/style.css)
- [`client.js`](src/vistreamasr/web/static/client.js)

HTML structure overview ([`index.html`](src/vistreamasr/web/static/index.html))
The page establishes a minimal, focused layout:

- Title/header:
  - h1: ViStreamASR Web Interface
- Controls region:
  - div#controls: container for input and visualization
    - div#waveformContainer: container framing the canvas
      - canvas#waveform: rendering target for the live waveform
    - button#recordButton: circular primary control that toggles recording. The button displays a microphone icon (🎤) when idle and a stop icon (⏹️) when recording. It includes aria-labels for accessibility, initially set to “Start Recording” and toggled to “Stop Recording” while recording.
- Status region:
  - div#status: shows connection and recording status, e.g., “Status: Idle”, “Connecting to WebSocket…”
- Transcript region:
  - div#transcriptContainer: scrollable container holding:
    - dynamically added .transcript-entry (finalized utterances)
    - div#partialTranscript: continuously updated with partial results
- Script bundle:
  - script src="client.js": client logic binding UI and audio/WebSocket

Styling and animations ([`style.css`](src/vistreamasr/web/static/style.css))
Layout and typography

- Global:
  - body uses a system-ui font stack and centers column layout with generous spacing, light background (#f8f9fa), and dark text.
  - h1 styled with stronger weight and spacing for prominence.
- Controls:
  - #controls arranged vertically with gap for consistent spacing.
- Waveform:
  - #waveformContainer: responsive card with fixed height, border, and rounded corners; contains the canvas.
  - #waveform: takes full container width/height.
- Record button:
  - #recordButton: circular (80×80), primary blue background (#007bff), hover scaling, subtle shadow, disabled state styling.
  - .recording state switches to danger red (#dc3545) and applies a pulse animation.
- Pulse animation:
  - @keyframes pulse controls a ring-like expanding box shadow to indicate active recording.
- Status:
  - #status styled as subdued informational text.
- Transcript:
  - #transcriptContainer: card-like panel with border, rounded corners, shadow, min/max height with vertical scroll.
  - .transcript-entry: pill-like blocks with subtle background and padding.
  - .speaker-1 and .speaker-2:
    - Alternate background colors and text alignment to visually distinguish speakers.
  - .speaker-label:
    - Bold label prefix (e.g., “Speaker 1:”).
  - #partialTranscript:
    - Muted color, italicized, separated by a top border for clarity.

Client-side logic ([`client.js`](src/vistreamasr/web/static/client.js))
Key modules and state

- DOM references:
  - recordButton, statusDiv, transcriptContainer, partialTranscriptDiv, waveformCanvas (+ 2D context)
- Audio/recording:
  - MediaRecorder instance and a ring buffer of audioChunks
  - AudioContext with AnalyserNode + ScriptProcessorNode for waveform sampling
- WebSocket:
  - Connection to ws://{hostname}:{port}/asr
- UI state:
  - isRecording boolean; currentSpeaker simple alternator starting at 1

Initialization

- Canvas resizing:
  - resizeCanvas() sets canvas width/height from offset sizes; runs on load and window resize to maintain crisp rendering.
- WebSocket connect:
  - connectWebSocket():
    - onopen: enables the record button and updates status.
    - onmessage: parses JSON payloads; handles:
      - type === "partial": update partialTranscriptDiv with data.text
      - type === "final": if non-empty, add a finalized transcript entry and clear partial; alternates currentSpeaker (1 ↔ 2)
    - onclose/onerror: disables record button, updates status, and if recording, stops capture.
  - Page load hook:
    - Shows “Connecting to WebSocket…”, disables record button, then calls connectWebSocket().

Recording workflow

- Start:
  - startRecording():
    - Requests microphone via navigator.mediaDevices.getUserMedia({ audio: true }).
    - Sets up AudioContext:
      - createAnalyser(), createMediaStreamSource(stream), createScriptProcessor(2048, 1, 1).
      - Configures analyser (smoothingTimeConstant=0.8, fftSize=1024).
      - Hooks onaudioprocess to drawWaveform() when isRecording is true.
    - Creates MediaRecorder(stream) and starts with a 100 ms timeslice for chunking.
    - ondataavailable:
      - For each chunk, send event.data binary to server if WebSocket is OPEN.
    - onstop:
      - Clears audioChunks and closes AudioContext to release resources.
    - UI updates:
      - isRecording = true, button text → “Stop Recording”, add .recording class, status “Recording…”, kick off drawWaveform().
    - Error handling:
      - If microphone access fails, updates status and alerts user.
- Stop:
  - stopRecording():
    - Stops MediaRecorder, stops tracks, resets UI (text → “Start Recording”, remove .recording), clears waveform via cancelAnimationFrame and canvas clear.

Waveform visualization

- drawWaveform():
  - Uses analyser.getByteTimeDomainData() on a Uint8Array of frequencyBinCount size.
  - Clears canvas and draws a 2px #007bff path across the center line using a computed sliceWidth, mapping amplitudes around the vertical midpoint.
  - Scheduled with requestAnimationFrame while recording.

Transcript rendering

- addTranscriptEntry(text, speaker):
  - Creates .transcript-entry with:
    - .speaker-label containing “Speaker X:”
    - .transcript-text with the final text
  - Applies a CSS class speaker-{X} by parsing the speaker label, enabling alternate colors/layout.
  - Inserts before partialTranscriptDiv and auto-scrolls to the bottom.
- Partial vs final:
  - partialTranscriptDiv shows non-committed interim hypotheses.
  - Final messages are appended as discrete entries and clear the partial line.

WebSocket message format

- Expected inbound shapes from server:
  - Partial:
    - {"type": "partial", "text": "<string>"}
  - Final:
    - {"type": "final", "text": "<string>", "speaker": "Speaker 1" | "Speaker 2" | null}
- Outbound audio:
  - Raw binary audio chunks from MediaRecorder (Blob). The exact MIME type commonly defaults to audio/webm; servers should be prepared to demux/convert.

User interactions

- Primary control is the Record button:
  - Toggles start/stopRecording().
  - Disabled until WebSocket is connected.
  - Visual pulsing and red color indicate active capture.

Error handling and resilience

- Graceful handling of:
  - WebSocket failures (disables recording; stops if active).
  - Microphone permission denial with user alert.
- AudioContext cleanly closed on stop to free resources.
- onbeforeunload closes WebSocket and stops any active recording.

Customization and extension points

- WebSocket path and host:
  - Constructed from location host/port to “/asr”; adjust if server mapping changes.
- Speaker labeling:
  - currentSpeaker alternation is a placeholder; replace with true diarization labels if provided by the backend.
- MediaRecorder codec/timeslice:
  - Adjust start(timeslice) or specify mimeType if the backend prefers a particular container/codec.
- Waveform styling:
  - Modify stroke color/width and background fill for different aesthetics.
- Accessibility:
  - Add ARIA labels and keyboard bindings as needed.

Dependencies

- Native Web APIs only (no external JS libs):
  - MediaRecorder, Web Audio API (AudioContext/AnalyserNode), WebSocket, Canvas 2D.

Security and privacy considerations

- Microphone access:
  - Requires HTTPS in browsers for production and explicit user permission.
- Data handling:
  - Audio chunks sent to backend in near-real-time; ensure transport security and server-side validation.

Operational notes

- The web static assets are intended to be served by the FastAPI server component alongside a WebSocket endpoint at /asr.
- The interface assumes that upon connecting, the WebSocket is ready to accept raw audio chunks and will return well-formed JSON messages for transcription updates.

Change log summary (redesign scope)

- Introduced responsive, card-based UI for waveform and transcript.
- Added pulsing circular record button with recording state feedback.
- Implemented canvas-based waveform visualization via AnalyserNode.
- Added partial vs final transcript distinction with auto-scroll.
- Hardened WebSocket lifecycle handling and UI-disable on errors.
- Modularized render and update helpers for maintainability.

File index

- HTML: [`index.html`](src/vistreamasr/web/static/index.html)
- Styles: [`style.css`](src/vistreamasr/web/static/style.css)
- Client logic: [`client.js`](src/vistreamasr/web/static/client.js)
