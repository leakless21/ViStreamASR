let mediaRecorder;
let audioChunks = [];
let websocket;
let isRecording = false;
let audioContext;
let analyser;
let microphone;
let javascriptNode;
let animationId;

const recordButton = document.getElementById("recordButton");
const statusDiv = document.getElementById("status");
const transcriptContainer = document.getElementById("transcriptContainer");
const partialTranscriptDiv = document.getElementById("partialTranscript");
const waveformCanvas = document.getElementById("waveform");
const waveformCtx = waveformCanvas.getContext("2d");

// Set canvas size
function resizeCanvas() {
  waveformCanvas.width = waveformCanvas.offsetWidth;
  waveformCanvas.height = waveformCanvas.offsetHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// WebSocket connection
function connectWebSocket() {
  const wsUrl = `ws://${window.location.hostname}:${window.location.port}/asr`;
  websocket = new WebSocket(wsUrl);

  websocket.onopen = () => {
    console.log("WebSocket connection established.");
    updateStatus("WebSocket connected. Ready to record.");
    recordButton.disabled = false;
  };

  websocket.onmessage = (event) => {
    console.log("Message from server:", event.data);

    try {
      const data = JSON.parse(event.data);

      if (data.type === "partial") {
        partialTranscriptDiv.textContent = data.text || "";
      } else if (data.type === "final") {
        if (data.text && data.text.trim() !== "") {
          addTranscriptEntry(data.text, data.timestamp);
          partialTranscriptDiv.textContent = "";
        }
      }
    } catch (error) {
      console.error("Error parsing WebSocket message:", error);
      if (typeof event.data === "string") {
        addTranscriptEntry(event.data, "System"); // Fallback for non-JSON messages
      }
    }
  };

  websocket.onclose = (event) => {
    console.log("WebSocket connection closed:", event.code, event.reason);
    updateStatus("WebSocket disconnected.");
    recordButton.disabled = true;
    if (isRecording) {
      stopRecording();
    }
  };

  websocket.onerror = (error) => {
    console.error("WebSocket error:", error);
    updateStatus("WebSocket error. Check console.");
    recordButton.disabled = true;
    if (isRecording) {
      stopRecording();
    }
  };
}

function updateStatus(message) {
  statusDiv.textContent = `Status: ${message}`;
}

function addTranscriptEntry(text, timestamp) {
  const entryDiv = document.createElement("div");
  entryDiv.className = "transcript-entry";

  const timestampSpan = document.createElement("span");
  timestampSpan.className = "transcript-timestamp";
  // Format timestamp, assuming it's in seconds or a format that can be simply displayed
  // If timestamp is not provided, it will default to "Just now"
  timestampSpan.textContent = timestamp
    ? new Date(timestamp * 1000).toLocaleTimeString()
    : "Just now";

  const textSpan = document.createElement("span");
  textSpan.className = "transcript-text";
  textSpan.textContent = text;

  entryDiv.appendChild(timestampSpan);
  entryDiv.appendChild(textSpan);

  // Insert before the partial transcript div
  transcriptContainer.insertBefore(entryDiv, partialTranscriptDiv);
  transcriptContainer.scrollTop = transcriptContainer.scrollHeight; // Auto-scroll
}

async function startRecording() {
  if (isRecording) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    // Setup audio context for visualization
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    microphone = audioContext.createMediaStreamSource(stream);
    javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);

    analyser.smoothingTimeConstant = 0.8;
    analyser.fftSize = 1024;

    microphone.connect(analyser);
    analyser.connect(javascriptNode);
    javascriptNode.connect(audioContext.destination);

    javascriptNode.onaudioprocess = () => {
      if (isRecording) {
        drawWaveform();
      }
    };

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(event.data);
        } else {
          console.warn("WebSocket is not open. Cannot send audio data.");
        }
      }
    };

    mediaRecorder.onstop = () => {
      console.log("MediaRecorder stopped.");
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        // websocket.send(JSON.stringify({ type: 'audio_end' }));
      }
      audioChunks = [];
      if (audioContext) {
        audioContext.close(); // Close audio context to save resources
      }
    };

    audioChunks = [];
    mediaRecorder.start(100); // Collect data every 100ms
    isRecording = true;
    recordButton.textContent = "⏹️";
    recordButton.setAttribute("aria-label", "Stop Recording");
    recordButton.classList.add("recording");
    updateStatus("Recording...");
    drawWaveform(); // Start drawing
  } catch (err) {
    console.error("Error accessing microphone:", err);
    updateStatus("Error: Could not access microphone.");
    alert(
      "Could not access your microphone. Please ensure you have granted permission."
    );
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder || mediaRecorder.state === "inactive")
    return;

  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach((track) => track.stop());

  isRecording = false;
  recordButton.textContent = "🎤";
  recordButton.setAttribute("aria-label", "Start Recording");
  recordButton.classList.remove("recording");
  updateStatus("Recording stopped.");
  cancelAnimationFrame(animationId); // Stop drawing waveform
  waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height); // Clear canvas
}

function drawWaveform() {
  if (!isRecording || !analyser) return;

  animationId = requestAnimationFrame(drawWaveform);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);

  waveformCtx.fillStyle = "#fff";
  waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

  waveformCtx.lineWidth = 2;
  waveformCtx.strokeStyle = "#007bff";
  waveformCtx.beginPath();

  const sliceWidth = (waveformCanvas.width * 1.0) / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * waveformCanvas.height) / 2;

    if (i === 0) {
      waveformCtx.moveTo(x, y);
    } else {
      waveformCtx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
  waveformCtx.stroke();
}

recordButton.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// Initialize WebSocket connection on page load
window.addEventListener("load", () => {
  updateStatus("Connecting to WebSocket...");
  recordButton.disabled = true;
  connectWebSocket();
});

// Handle page unload
window.addEventListener("beforeunload", () => {
  if (websocket) {
    websocket.close();
  }
  if (isRecording) {
    stopRecording();
  }
});
