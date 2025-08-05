from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging
import json
import numpy as np
from pathlib import Path # Ensure Path is imported at the top

# ViStreamASR imports
from vistreamasr.core import ASREngine
from vistreamasr.config import get_settings
from vistreamasr.streaming import StreamingASR
from ..ffmpeg_manager import FFmpegManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup...")
    settings = get_settings()
    # Assuming ModelConfig has a 'debug' field, or handle its absence.
    # For now, defaulting to False if not present.
    model_debug_mode = getattr(settings.model, 'debug', False)
    logger.info(f"Loaded settings: Model chunk size {settings.model.chunk_size_ms}ms, VAD enabled: {settings.vad.enabled}, Model Debug: {model_debug_mode}")
    
    # Initialize StreamingASR, which in turn initializes ASREngine
    # The StreamingASR instance will be shared across WebSocket connections.
    # The ASREngine's state will be reset per connection.
    streaming_asr_service = StreamingASR(settings=settings, debug=model_debug_mode)
    logger.info(f"StreamingASR service instance created. Engine attribute before explicit init: {streaming_asr_service.engine}")
    
    # Explicitly trigger engine initialization for diagnostic purposes
    try:
        logger.info("Attempting explicit initialization of StreamingASR engine...")
        streaming_asr_service._ensure_engine_initialized()
        logger.info(f"StreamingASR engine initialized successfully. Engine: {streaming_asr_service.engine}")
        if streaming_asr_service.engine and streaming_asr_service.engine.state.acoustic_model:
            logger.info("StreamingASR engine's acoustic model is loaded.")
        else:
            logger.warning("StreamingASR engine initialized, but acoustic model might not be loaded.")
    except Exception as e:
        logger.error(f"Error during explicit StreamingASR engine initialization in lifespan: {e}", exc_info=True)
        # Decide if the app should fail to start or continue with a degraded state
        # For now, we'll let it continue, but the WebSocket will likely fail.

    app.state.streaming_asr_service = streaming_asr_service
    logger.info("StreamingASR service (potentially initialized) stored in app state.")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown...")
    # Potentially add cleanup for streaming_asr_service if needed in the future
    logger.info("StreamingASR service shutdown complete.")

app = FastAPI(lifespan=lifespan)

# Serve static files from the 'static' directory
# Define static_dir_path outside the try-except to ensure it's always bound for the except block.
static_dir_path = Path(__file__).parent / "static"
try:
    app.mount("/static", StaticFiles(directory=str(static_dir_path)), name="static")
    logger.info(f"Mounted static files directory '{static_dir_path}' at /static")
except RuntimeError as e:
    logger.error(f"Failed to mount static files: {e}. Ensure 'static' directory exists at '{static_dir_path}'")
except Exception as e: # Catch other potential errors during mounting
    logger.error(f"An unexpected error occurred while mounting static files: {e}")


@app.get("/")
async def get():
    """
    Serves the main index.html file.
    """
    return {"message": "ViStreamASR Web Server. Go to /static/index.html for the UI."}

async def ffmpeg_writer(websocket: WebSocket, ffmpeg_manager: FFmpegManager):
    try:
        while True:
            data = await websocket.receive_bytes()
            await ffmpeg_manager.write_data(data)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected, stopping ffmpeg writer.")

async def ffmpeg_reader(websocket: WebSocket, ffmpeg_manager: FFmpegManager, streaming_asr: StreamingASR):
    try:
        if not streaming_asr.engine:
            logger.error("ASR engine not initialized.")
            return

        while True:
            audio_chunk_raw = await ffmpeg_manager.read_data(4096 * 2)
            if not audio_chunk_raw:
                await asyncio.sleep(0.1)
                continue

            audio_chunk = np.frombuffer(audio_chunk_raw, dtype=np.int16).astype(np.float32) / 32768.0
            result = streaming_asr.engine.process_audio(audio_chunk, is_last=False)

            if result:
                if result.get('current_transcription'):
                    partial_text = result['current_transcription']
                    await websocket.send_text(json.dumps({"type": "partial", "text": partial_text}))
                    logger.debug(f"Sent partial transcription: '{partial_text}'")
                
                if result.get('new_final_text'):
                    final_text = result['new_final_text']
                    await websocket.send_text(json.dumps({"type": "final", "text": final_text}))
                    logger.info(f"Sent final transcription: '{final_text}'")

    except Exception as e:
        logger.error("Error in ffmpeg_reader: %s", e, exc_info=True)

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    streaming_asr: StreamingASR = websocket.app.state.streaming_asr_service
    ffmpeg_manager = FFmpegManager()

    writer_task = asyncio.create_task(ffmpeg_writer(websocket, ffmpeg_manager))
    reader_task = asyncio.create_task(ffmpeg_reader(websocket, ffmpeg_manager, streaming_asr))

    try:
        if not await ffmpeg_manager.start():
            await websocket.close(code=1011, reason="FFmpeg failed to start.")
            return

        if streaming_asr.engine:
            streaming_asr.engine.reset_state()
        logger.info("ASREngine state reset for new WebSocket connection.")

        await asyncio.gather(writer_task, reader_task)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
    finally:
        writer_task.cancel()
        reader_task.cancel()
        await ffmpeg_manager.stop()
        logger.info("FFmpeg process stopped.")

if __name__ == "__main__":
    import uvicorn
    # Path is already imported at the top, but for standalone script clarity, it's fine.
    # from pathlib import Path 

    # To run this server directly (for development):
    # Ensure you are in the 'src/vistreamasr/web' directory,
    # or adjust the 'app' and 'static_dir' paths if running from elsewhere.
    # Example: python src/vistreamasr/web/server.py
    # The static files need to be accessible relative to where uvicorn runs.
    # If running from `src/vistreamasr/web`, then `static` directory is correct.
    # If running from project root, it would be `src/vistreamasr/web/static`.

    # For the purpose of this task, we assume the CLI will handle running the server
    # from the correct context or with appropriate path configurations.
    print("Starting Uvicorn server on http://127.0.0.1:8000")
    print("WebSocket endpoint available at ws://127.0.0.1:8000/asr")
    print(f"Static files served from /static (e.g., http://127.0.0.1:8000/static/index.html)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
