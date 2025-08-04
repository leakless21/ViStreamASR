"""
ViStreamASR Streaming Interface

This module provides the high-level StreamingASR interface that wraps
the low-level ASREngine for easy-to-use streaming ASR functionality.
"""

import os
import time
import torch
import torchaudio
from typing import Generator, Dict, Any, Optional
from pathlib import Path
import sys
import numpy as np
import sounddevice as sd
from .core import ASREngine
from .vad import VADProcessor, VADASRCoordinator  # New import for VAD
from .logging import log_with_symbol
from loguru import logger

# Define symbols that work across platforms
symbols = {
    'tool': '🔧' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[CONFIG]',
    'check': '✅' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[OK]',
    'ruler': '📏' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SIZE]',
    'folder': '📁' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[FILE]',
    'wave': '🎵' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[AUDIO]',
}

class FileStreamer:
    """
    Handles streaming ASR results from an audio file.
    """
    def __init__(self,
                 engine: ASREngine,
                 vad_coordinator: Optional[VADASRCoordinator],
                 asr_instance_config: Dict[str, Any], # Config from main ASR instance
                 logger_instance,
                 symbols_dict: Dict[str, str]):
        self.engine = engine
        self.vad_coordinator = vad_coordinator
        self.config = asr_instance_config # Stores chunk_size_ms, debug, etc.
        self.logger = logger_instance
        self.symbols = symbols_dict

    def _load_audio_file(self, audio_file: str) -> Optional[Dict[str, Any]]:
        """Load and prepare audio file for ASR processing."""
        if not os.path.exists(audio_file):
            if self.config.get('debug', False):
                self.logger.error(f"Audio file not found: {audio_file}")
            return None
        
        try:
            if self.config.get('debug', False):
                self.logger.debug(f"Loading audio file: {audio_file}")

            # Check for valid audio file format
            if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                if self.config.get('debug', False):
                    self.logger.error(f"Unsupported audio format: {audio_file}")
                return None
            
            # Load with torchaudio
            waveform, original_sr = torchaudio.load(audio_file)
            
            # Convert to mono if stereo
            if waveform.shape > 1: # Should be waveform.shape[0] > 1 for channels
                waveform = waveform.mean(dim=0, keepdim=True)
                if self.config.get('debug', False):
                    self.logger.debug("Converted stereo to mono")
            
            # Prepare audio for ASR (convert to 16kHz and normalize)
            prepared_audio = self._prepare_audio_for_asr(waveform.squeeze(), original_sr)
            
            duration = len(waveform.squeeze()) / original_sr
            
            if self.config.get('debug', False):
                log_with_symbol(self.symbols['check'], f"Audio prepared: {duration:.2f}s, {original_sr}Hz → 16kHz")
            
            return {
                'waveform': prepared_audio,
                'original_sample_rate': original_sr,
                'duration': duration
            }
        except RuntimeError as e:
            if "Error opening file" in str(e) or "failed to load" in str(e):
                if self.config.get('debug', False):
                    self.logger.error(f"Error loading audio file: {e}")
                return None
            else:
                raise e
        except Exception as e:
            if self.config.get('debug', False):
                self.logger.error(f"Unexpected error loading audio file: {e}")
            return None
    
    def _prepare_audio_for_asr(self, audio_data, sample_rate):
        """Prepare audio data for ASR engine (convert to 16kHz mono and normalize)."""
        target_sample_rate = 16000
        
        # Convert to tensor if needed
        if not isinstance(audio_data, torch.Tensor):
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        else:
            audio_tensor = audio_data.float()
        
        # Convert stereo to mono if needed
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[0] > 1 : # Check if actually stereo
            audio_tensor = audio_tensor.mean(axis=0)
        elif len(audio_tensor.shape) > 1 and audio_tensor.shape[0] == 1: # Already mono but has channel dim
            audio_tensor = audio_tensor.squeeze(0)

        # Resample if needed
        if sample_rate != target_sample_rate:
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension for resampler
            
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio_tensor = resampler(audio_tensor).squeeze()
        
        # Normalize
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            audio_tensor = audio_tensor / max_val
        
        # Convert back to numpy
        return audio_tensor.numpy()

    def stream(self, audio_file: str, chunk_size_ms: Optional[int] = None, process_vad_chunk_func: Optional[callable] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream ASR results from an audio file.
        
        Args:
            audio_file: Path to audio file
            chunk_size_ms: Override chunk size for this session
            process_vad_chunk_func: Function to process chunks with VAD (from StreamingASR)
            
        Yields:
            dict: Results with keys:
                - 'partial': True if partial transcription
                - 'final': True if final transcription  
                - 'text': Transcription text
                - 'chunk_info': Processing info (samples, duration, etc.)
        """
        chunk_size = chunk_size_ms or self.config.get('chunk_size_ms', 640)
        
        if self.config.get('debug', False):
            self.logger.debug(f"Streaming from file: {audio_file} with chunk size: {chunk_size}ms")
        
        # Load and prepare audio
        audio_data = self._load_audio_file(audio_file)
        if audio_data is None:
            return
        
        prepared_audio = audio_data['waveform']
        # duration = audio_data['duration'] # Not used directly in the loop
        
        if self.config.get('debug', False):
            log_with_symbol(self.symbols['check'], f"Audio loaded: {audio_data['duration']:.2f}s, {len(prepared_audio)} samples")
        
        # Reset engine state (done by StreamingASR before calling this)
        # self.engine.reset_state()
        # if self.vad_coordinator:
        #     self.vad_coordinator.reset()
        
        # Calculate chunk parameters
        chunk_size_samples = int(16000 * chunk_size / 1000.0)
        total_chunks = (len(prepared_audio) + chunk_size_samples - 1) // chunk_size_samples
        
        if self.config.get('debug', False):
            self.logger.debug(f"Processing {total_chunks} chunks of {chunk_size_samples} samples each")
        
        # Process chunks
        start_time = time.time()
        
        for i in range(total_chunks):
            start_sample = i * chunk_size_samples
            end_sample = min(start_sample + chunk_size_samples, len(prepared_audio))
            chunk = prepared_audio[start_sample:end_sample]
            
            is_last_chunk_of_file = (i == total_chunks - 1)
            
            if self.config.get('debug', False):
                self.logger.debug(f"Processing chunk {i+1}/{total_chunks}, samples: {len(chunk)}")
            
            if self.vad_coordinator is not None and process_vad_chunk_func:
                # VAD is enabled: process chunk through helper method from StreamingASR
                yield from process_vad_chunk_func(
                    audio_chunk=chunk,
                    main_chunk_id=i + 1,
                    total_main_chunks=total_chunks,
                    is_last_main_chunk=is_last_chunk_of_file,
                    samplerate=16000 
                )
            else:
                # VAD is disabled: process chunk directly with ASR engine
                result = self.engine.process_audio(chunk, is_last=is_last_chunk_of_file)
                
                chunk_info = {
                    'chunk_id': i + 1,
                    'total_chunks': total_chunks,
                    'samples': len(chunk),
                    'duration_ms': len(chunk) / 16000 * 1000,
                    'is_last': is_last_chunk_of_file,
                    'vad_status': 'N/A'
                }
                
                if result.get('current_transcription'):
                    yield {
                        'partial': True,
                        'final': False,
                        'text': result['current_transcription'],
                        'chunk_info': chunk_info
                    }
                if result.get('new_final_text'):
                    yield {
                        'partial': False,
                        'final': True,
                        'text': result['new_final_text'],
                        'chunk_info': chunk_info
                    }
        
        # Final statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.config.get('debug', False):
            self.logger.debug(f"Streaming completed in {total_time:.2f}s")

class MicrophoneStreamer:
    """
    Handles streaming ASR results from microphone input.
    """
    def __init__(self,
                 engine: ASREngine,
                 vad_coordinator: Optional[VADASRCoordinator],
                 asr_instance_config: Dict[str, Any], # Config from main ASR instance
                 logger_instance):
        self.engine = engine
        self.vad_coordinator = vad_coordinator
        self.config = asr_instance_config # Stores chunk_size_ms, debug, etc.
        self.logger = logger_instance

    def stream(self, duration_seconds: Optional[float] = None, process_vad_chunk_func: Optional[callable] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream ASR results from microphone input.

        Args:
            duration_seconds: Maximum duration to record (None for infinite)
            process_vad_chunk_func: Function to process chunks with VAD (from StreamingASR)

        Yields:
            dict: Same format as FileStreamer.stream()
        """
        samplerate = 16000
        chunk_size_samples = int(samplerate * self.config.get('chunk_size_ms', 640) / 1000.0)
        
        if self.config.get('debug', False):
            self.logger.debug(f"Starting microphone streaming with chunk size: {chunk_size_samples} samples")

        # Reset engine state (done by StreamingASR before calling this)
        # self.engine.reset_state()
        # if self.vad_coordinator:
        #     self.vad_coordinator.reset()

        start_time = time.time()
        buffer = np.zeros((0,), dtype=np.float32)
        mic_chunk_id = 0

        def callback(indata, frames, time_info, status):
            nonlocal buffer
            if status:
                self.logger.error(status)
            buffer = np.concatenate((buffer, indata[:, 0]))

        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback):
            while True:
                is_last_mic_chunk = False
                if duration_seconds is not None and (time.time() - start_time) > duration_seconds:
                    is_last_mic_chunk = True
                
                if len(buffer) >= chunk_size_samples or is_last_mic_chunk:
                    mic_chunk = buffer[:chunk_size_samples]
                    buffer = buffer[chunk_size_samples:]
                    if len(mic_chunk) == 0 and is_last_mic_chunk:
                        break
                    if len(mic_chunk) == 0 and not is_last_mic_chunk:
                        time.sleep(0.01)
                        continue
                    
                    mic_chunk_id += 1
                    if self.config.get('debug', False):
                        self.logger.debug(f"Processing microphone chunk {mic_chunk_id}, samples: {len(mic_chunk)}")
                    
                    if self.vad_coordinator is not None and process_vad_chunk_func:
                        yield from process_vad_chunk_func(
                            audio_chunk=mic_chunk,
                            main_chunk_id=mic_chunk_id,
                            total_main_chunks=None,
                            is_last_main_chunk=is_last_mic_chunk,
                            samplerate=samplerate
                        )
                    else:
                        result = self.engine.process_audio(mic_chunk, is_last=is_last_mic_chunk)
                        
                        chunk_info = {
                            'chunk_id': mic_chunk_id,
                            'total_chunks': None,
                            'samples': len(mic_chunk),
                            'duration_ms': len(mic_chunk) / samplerate * 1000,
                            'is_last': is_last_mic_chunk,
                            'vad_status': 'N/A'
                        }

                        if result.get('current_transcription'):
                            yield {
                                'partial': True,
                                'final': False,
                                'text': result['current_transcription'],
                                'chunk_info': chunk_info
                            }
                        if result.get('new_final_text'):
                            yield {
                                'partial': False,
                                'final': True,
                                'text': result['new_final_text'],
                                'chunk_info': chunk_info
                            }
                    if is_last_mic_chunk:
                        break
                else:
                    time.sleep(0.01)
        
        if self.config.get('debug', False):
            self.logger.debug("Microphone streaming completed")


class StreamingASR:
    """
    Simple streaming ASR interface.
    
    Example usage:
        asr = StreamingASR()
        
        # Process file in streaming fashion
        for result in asr.stream_from_file("audio.wav", chunk_size_ms=640):
            if result['partial']:
                # Handle partial result
            if result['final']:
                # Handle final result
    """
    
    def __init__(self, settings=None, chunk_size_ms: int = 640, auto_finalize_after: float = 15.0, debug: bool = False, vad_config: Optional[Dict[str, Any]] = None):
        """
        Initialize StreamingASR.
        
        Args:
            settings: ViStreamASRSettings configuration object
            chunk_size_ms: Chunk size in milliseconds (default: 640ms for optimal performance)
            auto_finalize_after: Maximum duration in seconds before auto-finalizing a segment (default: 15.0s)
            debug: Enable debug logging
            vad_config: Configuration for VAD processing (default: None for disabled)
        """
        if settings is not None:
            self.settings = settings
            self.chunk_size_ms = settings.model.chunk_size_ms
            self.auto_finalize_after = settings.model.auto_finalize_after
            self.debug = settings.model.debug
            self.vad_config = settings.vad.model_dump() if settings.vad.enabled else None
        else:
            # Legacy support for direct parameters
            self.settings = None
            self.chunk_size_ms = chunk_size_ms
            self.auto_finalize_after = auto_finalize_after
            self.debug = debug
            self.vad_config = vad_config or {}
        
        self.engine: Optional[ASREngine] = None
        self.vad_processor: Optional[VADProcessor] = None 
        self.vad_coordinator: Optional[VADASRCoordinator] = None
        
        # Initialize VADProcessor if VAD is enabled (kept for potential direct use or stats)
        if self.vad_config and self.vad_config.get('enabled', False):
            try:
                self.vad_processor = VADProcessor(
                    sample_rate=self.vad_config.get('sample_rate', 16000),
                    threshold=self.vad_config.get('threshold', 0.5),
                    min_speech_duration_ms=self.vad_config.get('min_speech_duration_ms', 250),
                    min_silence_duration_ms=self.vad_config.get('min_silence_duration_ms', 100),
                    speech_pad_ms=self.vad_config.get('speech_pad_ms', 30)
                )
                if self.debug:
                    logger.debug("VAD processor initialized")
            except Exception as e:
                if self.debug:
                    logger.error(f"Failed to initialize VAD processor: {e}")
                self.vad_processor = None
        
        if self.debug:
            vad_enabled = self.vad_config.get('enabled', False) if self.vad_config else False
            vad_status = "enabled" if vad_enabled else "disabled"
            logger.debug(f"VAD status: {vad_status}")

        self._file_streamer: Optional[FileStreamer] = None
        self._microphone_streamer: Optional[MicrophoneStreamer] = None

    def _ensure_engine_initialized(self):
        """Lazy initialization of the ASR engine, VAD coordinator, and streamers."""
        if self.engine is None:
            if self.debug:
                logger.debug("Initializing ASR engine...")
            
            if self.settings is not None:
                self.engine = ASREngine(
                    chunk_size_ms=self.settings.model.chunk_size_ms,
                    max_duration_before_forced_finalization=self.settings.model.auto_finalize_after,
                    debug_mode=self.settings.model.debug
                )
            else:
                self.engine = ASREngine(
                    chunk_size_ms=self.chunk_size_ms,
                    max_duration_before_forced_finalization=self.auto_finalize_after,
                    debug_mode=self.debug
                )
            self.engine.initialize_models()
            
            # Initialize VAD coordinator if VAD is enabled
            if self.vad_config and self.vad_config.get('enabled', False):
                self.vad_coordinator = VADASRCoordinator(self.vad_config)
                if self.debug:
                    logger.debug("VAD coordinator initialized")
            
            # Initialize streamers
            asr_config = {
                'chunk_size_ms': self.chunk_size_ms if self.settings is None else self.settings.model.chunk_size_ms,
                'auto_finalize_after': self.auto_finalize_after if self.settings is None else self.settings.model.auto_finalize_after,
                'debug': self.debug if self.settings is None else self.settings.model.debug,
                'vad_config': self.vad_config
            }
            self._file_streamer = FileStreamer(
                engine=self.engine,
                vad_coordinator=self.vad_coordinator,
                asr_instance_config=asr_config,
                logger_instance=logger,
                symbols_dict=symbols
            )
            self._microphone_streamer = MicrophoneStreamer(
                engine=self.engine,
                vad_coordinator=self.vad_coordinator,
                asr_instance_config=asr_config,
                logger_instance=logger
            )

            if self.debug:
                log_with_symbol(symbols['check'], "ASR engine and streamers initialized successfully")
    
    def _process_vad_chunk(self, audio_chunk: np.ndarray, main_chunk_id: int, 
                           total_main_chunks: Optional[int], is_last_main_chunk: bool, 
                           samplerate: int) -> Generator[Dict[str, Any], None, None]:
        """
        Helper method to process an audio chunk through VAD and then ASR.
        This method is called by FileStreamer and MicrophoneStreamer.
        
        Args:
            audio_chunk: The audio chunk to process (numpy array).
            main_chunk_id: ID of the current main chunk (file or microphone).
            total_main_chunks: Total number of main chunks (for file streaming), None for mic.
            is_last_main_chunk: Boolean indicating if this is the last main chunk.
            samplerate: The sample rate of the audio chunk.
            
        Yields:
            dict: ASR results for speech segments detected by VAD.
        """
        if not self.vad_coordinator or not self.engine:
            return

        vad_chunk_size = 512  # Supported chunk size for 16kHz Silero VAD
        num_vad_sub_chunks = (len(audio_chunk) + vad_chunk_size - 1) // vad_chunk_size
        
        for j in range(num_vad_sub_chunks):
            start_vad = j * vad_chunk_size
            end_vad = min(start_vad + vad_chunk_size, len(audio_chunk))
            vad_sub_chunk = audio_chunk[start_vad:end_vad]
            
            if len(vad_sub_chunk) < vad_chunk_size:
                padding_needed = vad_chunk_size - len(vad_sub_chunk)
                vad_sub_chunk = np.pad(vad_sub_chunk, (0, padding_needed), 'constant')
            
            is_last_vad_sub_chunk = is_last_main_chunk and (j == num_vad_sub_chunks - 1)
            
            for speech_segment_tensor in self.vad_coordinator.process_audio_chunk(vad_sub_chunk, is_last=is_last_vad_sub_chunk):
                speech_segment_np = speech_segment_tensor.numpy()
                
                asr_is_last = is_last_vad_sub_chunk
                if total_main_chunks is not None: 
                     asr_is_last = is_last_vad_sub_chunk and main_chunk_id == total_main_chunks and j == num_vad_sub_chunks -1
                else: 
                     asr_is_last = is_last_vad_sub_chunk and j == num_vad_sub_chunks -1

                asr_result = self.engine.process_audio(speech_segment_np, is_last=asr_is_last)
                
                chunk_info = {
                    'chunk_id': main_chunk_id,
                    'total_chunks': total_main_chunks, 
                    'samples': len(speech_segment_np),
                    'duration_ms': len(speech_segment_np) / samplerate * 1000,
                    'is_last': is_last_main_chunk, 
                    'vad_status': 'speech_segment'
                }
                
                if asr_result.get('current_transcription'):
                    yield {
                        'partial': True,
                        'final': False,
                        'text': asr_result['current_transcription'],
                        'chunk_info': chunk_info
                    }
                if asr_result.get('new_final_text'):
                    yield {
                        'partial': False,
                        'final': True,
                        'text': asr_result['new_final_text'],
                        'chunk_info': chunk_info
                    }
                if self.debug:
                    logger.debug(f"Processed VAD sub-chunk {j+1}/{num_vad_sub_chunks} for main chunk {main_chunk_id}")
    
    def stream_from_file(self, audio_file: str, chunk_size_ms: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream ASR results from an audio file.
        
        Args:
            audio_file: Path to audio file
            chunk_size_ms: Override chunk size for this session
            
        Yields:
            dict: Results with keys:
                - 'partial': True if partial transcription
                - 'final': True if final transcription  
                - 'text': Transcription text
                - 'chunk_info': Processing info (samples, duration, etc.)
        """
        self._ensure_engine_initialized()
        if not self._file_streamer:
            # This should not happen if _ensure_engine_initialized works correctly
            logger.error("FileStreamer not initialized.")
            return

        # Reset engine and VAD coordinator state before starting a new stream
        if self.engine:
            self.engine.reset_state()
        if self.vad_coordinator:
            self.vad_coordinator.reset()
            
        yield from self._file_streamer.stream(audio_file, chunk_size_ms, self._process_vad_chunk)

    def stream_from_microphone(self, duration_seconds: Optional[float] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream ASR results from microphone input.

        Args:
            duration_seconds: Maximum duration to record (None for infinite)

        Yields:
            dict: Same format as stream_from_file()
        """
        self._ensure_engine_initialized()
        if not self._microphone_streamer:
            logger.error("MicrophoneStreamer not initialized.")
            return

        # Reset engine and VAD coordinator state before starting a new stream
        if self.engine:
            self.engine.reset_state()
        if self.vad_coordinator:
            self.vad_coordinator.reset()

        yield from self._microphone_streamer.stream(duration_seconds, self._process_vad_chunk)