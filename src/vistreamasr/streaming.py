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
        self.vad_processor: Optional[VADProcessor] = None # Kept for potential direct use or stats
        self.vad_coordinator: Optional[VADASRCoordinator] = None
        
        # Initialize VAD if enabled (VADProcessor is part of VADASRCoordinator now)
        # This direct VADProcessor instantiation might be redundant if only used through coordinator
        # but kept for now if any direct VAD stats or checks are needed outside coordinator logic.
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

    def _ensure_engine_initialized(self):
        """Lazy initialization of the ASR engine and VAD coordinator."""
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
                # The asr_engine is no longer passed to VADASRCoordinator
                self.vad_coordinator = VADASRCoordinator(self.vad_config)
                if self.debug:
                    logger.debug("VAD coordinator initialized")
            
            if self.debug:
                log_with_symbol(symbols['check'], "ASR engine initialized successfully")
    
    
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
        
        chunk_size = chunk_size_ms or self.chunk_size_ms
        
        if self.debug:
            logger.debug(f"Streaming from file: {audio_file} with chunk size: {chunk_size}ms")
        
        # Load and prepare audio
        audio_data = self._load_audio_file(audio_file)
        if audio_data is None:
            return
        
        prepared_audio = audio_data['waveform']
        duration = audio_data['duration']
        
        if self.debug:
            log_with_symbol(symbols['check'], f"Audio loaded: {duration:.2f}s, {len(prepared_audio)} samples")
        
        # Reset engine state
        self._ensure_engine_initialized() # Ensure engine is ready
        if self.engine: # Check if engine is initialized
            self.engine.reset_state()
        if self.vad_coordinator: # Check if coordinator is initialized
            self.vad_coordinator.reset()
        
        # Calculate chunk parameters
        chunk_size_samples = int(16000 * chunk_size / 1000.0)
        total_chunks = (len(prepared_audio) + chunk_size_samples - 1) // chunk_size_samples
        
        if self.debug:
            logger.debug(f"Processing {total_chunks} chunks of {chunk_size_samples} samples each")
        
        # Process chunks
        start_time = time.time()
        
        for i in range(total_chunks):
            start_sample = i * chunk_size_samples
            end_sample = min(start_sample + chunk_size_samples, len(prepared_audio))
            chunk = prepared_audio[start_sample:end_sample]
            
            is_last_chunk_of_file = (i == total_chunks - 1)
            
            if self.debug:
                logger.debug(f"Processing chunk {i+1}/{total_chunks}, samples: {len(chunk)}")
            
            if self.vad_coordinator is not None:
                # VAD is enabled: process chunk through coordinator, then segments through ASR
                vad_chunk_size = 512  # Supported chunk size for 16kHz Silero VAD
                num_vad_sub_chunks = (len(chunk) + vad_chunk_size - 1) // vad_chunk_size
                
                for j in range(num_vad_sub_chunks):
                    start_vad = j * vad_chunk_size
                    end_vad = min(start_vad + vad_chunk_size, len(chunk))
                    vad_sub_chunk = chunk[start_vad:end_vad]
                    
                    # Pad the last sub-chunk if it's smaller
                    if len(vad_sub_chunk) < vad_chunk_size:
                        padding_needed = vad_chunk_size - len(vad_sub_chunk)
                        vad_sub_chunk = np.pad(vad_sub_chunk, (0, padding_needed), 'constant')
                    
                    is_last_vad_sub_chunk = (is_last_chunk_of_file and j == num_vad_sub_chunks - 1)
                    
                    # The coordinator now yields speech segments
                    for speech_segment_tensor in self.vad_coordinator.process_audio_chunk(vad_sub_chunk, is_last=is_last_vad_sub_chunk):
                        if self.engine: # Check if engine is initialized
                            # Convert tensor to numpy for ASR engine
                            speech_segment_np = speech_segment_tensor.numpy()
                            # Process the finalized speech segment with ASR
                            # is_last for ASR should be true only if it's the very last segment of the entire audio
                            asr_result = self.engine.process_audio(speech_segment_np, is_last=is_last_vad_sub_chunk and i == total_chunks -1 and j == num_vad_sub_chunks -1)

                            chunk_info = {
                                'chunk_id': i + 1,
                                'total_chunks': total_chunks,
                                'samples': len(speech_segment_np),
                                'duration_ms': len(speech_segment_np) / 16000 * 1000,
                                'is_last': is_last_chunk_of_file, # Reflects the main file chunk
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
                            logger.debug(f"Processed VAD sub-chunk {j+1}/{num_vad_sub_chunks}")
            else:
                # VAD is disabled: process chunk directly with ASR engine
                if self.engine: # Check if engine is initialized
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
        # rtf = self.engine.get_asr_rtf() # Ensure engine is not None
        
        if self.debug:
            logger.debug(f"Streaming completed in {total_time:.2f}s")
            # if self.engine and hasattr(self.engine, 'get_asr_rtf'): # Check method existence
            #     rtf = self.engine.get_asr_rtf()
            #     logger.debug(f"ASR RTF: {rtf:.3f}")

    def stream_from_microphone(self, duration_seconds: Optional[float] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream ASR results from microphone input.

        Args:
            duration_seconds: Maximum duration to record (None for infinite)

        Yields:
            dict: Same format as stream_from_file()
        """
        self._ensure_engine_initialized()
        samplerate = 16000
        chunk_size_samples = int(samplerate * self.chunk_size_ms / 1000.0)
        
        if self.debug:
            logger.debug(f"Starting microphone streaming with chunk size: {chunk_size_samples} samples")
        
        if self.engine: # Check if engine is initialized
            self.engine.reset_state()
        if self.vad_coordinator: # Check if coordinator is initialized
            self.vad_coordinator.reset()

        start_time = time.time()
        buffer = np.zeros((0,), dtype=np.float32)
        mic_chunk_id = 0

        def callback(indata, frames, time_info, status):
            nonlocal buffer
            if status:
                logger.error(status) # Use logger.error for error logging
            buffer = np.concatenate((buffer, indata[:, 0]))

        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback):
            while True:
                is_last_mic_chunk = False
                if duration_seconds is not None and (time.time() - start_time) > duration_seconds:
                    is_last_mic_chunk = True
                
                if len(buffer) >= chunk_size_samples or is_last_mic_chunk:
                    mic_chunk = buffer[:chunk_size_samples]
                    buffer = buffer[chunk_size_samples:]
                    if len(mic_chunk) == 0 and is_last_mic_chunk: # No more data and it's the last
                        break
                    if len(mic_chunk) == 0 and not is_last_mic_chunk: # Not enough data yet, keep waiting
                        time.sleep(0.01)
                        continue
                    
                    mic_chunk_id += 1
                    if self.debug:
                        logger.debug(f"Processing microphone chunk {mic_chunk_id}, samples: {len(mic_chunk)}")
                    
                    if self.vad_coordinator is not None:
                        # VAD is enabled: process chunk through coordinator, then segments through ASR
                        vad_chunk_size = 512  # Supported chunk size for 16kHz Silero VAD
                        num_vad_sub_chunks = (len(mic_chunk) + vad_chunk_size - 1) // vad_chunk_size
                        for j in range(num_vad_sub_chunks):
                            start_vad = j * vad_chunk_size
                            end_vad = min(start_vad + vad_chunk_size, len(mic_chunk))
                            vad_sub_chunk = mic_chunk[start_vad:end_vad]

                            if len(vad_sub_chunk) < vad_chunk_size:
                                padding_needed = vad_chunk_size - len(vad_sub_chunk)
                                vad_sub_chunk = np.pad(vad_sub_chunk, (0, padding_needed), 'constant')
                            
                            # is_last for VAD coordinator: true if this is the last sub-chunk of the last mic chunk
                            is_last_vad_sub_chunk = is_last_mic_chunk and (j == num_vad_sub_chunks - 1)
                            for speech_segment_tensor in self.vad_coordinator.process_audio_chunk(vad_sub_chunk, is_last=is_last_vad_sub_chunk):
                                if self.engine: # Check if engine is initialized
                                    speech_segment_np = speech_segment_tensor.numpy()
                                    # is_last for ASR: true if this is the very last audio segment overall
                                    asr_is_last = is_last_vad_sub_chunk and j == num_vad_sub_chunks -1
                                    asr_result = self.engine.process_audio(speech_segment_np, is_last=asr_is_last)

                                    chunk_info = {
                                        'chunk_id': mic_chunk_id,
                                        'samples': len(speech_segment_np),
                                        'duration_ms': len(speech_segment_np) / samplerate * 1000,
                                        'is_last': is_last_mic_chunk, # Reflects the main microphone chunk
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
                                        logger.debug(f"Processed VAD sub-chunk {j+1}/{num_vad_sub_chunks}")
                    else:
                        # VAD is disabled: process chunk directly with ASR engine
                        if self.engine: # Check if engine is initialized
                            result = self.engine.process_audio(mic_chunk, is_last=is_last_mic_chunk)
                            
                            chunk_info = {
                                'chunk_id': mic_chunk_id,
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
        
        if self.debug:
            logger.debug("Microphone streaming completed")
            # if self.vad_processor is not None: # This was the standalone VADProcessor
            if self.vad_coordinator is not None:
                # Placeholder for any VAD stats if needed
                pass

    def _load_audio_file(self, audio_file: str) -> Optional[Dict[str, Any]]:
        """Load and prepare audio file for ASR processing."""
        if not os.path.exists(audio_file):
            if self.debug:
                logger.error(f"Audio file not found: {audio_file}")
            return None
        
        try:
            if self.debug:
                logger.debug(f"Loading audio file: {audio_file}")

            # Check for valid audio file format
            if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                if self.debug:
                    logger.error(f"Unsupported audio format: {audio_file}")
                return None
            
            # Load with torchaudio
            waveform, original_sr = torchaudio.load(audio_file)
            
            # Convert to mono if stereo
            if waveform.shape > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                if self.debug:
                    logger.debug("Converted stereo to mono")
            
            # Prepare audio for ASR (convert to 16kHz and normalize)
            prepared_audio = self._prepare_audio_for_asr(waveform.squeeze(), original_sr)
            
            duration = len(waveform.squeeze()) / original_sr
            
            if self.debug:
                log_with_symbol(symbols['check'], f"Audio prepared: {duration:.2f}s, {original_sr}Hz → 16kHz")
            
            return {
                'waveform': prepared_audio,
                'original_sample_rate': original_sr,
                'duration': duration
            }
        except RuntimeError as e:
            if "Error opening file" in str(e) or "failed to load" in str(e):
                if self.debug:
                    logger.error(f"Error loading audio file: {e}")
                return None
            else:
                raise e
        except Exception as e:
            if self.debug:
                logger.error(f"Unexpected error loading audio file: {e}")
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
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.mean(axis=0)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio_tensor = resampler(audio_tensor).squeeze()
        
        # Normalize
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            audio_tensor = audio_tensor / max_val
        
        # Convert back to numpy
        return audio_tensor.numpy()