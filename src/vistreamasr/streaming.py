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


# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
                print(f"Partial: {result['text']}")
            if result['final']:
                print(f"Final: {result['text']}")
    """
    
    def __init__(self, chunk_size_ms: int = 640, auto_finalize_after: float = 15.0, debug: bool = False, vad_config: Optional[Dict[str, Any]] = None):
        """
        Initialize StreamingASR.
        
        Args:
            chunk_size_ms: Chunk size in milliseconds (default: 640ms for optimal performance)
            auto_finalize_after: Maximum duration in seconds before auto-finalizing a segment (default: 15.0s)
            debug: Enable debug logging
            vad_config: Configuration for VAD processing (default: None for disabled)
        """
        self.chunk_size_ms = chunk_size_ms
        self.auto_finalize_after = auto_finalize_after
        self.debug = debug
        self.vad_config = vad_config or {}
        self.engine: Optional[ASREngine] = None
        self.vad_processor: Optional[VADProcessor] = None
        self.vad_coordinator: Optional[VADASRCoordinator] = None
        
        # Initialize VAD if enabled
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
                    print(f"{symbols['tool']} [StreamingASR] VAD processor initialized")
            except Exception as e:
                if self.debug:
                    print(f"{symbols['warning']}  [StreamingASR] Failed to initialize VAD processor: {e}")
                self.vad_processor = None
        
        if self.debug:
            vad_status = "enabled" if self.vad_processor else "disabled"
            print(f"{symbols['tool']} [StreamingASR] Initialized with {chunk_size_ms}ms chunks, auto-finalize after {auto_finalize_after}s, debug={debug}, VAD={vad_status}")
    def _ensure_engine_initialized(self):
        """Lazy initialization of the ASR engine."""
        if self.engine is None:
            if self.debug:
                print(f"{symbols['tool']} [StreamingASR] Initializing ASR engine...")
            
            self.engine = ASREngine(
                chunk_size_ms=self.chunk_size_ms,
                max_duration_before_forced_finalization=self.auto_finalize_after,
                debug_mode=self.debug
            )
            self.engine.initialize_models()
            
            # Initialize VAD coordinator if VAD is enabled
            if self.vad_processor is not None:
                self.vad_coordinator = VADASRCoordinator(self.vad_config, self.engine)
                if self.debug:
                    print(f"{symbols['tool']} [StreamingASR] VAD-ASR coordinator initialized")
            
            if self.debug:
                print(f"{symbols['check']} [StreamingASR] ASR engine ready")
    
    
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
            print(f"{symbols['wave']} [StreamingASR] Starting file stream: {audio_file}")
            print(f"{symbols['ruler']} [StreamingASR] Chunk size: {chunk_size}ms")
        
        # Load and prepare audio
        audio_data = self._load_audio_file(audio_file)
        if audio_data is None:
            return
        
        prepared_audio = audio_data['waveform']
        duration = audio_data['duration']
        
        if self.debug:
            print(f"{symbols['wave']} [StreamingASR] Audio loaded: {duration:.2f}s, {len(prepared_audio)} samples")
        
        # Reset engine state
        self._ensure_engine_initialized()
        self.engine.reset_state()
        
        # Calculate chunk parameters
        chunk_size_samples = int(16000 * chunk_size / 1000.0)
        total_chunks = (len(prepared_audio) + chunk_size_samples - 1) // chunk_size_samples
        
        if self.debug:
            print(f"{symbols['check']} [StreamingASR] Processing {total_chunks} chunks of {chunk_size_samples} samples each")
        
        # Process chunks
        start_time = time.time()
        
        for i in range(total_chunks):
            start_sample = i * chunk_size_samples
            end_sample = min(start_sample + chunk_size_samples, len(prepared_audio))
            chunk = prepared_audio[start_sample:end_sample]
            
            is_last = (i == total_chunks - 1)
            
            if self.debug:
                print(f"\n{symbols['tool']} [StreamingASR] Processing chunk {i+1}/{total_chunks} ({len(chunk)} samples)")
            
            # Process chunk with VAD if enabled
            if self.vad_coordinator is not None:
                # Initialize result to avoid unbound variable errors
                result = {'current_transcription': '', 'new_final_text': None}
                # Process the chunk in smaller pieces for the VAD
                vad_chunk_size = 512  # Supported chunk size for 16kHz
                num_vad_chunks = (len(chunk) + vad_chunk_size - 1) // vad_chunk_size
                for j in range(num_vad_chunks):
                    start = j * vad_chunk_size
                    end = min(start + vad_chunk_size, len(chunk))
                    vad_chunk = chunk[start:end]
                    
                    # Pad the last chunk if it's smaller than the required size
                    if len(vad_chunk) < vad_chunk_size:
                        vad_chunk = np.pad(vad_chunk, (0, vad_chunk_size - len(vad_chunk)), 'constant')

                    result = self.vad_coordinator.process_audio_chunk(vad_chunk, is_last=(is_last and j == num_vad_chunks - 1))
                    if self.debug:
                        vad_status = "speech" if result.get('current_transcription') or result.get('new_final_text') else "silence"
                        print(f"{symbols['tool']} [StreamingASR] VAD processing: {vad_status}")
            else:
                # Process chunk directly with ASR engine
                result = self.engine.process_audio(chunk, is_last=is_last)
            
            # Prepare output
            chunk_info = {
                'chunk_id': i + 1,
                'total_chunks': total_chunks,
                'samples': len(chunk),
                'duration_ms': len(chunk) / 16000 * 1000,
                'is_last': is_last
            }
            
            # Add VAD status to chunk info if enabled
            if self.vad_processor is not None:
                chunk_info['vad_status'] = "speech" if result.get('current_transcription') or result.get('new_final_text') else "silence"
            
            # Yield partial results
            if result.get('current_transcription'):
                yield {
                    'partial': True,
                    'final': False,
                    'text': result['current_transcription'],
                    'chunk_info': chunk_info
                }
            
            # Yield final results
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
        rtf = self.engine.get_asr_rtf()
        
        if self.debug:
            print(f"\n{symbols['check']} [StreamingASR] Processing complete")
            print(f"{symbols['ruler']}  Total time: {total_time:.2f}s")
            print(f"{symbols['check']} �� RTF: {rtf:.2f}x")
            print(f"{symbols['check']} ⚡ Speedup: {duration/total_time:.1f}x")

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
        chunk_size = self.chunk_size_ms
        chunk_size_samples = int(samplerate * chunk_size / 1000.0)
        if self.debug:
            print(
                f"{symbols['wave']} [StreamingASR] Starting microphone stream at {samplerate}Hz, chunk size: {chunk_size}ms ({chunk_size_samples} samples)")
        self._ensure_engine_initialized()
        self.engine.reset_state()
        start_time = time.time()
        buffer = np.zeros((0,), dtype=np.float32)
        chunk_id = 0

        def callback(indata, frames, time_info, status):
            nonlocal buffer
            if status:
                print(status, file=sys.stderr)
            buffer = np.concatenate((buffer, indata[:, 0]))

        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback):
            while True:
                if duration_seconds is not None and (time.time() - start_time) > duration_seconds:
                    is_last = True
                else:
                    is_last = False
                if len(buffer) >= chunk_size_samples or is_last:
                    chunk = buffer[:chunk_size_samples]
                    buffer = buffer[chunk_size_samples:]
                    if len(chunk) == 0:
                        if is_last:
                            break
                        else:
                            time.sleep(0.01)
                            continue
                    chunk_id += 1
                    if self.debug:
                        print(
                            f"{symbols['tool']} [StreamingASR] Processing mic chunk {chunk_id} ({len(chunk)} samples)")
                    
                    # Process chunk with VAD if enabled
                    if self.vad_coordinator is not None:
                        result = self.vad_coordinator.process_audio_chunk(chunk, is_last=is_last)
                        if self.debug:
                            vad_status = "speech" if result.get('current_transcription') or result.get('new_final_text') else "silence"
                            print(f"{symbols['tool']} [StreamingASR] VAD processing: {vad_status}")
                    else:
                        # Process chunk directly with ASR engine
                        result = self.engine.process_audio(chunk, is_last=is_last)
                    
                    chunk_info = {
                        'chunk_id': chunk_id,
                        'samples': len(chunk),
                        'duration_ms': len(chunk) / samplerate * 1000,
                        'is_last': is_last
                    }
                    
                    # Add VAD status to chunk info if enabled
                    if self.vad_processor is not None:
                        chunk_info['vad_status'] = "speech" if result.get('current_transcription') or result.get('new_final_text') else "silence"
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
                    if is_last:
                        break
                else:
                    time.sleep(0.01)
        if self.debug:
            print(f"{symbols['check']} [StreamingASR] Microphone streaming complete.")
            
            # Add VAD statistics if enabled
            if self.vad_processor is not None:
                print(f"{symbols['tool']} [StreamingASR] VAD processing enabled")
    
    def _load_audio_file(self, audio_file: str) -> Optional[Dict[str, Any]]:
        """Load and prepare audio file for ASR processing."""
        if not os.path.exists(audio_file):
            if self.debug:
                print(f"{symbols['folder']} [StreamingASR] File not found: {audio_file}")
            return None
        
        try:
            if self.debug:
                print(f"{symbols['folder']} [StreamingASR] Loading: {audio_file}")

            # Check for valid audio file format
            if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                if self.debug:
                    print(f"{symbols['folder']} [StreamingASR] Invalid audio file format: {audio_file}")
                return None
            
            # Load with torchaudio
            waveform, original_sr = torchaudio.load(audio_file)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                if self.debug:
                    print(f"{symbols['tool']} [StreamingASR] Converted stereo to mono")
            
            # Prepare audio for ASR (convert to 16kHz and normalize)
            prepared_audio = self._prepare_audio_for_asr(waveform.squeeze(), original_sr)
            
            duration = len(waveform.squeeze()) / original_sr
            
            if self.debug:
                print(f"{symbols['check']} [StreamingASR] Audio prepared: {len(prepared_audio)} samples at 16kHz")
            
            return {
                'waveform': prepared_audio,
                'original_sample_rate': original_sr,
                'duration': duration
            }
        except RuntimeError as e:
            if "Error opening file" in str(e) or "failed to load" in str(e):
                if self.debug:
                    print(f"{symbols['folder']} [StreamingASR] Invalid audio file: {audio_file}")
                return None
            else:
                raise e
        except Exception as e:
            if self.debug:
                print(f"{symbols['folder']} [StreamingASR] Error loading audio file: {e}")
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