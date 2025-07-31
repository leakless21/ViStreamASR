"""
ViStreamASR VAD Module

This module provides Voice Activity Detection functionality using Silero-VAD
for filtering silence periods in audio streams before ASR processing.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

# Setup logging
logger = logging.getLogger(__name__)

class VADProcessor:
    """
    Handles voice activity detection using the Silero-VAD model.
    
    This class provides functionality to detect speech segments in audio streams,
    filter out silence periods, and prepare audio chunks for ASR processing.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30):
        """
        Initialize the VADProcessor with configuration parameters.
        
        Args:
            sample_rate (int): Audio sample rate (8000 or 16000). Default: 16000
            threshold (float): Speech probability threshold (0.0-1.0). Default: 0.5
            min_speech_duration_ms (int): Minimum speech duration in milliseconds. Default: 250
            min_silence_duration_ms (int): Minimum silence duration in milliseconds. Default: 100
            speech_pad_ms (int): Padding added to speech segments in milliseconds. Default: 30
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If model fails to load
        """
        # Validate parameters
        if sample_rate not in [8000, 16000]:
            raise ValueError("Sample rate must be 8000 or 16000")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if min_speech_duration_ms <= 0:
            raise ValueError("Minimum speech duration must be positive")
        if min_silence_duration_ms <= 0:
            raise ValueError("Minimum silence duration must be positive")
        if speech_pad_ms < 0:
            raise ValueError("Speech padding cannot be negative")
            
        # Store configuration
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # Calculate thresholds in samples
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)
        
        # Initialize model and state
        self.model = None  # type: ignore
        self._load_model()
        self.reset_states()
        
    def _load_model(self):
        """Load the Silero-VAD model."""
        try:
            # Try to load using pip package first
            try:
                from silero_vad import load_silero_vad
                self.model = load_silero_vad()
                logger.info("Silero-VAD model loaded successfully using pip package")
            except ImportError:
                # Fallback to torch.hub
                self.model = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                logger.info("Silero-VAD model loaded successfully using torch.hub")
                
        except Exception as e:
            logger.error(f"Failed to load Silero-VAD model: {e}")
            raise RuntimeError(f"Failed to load Silero-VAD model: {e}")
    
    def reset_states(self):
        """Reset VAD internal states for new audio session."""
        if self.model is not None:
            try:
                if hasattr(self.model, 'reset_states'):
                    getattr(self.model, 'reset_states')()
            except Exception as e:
                logger.warning(f"Failed to reset VAD model states: {e}")
            
        # Reset internal state tracking
        self.state = "silence"  # "silence" or "speech"
        self.silence_counter = 0
        self.speech_counter = 0
        self.buffer = []
        self.current_speech_start = None
        
    def process_chunk(self, audio_chunk: Union[np.ndarray, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Process an audio chunk and return speech segment if detected.
        
        Args:
            audio_chunk (Union[np.ndarray, torch.Tensor]): Audio chunk as numpy array or torch tensor
            
        Returns:
            Optional[torch.Tensor]: Speech segment if detected, None otherwise
            
        Raises:
            ValueError: If audio_chunk is invalid
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None
            
        # Convert to tensor if needed
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.float()
            
        # Ensure correct shape (1D tensor)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()
            
        # Get speech probability
        try:
            with torch.no_grad():
                if self.model is not None:
                    speech_prob = getattr(self.model, '__call__')(audio_tensor, self.sample_rate).item()
                else:
                    speech_prob = 0.0
        except Exception as e:
            logger.warning(f"Error processing audio chunk with VAD: {e}")
            return None
            
        # Update state based on probability
        is_speech = speech_prob >= self.threshold
        
        if is_speech:
            # Handle speech detection
            self.silence_counter = 0
            self.speech_counter += len(audio_tensor)
            
            if self.state == "silence":
                # Transition from silence to speech
                self.state = "speech"
                self.current_speech_start = len(self.buffer) * audio_tensor.shape[0] if self.buffer else 0
                # Add current chunk to buffer
                self.buffer.append(audio_tensor)
            else:
                # Continue speech
                self.buffer.append(audio_tensor)
                
                # Check if we have enough speech
                if self.speech_counter >= self.min_speech_samples:
                    # We have a valid speech segment, but continue buffering
                    # until we detect silence or end of stream
                    pass
                    
        else:
            # Handle silence detection
            self.speech_counter = 0
            self.silence_counter += len(audio_tensor)
            
            if self.state == "speech":
                # We were in speech state, check if silence is long enough
                if self.silence_counter >= self.min_silence_samples:
                    # End of speech segment detected
                    return self._finalize_speech_segment()
                else:
                    # Not enough silence yet, continue buffering
                    self.buffer.append(audio_tensor)
            else:
                # Continue silence
                # For silence, we don't buffer unless we're at the end of a stream
                pass
                
        return None  # No complete speech segment yet
    
    def get_speech_probability(self, audio_chunk: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Get the speech probability for an audio chunk.
        
        Args:
            audio_chunk (Union[np.ndarray, torch.Tensor]): Audio chunk as numpy array or torch tensor
            
        Returns:
            float: Speech probability (0.0-1.0)
            
        Raises:
            ValueError: If audio_chunk is invalid
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0
            
        # Convert to tensor if needed
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.float()
            
        # Ensure correct shape (1D tensor)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()
            
        # Get speech probability
        try:
            with torch.no_grad():
                if self.model is not None:
                    speech_prob = getattr(self.model, '__call__')(audio_tensor, self.sample_rate).item()
                    return float(speech_prob)
                else:
                    return 0.0
        except Exception as e:
            logger.warning(f"Error processing audio chunk with VAD: {e}")
            return 0.0
    
    def _finalize_speech_segment(self) -> Optional[torch.Tensor]:
        """
        Finalize and return the current speech segment with padding.
        
        Returns:
            Optional[torch.Tensor]: Speech segment with padding, or None if buffer is empty
        """
        if not self.buffer:
            return None
            
        # Concatenate all buffered chunks
        speech_segment = torch.cat(self.buffer, dim=0)
        
        # Apply padding if needed
        if self.speech_pad_samples > 0:
            pad_tensor = torch.zeros(self.speech_pad_samples, dtype=speech_segment.dtype)
            speech_segment = torch.cat([pad_tensor, speech_segment, pad_tensor], dim=0)
            
        # Reset buffer and state
        self.buffer = []
        self.state = "silence"
        self.silence_counter = 0
        self.speech_counter = 0
        self.current_speech_start = None
        
        return speech_segment
    
    def flush(self) -> Optional[torch.Tensor]:
        """
        Flush any remaining audio in buffer as a final speech segment.
        
        This should be called at the end of audio processing to ensure
        any trailing speech is processed.
        
        Returns:
            Optional[torch.Tensor]: Remaining speech segment, or None if buffer is empty
        """
        if self.buffer and self.state == "speech":
            return self._finalize_speech_segment()
        return None
    
    def is_speech(self, audio_chunk: Union[np.ndarray, torch.Tensor]) -> bool:
        """
        Determine if an audio chunk contains speech.
        
        Args:
            audio_chunk (Union[np.ndarray, torch.Tensor]): Audio chunk as numpy array or torch tensor
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        speech_prob = self.get_speech_probability(audio_chunk)
        return speech_prob >= self.threshold


class VADASRCoordinator:
    """
    Coordinator class that integrates VAD with the ASR engine.
    
    This class manages the interaction between VAD processing and ASR transcription,
    ensuring efficient processing of audio streams by filtering out silence periods.
    """
    
    def __init__(self, vad_config: Dict[str, Any], asr_engine):
        """
        Initialize the VAD-ASR coordinator.
        
        Args:
            vad_config (Dict[str, Any]): Configuration parameters for VAD
            asr_engine: ASR engine instance for processing speech segments
        """
        self.vad_config = vad_config
        self.asr_engine = asr_engine
        
        # Initialize VAD processor if enabled or if vad_config is provided
        self.vad_processor = None
        if vad_config and (vad_config.get('enabled', False) or len(vad_config) > 0):
            try:
                self.vad_processor = VADProcessor(
                    sample_rate=vad_config.get('sample_rate', 16000),
                    threshold=vad_config.get('threshold', 0.5),
                    min_speech_duration_ms=vad_config.get('min_speech_duration_ms', 250),
                    min_silence_duration_ms=vad_config.get('min_silence_duration_ms', 100),
                    speech_pad_ms=vad_config.get('speech_pad_ms', 30)
                )
                logger.info("VAD processor initialized successfully")
            except ValueError as e:
                logger.warning(f"Invalid VAD configuration: {e}. VAD will be disabled.")
                self.vad_processor = None
            except Exception as e:
                logger.warning(f"Failed to initialize VAD processor: {e}. VAD will be disabled.")
                self.vad_processor = None
    
    def process_audio_chunk(self, audio_chunk: Union[np.ndarray, torch.Tensor], is_last: bool = False) -> Dict[str, Any]:
        """
        Process an audio chunk with VAD filtering and ASR transcription.
        
        Args:
            audio_chunk (Union[np.ndarray, torch.Tensor]): Audio chunk data
            is_last (bool): Flag indicating if this is the last chunk
            
        Returns:
            Dict[str, Any]: Results with 'partial' and 'final' transcriptions
        """
        # If VAD is not enabled or not available, process directly with ASR
        if self.vad_processor is None:
            return self.asr_engine.process_audio(audio_chunk, is_last=is_last)
        
        # Process with VAD
        speech_segment = self.vad_processor.process_chunk(audio_chunk)
        
        # If we have a speech segment, process it with ASR
        if speech_segment is not None:
            return self.asr_engine.process_audio(speech_segment.numpy(), is_last=False)
        
        # If this is the last chunk, flush any remaining audio
        if is_last:
            remaining_segment = self.vad_processor.flush()
            if remaining_segment is not None:
                result = self.asr_engine.process_audio(remaining_segment.numpy(), is_last=True)
                # Reset VAD state for next session
                self.vad_processor.reset_states()
                return result
        
        # No speech detected or not enough to form a segment
        return {
            'current_transcription': getattr(self.asr_engine, 'current_transcription', ''),
            'new_final_text': None
        }
    
    def reset(self):
        """Reset both VAD and ASR states for new audio session."""
        if self.vad_processor is not None:
            self.vad_processor.reset_states()
        if self.asr_engine is not None:
            self.asr_engine.reset_state()


# Backward compatibility
VADProcessorWrapper = VADProcessor