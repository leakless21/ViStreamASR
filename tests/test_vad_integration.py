"""
Unit tests for Silero-VAD integration with ViStreamASR
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vistreamasr.vad import VADProcessor, VADASRCoordinator

class TestVADProcessor:
    """Test cases for the VADProcessor class."""
    
    def test_initialization_default(self):
        """Test VADProcessor initialization with default parameters."""
        vad = VADProcessor()
        assert vad.sample_rate == 16000
        assert vad.threshold == 0.5
        assert vad.min_speech_duration_ms == 250
        assert vad.min_silence_duration_ms == 250
        assert vad.speech_pad_ms == 50
    
    def test_initialization_custom(self):
        """Test VADProcessor initialization with custom parameters."""
        vad = VADProcessor(
            sample_rate=8000,
            threshold=0.7,
            min_speech_duration_ms=300,
            min_silence_duration_ms=200,
            speech_pad_ms=100
        )
        assert vad.sample_rate == 8000
        assert vad.threshold == 0.7
        assert vad.min_speech_duration_ms == 300
        assert vad.min_silence_duration_ms == 200
        assert vad.speech_pad_ms == 100
    
    def test_process_chunk_speech(self):
        """Test VAD processing of a speech-like audio chunk."""
        vad = VADProcessor()
        
        # Create a mock speech-like audio chunk
        audio_chunk = torch.randn(512)  # 32ms at 16kHz
        
        # Mock the model to return a high probability for speech
        with patch.object(vad.model, '__call__', return_value=torch.tensor([0.8])):
            speech_segment = vad.process_chunk(audio_chunk)
            # process_chunk should return None for a single chunk (needs more context)
            assert speech_segment is None
    
    def test_process_chunk_silence(self):
        """Test VAD processing of a silence-like audio chunk."""
        vad = VADProcessor()
        
        # Create a mock silence-like audio chunk
        audio_chunk = torch.zeros(512)
        
        # Mock the model to return a low probability for speech
        with patch.object(vad.model, '__call__', return_value=torch.tensor([0.2])):
            speech_segment = vad.process_chunk(audio_chunk)
            # process_chunk should return None for silence
            assert speech_segment is None
    
    def test_is_speech_positive(self):
        """Test speech detection with high probability."""
        vad = VADProcessor(threshold=0.5)
        
        # Create a mock audio chunk
        audio_chunk = torch.randn(512)
        
        # Mock the model to return a high probability for speech
        with patch.object(vad.model, '__call__', return_value=torch.tensor([0.8])):
            result = vad.is_speech(audio_chunk)
            assert result is True
    
    def test_is_speech_negative(self):
        """Test speech detection with low probability."""
        vad = VADProcessor(threshold=0.5)
        
        # Create a mock audio chunk
        audio_chunk = torch.randn(512)
        
        # Mock the model to return a low probability for speech
        with patch.object(vad.model, '__call__', return_value=torch.tensor([0.3])):
            result = vad.is_speech(audio_chunk)
            assert result is False
    
    def test_reset_states(self):
        """Test VAD state reset functionality."""
        vad = VADProcessor()
        
        # Mock the model's reset_states method
        with patch.object(vad.model, 'reset_states') as mock_reset:
            vad.reset_states()
            mock_reset.assert_called_once()

class TestVADASRCoordinator:
    """Test cases for the VADASRCoordinator class."""
    
    def test_initialization(self):
        """Test VADASRCoordinator initialization."""
        vad_config = {
            'sample_rate': 16000,
            'threshold': 0.5
        }
        asr_engine = Mock()
        asr_engine.process_audio = Mock(return_value={'partial': '', 'final': None})
        asr_engine.reset_state = Mock()
        
        coordinator = VADASRCoordinator(vad_config, asr_engine)
        assert coordinator is not None
    
    def test_process_audio_chunk_speech(self):
        """Test processing of audio chunk with speech."""
        vad_config = {
            'sample_rate': 16000,
            'threshold': 0.5
        }
        asr_engine = Mock()
        asr_engine.process_audio = Mock(return_value={'partial': 'test transcription', 'final': None})
        asr_engine.reset_state = Mock()
        
        coordinator = VADASRCoordinator(vad_config, asr_engine)
        
        # Create a mock audio chunk
        audio_chunk = np.random.randn(1024).astype(np.float32)
        
        # Mock the VAD to detect speech and return a speech segment
        with patch.object(coordinator.vad_processor, 'process_chunk', return_value=torch.tensor(audio_chunk)):
            result = coordinator.process_audio_chunk(audio_chunk)
            assert 'partial' in result
            assert result['partial'] == 'test transcription'
            asr_engine.process_audio.assert_called_once()
    
    def test_process_audio_chunk_silence(self):
        """Test processing of audio chunk with silence."""
        vad_config = {
            'sample_rate': 16000,
            'threshold': 0.5
        }
        asr_engine = Mock()
        asr_engine.process_audio = Mock(return_value={'partial': '', 'final': None})
        asr_engine.reset_state = Mock()
        
        coordinator = VADASRCoordinator(vad_config, asr_engine)
        
        # Create a mock audio chunk
        audio_chunk = np.random.randn(1024).astype(np.float32)
        
        # Mock the VAD to detect silence (return None)
        with patch.object(coordinator.vad_processor, 'process_chunk', return_value=None):
            result = coordinator.process_audio_chunk(audio_chunk)
            # When silence is detected, no ASR processing should occur
            # The result should be empty or indicate no processing
            assert result is not None
            assert 'current_transcription' in result
            asr_engine.process_audio.assert_not_called()
    
    def test_reset(self):
        """Test coordinator reset functionality."""
        vad_config = {
            'sample_rate': 16000,
            'threshold': 0.5
        }
        asr_engine = Mock()
        asr_engine.process_audio = Mock(return_value={'partial': '', 'final': None})
        asr_engine.reset_state = Mock()
        
        coordinator = VADASRCoordinator(vad_config, asr_engine)
        
        # Mock the reset methods of VAD and ASR components
        with patch.object(coordinator.vad_processor, 'reset_states') as mock_vad_reset:
            coordinator.reset()
            mock_vad_reset.assert_called_once()
            asr_engine.reset_state.assert_called_once()

# Integration tests that can run without the actual VAD implementation
class TestVADIntegrationConcepts:
    """Test conceptual integration of VAD with ViStreamASR."""
    
    def test_vad_asr_workflow(self):
        """Test the conceptual workflow of VAD filtering before ASR processing."""
        # This test represents the intended workflow
        
        # 1. Audio chunk is received
        audio_chunk = np.random.randn(1024).astype(np.float32)
        
        # 2. VAD processing (conceptual)
        # In reality, this would call the VAD processor
        is_speech = True  # Mock result
        
        # 3. If speech is detected, forward to ASR
        if is_speech:
            # ASR processing would occur here
            asr_result = "mock transcription"
            assert asr_result is not None
        else:
            # Silence is filtered out
            pass
    
    def test_vad_parameter_effects(self):
        """Test how different VAD parameters affect processing."""
        # Test different threshold values
        thresholds = [0.3, 0.5, 0.7]
        
        # Mock probabilities
        probabilities = [0.4, 0.6, 0.8]
        
        for threshold in thresholds:
            for prob in probabilities:
                is_speech = prob >= threshold
                # This is just to ensure the logic works
                assert isinstance(is_speech, bool)
    
    def test_vietnamese_speech_considerations(self):
        """Test considerations for Vietnamese speech processing."""
        # Vietnamese speech characteristics to consider:
        # 1. Tonal nature - ensure VAD doesn't filter out tonal components
        # 2. Short syllables - adjust minimum speech duration
        # 3. Dialect variations - test with different dialect patterns
        
        # This test is conceptual and would be expanded with actual Vietnamese audio samples
        assert True  # Placeholder for actual implementation

# Performance tests (conceptual)
class TestVADPerformance:
    """Conceptual performance tests for VAD integration."""
    
    def test_processing_time_requirements(self):
        """Test that VAD processing meets time requirements."""
        # Requirement: Each 30ms+ chunk should process in <1ms
        # This would be measured with actual implementation
        max_processing_time = 0.001  # 1ms in seconds
        assert isinstance(max_processing_time, float)
        assert max_processing_time > 0
    
    def test_memory_usage(self):
        """Test memory usage requirements."""
        # Requirement: Model size < 5MB, runtime memory < 10MB
        # This would be measured with actual implementation
        max_model_size = 5 * 1024 * 1024  # 5MB in bytes
        max_runtime_memory = 10 * 1024 * 1024  # 10MB in bytes
        assert isinstance(max_model_size, int)
        assert isinstance(max_runtime_memory, int)

if __name__ == "__main__":
    pytest.main([__file__])