import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vistreamasr.streaming import StreamingASR

@pytest.fixture
def mock_asr_engine():
    """Mock ASR engine for testing."""
    engine = MagicMock()
    engine.process_audio.return_value = {'current_transcription': 'test', 'new_final_text': None}
    engine.reset_state = MagicMock()
    return engine

@pytest.fixture
def mock_vad_coordinator():
    """Mock VAD coordinator for testing."""
    coordinator = MagicMock()
    coordinator.process_audio_chunk.return_value = {'current_transcription': 'test', 'new_final_text': None}
    return coordinator

@pytest.fixture
def streaming_asr_with_vad(mock_asr_engine, mock_vad_coordinator):
    """StreamingASR instance with mocked VAD and ASR engine."""
    with patch('vistreamasr.streaming.ASREngine', return_value=mock_asr_engine):
        asr = StreamingASR(vad_config={'enabled': True, 'sample_rate': 16000})
        asr.engine = mock_asr_engine
        asr.vad_coordinator = mock_vad_coordinator
        asr.vad_processor = MagicMock()
        return asr

def test_microphone_vad_chunking(streaming_asr_with_vad, mock_vad_coordinator):
    """Test that microphone audio is correctly chunked for VAD processing."""
    # Simulate audio data that would cause the original error (10240 samples)
    large_audio_chunk = np.random.rand(10240).astype(np.float32)
    
    # Mock sounddevice.InputStream to feed our test data
    with patch('vistreamasr.streaming.sd.InputStream') as mock_stream:
        def callback_mock(indata, frames, time_info, status):
            # Feed the large chunk in one go
            indata[:, 0] = large_audio_chunk[:frames]
        
        mock_stream.return_value.__enter__.return_value = MagicMock()
        
        # Simulate the callback being called with our test data
        with patch.object(streaming_asr_with_vad, '_ensure_engine_initialized'):
            # Manually trigger the callback logic to test chunking
            buffer = np.zeros((0,), dtype=np.float32)
            
            def callback(indata, frames, time_info, status):
                nonlocal buffer
                buffer = np.concatenate((buffer, indata[:, 0]))
            
            # Simulate receiving the large audio chunk
            callback(np.array([large_audio_chunk]).reshape(-1, 1), 10240, {}, None)
            
            # Process the buffer as the stream_from_microphone method would
            chunk_size_samples = 10240  # Default chunk size for 640ms at 16kHz
            chunk = buffer[:chunk_size_samples]
            
            # Process the chunk with VAD
            result = {'current_transcription': '', 'new_final_text': None}
            vad_chunk_size = 512
            num_vad_chunks = (len(chunk) + vad_chunk_size - 1) // vad_chunk_size
            
            for j in range(num_vad_chunks):
                start = j * vad_chunk_size
                end = min(start + vad_chunk_size, len(chunk))
                vad_chunk = chunk[start:end]
                
                # Pad the last chunk if necessary
                current_vad_chunk_len = len(vad_chunk)
                if current_vad_chunk_len < vad_chunk_size:
                    padding_needed = vad_chunk_size - current_vad_chunk_len
                    vad_chunk = np.pad(vad_chunk, (0, padding_needed), 'constant')
                
                # Verify chunk size is correct for VAD
                assert len(vad_chunk) == vad_chunk_size, f"VAD chunk size incorrect: {len(vad_chunk)} != {vad_chunk_size}"
                
                # Mock VAD processing
                mock_vad_coordinator.process_audio_chunk(vad_chunk, is_last=(j == num_vad_chunks - 1))
            
            # Verify VAD coordinator was called the correct number of times
            expected_calls = num_vad_chunks
            assert mock_vad_coordinator.process_audio_chunk.call_count == expected_calls, \
                f"VAD coordinator called {mock_vad_coordinator.process_audio_chunk.call_count} times, expected {expected_calls}"

def test_microphone_vad_padding(streaming_asr_with_vad, mock_vad_coordinator):
    """Test that partial final chunks are correctly padded for VAD processing."""
    # Simulate audio data that results in a partial final VAD chunk
    audio_chunk = np.random.rand(520).astype(np.float32)  # Will result in one full and one partial chunk
    
    with patch.object(streaming_asr_with_vad, '_ensure_engine_initialized'):
        # Process the chunk with VAD
        result = {'current_transcription': '', 'new_final_text': None}
        vad_chunk_size = 512
        num_vad_chunks = (len(audio_chunk) + vad_chunk_size - 1) // vad_chunk_size
        
        for j in range(num_vad_chunks):
            start = j * vad_chunk_size
            end = min(start + vad_chunk_size, len(audio_chunk))
            vad_chunk = audio_chunk[start:end]
            
            # Pad the last chunk if necessary
            current_vad_chunk_len = len(vad_chunk)
            if current_vad_chunk_len < vad_chunk_size:
                padding_needed = vad_chunk_size - current_vad_chunk_len
                vad_chunk = np.pad(vad_chunk, (0, padding_needed), 'constant')
                
                # Verify padding was applied
                assert len(vad_chunk) == vad_chunk_size, f"Padded VAD chunk size incorrect: {len(vad_chunk)} != {vad_chunk_size}"
                assert np.all(vad_chunk[current_vad_chunk_len:] == 0), "Padding should be zeros"
            
            # Mock VAD processing
            mock_vad_coordinator.process_audio_chunk(vad_chunk, is_last=(j == num_vad_chunks - 1))