"""
ViStreamASR Core Module

This module provides the complete streaming ASR functionality including:
- High-level StreamingASR interface for easy usage
- Low-level ASREngine for advanced control
- Model loading and caching
- Incremental ASR processing

All functionality is consolidated in this single module.
"""

import sys
import os
import torch
from typing import Tuple, List, Generator, Optional, Dict, Any
import torchaudio.compliance.kaldi as kaldi
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time
import tarfile
import tempfile
import requests
from pathlib import Path

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
    'download': '📥' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[DOWN]',
    'loading': '🔄' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[LOAD]',
    'rocket': '🚀' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[GPU]',
    'ruler': '📏' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SIZE]',
    'warning': '⚠️' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[WARN]',
    'green': '🟢' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[PROC]',
    'chart': '📊' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[DATA]',
    'buffer': '📈' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[BUF]',
    'memo': '📝' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[TEXT]',
    'clean': '🧹' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[CLEAN]',
    'clock': '⏰' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[TIME]',
    'finish': '🏁' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[END]',
    'skip': '⏭️' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SKIP]',
}

# GPU configuration
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

def get_cache_dir():
    """Get the ViStreamASR cache directory."""
    cache_dir = Path.home() / ".cache" / "ViStreamASR"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors."""
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels."""
    _sos = torch.tensor([sos], dtype=torch.long, requires_grad=False, device=ys_pad.device)
    _eos = torch.tensor([eos], dtype=torch.long, requires_grad=False, device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

def reverse_pad_list(ys_pad: torch.Tensor, ys_lens: torch.Tensor, pad_value: float = -1.0) -> torch.Tensor:
    """Reverse padding for the list of tensors."""
    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True, pad_value)
    return r_ys_pad

def compute_fbank(wav_path=None, waveform=None, sample_rate=16000, 
                  num_mel_bins=80, frame_length=25, frame_shift=10):
    """Extract fbank features."""
    if waveform is None:
        waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                     num_mel_bins=num_mel_bins,
                     frame_length=frame_length,
                     frame_shift=frame_shift,
                     sample_frequency=sample_rate)
    return mat

def load_models(debug_mode=False):
    """Load models with automatic download and caching to ~/.cache/ViStreamASR."""
    
    # Use cache directory
    cache_dir = get_cache_dir()
    model_path = cache_dir / "pytorch_model.bin"
    
    # Default Hugging Face model URL
    model_url = "https://huggingface.co/nguyenvulebinh/ViStreamASR/resolve/main/pytorch_model.bin"
    
    if debug_mode:
        print(f"{symbols['tool']} [ENGINE] Cache directory: {cache_dir}")
        print(f"{symbols['tool']} [ENGINE] Model path: {model_path}")
    
    # Check if model exists in cache, if not download it
    if not model_path.exists():
        if debug_mode:
            print(f"{symbols['download']} [ENGINE] Model not found in cache, downloading...")
            print(f"{symbols['download']} [ENGINE] URL: {model_url}")
        
        try:
            # Download the model with progress indication
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            print(f"{symbols['download']} Downloading ViStreamASR model to cache...")
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownloading... {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
            
            print(f"\n{symbols['check']} Model downloaded successfully to cache")
            
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to download model from {model_url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error downloading model: {e}")
    else:
        if debug_mode:
            print(f"{symbols['check']} [ENGINE] Model found in cache")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if debug_mode:
        print(f"{symbols['loading']} [ENGINE] Loading acoustic model from cache...")
    
    with tarfile.open(model_path, 'r') as tar:
        # Load acoustic model
        try:
            model_member = tar.getmember('model/acoustic/model.pt')
            model_file = tar.extractfile(model_member)
            
            # Create a temporary file to load the torch model
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_model:
                temp_model.write(model_file.read())
                temp_model.flush()
                
                # Load the model
                model = torch.jit.load(temp_model.name)
                
                # Clean up temp file
                os.unlink(temp_model.name)
                
            if use_gpu:
                model = model.cuda()
                if debug_mode:
                    print(f"{symbols['rocket']} [ENGINE] Model loaded on GPU")
            else:
                if debug_mode:
                    print(f"{symbols['tool']} [ENGINE] Model loaded on CPU")
                
        except KeyError as e:
            raise FileNotFoundError(f"Acoustic model not found in model file: {e}")
    
    if debug_mode:
        print(f"{symbols['loading']} [ENGINE] Loading language models from cache...")
    
    with tarfile.open(model_path, 'r') as tar:
        # Extract text files to temporary files for the decoder
        temp_files = {}
        
        try:
            # Extract lexicon file
            lexicon_member = tar.getmember('model/lm/lexicon.txt')
            lexicon_file = tar.extractfile(lexicon_member)
            temp_lexicon = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_lexicon.write(lexicon_file.read().decode('utf-8'))
            temp_lexicon.flush()
            temp_files['lexicon'] = temp_lexicon.name
            
            # Extract tokens file
            tokens_member = tar.getmember('model/lm/tokens.txt')
            tokens_file = tar.extractfile(tokens_member)
            temp_tokens = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_tokens.write(tokens_file.read().decode('utf-8'))
            temp_tokens.flush()
            temp_files['tokens'] = temp_tokens.name
            
            # Extract language model binary
            lm_member = tar.getmember('model/lm/vi_lm_5grams.bin')
            lm_file = tar.extractfile(lm_member)
            temp_lm = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
            temp_lm.write(lm_file.read())
            temp_lm.flush()
            temp_files['lm'] = temp_lm.name
            
        except KeyError as e:
            # Clean up any created temp files
            for temp_path in temp_files.values():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise FileNotFoundError(f"Language model file not found in model file: {e}")
    
    try:
        lm_files = {
            "lexicon": temp_files['lexicon'],
            "tokens": temp_files['tokens'],
            "lm": temp_files['lm'],
            "blank_token": '<blank>', 
            "sil_token": '<sos/eos>',
            "beam_size": 100,
            "LM_WEIGHT": 0.5,
        }
        
        ngram_beam_search_decoder = ctc_decoder(
            lexicon=lm_files['lexicon'],
            tokens=lm_files['tokens'],
            lm=lm_files['lm'],
            nbest=20,
            beam_size=lm_files['beam_size'],
            lm_weight=lm_files['LM_WEIGHT'],
            log_add=True,
            blank_token=lm_files['blank_token'],
            sil_token=lm_files['sil_token']
        )
        
        beam_search_decoder = ctc_decoder(
            lexicon=lm_files['lexicon'],
            tokens=lm_files['tokens'],
            lm=None,
            nbest=20,
            beam_size=16,
            blank_token=lm_files['blank_token'],
            sil_token=lm_files['sil_token']
        )
        
    finally:
        # Clean up temporary files
        for temp_path in temp_files.values():
            try:
                os.unlink(temp_path)
            except:
                pass
    
    if debug_mode:
        print(f"{symbols['check']} [ENGINE] Models loaded successfully from cache!")
    
    return model, ngram_beam_search_decoder, beam_search_decoder

def ngram_beam_search(ngram_beam_search_decoder, emission):
    """Perform n-gram beam search decoding."""
    ngram_beam_search_result = ngram_beam_search_decoder(emission.cpu())
    decoder_output_tokens = []
    decoder_output_transcript = []
    decoder_ngram_best_transcipts = []
    
    for sample_output in ngram_beam_search_result:
        decoder_output_tokens.append([])
        decoder_output_transcript.append([])
        for beam_output in sample_output:
            decoder_output_tokens[-1].append((beam_output.tokens[1:-1], beam_output.score))
            decoder_output_transcript[-1].append(' '.join(beam_output.words))
        decoder_ngram_best_transcipts.append(' '.join(sample_output[0].words))
    return decoder_ngram_best_transcipts, decoder_output_tokens, decoder_output_transcript

def clm_beam_ranking(model, encoder_out, decoder_output_tokens, decoder_output_transcript, reverse_weight=0.3):
    """Perform CTC Language Model beam ranking."""
    hyps = [y[0] for x in decoder_output_tokens for y in x]
    sample_num_hyps = [len(item) for item in decoder_output_tokens]
    hyps_pad = pad_sequence(
        hyps, 
        batch_first=True, padding_value=model.ignore_id,
    )
    if use_gpu:
        hyps_pad = hyps_pad.cuda()
    ori_hyps_pad = hyps_pad
    hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)
    hyps_pad, _ = add_sos_eos(hyps_pad, model.sos, model.eos, model.ignore_id)
    hyps_lens = hyps_lens + 1

    encoder_out_repeat = []
    for i_encoder_out, num_hyps in zip(encoder_out, sample_num_hyps):
        encoder_out_repeat.append(i_encoder_out.unsqueeze(0).repeat(num_hyps, 1, 1))
    encoder_out_repeat = torch.cat(encoder_out_repeat, dim=0)

    encoder_repeat_mask = torch.ones(len(hyps_pad), 1, encoder_out_repeat.size(1), dtype=torch.bool, device=device)
    r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, model.ignore_id)
    r_hyps_pad, _ = add_sos_eos(r_hyps_pad, model.sos, model.eos, model.ignore_id)
    decoder_out, r_decoder_out, _ = model.decoder.forward(encoder_out_repeat, encoder_repeat_mask, hyps_pad, hyps_lens, r_hyps_pad, reverse_weight=reverse_weight)
    decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
    decoder_out = decoder_out.cpu()
    r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
    r_decoder_out = r_decoder_out.cpu()

    clm_scores = []
    for i, hyp in enumerate(hyps):
        score = torch.gather(decoder_out[i, :len(hyps[i]), :], 1, hyps[i].unsqueeze(1)).sum()
        score += decoder_out[i][len(hyp)][model.eos]
        if reverse_weight > 0:
            r_score = torch.gather(torch.flip(r_decoder_out[i, :len(hyps[i]), :], dims=[0]), 1, hyps[i].unsqueeze(1)).sum()
            r_score += r_decoder_out[i][len(hyp)][model.eos]
            score = score * (1 - reverse_weight) + r_score * reverse_weight
        clm_scores.append(score.detach().numpy().tolist())

    idx = 0
    clm_scores_by_sample = []
    for num in sample_num_hyps:
        clm_scores_by_sample.append(clm_scores[idx: idx + num])
        idx += num
    
    decoder_clm_best_transcipts = []
    for sample_candidates, scores in zip(decoder_output_transcript, clm_scores_by_sample):
        sort_candidate = list(zip(sample_candidates, scores))
        sort_candidate.sort(key = lambda value: value[1], reverse=True)
        decoder_clm_best_transcipts.append(sort_candidate)
    return [item[0][0] for item in decoder_clm_best_transcipts]


class IncrementalASR():
    def __init__(self, model, device='cpu'):
        assert model.encoder.static_chunk_size > 0 or model.encoder.use_dynamic_chunk
        self.model = model
        self.decoding_chunk_size = 16
        self.num_decoding_left_chunks = -1
        self.fbank_interval = 1600
        self.fbank_future = 400
        self.fbank_window = self.fbank_interval + self.fbank_future
        self.num_frame = 10
        self.subsampling = model.subsampling_rate()
        self.context = model.right_context() + 1
        self.stride = self.subsampling * self.decoding_chunk_size
        self.decoding_window = (self.decoding_chunk_size - 1) * self.subsampling + self.context
        self.required_cache_size = self.decoding_chunk_size * self.num_decoding_left_chunks
        self.device = device
        self.reset_cache()
        
    def reset_cache(self):
        self.cache = {
            'offset': 0,
            'att_cache': torch.zeros((0, 0, 0, 0), device=self.device),
            'cnn_cache': torch.zeros((0, 0, 0, 0), device=self.device),
            'audio_remain': torch.tensor([], dtype=torch.float32, device=self.device),
            'fbank_features_remain': torch.tensor([], dtype=torch.float32, device=self.device),
        }
        
    def forward(self, audio_wav, last=False):
        """Process audio incrementally."""
        outputs = []
        emissions = []

        audio_wav = torch.cat([self.cache['audio_remain'], audio_wav])
        fbank_features = self.cache['fbank_features_remain']
        
        if len(audio_wav) > 0:
            if last:
                audio_wav = torch.cat([audio_wav, torch.tensor([0.]*2000, dtype=torch.float32, device=self.device)])
                
            for cur in range(0, len(audio_wav), self.fbank_interval):
                if last:
                    end = min(cur + self.fbank_window, len(audio_wav))
                else:
                    if cur + self.fbank_window > len(audio_wav):
                        self.cache['audio_remain'] = audio_wav[cur:]
                        continue
                    else:
                        end = cur + self.fbank_window
                fbank_features = torch.cat([fbank_features, compute_fbank(waveform=audio_wav[cur:end].unsqueeze(0))[:self.num_frame].unsqueeze(0)], 1)
            
            num_frames = fbank_features.size(1)

            if not last and num_frames < self.decoding_window:
                self.cache['fbank_features_remain'] = fbank_features
                return None, None

            for cur in range(0, num_frames, self.stride):
                if last:
                    end = min(cur + self.decoding_window, num_frames)
                    if end - cur < 10:
                        continue
                else:
                    if cur + self.decoding_window > num_frames:
                        self.cache['fbank_features_remain'] = fbank_features[:, cur:, :]
                        continue
                    else:
                        end = cur + self.decoding_window
                chunk_xs = fbank_features[:, cur:end, :]

                (y, self.cache['att_cache'], self.cache['cnn_cache']) = \
                    self.model.encoder.forward_chunk(
                        chunk_xs, self.cache['offset'], self.required_cache_size,
                        self.cache['att_cache'], self.cache['cnn_cache'])
                
                outputs.append(y)
                emissions.append(self.model.ctc.ctc_lo(y))
                self.cache['offset'] += y.size(1)

        if last:
            self.reset_cache()

        if len(outputs) == 0:
            return None, None

        return torch.cat(outputs, 1), torch.cat(emissions, 1)

class ASREngine:
    """Main ASR Engine class that handles all streaming functionality."""
    
    def __init__(self, chunk_size_ms=640, max_duration_before_forced_finalization=15.0, debug_mode=False):
        # Model components
        self.acoustic_model = None
        self.ngram_lm = None
        self.beam_search = None
        self.asr_realtime_model = None
        
        # Processing state
        self.current_transcription = ""
        self.buffer_emission = None
        self.buffer_encode_out = None
        self.chunks_since_last_finalization = 0
        
        # Configuration
        self.chunk_size_ms = chunk_size_ms
        self.max_duration_before_forced_finalization = max_duration_before_forced_finalization
        self.debug_mode = debug_mode
        
        # Calculate max_chunks_before_forced_finalization dynamically
        self.max_chunks_before_forced_finalization = int(
            self.max_duration_before_forced_finalization * 1000 / self.chunk_size_ms
        )
        
        if self.debug_mode:
            print(f"{symbols['tool']} [CONFIG] Chunk size: {self.chunk_size_ms}ms")
            print(f"{symbols['tool']} [CONFIG] Max duration before forced finalization: {self.max_duration_before_forced_finalization}s")
            print(f"{symbols['tool']} [CONFIG] Max chunks before forced finalization: {self.max_chunks_before_forced_finalization}")
        
        # Timing tracking for pure ASR performance
        self.asr_processing_time = 0.0
        self.asr_audio_duration = 0.0
        
    def initialize_models(self):
        """Initialize all models if not already loaded."""
        if self.acoustic_model is None:
            self.acoustic_model, self.ngram_lm, self.beam_search = load_models(debug_mode=self.debug_mode)
            self.asr_realtime_model = IncrementalASR(self.acoustic_model, device=device)
    
    def reset_state(self):
        """Reset ASR state and transcription."""
        if self.debug_mode:
            print(f"{symbols['clean']} [STATE] Resetting ASR state...")
        if self.asr_realtime_model is not None:
            self.asr_realtime_model.reset_cache()
            if self.debug_mode:
                print(f"{symbols['clean']} [STATE] Incremental ASR cache cleared")
        
        old_transcription = self.current_transcription
        self.current_transcription = ""
        self.buffer_emission = None
        self.buffer_encode_out = None
        self.chunks_since_last_finalization = 0
        
        # Reset ASR timing
        self.asr_processing_time = 0.0
        self.asr_audio_duration = 0.0
        
        if self.debug_mode:
            print(f"{symbols['clean']} [STATE] State reset complete. Old transcription: '{old_transcription}'")
    
    def get_asr_rtf(self):
        """Get the pure ASR processing RTF (computational performance only)."""
        if self.asr_audio_duration > 0:
            return self.asr_processing_time / self.asr_audio_duration
        return 0.0
    
    def _prepare_audio_tensor(self, audio_data):
        """Prepare audio tensor for processing."""
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device=device)
        else:
            audio_tensor = audio_data.to(device)
        
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.mean(axis=0)
        
        if torch.max(torch.abs(audio_tensor)) > 0:
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
        
        return audio_tensor
    
    def _update_buffers(self, emission, encoder_out):
        """Update emission and encoder buffers."""
        if self.buffer_emission is None:
            self.buffer_emission = emission
            self.buffer_encode_out = encoder_out
        else:
            self.buffer_emission = torch.cat([self.buffer_emission, emission], 1)
            self.buffer_encode_out = torch.cat([self.buffer_encode_out, encoder_out], 1)
    
    def _reset_buffers(self):
        """Reset buffers after finalization."""
        self.buffer_emission = None
        self.buffer_encode_out = None
        self.current_transcription = ""
    
    def _run_lm_pipeline(self, buffer_emission, buffer_encode_out):
        """Run full language model pipeline."""
        ngram_best, beam_tokens, beam_transcripts = ngram_beam_search(self.ngram_lm, buffer_emission)
        cls_best = clm_beam_ranking(self.acoustic_model, buffer_encode_out, beam_tokens, beam_transcripts)
        
        if len(cls_best) > 0 and len(cls_best[0]) > 0:
            return cls_best[0]
        return None
    
    def _create_default_result(self):
        """Create default result dictionary."""
        return {
            'current_transcription': self.current_transcription,
            'new_final_text': None
        }
    
    def process_audio_chunk(self, audio_data, sample_rate, is_last=False):
        """Process a single audio chunk (core streaming logic)."""
        # Check if audio_data has same size as chunk_size_ms
        expected_samples = int(sample_rate * self.chunk_size_ms / 1000.0)
        actual_samples = len(audio_data)
        
        if actual_samples != expected_samples:
            if is_last and actual_samples < expected_samples:
                # Last chunk can be shorter - this is normal
                if self.debug_mode:
                    print(f"{symbols['ruler']} [CHUNK-SIZE] Last chunk is shorter: {actual_samples}/{expected_samples} samples ({actual_samples/sample_rate*1000:.1f}ms/{self.chunk_size_ms}ms)")
            elif actual_samples < expected_samples:
                # Non-last chunk is shorter than expected - potential issue
                if self.debug_mode:
                    print(f"{symbols['warning']}  [CHUNK-SIZE] Chunk shorter than expected: {actual_samples}/{expected_samples} samples ({actual_samples/sample_rate*1000:.1f}ms/{self.chunk_size_ms}ms)")
            else:
                # Chunk is longer than expected - potential issue
                if self.debug_mode:
                    print(f"{symbols['warning']}  [CHUNK-SIZE] Chunk longer than expected: {actual_samples}/{expected_samples} samples ({actual_samples/sample_rate*1000:.1f}ms/{self.chunk_size_ms}ms)")
        else:
            # Perfect size match
            if self.debug_mode:
                print(f"{symbols['check']} [CHUNK-SIZE] Perfect size: {actual_samples} samples ({self.chunk_size_ms}ms)")
        
        if self.asr_realtime_model is None:
            return {
                'current_transcription': self.current_transcription,
                'new_final_text': None
            }
        
        # Track audio duration for ASR timing
        chunk_duration = len(audio_data) / sample_rate
        asr_start_time = time.time()
        
        # Debug info at start of processing
        buffer_size = self.buffer_emission.size(1) if self.buffer_emission is not None else 0
        self.chunks_since_last_finalization += 1
        if self.debug_mode:
            print(f"{symbols['tool']} [CHUNK] Audio: {len(audio_data)} samples | Buffer: {buffer_size} frames | is_last: {is_last} | Chunks since finalization: {self.chunks_since_last_finalization}")
        
        # Convert to tensor and normalize
        audio_tensor = self._prepare_audio_tensor(audio_data)
        
        # Skip processing if chunk is too short
        if len(audio_tensor) < 320:
            if self.debug_mode:
                print(f"{symbols['skip']}  [CHUNK] Skipping short chunk: {len(audio_tensor)} samples < 320")
            return {
                'current_transcription': self.current_transcription,
                'new_final_text': None
            }
        
        if self.debug_mode:
            print(f"{symbols['check']} [CHUNK] Processing valid chunk: {len(audio_tensor)} samples")
        
        result = self._create_default_result()
        
        # Handle last chunk
        if is_last:
            if self.debug_mode:
                print(f"{symbols['finish']} [LAST] Processing final chunk with is_last=True")
            
            if len(audio_tensor) > 0:
                try:
                    if self.debug_mode:
                        print(f"{symbols['finish']} [LAST] Processing final audio chunk first...")
                    encoder_out, emission = self.asr_realtime_model.forward(audio_tensor, last=False)
                    
                    if emission is not None:
                        emission_frames = emission.size(1)
                        if self.debug_mode:
                            print(f"{symbols['finish']} [LAST] Final chunk added {emission_frames} frames to buffer")
                        self._update_buffers(emission, encoder_out)
                except Exception as e:
                    if self.debug_mode:
                        print(f"{symbols['warning']}  [LAST] Error processing final audio chunk: {e}")
            
            if self.buffer_emission is not None:
                buffer_size = self.buffer_emission.size(1)
                if self.debug_mode:
                    print(f"{symbols['finish']} [LAST] Buffer has {buffer_size} frames for final processing")
                try:
                    encoder_out, emission = self.asr_realtime_model.forward(torch.tensor([], dtype=torch.float32, device=device), last=True)
                    if self.debug_mode:
                        print(f"{symbols['finish']} [LAST] Incremental ASR finalization complete")
                    
                    if self.buffer_emission.size(1) > 0:
                        if self.debug_mode:
                            print(f"{symbols['finish']} [LAST] Running final LM pipeline...")
                        final_text = self._run_lm_pipeline(self.buffer_emission, self.buffer_encode_out)
                        if self.debug_mode:
                            print(f"{symbols['finish']} [LAST] Final text set: '{final_text}'")
                        result['new_final_text'] = final_text
                    else:
                        if self.debug_mode:
                            print(f"{symbols['finish']} [LAST] Buffer is empty, no final processing needed")
                    
                    if self.debug_mode:
                        print(f"{symbols['finish']} [LAST] Resetting buffers after final processing")
                    self._reset_buffers()
                except Exception as e:
                    if self.debug_mode:
                        print(f"{symbols['warning']}  [LAST] Final chunk processing error (ignored): {e}")
            else:
                if self.debug_mode:
                    print(f"{symbols['finish']} [LAST] No buffer for final processing")
            
            asr_end_time = time.time()
            self.asr_processing_time += (asr_end_time - asr_start_time)
            self.asr_audio_duration += chunk_duration
            
            return result

        # Process audio as speech
        if len(audio_tensor) > 0:
            try:
                if self.debug_mode:
                    print(f"{symbols['green']} [SPEECH] Processing speech chunk...")
                
                encoder_out, emission = self.asr_realtime_model.forward(audio_tensor, last=False)
                
                if emission is not None:
                    emission_frames = emission.size(1)
                    encoder_frames = encoder_out.size(1)
                    if self.debug_mode:
                        print(f"{symbols['chart']} [ASR] New frames - Emission: {emission_frames}, Encoder: {encoder_frames}")
                    
                    old_buffer_size = self.buffer_emission.size(1) if self.buffer_emission is not None else 0
                    if self.buffer_emission is None:
                        self._update_buffers(emission, encoder_out)
                        if self.debug_mode:
                            print(f"{symbols['buffer']} [BUFFER] Created new buffer with {emission_frames} frames")
                    else:
                        self._update_buffers(emission, encoder_out)
                        if self.debug_mode:
                            print(f"{symbols['buffer']} [BUFFER] Extended buffer: {old_buffer_size} → {self.buffer_emission.size(1)} frames")
                    
                    beam_result = ngram_beam_search(self.beam_search, self.buffer_emission)
                    new_transcription = beam_result[0][0] if beam_result[0] else ""
                    
                    if new_transcription != self.current_transcription:
                        if self.debug_mode:
                            print(f"{symbols['memo']} [PARTIAL] '{self.current_transcription}' → '{new_transcription}'")
                        self.current_transcription = new_transcription
                    else:
                        if self.debug_mode:
                            print(f"{symbols['memo']} [PARTIAL] No change: '{new_transcription}'")
                    
                    result['current_transcription'] = self.current_transcription
                else:
                    if self.debug_mode:
                        print(f"{symbols['warning']}  [ASR] No emission returned from forward pass")
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"{symbols['warning']}  ASR processing error (chunk skipped): {e}")
                result['current_transcription'] = self.current_transcription
        
        # Check for forced finalization
        force_finalization = (self.chunks_since_last_finalization >= self.max_chunks_before_forced_finalization and 
                             self.buffer_emission is not None and 
                             self.buffer_emission.size(1) > 100)
        
        if force_finalization:
            if self.debug_mode:
                print(f"{symbols['clock']} [FORCED] Triggering forced finalization after {self.chunks_since_last_finalization} chunks")
            try:
                final_text = self._run_lm_pipeline(self.buffer_emission, self.buffer_encode_out)
                if self.debug_mode:
                    print(f"{symbols['finish']} [FORCED-FINAL] Forced finalization: '{final_text}'")
                
                result['new_final_text'] = final_text
                self.chunks_since_last_finalization = 0
                self._reset_buffers()
                
                # Reset the incremental ASR cache for complete state consistency
                if self.asr_realtime_model is not None:
                    self.asr_realtime_model.reset_cache()
                    if self.debug_mode:
                        print(f"{symbols['clean']} [FORCED] IncrementalASR cache reset for complete state consistency")
                
                asr_end_time = time.time()
                self.asr_processing_time += (asr_end_time - asr_start_time)
                self.asr_audio_duration += chunk_duration
                
                return result
                
            except Exception as e:
                if self.debug_mode:
                    print(f"{symbols['warning']}  [FORCED] Forced finalization error: {e}")
            
            self.chunks_since_last_finalization = 0
        
        asr_end_time = time.time()
        self.asr_processing_time += (asr_end_time - asr_start_time)
        self.asr_audio_duration += chunk_duration
        
        return result
    
    def process_audio(self, audio_data, is_last=False):
        """Process audio data (source-agnostic)."""
        if audio_data is None or len(audio_data) == 0:
            return {
                'current_transcription': self.current_transcription,
                'new_final_text': None
            }
        
        return self.process_audio_chunk(audio_data, 16000, is_last) 