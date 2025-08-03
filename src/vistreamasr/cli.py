#!/usr/bin/env python3
"""
ViStreamASR CLI Tool

Command-line interface for demonstrating streaming ASR functionality.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Initialize logging first
from .logging import initialize_logging, log_with_symbol

# Define symbols that work across platforms
symbols = {
    'mic': '🎤' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[MIC]',
    'folder': '📁' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[FILE]',
    'ruler': '📏' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SIZE]',
    'clock': '⏰' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[TIME]',
    'tool': '🔧' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[CONFIG]',
    'check': '✅' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[OK]',
    'wave': '🎵' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[AUDIO]',
    'memo': '📝' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[TEXT]',
    'book': '📖' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[INFO]',
    'home': '🏠' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[HOME]',
    'brain': '🧠' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[MODEL]',
    'rocket': '🚀' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[GPU]',
    'stopwatch': '⏱️' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[TIME]',
    'speaker': '🔊' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SPEAKER]',
    'stop': '⏹️' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[STOP]',
}

# Handle import for both installed package and development mode
try:
    from .streaming import StreamingASR
    from .config import ViStreamASRSettings
except ImportError:
    from streaming import StreamingASR
    from config import ViStreamASRSettings


def transcribe_file_streaming(audio_file, settings: ViStreamASRSettings):
    """
    Transcribe an audio file using streaming ASR.
    
    Args:
        audio_file: Path to audio file
        settings: ViStreamASRSettings configuration
    """
    log_with_symbol(symbols['mic'], f"ViStreamASR File Transcription")
    log_with_symbol(symbols['folder'], f"Audio file: {audio_file}")
    log_with_symbol(symbols['ruler'], f"Chunk size: {settings.model.chunk_size_ms}ms")
    log_with_symbol(symbols['clock'], f"Auto-finalize after: {settings.model.auto_finalize_after}s")
    log_with_symbol(symbols['tool'], f"Debug mode: {settings.model.debug}")
    
    if not os.path.exists(audio_file):
        log_with_symbol(symbols['stop'], f"Error: Audio file not found: {audio_file}", "error")
        return 1
    
    # Initialize StreamingASR
    log_with_symbol(symbols['tool'], "Initializing ViStreamASR...")
    
    # Prepare VAD configuration
    vad_config = None
    if settings.vad.enabled:
        vad_config = settings.vad.model_dump()
        log_with_symbol(symbols['tool'], f"VAD enabled with aggressiveness={settings.vad.aggressiveness}")
    
    asr = StreamingASR(
        chunk_size_ms=settings.model.chunk_size_ms,
        auto_finalize_after=settings.model.auto_finalize_after,
        debug=settings.model.debug,
        vad_config=vad_config
    )
    
    # Collect results
    final_segments = []
    current_partial = ""
    
    # Start streaming
    log_with_symbol(symbols['wave'], "Starting streaming transcription...")
    
    start_time = time.time()
    
    try:
        for result in asr.stream_from_file(audio_file, chunk_size_ms=settings.model.chunk_size_ms):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial') and result.get('text'):
                current_partial = result['text']
                log_with_symbol(
                    symbols['memo'], 
                    f"[PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {current_partial}",
                    "debug" if settings.model.debug else "info"
                )
            
            if result.get('final') and result.get('text'):
                final_text = result['text']
                final_segments.append(final_text)
                current_partial = ""
                log_with_symbol(
                    symbols['check'], 
                    f"[FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}",
                    "info"
                )
        
    except KeyboardInterrupt:
        log_with_symbol(symbols['stop'], "Interrupted by user")
        return 0
    except Exception as e:
        log_with_symbol(symbols['stop'], f"Error during streaming: {e}", "error")
        return 1
    
    # Final results
    end_time = time.time()
    total_time = end_time - start_time
    
    log_with_symbol(symbols['stopwatch'], f"Processing time: {total_time:.2f} seconds")
    log_with_symbol(symbols['memo'], f"Final segments: {len(final_segments)}")
    
    log_with_symbol(symbols['memo'], "Complete Transcription:")
    complete_transcription = " ".join(final_segments)
    # Wrap text at 80 characters for better readability
    words = complete_transcription.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= 80:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    for line in lines:
        print(line)
    
    log_with_symbol(symbols['check'], "Transcription completed successfully!")
    return 0


def transcribe_microphone_streaming(duration_seconds, settings: ViStreamASRSettings):
    """
    Transcribe from microphone using streaming ASR.
    
    Args:
        duration_seconds: Maximum duration to record (None for infinite)
        settings: ViStreamASRSettings configuration
    """
    log_with_symbol(symbols['mic'], "ViStreamASR Microphone Transcription")
    log_with_symbol(symbols['speaker'], "Recording from microphone")
    log_with_symbol(symbols['ruler'], f"Chunk size: {settings.model.chunk_size_ms}ms")
    log_with_symbol(symbols['clock'], f"Auto-finalize after: {settings.model.auto_finalize_after}s")
    if duration_seconds:
        log_with_symbol(symbols['stopwatch'], f"Duration: {duration_seconds}s")
    else:
        log_with_symbol(symbols['stopwatch'], "Duration: Unlimited (Press Ctrl+C to stop)")
    log_with_symbol(symbols['tool'], f"Debug mode: {settings.model.debug}")
    
    # Check if microphone is available
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            log_with_symbol(symbols['stop'], "Error: No microphone devices found", "error")
            return 1
        log_with_symbol(symbols['check'], f"Found {len(input_devices)} microphone device(s)")
    except ImportError:
        log_with_symbol(symbols['stop'], "Error: sounddevice library not installed. Install with: pip install sounddevice", "error")
        return 1
    except Exception as e:
        log_with_symbol(symbols['stop'], f"Error checking microphone: {e}", "error")
        return 1
    
    # Initialize StreamingASR
    log_with_symbol(symbols['tool'], "Initializing ViStreamASR...")
    
    # Prepare VAD configuration
    vad_config = None
    if settings.vad.enabled:
        vad_config = settings.vad.model_dump()
        log_with_symbol(symbols['tool'], f"VAD enabled with aggressiveness={settings.vad.aggressiveness}")
    
    asr = StreamingASR(
        chunk_size_ms=settings.model.chunk_size_ms,
        auto_finalize_after=settings.model.auto_finalize_after,
        debug=settings.model.debug,
        vad_config=vad_config
    )
    
    # Collect results
    final_segments = []
    current_partial = ""
    
    # Start streaming
    log_with_symbol(symbols['wave'], "Starting microphone streaming...")
    log_with_symbol(symbols['speaker'], "Please speak into your microphone...")
    
    start_time = time.time()
    
    try:
        for result in asr.stream_from_microphone(duration_seconds=duration_seconds):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial') and result.get('text'):
                current_partial = result['text']
                log_with_symbol(
                    symbols['memo'], 
                    f"[PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {current_partial}",
                    "debug" if settings.model.debug else "info"
                )
            
            if result.get('final') and result.get('text'):
                final_text = result['text']
                final_segments.append(final_text)
                current_partial = ""
                log_with_symbol(
                    symbols['check'], 
                    f"[FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}",
                    "info"
                )
        
    except KeyboardInterrupt:
        log_with_symbol(symbols['stop'], "Microphone streaming stopped by user")
        return 0
    except Exception as e:
        log_with_symbol(symbols['stop'], f"Error during microphone streaming: {e}", "error")
        return 1
    
    # Final results
    end_time = time.time()
    total_time = end_time - start_time
    
    log_with_symbol(symbols['stopwatch'], f"Recording time: {total_time:.2f} seconds")
    log_with_symbol(symbols['memo'], f"Final segments: {len(final_segments)}")
    
    if final_segments:
        log_with_symbol(symbols['memo'], "Complete Transcription:")
        complete_transcription = " ".join(final_segments)
        # Wrap text at 80 characters for better readability
        words = complete_transcription.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= 80:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            print(line)
    else:
        log_with_symbol(symbols['memo'], "No speech detected or transcribed during recording")
    
    log_with_symbol(symbols['check'], "Microphone transcription completed successfully!")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ViStreamASR - Vietnamese Streaming ASR Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s transcribe audio.wav                                    # Basic file transcription
  %(prog)s transcribe audio.wav --config path/to/config.toml       # Use custom config
  %(prog)s transcribe audio.wav --model.debug                      # Enable debug logging
  %(prog)s transcribe audio.wav --vad.enabled                      # Enable VAD
   
  %(prog)s microphone                                              # Record from microphone indefinitely
  %(prog)s microphone --duration 30                               # Record for 30 seconds
  %(prog)s microphone --config path/to/config.toml                # Use custom config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to TOML configuration file (default: looks for vistreamasr.toml in current and parent directories)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Transcribe command (for audio files)
    transcribe_parser = subparsers.add_parser(
        'transcribe', 
        help='Transcribe audio file using streaming ASR',
        description='Stream process an audio file and show real-time transcription results'
    )
    transcribe_parser.add_argument(
        'audio_file',
        help='Path to audio file (WAV, MP3, etc.)'
    )
    
    # Model arguments
    model_group = transcribe_parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--model.chunk-size-ms',
        type=int,
        help='Chunk size in milliseconds (default: 640ms)'
    )
    model_group.add_argument(
        '--model.auto-finalize-after',
        type=float,
        help='Maximum duration in seconds before auto-finalizing a segment (default: 15.0s)'
    )
    model_group.add_argument(
        '--model.debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # VAD arguments
    vad_group = transcribe_parser.add_argument_group('VAD Options')
    vad_group.add_argument(
        '--vad.enabled',
        action='store_true',
        help='Enable Voice Activity Detection to filter silence'
    )
    vad_group.add_argument(
        '--vad.threshold',
        type=float,
        help='VAD speech probability threshold (default: 0.5)'
    )
    vad_group.add_argument(
        '--vad.min-speech-duration-ms',
        type=int,
        help='Minimum speech duration in milliseconds (default: 250ms)'
    )
    vad_group.add_argument(
        '--vad.min-silence-duration-ms',
        type=int,
        help='Minimum silence duration in milliseconds (default: 100ms)'
    )
    vad_group.add_argument(
        '--vad.speech-pad-ms',
        type=int,
        help='Padding added to speech segments in milliseconds (default: 30ms)'
    )
    
    # Microphone command
    microphone_parser = subparsers.add_parser(
        'microphone',
        aliases=['mic'],
        help='Transcribe from microphone using streaming ASR',
        description='Record from microphone and show real-time transcription results'
    )
    microphone_parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Maximum duration to record in seconds (default: unlimited, press Ctrl+C to stop)'
    )
    
    # Model arguments for microphone
    mic_model_group = microphone_parser.add_argument_group('Model Options')
    mic_model_group.add_argument(
        '--model.chunk-size-ms',
        type=int,
        help='Chunk size in milliseconds (default: 640ms)'
    )
    mic_model_group.add_argument(
        '--model.auto-finalize-after',
        type=float,
        help='Maximum duration in seconds before auto-finalizing a segment (default: 15.0s)'
    )
    mic_model_group.add_argument(
        '--model.debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # VAD arguments for microphone
    mic_vad_group = microphone_parser.add_argument_group('VAD Options')
    mic_vad_group.add_argument(
        '--vad.enabled',
        action='store_true',
        help='Enable Voice Activity Detection to filter silence'
    )
    mic_vad_group.add_argument(
        '--vad.threshold',
        type=float,
        help='VAD speech probability threshold (default: 0.5)'
    )
    mic_vad_group.add_argument(
        '--vad.min-speech-duration-ms',
        type=int,
        help='Minimum speech duration in milliseconds (default: 250ms)'
    )
    mic_vad_group.add_argument(
        '--vad.min-silence-duration-ms',
        type=int,
        help='Minimum silence duration in milliseconds (default: 100ms)'
    )
    mic_vad_group.add_argument(
        '--vad.speech-pad-ms',
        type=int,
        help='Padding added to speech segments in milliseconds (default: 30ms)'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show library information'
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Show version information'
    )
    
    args = parser.parse_args()
    # DEBUG: Log the parsed args namespace
    print(f"DEBUG cli.py: Parsed args namespace: {vars(args)}")
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    settings = initialize_logging(config_path)
    
    # Override settings with CLI arguments
    if args.command == 'transcribe':
        if getattr(args, 'model.chunk_size_ms', None) is not None:
            settings.model.chunk_size_ms = getattr(args, 'model.chunk_size_ms')
        if getattr(args, 'model.auto_finalize_after', None) is not None:
            settings.model.auto_finalize_after = getattr(args, 'model.auto_finalize_after')
        if getattr(args, 'model.debug', False):
            settings.model.debug = getattr(args, 'model.debug')
        if getattr(args, 'vad.enabled', False):
            settings.vad.enabled = getattr(args, 'vad.enabled')
        if getattr(args, 'vad.threshold', None) is not None:
            settings.vad.aggressiveness = getattr(args, 'vad.threshold')
        if getattr(args, 'vad.min_speech_duration_ms', None) is not None:
            settings.vad.min_speech_duration_ms = getattr(args, 'vad.min_speech_duration_ms')
        if getattr(args, 'vad.min_silence_duration_ms', None) is not None:
            settings.vad.min_silence_duration_ms = getattr(args, 'vad.min_silence_duration_ms')
        if getattr(args, 'vad.speech_pad_ms', None) is not None:
            settings.vad.speech_pad_ms = getattr(args, 'vad.speech_pad_ms')
        
        return transcribe_file_streaming(args.audio_file, settings)
    
    elif args.command == 'microphone':
        if getattr(args, 'model.chunk_size_ms', None) is not None:
            settings.model.chunk_size_ms = getattr(args, 'model.chunk_size_ms')
        if getattr(args, 'model.auto_finalize_after', None) is not None:
            settings.model.auto_finalize_after = getattr(args, 'model.auto_finalize_after')
        if getattr(args, 'model.debug', False):
            settings.model.debug = getattr(args, 'model.debug')
        if getattr(args, 'vad.enabled', False):
            settings.vad.enabled = getattr(args, 'vad.enabled')
        if getattr(args, 'vad.threshold', None) is not None:
            settings.vad.aggressiveness = getattr(args, 'vad.threshold')
        if getattr(args, 'vad.min_speech_duration_ms', None) is not None:
            settings.vad.min_speech_duration_ms = getattr(args, 'vad.min_speech_duration_ms')
        if getattr(args, 'vad.min_silence_duration_ms', None) is not None:
            settings.vad.min_silence_duration_ms = getattr(args, 'vad.min_silence_duration_ms')
        if getattr(args, 'vad.speech_pad_ms', None) is not None:
            settings.vad.speech_pad_ms = getattr(args, 'vad.speech_pad_ms')
        
        return transcribe_microphone_streaming(args.duration, settings)
    
    elif args.command == 'info':
        log_with_symbol(symbols['mic'], "ViStreamASR - Vietnamese Streaming ASR Library")
        log_with_symbol(symbols['book'], "Description: Simple and efficient streaming ASR for Vietnamese")
        log_with_symbol(symbols['home'], "Cache directory: ~/.cache/ViStreamASR")
        log_with_symbol(symbols['brain'], "Model: ViStreamASR (U2-based)")
        log_with_symbol(symbols['tool'], f"Optimal chunk size: {settings.model.chunk_size_ms}ms")
        log_with_symbol(symbols['clock'], f"Default auto-finalize: {settings.model.auto_finalize_after} seconds")
        
        # Check if model is cached
        try:
            from .core import get_cache_dir
        except ImportError:
            from core import get_cache_dir
            
        cache_dir = get_cache_dir()
        model_path = cache_dir / "pytorch_model.bin"
        
        if model_path.exists():
            model_size = model_path.stat().st_size / (1024 * 1024 * 1024)  # GB
            log_with_symbol(symbols['check'], f"Model status: Cached ({model_size:.1f} GB)")
        else:
            log_with_symbol(symbols['check'], "Model status: Not cached (will download on first use)")
        
        return 0
    
    elif args.command == 'version':
        from . import __version__
        print(f"ViStreamASR version {__version__}")
        return 0
    
    else:
        parser.print_help()
        return 1


def cli_main():
    """Entry point for console script."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_with_symbol(symbols['stop'], "Interrupted by user")
        sys.exit(0)
    except Exception as e:
        log_with_symbol(symbols['stop'], f"Unexpected error: {e}", "error")
        sys.exit(1)


if __name__ == '__main__':
    cli_main()