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
except ImportError:
    from streaming import StreamingASR


def transcribe_file_streaming(audio_file, chunk_size_ms=640, auto_finalize_after=15.0, debug=False):
    """
    Transcribe an audio file using streaming ASR.
    
    Args:
        audio_file: Path to audio file
        chunk_size_ms: Chunk size in milliseconds  
        auto_finalize_after: Maximum duration before auto-finalization (seconds)
        debug: Enable debug logging
    """
    print(f"{symbols['mic']} ViStreamASR File Transcription")
    print(f"=" * 50)
    print(f"{symbols['folder']} Audio file: {audio_file}")
    print(f"{symbols['ruler']} Chunk size: {chunk_size_ms}ms")
    print(f"{symbols['clock']} Auto-finalize after: {auto_finalize_after}s")
    print(f"{symbols['tool']} Debug mode: {debug}")
    print()
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        return 1
    
    # Initialize StreamingASR
    print(f"🔄 Initializing ViStreamASR...")
    asr = StreamingASR(
        chunk_size_ms=chunk_size_ms, 
        auto_finalize_after=auto_finalize_after,
        debug=debug
    )
    
    # Collect results
    final_segments = []
    current_partial = ""
    
    # Start streaming
    print(f"\n🎵 Starting streaming transcription...")
    print(f"=" * 60)
    
    start_time = time.time()
    
    try:
        for result in asr.stream_from_file(audio_file, chunk_size_ms=chunk_size_ms):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial') and result.get('text'):
                current_partial = result['text']
                print(f"{symbols['memo']} [PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {current_partial}")
            
            if result.get('final') and result.get('text'):
                final_text = result['text']
                final_segments.append(final_text)
                current_partial = ""
                print(f"{symbols['check']} [FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}")
                print(f"-" * 60)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        return 1
    
    # Final results
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📊 TRANSCRIPTION RESULTS")
    print(f"=" * 50)
    print(f"{symbols['stopwatch']}  Processing time: {total_time:.2f} seconds")
    print(f"{symbols['memo']} Final segments: {len(final_segments)}")
    
    print(f"\n{symbols['memo']} Complete Transcription:")
    print(f"=" * 60)
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
    
    print(f"\n{symbols['check']} Transcription completed successfully!")
    return 0


def transcribe_microphone_streaming(duration_seconds=None, chunk_size_ms=640, auto_finalize_after=15.0, debug=False):
    """
    Transcribe from microphone using streaming ASR.
    
    Args:
        duration_seconds: Maximum duration to record (None for infinite)
        chunk_size_ms: Chunk size in milliseconds  
        auto_finalize_after: Maximum duration before auto-finalization (seconds)
        debug: Enable debug logging
    """
    print(f"{symbols['mic']} ViStreamASR Microphone Transcription")
    print(f"=" * 50)
    print(f"{symbols['speaker']} Recording from microphone")
    print(f"{symbols['ruler']} Chunk size: {chunk_size_ms}ms")
    print(f"{symbols['clock']} Auto-finalize after: {auto_finalize_after}s")
    if duration_seconds:
        print(f"{symbols['stopwatch']} Duration: {duration_seconds}s")
    else:
        print(f"{symbols['stopwatch']} Duration: Unlimited (Press Ctrl+C to stop)")
    print(f"{symbols['tool']} Debug mode: {debug}")
    print()
    
    # Check if microphone is available
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            print(f"❌ Error: No microphone devices found")
            return 1
        print(f"{symbols['check']} Found {len(input_devices)} microphone device(s)")
    except ImportError:
        print(f"❌ Error: sounddevice library not installed. Install with: pip install sounddevice")
        return 1
    except Exception as e:
        print(f"❌ Error checking microphone: {e}")
        return 1
    
    # Initialize StreamingASR
    print(f"🔄 Initializing ViStreamASR...")
    asr = StreamingASR(
        chunk_size_ms=chunk_size_ms, 
        auto_finalize_after=auto_finalize_after,
        debug=debug
    )
    
    # Collect results
    final_segments = []
    current_partial = ""
    
    # Start streaming
    print(f"\n🎤 Starting microphone streaming...")
    print(f"{symbols['speaker']} Please speak into your microphone...")
    print(f"=" * 60)
    
    start_time = time.time()
    
    try:
        for result in asr.stream_from_microphone(duration_seconds=duration_seconds):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial') and result.get('text'):
                current_partial = result['text']
                print(f"{symbols['memo']} [PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {current_partial}")
            
            if result.get('final') and result.get('text'):
                final_text = result['text']
                final_segments.append(final_text)
                current_partial = ""
                print(f"{symbols['check']} [FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}")
                print(f"-" * 60)
        
    except KeyboardInterrupt:
        print(f"\n{symbols['stop']} Microphone streaming stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during microphone streaming: {e}")
        return 1
    
    # Final results
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📊 MICROPHONE TRANSCRIPTION RESULTS")
    print(f"=" * 50)
    print(f"{symbols['stopwatch']}  Recording time: {total_time:.2f} seconds")
    print(f"{symbols['memo']} Final segments: {len(final_segments)}")
    
    if final_segments:
        print(f"\n{symbols['memo']} Complete Transcription:")
        print(f"=" * 60)
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
        print(f"\n{symbols['memo']} No speech detected or transcribed during recording")
    
    print(f"\n{symbols['check']} Microphone transcription completed successfully!")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ViStreamASR - Vietnamese Streaming ASR Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s transcribe audio.wav                                    # Basic file transcription
  %(prog)s transcribe audio.wav --chunk-size 640                  # Use 640ms chunks
  %(prog)s transcribe audio.wav --show-debug                      # Enable debug logging
  
  %(prog)s microphone                                              # Record from microphone indefinitely
  %(prog)s microphone --duration 30                               # Record for 30 seconds
  %(prog)s microphone --chunk-size 500 --show-debug               # Custom settings with debug
        """
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
    transcribe_parser.add_argument(
        '--chunk-size',
        type=int,
        default=640,
        help='Chunk size in milliseconds (default: 640ms for optimal performance)'
    )
    transcribe_parser.add_argument(
        '--auto-finalize-after',
        type=float,
        default=15.0,
        help='Maximum duration in seconds before auto-finalizing a segment (default: 15.0s)'
    )
    transcribe_parser.add_argument(
        '--show-debug',
        action='store_true',
        help='Enable debug logging (show detailed processing information)'
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
    microphone_parser.add_argument(
        '--chunk-size',
        type=int,
        default=640,
        help='Chunk size in milliseconds (default: 640ms for optimal performance)'
    )
    microphone_parser.add_argument(
        '--auto-finalize-after',
        type=float,
        default=15.0,
        help='Maximum duration in seconds before auto-finalizing a segment (default: 15.0s)'
    )
    microphone_parser.add_argument(
        '--show-debug',
        action='store_true',
        help='Enable debug logging (show detailed processing information)'
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
    
    if args.command == 'transcribe':
        return transcribe_file_streaming(
            args.audio_file, 
            chunk_size_ms=args.chunk_size,
            auto_finalize_after=args.auto_finalize_after,
            debug=args.show_debug
        )
    
    elif args.command == 'microphone':
        return transcribe_microphone_streaming(
            duration_seconds=args.duration,
            chunk_size_ms=args.chunk_size,
            auto_finalize_after=args.auto_finalize_after,
            debug=args.show_debug
        )
    
    elif args.command == 'info':
        print(f"{symbols['mic']} ViStreamASR - Vietnamese Streaming ASR Library")
        print(f"=" * 50)
        print(f"{symbols['book']} Description: Simple and efficient streaming ASR for Vietnamese")
        print(f"{symbols['home']} Cache directory: ~/.cache/ViStreamASR")
        print(f"{symbols['brain']} Model: ViStreamASR (U2-based)")
        print(f"{symbols['tool']} Optimal chunk size: 640ms")
        print(f"{symbols['clock']} Default auto-finalize: 15 seconds")
        print(f"{symbols['rocket']} GPU support: {'Available' if StreamingASR(debug=False)._ensure_engine_initialized() or True else 'Not available'}")
        
        print(f"\nUsage examples:")
        print(f"  vistream-asr transcribe audio.wav")
        print(f"  vistream-asr transcribe audio.wav --chunk-size 500")
        print(f"  vistream-asr transcribe audio.wav --auto-finalize-after 20")
        print(f"  vistream-asr transcribe audio.wav --show-debug")
        print(f"  vistream-asr microphone")
        print(f"  vistream-asr microphone --duration 30")
        print(f"  vistream-asr microphone --chunk-size 500")
        print(f"  vistream-asr microphone --show-debug")
        
        # Check if model is cached
        try:
            from .core import get_cache_dir
        except ImportError:
            from core import get_cache_dir
            
        cache_dir = get_cache_dir()
        model_path = cache_dir / "pytorch_model.bin"
        
        if model_path.exists():
            model_size = model_path.stat().st_size / (1024 * 1024 * 1024)  # GB
            print(f"💾 Model status: Cached ({model_size:.1f} GB)")
        else:
            print(f"💾 Model status: Not cached (will download on first use)")
        
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
        print(f"\n⏹️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli_main() 