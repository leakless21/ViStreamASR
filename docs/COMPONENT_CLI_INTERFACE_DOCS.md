# Component: CLI Interface Documentation

This document provides comprehensive documentation for the CLI Interface Component, which offers a command-line interface for ViStreamASR functionality including file transcription, real-time microphone processing, and system information.

## Overview

The CLI Interface Component serves as the primary user interface for ViStreamASR, providing:

- **File Transcription**: Process audio files with streaming ASR
- **Real-time Microphone**: Live speech recognition from audio input devices
- **System Information**: Library status, version, and configuration details
- **User-friendly Output**: Formatted transcription results with progress indicators
- **Error Handling**: Comprehensive error reporting and recovery

## Component Architecture

### Core Classes and Responsibilities

| Class/Function | Location | Primary Responsibility | Key Features |
|----------------|----------|----------------------|-------------|
| `main()` | [`src/cli.py:262`](src/cli.py:262) | Main CLI entry point and argument parsing | Command routing, help system |
| `transcribe_file_streaming()` | [`src/cli.py:52`](src/cli.py:52) | File transcription with streaming ASR | Progress tracking, result formatting |
| `transcribe_microphone_streaming()` | [`src/cli.py:147`](src/cli.py:147) | Real-time microphone transcription | Live processing, duration control |
| `cli_main()` | [`src/cli.py:418`](src/cli.py:418) | Console script entry point | Exception handling, cleanup |

## 1. Command Reference

### Available Commands

The CLI interface provides several commands for different use cases:

#### `transcribe` - Audio File Transcription

```bash
vistream-asr transcribe <audio_file> [options]
```

**Description:**
Stream process an audio file and show real-time transcription results with partial and final segments.

**Parameters:**
- `audio_file`: Path to audio file (required) - Supports WAV, MP3, FLAC, and other torchaudio-compatible formats

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--chunk-size` | 640 | Chunk size in milliseconds (100-2000ms) |
| `--auto-finalize-after` | 15.0 | Maximum duration before auto-finalizing segments (seconds) |
| `--show-debug` | False | Enable debug logging with detailed processing information |

**Examples:**
```bash
# Basic file transcription
vistream-asr transcribe audio.wav

# Custom chunk size for lower latency
vistream-asr transcribe audio.wav --chunk-size 300

# Enable debug logging
vistream-asr transcribe audio.wav --show-debug

# Custom auto-finalization timing
vistream-asr transcribe audio.wav --auto-finalize-after 20.0
```

#### `microphone` - Real-time Microphone Transcription

```bash
vistream-asr microphone [options]
```

**Description:**
Record from microphone and show real-time transcription results. Press Ctrl+C to stop.

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--duration` | None | Maximum recording duration in seconds (None for unlimited) |
| `--chunk-size` | 640 | Chunk size in milliseconds (100-2000ms) |
| `--auto-finalize-after` | 15.0 | Maximum duration before auto-finalizing segments (seconds) |
| `--show-debug` | False | Enable debug logging with detailed processing information |

**Examples:**
```bash
# Infinite microphone recording (press Ctrl+C to stop)
vistream-asr microphone

# Record for 30 seconds
vistream-asr microphone --duration 30

# Custom chunk size with debug
vistream-asr microphone --chunk-size 500 --show-debug
```

#### `info` - System Information

```bash
vistream-asr info
```

**Description:**
Display library information including version, model status, cache location, and usage examples.

**Output Information:**
- Library description and version
- Cache directory location
- Model status (cached/download size)
- GPU availability
- Configuration defaults
- Usage examples

#### `version` - Version Information

```bash
vistream-asr version
```

**Description:**
Show the current ViStreamASR version number.

### Command Line Arguments

#### Global Arguments

All commands support these global options:

| Argument | Description | Example |
|----------|-------------|---------|
| `--help`, `-h` | Show help message | `vistream-asr transcribe --help` |
| `--version` | Show version information | `vistream-asr --version` |

#### Subcommand Arguments

Each command has its own specific arguments as documented above.

## 2. Usage Examples

### Basic File Transcription

```bash
# Simple file transcription
vistream-asr transcribe speech.wav

# Output example:
🎤 ViStreamASR File Transcription
==================================================
📁 Audio file: speech.wav
📏 Chunk size: 640ms
⏰ Auto-finalize after: 15.0s
🔧 Debug mode: False

🎵 Starting streaming transcription...
==================================================
📝 [PARTIAL   1]xin chào tôi là
✅ [FINAL     1]xin chào tôi là john
--------------------------------------------------
📝 [PARTIAL   2]john đến từ hà nội
✅ [FINAL     2]john đến từ hà nội
--------------------------------------------------

📊 TRANSCRIPTION RESULTS
==================================================
⏱️  Processing time: 3.24 seconds
📝 Final segments: 2

📝 Complete Transcription:
==================================================
xin chào tôi là john john đến từ hà nội
```

### Microphone Transcription

```bash
# Real-time microphone transcription
vistream-asr microphone --duration 10

# Output example:
🎤 ViStreamASR Microphone Transcription
==================================================
🔊 Recording from microphone
📏 Chunk size: 640ms
⏰ Auto-finalize after: 15.0s
⏱️ Duration: 10 seconds
🔧 Debug mode: False

🎤 Starting microphone streaming...
🔊 Please speak into your microphone...
==================================================
📝 [PARTIAL   1]xin chào
✅ [FINAL     1]xin chào
--------------------------------------------------
📝 [PARTIAL   2]bạn có khỏe không
✅ [FINAL     2]bạn có khỏe không
--------------------------------------------------

📊 MICROPHONE TRANSCRIPTION RESULTS
==================================================
⏱️  Recording time: 10.02 seconds
📝 Final segments: 2

📝 Complete Transcription:
==================================================
xin chào bạn có khỏe không
```

### Debug Mode Usage

```bash
# Enable detailed debug information
vistream-asr transcribe audio.wav --show-debug

# Debug output includes:
# 🔄 Initializing ViStreamASR...
# 📊 [StreamingASR] Starting file stream: audio.wav
# 📏 [StreamingASR] Chunk size: 640ms
# 📁 [StreamingASR] Loading: audio.wav
# ✅ [StreamingASR] Audio prepared: 44100 samples at 16kHz
# ✅ [StreamingASR] Processing 7 chunks of 10240 samples each
# 🔧 [StreamingASR] Processing chunk 1/7 (10240 samples)
# 🔧 [ENGINE] Audio: 10240 samples | Buffer: 0 frames | is_last: False
# ✅ [CHUNK-SIZE] Perfect size: 10240 samples (640ms)
# 🟢 [SPEECH] Processing speech chunk...
# 📊 [ASR] New frames - Emission: 16, Encoder: 16
# 📝 [PARTIAL 1]xin chào
```

### System Information

```bash
# Display system information
vistream-asr info

# Output example:
🎤 ViStreamASR - Vietnamese Streaming ASR Library
==================================================
📖 Description: Simple and efficient streaming ASR for Vietnamese
🏠 Cache directory: ~/.cache/ViStreamASR
🧠 Model: ViStreamASR (U2-based)
🔧 Optimal chunk size: 640ms
⏰ Default auto-finalize: 15 seconds
🚀 GPU support: Available

Usage examples:
  vistream-asr transcribe audio.wav
  vistream-asr transcribe audio.wav --chunk-size 500
  vistream-asr transcribe audio.wav --auto-finalize-after 20
  vistream-asr transcribe audio.wav --show-debug
  vistream-asr microphone
  vistream-asr microphone --duration 30
  vistream-asr microphone --chunk-size 500
  vistream-asr microphone --show-debug

💾 Model status: Cached (2.7 GB)
```

## 3. Error Handling and Troubleshooting

### Common Error Scenarios

#### File Not Found Errors

```bash
$ vistream-asr transcribe missing_file.wav
❌ Error: Audio file not found: missing_file.wav
```

**Solution:** Verify file path exists and is accessible.

#### Microphone Device Issues

```bash
$ vistream-asr microphone
❌ Error: No microphone devices found
```

**Solution:** 
- Check microphone connections
- Verify permissions (especially on macOS/Linux)
- Install required dependencies: `pip install sounddevice`

#### Missing Dependencies

```bash
$ vistream-asr transcribe audio.wav
❌ Error: sounddevice library not installed. Install with: pip install sounddevice
```

**Solution:** Install required dependencies:
```bash
pip install sounddevice torchaudio numpy torch
```

#### Model Download Issues

```bash
$ vistream-asr transcribe audio.wav
❌ Error: Failed to download model from https://...
```

**Solution:**
- Check internet connection
- Verify Hugging Face model access
- Clear cache: `rm -rf ~/.cache/ViStreamASR/`

### Debug Mode for Troubleshooting

Enable debug mode to diagnose issues:

```bash
# File processing with debug
vistream-asr transcribe audio.wav --show-debug

# Microphone processing with debug
vistream-asr microphone --show-debug
```

**Debug Information Provided:**
- Model loading status
- Audio file processing details
- Chunk processing information
- Buffer management status
- Performance metrics (RTF, processing time)
- Error stack traces when applicable

### Performance Issues

#### High Latency

**Symptom:** Transcription results appear significantly delayed after speech.

**Solutions:**
```bash
# Reduce chunk size for lower latency
vistream-asr transcribe audio.wav --chunk-size 300

# Enable GPU acceleration (if available)
# System automatically detects and uses GPU
```

#### Memory Issues

**Symptom:** Program crashes with memory errors on large files.

**Solutions:**
```bash
# Use smaller chunk sizes to reduce memory usage
vistream-asr transcribe large_audio.wav --chunk-size 500

# Process in shorter segments for very long files
# Split file and process individually
```

#### Audio Quality Issues

**Symptom:** Poor transcription accuracy or garbled output.

**Solutions:**
```bash
# Check audio file integrity
# Ensure proper sample rate (16kHz recommended)
vistream-asr transcribe audio.wav --show-debug  # Check audio preparation
```

## 4. Command Implementation Details

### Main Entry Point

The CLI interface uses a structured argument parsing approach:

```python
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
    
    # Subcommands setup
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Configure each subcommand
    # ... (transcribe, microphone, info, version subparsers)
```

### Command Routing

The system routes commands to appropriate handlers:

```python
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
    # Display system information
    return 0
    
elif args.command == 'version':
    # Show version information
    return 0
```

### File Transcription Implementation

The file transcription function provides comprehensive feedback:

```python
def transcribe_file_streaming(audio_file, chunk_size_ms=640, auto_finalize_after=15.0, debug=False):
    """
    Transcribe an audio file using streaming ASR.
    """
    print(f"{symbols['mic']} ViStreamASR File Transcription")
    print("=" * 50)
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
```

### Microphone Transcription Implementation

The microphone transcription includes robust device detection:

```python
def transcribe_microphone_streaming(duration_seconds=None, chunk_size_ms=640, auto_finalize_after=15.0, debug=False):
    """
    Transcribe from microphone using streaming ASR.
    """
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
```

## 5. Output Formatting

### Visual Indicators

The CLI uses consistent visual symbols for different types of information:

| Symbol | Meaning | Usage |
|--------|---------|-------|
| 🎤 | Microphone/ASR | Main interface indicator |
| 📁 | File operations | Audio file paths |
| 📏 | Size/Configuration | Chunk sizes, durations |
| ⏰ | Time operations | Auto-finalize timing |
| 🔧 | Configuration | Debug mode, settings |
| ✅ | Success | Final transcriptions, completion |
| 📝 | Text content | Partial and final transcriptions |
| 📊 | Statistics | Performance metrics |
| ❌ | Errors | Error conditions |
| 🔄 | Processing | Initialization, model loading |
| 🚀 | GPU acceleration | GPU availability status |

### Output Structure

#### File Transcription Output

```
🎤 ViStreamASR File Transcription
==================================================
📁 Audio file: speech.wav
📏 Chunk size: 640ms
⏰ Auto-finalize after: 15.0s
🔧 Debug mode: False

🎵 Starting streaming transcription...
==================================================
📝 [PARTIAL   1]xin chào tôi là
✅ [FINAL     1]xin chào tôi là john
--------------------------------------------------
📝 [PARTIAL   2]john đến từ hà nội
✅ [FINAL     2]john đến từ hà nội
--------------------------------------------------

📊 TRANSCRIPTION RESULTS
==================================================
⏱️  Processing time: 3.24 seconds
📝 Final segments: 2

📝 Complete Transcription:
==================================================
xin chào tôi là john john đến từ hà nội
```

#### Microphone Transcription Output

```
🎤 ViStreamASR Microphone Transcription
==================================================
🔊 Recording from microphone
📏 Chunk size: 640ms
⏰ Auto-finalize after: 15.0s
⏱️ Duration: 10 seconds
🔧 Debug mode: False

🎤 Starting microphone streaming...
🔊 Please speak into your microphone...
==================================================
📝 [PARTIAL   1]xin chào
✅ [FINAL     1]xin chào
--------------------------------------------------
📝 [PARTIAL   2]bạn có khỏe không
✅ [FINAL     2]bạn có khỏe không
--------------------------------------------------

📊 MICROPHONE TRANSCRIPTION RESULTS
==================================================
⏱️  Recording time: 10.02 seconds
📝 Final segments: 2

📝 Complete Transcription:
==================================================
xin chào bạn có khỏe không
```

### Progress Tracking

The CLI provides detailed progress information:

- **Chunk Progress**: `[PARTIAL X/Y]` and `[FINAL X/Y]` indicators
- **Processing Statistics**: Total time, RTF, speedup factors
- **Audio Information**: Sample counts, durations, chunk sizes
- **System Status**: Model loading, device detection, GPU availability

## 6. Configuration and Customization

### Environment Variables

The CLI can be configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VISTREAMASR_DEBUG` | False | Enable debug mode globally |
| `VISTREAMASR_CHUNK_SIZE` | 640 | Default chunk size in ms |
| `VISTREAMASR_CACHE_DIR` | ~/.cache/ViStreamASR | Model cache directory |

### Configuration File Support

Create a `~/.vistreamasr.conf` file for persistent configuration:

```ini
[default]
chunk_size = 640
auto_finalize_after = 15.0
debug = false

[microphone]
default_duration = 30
chunk_size = 640

[file]
max_file_size = 100MB
supported_formats = wav,mp3,flac,m4a
```

### Custom Themes and Formatting

The CLI supports customizable output formatting through configuration:

```python
# Custom color scheme (if terminal supports colors)
color_schemes = {
    'dark': {
        'partial': '\033[94m',    # Blue
        'final': '\033[92m',      # Green  
        'error': '\033[91m',      # Red
        'reset': '\033[0m'
    },
    'light': {
        'partial': '\033[94m',    # Blue
        'final': '\033[92m',      # Green
        'error': '\033[91m',      # Red
        'reset': '\033[0m'
    }
}
```

## 7. Integration Patterns

### Shell Script Integration

The CLI can be easily integrated into shell scripts:

```bash
#!/bin/bash
# transcribe_script.sh

# Check if file exists
if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found"
    exit 1
fi

# Transcribe file and save results
result=$(vistream-asr transcribe "$1" --chunk-size 500)
echo "$result" > transcription.txt

# Extract just the transcription text
grep "Complete Transcription:" transcription.txt | tail -1 | cut -d' ' -f3- > final_text.txt
```

### Pipeline Integration

The CLI works well with Unix pipelines:

```bash
# Process multiple files
for file in *.wav; do
    echo "Processing $file..."
    vistream-asr transcribe "$file" --chunk-size 300
done

# Process audio stream from another program
arecord -f cd -c 1 -r 16000 | vistream-asr microphone --duration 30
```

### System Service Integration

Create a systemd service for continuous transcription:

```ini
# /etc/systemd/system/vistreamasr.service
[Unit]
Description=ViStreamASR Transcription Service
After=sound.target

[Service]
Type=simple
User=asr
ExecStart=/usr/local/bin/vistream-asr microphone --duration 3600
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 8. Performance Optimization

### Chunk Size Optimization

Choose appropriate chunk sizes based on use case:

| Use Case | Recommended Chunk Size | Latency | Accuracy |
|----------|----------------------|---------|----------|
| Real-time chat | 200-300ms | Very low | Good |
| Meeting transcription | 500-640ms | Low | Very good |
| High accuracy | 1000-2000ms | Medium | Excellent |
| Low bandwidth | 300-500ms | Low | Good |

### Memory Optimization

For large files or limited memory:

```bash
# Use smaller chunks to reduce memory usage
vistream-asr transcribe large_file.wav --chunk-size 400

# Process in segments for very large files
ffmpeg -i large_file.wav -f segment -segment_time 30 -c copy chunk_%03d.wav
for chunk in chunk_*.wav; do
    vistream-asr transcribe "$chunk"
done
```

### GPU Acceleration

The CLI automatically detects and uses GPU acceleration:

```bash
# Check GPU availability
vistream-asr info

# GPU will be used automatically if available
# No additional configuration required
```

## 9. Advanced Usage

### Batch Processing

Process multiple files with a script:

```bash
#!/bin/bash
# batch_transcribe.sh

for file in "$@"; do
    echo "Processing: $file"
    output_file="${file%.*}.txt"
    vistream-asr transcribe "$file" > "$output_file"
    echo "Results saved to: $output_file"
done
```

### Quality Assessment

Evaluate transcription quality:

```bash
# Compare with reference text
reference="reference.txt"
transcription="output.txt"

# Calculate word error rate (requires wer-calc tool)
wer-calc "$reference" "$transcription"

# Or use simple word count comparison
ref_words=$(wc -w < "$reference")
trans_words=$(wc -w < "$transcription")
echo "Reference words: $ref_words"
echo "Transcription words: $trans_words"
```

### Custom Output Formats

Extract specific information from CLI output:

```bash
# Extract just the final transcription
vistream-asr transcribe audio.wav | \
    grep "Complete Transcription:" | \
    tail -1 | \
    cut -d' ' -f3- > clean_transcription.txt

# Extract processing statistics
vistream-asr transcribe audio.wav | \
    grep "Processing time:" | \
    cut -d' ' -f3 | \
    tr -d 's' > processing_time.txt
```

## 10. Summary

The CLI Interface Component provides a comprehensive, user-friendly command-line interface for ViStreamASR with:

- **Multiple Commands**: File transcription, microphone recording, system information
- **Rich Output**: Formatted results with progress indicators and visual feedback
- **Error Handling**: Comprehensive error reporting and troubleshooting guidance
- **Performance Optimization**: Configurable parameters for different use cases
- **Integration Ready**: Designed for shell scripts, pipelines, and system services
- **User-friendly**: Clear help system, examples, and intuitive parameter names

The component successfully abstracts the complexity of the underlying streaming ASR system while providing powerful command-line tools suitable for both casual users and automation scripts.

## Related Files

- **[`src/cli.py`](src/cli.py)**: Main CLI implementation
- **[`src/streaming.py`](src/streaming.py)**: Streaming interface used by CLI
- **[`src/core.py`](src/core.py)**: Core ASR engine powering the CLI

## Dependencies

- **argparse**: Command-line argument parsing
- **sounddevice**: Microphone input for real-time processing
- **torchaudio**: Audio file loading and processing
- **torch**: Underlying deep learning framework
- **numpy**: Numerical computations