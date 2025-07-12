#!/usr/bin/env python3
"""
Example script demonstrating ViStreamASR usage.
This script tests all features using the local codebase before building.
"""

import os
import sys
import argparse

# Add src directory to path to use local codebase instead of installed library
sys.path.insert(0, 'src')

def test_file_streaming(audio_file, chunk_size=640, debug=True):
    """Test file streaming functionality."""
    print(f"🎵 Testing File Streaming")
    print(f"=" * 50)
    
    try:
        from streaming import StreamingASR
        print("✅ Local StreamingASR imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import local StreamingASR: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    print(f"📁 Using audio file: {audio_file}")
    
    # Initialize StreamingASR
    print(f"\n🔄 Initializing StreamingASR...")
    asr = StreamingASR(chunk_size_ms=chunk_size, debug=debug)
    
    # Run streaming transcription
    print(f"\n🎵 Starting file streaming transcription...")
    print(f"=" * 60)
    
    final_segments = []
    partial_count = 0
    
    try:
        for result in asr.stream_from_file(audio_file):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial'):
                partial_count += 1
                text = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
                print(f"📝 [PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {text}")
            
            if result.get('final'):
                final_text = result['text']
                final_segments.append(final_text)
                print(f"✅ [FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}")
                print(f"-" * 60)
    
    except KeyboardInterrupt:
        print(f"\n⏹️  File streaming interrupted by user")
        return True
    except Exception as e:
        print(f"\n❌ Error during file streaming: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Show results
    print(f"\n📊 FILE STREAMING RESULTS")
    print(f"=" * 40)
    print(f"📝 Partial updates: {partial_count}")
    print(f"📝 Final segments: {len(final_segments)}")
    
    if final_segments:
        print(f"\n📝 Complete Transcription:")
        print(f"-" * 40)
        complete_text = " ".join(final_segments)
        print(f"{complete_text}")
    
    print(f"\n✅ File streaming test completed successfully!")
    return True

def test_microphone_streaming(duration=10, chunk_size=640, debug=True):
    """Test microphone streaming functionality."""
    print(f"\n🎤 Testing Microphone Streaming")
    print(f"=" * 50)
    
    try:
        from streaming import StreamingASR
        print("✅ Local StreamingASR imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import local StreamingASR: {e}")
        return False
    
    try:
        import sounddevice as sd
        print("✅ sounddevice library available")
        
        # Test if microphone is available
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            print("❌ No input devices (microphones) found")
            return False
        print(f"🎤 Found {len(input_devices)} input device(s)")
        
    except ImportError:
        print("❌ sounddevice library not installed. Install with: pip install sounddevice")
        return False
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")
        return False
    
    # Initialize StreamingASR
    print(f"\n🔄 Initializing StreamingASR for microphone...")
    asr = StreamingASR(chunk_size_ms=chunk_size, debug=debug)
    
    # Run microphone streaming
    print(f"\n🎤 Starting microphone streaming for {duration} seconds...")
    print(f"🔊 Please speak into your microphone...")
    print(f"=" * 60)
    
    final_segments = []
    partial_count = 0
    
    try:
        for result in asr.stream_from_microphone(duration_seconds=duration):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial'):
                partial_count += 1
                text = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
                print(f"🎙️  [PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {text}")
            
            if result.get('final'):
                final_text = result['text']
                final_segments.append(final_text)
                print(f"✅ [FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}")
                print(f"-" * 60)
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Microphone streaming interrupted by user")
        return True
    except Exception as e:
        print(f"\n❌ Error during microphone streaming: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Show results
    print(f"\n📊 MICROPHONE STREAMING RESULTS")
    print(f"=" * 40)
    print(f"📝 Partial updates: {partial_count}")
    print(f"📝 Final segments: {len(final_segments)}")
    
    if final_segments:
        print(f"\n📝 Transcribed from microphone:")
        print(f"-" * 40)
        complete_text = " ".join(final_segments)
        print(f"{complete_text}")
    else:
        print(f"ℹ️  No speech detected or transcribed during recording")
    
    print(f"\n✅ Microphone streaming test completed successfully!")
    return True

def test_asr_engine():
    """Test low-level ASREngine functionality."""
    print(f"\n🔧 Testing ASREngine (Low-level API)")
    print(f"=" * 50)
    
    try:
        from core import ASREngine
        print("✅ Local ASREngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import local ASREngine: {e}")
        return False
    
    try:
        # Test ASREngine initialization
        engine = ASREngine(chunk_size_ms=640, debug_mode=True)
        print("✅ ASREngine initialized successfully")
        
        # Test model initialization
        print("🔄 Initializing models (this may take a moment)...")
        engine.initialize_models()
        print("✅ Models initialized successfully")
        
        # Test RTF calculation
        rtf = engine.get_asr_rtf()
        print(f"📊 Current RTF: {rtf:.3f}x")
        
        # Test state reset
        engine.reset_state()
        print("✅ State reset successful")
        
        print(f"\n✅ ASREngine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during ASREngine testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="ViStreamASR Feature Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example.py                                    # Test all features
  python example.py --file-only                       # Test only file streaming
  python example.py --mic-only                        # Test only microphone streaming
  python example.py --mic-duration 15                 # Test microphone for 15 seconds
  python example.py --show-debug                        # Enable debug output
        """
    )
    
    parser.add_argument('--file-only', action='store_true',
                       help='Test only file streaming functionality')
    parser.add_argument('--mic-only', action='store_true',
                       help='Test only microphone streaming functionality')
    parser.add_argument('--engine-only', action='store_true',
                       help='Test only ASREngine low-level functionality')
    parser.add_argument('--audio-file', default="resource/linh_ref_long.wav",
                       help='Audio file for testing (default: resource/linh_ref_long.wav)')
    parser.add_argument('--mic-duration', type=int, default=10,
                       help='Duration for microphone test in seconds (default: 10)')
    parser.add_argument('--chunk-size', type=int, default=640,
                       help='Chunk size in milliseconds (default: 640)')
    parser.add_argument('--show-debug', action='store_true', default=False,
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    print("🎤 ViStreamASR Feature Test")
    print("=" * 50)
    print("🧪 Testing local codebase (src/) instead of installed library")
    print()
    
    debug_mode = args.show_debug
    test_results = []
    
    # Test file streaming
    if not args.mic_only and not args.engine_only:
        file_result = test_file_streaming(args.audio_file, args.chunk_size, debug_mode)
        test_results.append(("File Streaming", file_result))
    
    # Test microphone streaming
    if not args.file_only and not args.engine_only:
        mic_result = test_microphone_streaming(args.mic_duration, args.chunk_size, debug_mode)
        test_results.append(("Microphone Streaming", mic_result))
    
    # Test ASREngine
    if not args.file_only and not args.mic_only:
        engine_result = test_asr_engine()
        test_results.append(("ASREngine", engine_result))
    
    # Summary
    print(f"\n🏁 TEST SUMMARY")
    print(f"=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print(f"🎉 All tests passed! The codebase is ready for building.")
        return 0
    else:
        print(f"⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 