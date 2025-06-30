#!/usr/bin/env python3
"""
Example script demonstrating ViStreamASR usage.
"""

import os
import sys

def main():
    print("🎤 ViStreamASR Example")
    print("=" * 40)
    
    try:
        from ViStreamASR import StreamingASR
        print("✅ ViStreamASR imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ViStreamASR: {e}")
        print("💡 Try running: pip install -e .")
        return 1
    
    # Check if example audio file exists
    audio_file = "resource/linh_ref_long.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Example audio file not found: {audio_file}")
        return 1
    
    print(f"📁 Using audio file: {audio_file}")
    
    # Initialize StreamingASR
    print(f"\n🔄 Initializing StreamingASR...")
    asr = StreamingASR(chunk_size_ms=640, debug=True)
    
    # Run streaming transcription
    print(f"\n🎵 Starting streaming transcription...")
    print(f"=" * 60)
    
    final_segments = []
    
    try:
        for result in asr.stream_from_file(audio_file):
            chunk_info = result.get('chunk_info', {})
            
            if result.get('partial'):
                print(f"📝 [PARTIAL {chunk_info.get('chunk_id', '?'):3d}] {result['text']}")
            
            if result.get('final'):
                final_text = result['text']
                final_segments.append(final_text)
                print(f"✅ [FINAL   {chunk_info.get('chunk_id', '?'):3d}] {final_text}")
                print(f"-" * 60)
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Show results
    print(f"\n📊 RESULTS")
    print(f"=" * 40)
    print(f"📝 Final segments: {len(final_segments)}")
    
    print(f"\n📝 Complete Transcription:")
    print(f"-" * 40)
    complete_text = " ".join(final_segments)
    print(f"{complete_text}")
    
    print(f"\n✅ Example completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 