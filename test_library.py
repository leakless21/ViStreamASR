#!/usr/bin/env python3
"""
Test script for ViStreamASR library
This demonstrates that the core functionality works correctly.
"""

import sys
import os
import time

# Add src directory to path for imports
sys.path.insert(0, 'src')

def test_streaming_asr():
    """Test the streaming ASR functionality."""
    print("🧪 ViStreamASR Library Test")
    print("=" * 50)
    
    try:
        # Import from source
        from streaming import StreamingASR
        from core import ASREngine
        print("✅ Imports successful")
        
        # Initialize StreamingASR
        asr = StreamingASR(chunk_size_ms=640, debug=False)
        print(f"✅ StreamingASR initialized (chunk_size: {asr.chunk_size_ms}ms)")
        
        # Test with audio file
        audio_file = "resource/linh_ref_long.wav"
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        print(f"🎵 Testing with audio file: {audio_file}")
        
        # Process audio
        start_time = time.time()
        partial_count = 0
        final_count = 0
        final_segments = []
        
        print("\n📝 Processing audio...")
        for result in asr.stream_from_file(audio_file):
            if result.get('partial'):
                partial_count += 1
                if partial_count <= 3:  # Show first few partials
                    text = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
                    print(f"   Partial {partial_count}: {text}")
            
            if result.get('final'):
                final_count += 1
                final_text = result['text']
                final_segments.append(final_text)
                print(f"✅ Final {final_count}: {final_text}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Results
        print(f"\n🎯 Test Results:")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - Partial updates: {partial_count}")
        print(f"   - Final segments: {final_count}")
        print(f"   - Complete transcription:")
        
        complete_text = " ".join(final_segments)
        print(f"     {complete_text}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic library components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from streaming import StreamingASR
        from core import ASREngine
        
        # Test StreamingASR initialization
        asr = StreamingASR(chunk_size_ms=500, debug=True)
        print(f"✅ StreamingASR with custom chunk size: {asr.chunk_size_ms}ms")
        
        # Test ASREngine initialization  
        engine = ASREngine(chunk_size_ms=640, debug_mode=True)
        print(f"✅ ASREngine with debug mode")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting ViStreamASR tests...\n")
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    # Test streaming ASR
    streaming_ok = test_streaming_asr()
    
    print("\n" + "=" * 60)
    print("📊 OVERALL TEST RESULTS")
    print("=" * 60)
    print(f"Basic functionality: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Streaming ASR: {'✅ PASS' if streaming_ok else '❌ FAIL'}")
    
    if basic_ok and streaming_ok:
        print("\n🎉 All tests passed! ViStreamASR is working correctly.")
        print("\n💡 Usage example:")
        print("   sys.path.insert(0, 'src')")
        print("   from streaming import StreamingASR")
        print("   asr = StreamingASR()")
        print("   for result in asr.stream_from_file('audio.wav'):")
        print("       print(result['text'])")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 