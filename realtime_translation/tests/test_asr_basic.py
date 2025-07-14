import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path to enable imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from asr.whisper_ct2 import WhisperCT2Model, create_whisper_asr
    from asr.asr_engine import ASRManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install faster-whisper librosa soundfile")
    sys.exit(1)


def test_basic_asr():
    """Test basic ASR functionality"""
    print("Testing ASR Module")
    print("=" * 50)
    
    # Check for audio file
    audio_file = current_dir / "test_audio.wav"
    if not audio_file.exists():
        # Try different possible locations
        possible_files = [
            project_root.parent / "downloaded_audio.wav",
            project_root.parent / "downloaded_audio.webm", 
            project_root.parent / "downloaded_video.wav",
            current_dir / "test_audio.webm"
        ]
        
        for pf in possible_files:
            if pf.exists():
                audio_file = pf
                break
        else:
            print(f"No test audio file found. Looked for:")
            for pf in possible_files:
                print(f"   - {pf}")
            print("\nPlease provide an audio file as 'test_audio.wav' in the tests directory")
            return False
    
    print(f"Using audio file: {audio_file}")
    
    # Test 1: Create ASR engine
    print("\nTesting ASR Engine Creation")
    try:
        asr_engine = create_whisper_asr("tiny")
        print(f"ASR Engine created: {asr_engine.model_name}")
        print(f"   Device: {asr_engine.device}")
        print(f"   Compute type: {asr_engine.compute_type}")
        print(f"   Language: {asr_engine.language}")
    except Exception as e:
        print(f"Failed to create ASR engine: {e}")
        return False
    
    # Test 2: Load model
    print("\nTesting Model Loading")
    try:
        load_start = time.time()
        asr_engine.load_model()
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Basic transcription
    print("\nTesting Basic Transcription")
    try:
        # Create ASR manager for better performance tracking
        asr_manager = ASRManager(asr_engine)
        
        transcribe_start = time.time()
        result = asr_manager.transcribe(str(audio_file))
        transcribe_time = time.time() - transcribe_start
        
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
        print(f"   Audio duration: {result.audio_duration:.2f}s")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Real-time factor: {result.real_time_factor:.2f}x")
        print(f"   Language detected: {result.language}")
        print(f"   Overall confidence: {result.get_average_confidence():.3f}")
        print(f"   Segments: {len(result.segments)}")
        
        # Display transcription
        print(f"\nFull Transcription:")
        print("-" * 40)
        full_text = result.get_full_text()
        print(f"{full_text}")
        print("-" * 40)
        
        # Show segment details
        print(f"\nSegment Details:")
        for i, segment in enumerate(result.segments, 1):
            confidence_str = f"{segment.confidence:.3f}" if segment.confidence else "N/A"
            print(f"   [{i}] {segment.start:.1f}s - {segment.end:.1f}s (conf: {confidence_str})")
            print(f"       \"{segment.text.strip()}\"")
        
    except Exception as e:
        print(f"Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Performance statistics
    print("\nTesting Performance Monitoring")
    try:
        stats = asr_manager.get_performance_stats()
        print(f"Performance Statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Total audio duration: {stats['total_audio_duration']:.2f}s")
        print(f"   Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"   Average RTF: {stats['average_rtf']:.2f}x")
        print(f"   Overall RTF: {stats['overall_rtf']:.2f}x")
    except Exception as e:
        print(f"Performance monitoring failed: {e}")
        return False
    
    # Test 5: Model info
    print("\nTesting Model Information")
    try:
        model_info = asr_manager.get_model_info()
        print(f"Model Information:")
        for key, value in model_info.items():
            if key != 'config':  # Skip config details for brevity
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"Model info retrieval failed: {e}")
        return False
    
    # Test 6: Memory cleanup
    print("\nTesting Memory Cleanup")
    try:
        asr_engine.unload_model()
        print(f"Model unloaded successfully")
    except Exception as e:
        print(f"Model unloading failed: {e}")
        return False
    
    print("\nAll tests passed successfully!")
    return True


if __name__ == "__main__":
    print("Starting ASR Module Tests")
    
    # Run basic tests
    success = test_basic_asr()
    
    if success:
        print("\n" + "=" * 50)
        print("ASR Module is working correctly!")
        print("Ready for integration with the translation pipeline.")
    else:
        print("\n" + "=" * 50) 
        print("ASR Module tests failed!")
        print("Please check the error messages above.")
        sys.exit(1) 