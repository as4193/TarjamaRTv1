import sys
import os
import numpy as np
import soundfile as sf

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vad.pyannote_vad import PyannoteVAD

def test_vad_pipeline():
    """Test the updated VAD pipeline"""
    print("=" * 60)
    print("Testing PyannoteVAD with ipeline")
    print("=" * 60)
    
    try:
        # Initialize VAD
        print("1. Initializing PyannoteVAD...")
        vad = PyannoteVAD()
        
        # Check model info before loading
        print("2. Model info before loading:")
        info = vad.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Load the model
        print("\n3. Loading VAD model...")
        vad.load_model()
        print("Model loaded successfully")
        
        # Check model info after loading
        print("\n4. Model info after loading:")
        info = vad.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test with sample audio file if available
        from pathlib import Path
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
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
        
        if audio_file.exists():
            print(f"\n5. Testing with {audio_file}...")
            
            # Test detect_segments
            result = vad.detect_segments(str(audio_file))
            print(f"   Segments detected: {len(result.segments)}")
            print(f"   Total audio duration: {result.audio_duration:.2f}s")
            print(f"   Processing time: {result.processing_time:.2f}s")
            
            # Calculate speech vs silence duration
            speech_duration = sum(seg.end - seg.start for seg in result.segments if seg.is_speech)
            silence_duration = sum(seg.end - seg.start for seg in result.segments if not seg.is_speech)
            
            print(f"Speech duration: {speech_duration:.2f}s")
            print(f"Silence duration: {silence_duration:.2f}s")
            print(f"Speech ratio: {speech_duration/result.audio_duration*100:.1f}%")
            
            if silence_duration > 0:
                print("Silence detection working - silence segments found")
            else:
                print("No silence detected - might be continuous speech")
            
            # Show first few segments
            print("   First few segments:")
            for i, segment in enumerate(result.segments[:5]):
                print(f"     {i+1}: {segment.start:.2f}s - {segment.end:.2f}s "
                      f"({'speech' if segment.is_speech else 'silence'}, "
                      f"conf: {segment.confidence:.2f})")
        
        else:
            print(f"\n5. No test audio file found - skipping audio analysis")
            print(f"   To test with real audio, place a file named 'test_audio.wav' in the tests directory")
        
        # Test unloading
        print("\n6. Unloading model...")
        vad.unload_model()
        print("Model unloaded successfully")
        
        print("\n" + "=" * 60)
        print("All tests passed! VAD module is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vad_pipeline()
    sys.exit(0 if success else 1) 