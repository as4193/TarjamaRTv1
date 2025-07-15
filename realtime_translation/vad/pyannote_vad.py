import time
import numpy as np
import torch
from typing import List, Dict, Optional, Union, Iterator
from .vad_engine import VADEngine, VADResult, VADSegment


class PyannoteVAD(VADEngine):
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = "pyannote"
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self) -> None:
        """Load the Pyannote VAD model"""
        try:
            print(f"Loading Pyannote VAD model...")
            print(f"   Device: {self.device}")
            
            start_time = time.time()
            
            # Import pyannote
            try:
                from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError(
                    "Pyannote not installed. Install with: "
                    "pip install pyannote.audio"
                )
            
            # Load pre-trained VAD pipeline
            print(f"   Loading pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=""
            )
            
            # Move to GPU if available
            if self.device == "cuda":
                print(f"Using GPU acceleration")
                self.pipeline.to(torch.device("cuda"))
            
            print(f"Pre-trained VAD pipeline loaded successfully")
            
            load_time = time.time() - start_time
            print(f"Pyannote VAD model loaded successfully in {load_time:.2f} seconds")
            
            self.is_loaded = True
            
        except Exception as e:
            print(f"Failed to load Pyannote VAD: {e}")
            self.is_loaded = False
            raise
    
    def predict(self, audio: Union[np.ndarray, str]) -> Union[float, np.ndarray]:
        """
        Predict voice activity for audio
        Returns confidence score or array of scores
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Handle file input
            if isinstance(audio, str):
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio)
            else:
                audio_data = audio
                sample_rate = self.sample_rate
            
            # Convert to tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).float()
            else:
                audio_tensor = audio_data.float()
            
            # Ensure correct shape (channels, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Create temporary file for pyannote (it expects file input)
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_tensor.squeeze().numpy(), sample_rate)
                
                # Run VAD - Pyannote will auto-instantiate if needed
                vad_result = self.pipeline(tmp_file.name)
                
                # Calculate overall voice activity ratio
                total_duration = len(audio_tensor.squeeze()) / sample_rate
                voice_duration = sum(segment.end - segment.start for segment in vad_result.itersegments())
                
                confidence = voice_duration / total_duration if total_duration > 0 else 0.0
                
            # Clean up
            import os
            os.unlink(tmp_file.name)
            
            return confidence
            
        except Exception as e:
            print(f"Pyannote VAD prediction error: {e}")
            return 0.0
    
    def detect_segments(self, audio: Union[np.ndarray, str], **kwargs) -> VADResult:
        """Detect speech segments using Pyannote VAD"""

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Handle file input
            if isinstance(audio, str):
                audio_file = audio
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file)
            else:
                audio_data = audio
                sample_rate = self.sample_rate
                
                # Create temporary file
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    audio_file = tmp_file.name
            
            audio_duration = len(audio_data) / sample_rate
            
            # Run Pyannote VAD - will auto-instantiate if needed
            vad_output = self.pipeline(audio_file)
            
            # Convert to our segment format
            segments = []
            
            for segment in vad_output.itersegments():
                vad_segment = VADSegment(
                    start=segment.start,
                    end=segment.end,
                    confidence=0.9,  # Pyannote doesn't provide confidence scores
                    is_speech=True
                )
                segments.append(vad_segment)
            
            # Add silence segments between speech
            if segments:
                all_segments = []
                
                # Add initial silence if needed
                if segments[0].start > 0:
                    all_segments.append(VADSegment(
                        start=0.0,
                        end=segments[0].start,
                        confidence=0.1,
                        is_speech=False
                    ))
                
                # Add speech segments and silence between them
                for i, segment in enumerate(segments):
                    all_segments.append(segment)
                    
                    # Add silence after speech (if not last segment)
                    if i < len(segments) - 1:
                        silence_start = segment.end
                        silence_end = segments[i + 1].start
                        
                        if silence_end > silence_start:
                            all_segments.append(VADSegment(
                                start=silence_start,
                                end=silence_end,
                                confidence=0.1,
                                is_speech=False
                            ))
                
                # Add final silence if needed
                if segments[-1].end < audio_duration:
                    all_segments.append(VADSegment(
                        start=segments[-1].end,
                        end=audio_duration,
                        confidence=0.1,
                        is_speech=False
                    ))
                
                segments = all_segments
            else:
                # No speech detected - entire audio is silence
                segments = [VADSegment(
                    start=0.0,
                    end=audio_duration,
                    confidence=0.9,
                    is_speech=False
                )]
            
            processing_time = time.time() - start_time
            
            # Clean up temporary file if created
            if isinstance(audio, np.ndarray):
                import os
                os.unlink(audio_file)
            
            return VADResult(
                segments=segments,
                audio_duration=audio_duration,
                processing_time=processing_time,
                model_name=self.model_name,
                threshold=0.5
            )
            
        except Exception as e:
            print(f"Pyannote VAD segment detection error: {e}")
            # Return empty result
            return VADResult(
                segments=[],
                audio_duration=0.0,
                processing_time=time.time() - start_time,
                model_name=self.model_name,
                threshold=0.5
            )
    
    def detect_streaming(self, audio_chunks: Iterator[np.ndarray], **kwargs) -> Iterator[VADResult]:
        """Process streaming audio chunks"""
        chunk_index = 0
        
        for chunk in audio_chunks:
            try:
                result = self.detect_segments(chunk, **kwargs)
                
                # Adjust timing for streaming context
                chunk_duration = len(chunk) / self.sample_rate
                offset = chunk_index * chunk_duration
                
                # Offset all segment times
                for segment in result.segments:
                    segment.start += offset
                    segment.end += offset
                
                yield result
                
            except Exception as e:
                print(f"Streaming error on chunk {chunk_index}: {e}")
                # Yield empty result
                yield VADResult(
                    segments=[],
                    audio_duration=len(chunk) / self.sample_rate,
                    processing_time=0.0,
                    model_name=self.model_name,
                    threshold=0.5
                )
            
            chunk_index += 1
    
    def unload_model(self) -> None:
        """Unload the model"""
        self.pipeline = None
        self.is_loaded = False
        print("Pyannote VAD model unloaded successfully")
    
    def get_model_info(self) -> Dict:
        """Return model information"""
        return {
            'name': self.model_name,
            'type': 'pyannote_vad',
            'description': 'Pyannote voice activity detection with GPU support',
            'device': self.device,
            'is_loaded': self.is_loaded,
            'pipeline_model': 'pyannote/voice-activity-detection',
            'pretrained': True
        }


def create_pyannote_vad(config=None) -> PyannoteVAD:
    """Factory function to create Pyannote VAD instance"""
    return PyannoteVAD(config) 