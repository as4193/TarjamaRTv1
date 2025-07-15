import time
import numpy as np
import soundfile as sf
import scipy.signal
from typing import Union, Iterator, Optional, List
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("faster-whisper is required. Install with: pip install faster-whisper")

from .asr_engine import ASREngine, ASRResult, ASRSegment, calculate_audio_duration

def load_audio_soundfile(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio file using soundfile and resample if needed.
    Replacement for librosa.load() to avoid LLVM errors.
    """
    # Load audio file
    audio_data, original_sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if original_sr != target_sr:
        # Calculate resampling ratio
        resample_ratio = target_sr / original_sr
        
        # Resample using scipy
        audio_data = scipy.signal.resample(
            audio_data, 
            int(len(audio_data) * resample_ratio)
        ).astype(np.float32)
    
    return audio_data, target_sr

# Handle config import with fallback
try:
    from ..config import get_config
except ImportError:
    # Fallback for when running as standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from config import get_config
    except ImportError:
        # Create a mock config for testing
        def get_config():
            return type('Config', (), {
                'system': type('SystemConfig', (), {
                    'device': 'auto',
                    'max_workers': 4
                })(),
                'asr': type('ASRConfig', (), {
                    'model': 'tiny',
                    'language': 'arabic',  # Use full name
                    'compute_type': 'auto',
                    'beam_size': 1,
                    'best_of': 1,
                    'temperature': 0.0,
                    'condition_on_previous_text': True,
                    'chunk_length': 30,
                    'stride_length': 5
                })()
            })()


class WhisperCT2Model(ASREngine):
    """Faster-whisper implementation for Arabic ASR with streaming support"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.device = self._get_device()
        self.compute_type = self._get_compute_type()
        
    def _get_device(self) -> str:
        """Determine the best device for model execution"""
        try:
            system_config = get_config().system
            if system_config.device == "auto":
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            return system_config.device
        except:
            # Fallback
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except:
                return "cpu"
    
    def _get_compute_type(self) -> str:
        """Determine the best compute type based on device and config"""
        if self.config.compute_type != "auto":
            return self.config.compute_type
        
        if self.device == "cuda":
            return "float16"  # Faster on GPU
        else:
            return "int8"     # More efficient on CPU
    
    def load_model(self) -> None:
        """Load the faster-whisper model"""
        if self.is_loaded:
            return
        
        print(f"Loading Whisper model: {self.model_name}")
        print(f"   Device: {self.device}, Compute type: {self.compute_type}")
        
        start_time = time.time()
        
        try:
            # Get max workers with fallback
            try:
                max_workers = get_config().system.max_workers
            except:
                max_workers = 4
                
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=max_workers
            )
            
            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            self.is_loaded = True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def transcribe(self, audio: Union[np.ndarray, str], **kwargs) -> ASRResult:
        """
        Transcribe audio file or numpy array
        
        Args:
            audio: Audio data (numpy array) or file path (string)
            **kwargs: Additional transcription parameters
            
        Returns:
            ASRResult with transcription and metadata
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Handle different input types
        if isinstance(audio, str):
            # Load audio file
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio}")
            
            # Load with soundfile (avoiding librosa LLVM errors)
            audio_data, sr = load_audio_soundfile(audio, target_sr=16000)
            audio_duration = len(audio_data) / 16000
        elif isinstance(audio, np.ndarray):
            audio_data = audio
            audio_duration = calculate_audio_duration(audio_data, 16000)
        else:
            raise ValueError("Audio input must be numpy array or file path")
        
        # Map language to ISO code for Whisper compatibility
        whisper_language = None
        if self.language and self.language != 'auto':
            whisper_language = map_language_to_iso(self.language)
            print(f"Language mapping: '{self.language}' -> '{whisper_language}' for Whisper")
        
        # Transcription parameters - ACCURACY OPTIMIZED
        transcribe_params = {
            'beam_size': self.config.beam_size,
            'best_of': self.config.best_of,
            'temperature': self.config.temperature,
            'condition_on_previous_text': self.config.condition_on_previous_text,
            'language': whisper_language,  # Use mapped ISO code
            'word_timestamps': True,  # Enable word-level timestamps
            'vad_filter': getattr(self.config, 'vad_filter', True),  # Enable VAD filtering
            'repetition_penalty': getattr(self.config, 'repetition_penalty', 1.1),  # Reduce repetitions
            'compression_ratio_threshold': 2.0,  # More strict to avoid repetition
            'log_prob_threshold': -0.8,  # More strict for quality
            'no_speech_threshold': 0.5,  # More strict speech detection
            **kwargs
        }
        
        try:
            # Perform transcription
            segments, info = self.model.transcribe(audio_data, **transcribe_params)
            
            # Convert to ASRSegments
            asr_segments = []
            for segment in segments:
                # Map detected language back to full name
                detected_lang = info.language if hasattr(info, 'language') else whisper_language
                full_lang = map_language_to_full(detected_lang) if detected_lang else self.language
                
                asr_segment = ASRSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=getattr(segment, 'avg_logprob', None),
                    language=full_lang
                )
                
                # Add word-level information if available
                if hasattr(segment, 'words') and segment.words:
                    asr_segment.tokens = [word.word for word in segment.words]
                    asr_segment.token_timestamps = [word.start for word in segment.words]
                
                asr_segments.append(asr_segment)
            
            processing_time = time.time() - start_time
            
            # Map result language back to full name
            result_detected_lang = info.language if hasattr(info, 'language') else whisper_language
            result_full_lang = map_language_to_full(result_detected_lang) if result_detected_lang else self.language
            
            # Create result
            result = ASRResult(
                segments=asr_segments,
                language=result_full_lang,
                confidence=info.language_probability if hasattr(info, 'language_probability') else None,
                audio_duration=audio_duration,
                processing_time=processing_time,
                real_time_factor=processing_time / audio_duration if audio_duration > 0 else None,
                model_name=self.model_name
            )
            
            return result
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            raise
    
    def transcribe_streaming(self, audio_chunks: Iterator[np.ndarray], **kwargs) -> Iterator[ASRResult]:
        """
        Transcribe audio in streaming mode
        
        Args:
            audio_chunks: Iterator of audio chunks (16kHz numpy arrays)
            **kwargs: Additional transcription parameters
            
        Yields:
            ASRResult for each processed chunk
        """
        if not self.is_loaded:
            self.load_model()
        
        # Streaming buffer for overlapping chunks
        buffer = np.array([], dtype=np.float32)
        chunk_index = 0
        
        # Streaming parameters
        chunk_length = self.config.chunk_length  # seconds
        stride_length = self.config.stride_length  # seconds
        sample_rate = 16000
        
        chunk_samples = int(chunk_length * sample_rate)
        stride_samples = int(stride_length * sample_rate)
        
        for audio_chunk in audio_chunks:
            chunk_index += 1
            
            # Add to buffer
            buffer = np.concatenate([buffer, audio_chunk])
            
            # Process if we have enough samples
            if len(buffer) >= chunk_samples:
                # Extract chunk for processing
                process_chunk = buffer[:chunk_samples]
                
                # Transcribe chunk
                try:
                    result = self.transcribe(process_chunk, **kwargs)
                    
                    # Adjust timestamps for streaming context
                    base_time = (chunk_index - 1) * stride_length
                    for segment in result.segments:
                        segment.start += base_time
                        segment.end += base_time
                    
                    yield result
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_index}: {e}")
                    # Yield empty result to maintain stream
                    yield ASRResult(
                        segments=[],
                        model_name=self.model_name,
                        audio_duration=calculate_audio_duration(process_chunk),
                        processing_time=0.0
                    )
                
                # Update buffer (keep overlap for context)
                if len(buffer) > stride_samples:
                    buffer = buffer[stride_samples:]
                else:
                    buffer = np.array([], dtype=np.float32)
    
    def transcribe_file_chunked(self, audio_file: str, chunk_duration: float = 30.0) -> Iterator[ASRResult]:
        """
        Transcribe a long audio file in chunks
        
        Args:
            audio_file: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            
        Yields:
            ASRResult for each chunk
        """
        # Load audio file
        audio_data, sr = load_audio_soundfile(audio_file, target_sr=16000)
        total_duration = len(audio_data) / 16000
        
        chunk_samples = int(chunk_duration * 16000)
        
        print(f"Processing {audio_file} ({total_duration:.1f}s) in {chunk_duration}s chunks")
        
        for start_idx in range(0, len(audio_data), chunk_samples):
            end_idx = min(start_idx + chunk_samples, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            start_time = start_idx / 16000
            
            # Transcribe chunk
            result = self.transcribe(chunk)
            
            # Adjust timestamps
            for segment in result.segments:
                segment.start += start_time
                segment.end += start_time
            
            yield result
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear GPU memory if using CUDA
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        
        self.is_loaded = False
        print("Model unloaded successfully")
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        base_info = super().get_model_info()
        base_info.update({
            'device': self.device,
            'compute_type': self.compute_type,
            'chunk_length': self.config.chunk_length,
            'stride_length': self.config.stride_length,
            'beam_size': self.config.beam_size,
            'temperature': self.config.temperature
        })
        return base_info


def create_whisper_asr(model_name: Optional[str] = None) -> WhisperCT2Model:
    """
    Factory function to create a WhisperCT2Model instance
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Configured WhisperCT2Model instance
    """
    try:
        config = get_config().asr
        
        if model_name:
            # Create a copy of config with different model name
            import copy
            config = copy.deepcopy(config)
            config.model = model_name
    except:
        # Fallback config for testing
        config = type('Config', (), {
            'model': model_name or 'tiny',
            'language': 'arabic',  # Use full name
            'compute_type': 'auto',
            'beam_size': 1,
            'best_of': 1,
            'temperature': 0.0,
            'condition_on_previous_text': True,
            'chunk_length': 30,
            'stride_length': 5
        })()
    
    return WhisperCT2Model(config)


# Utility functions for Whisper-specific operations
def validate_whisper_model(model_name: str) -> bool:
    """Validate if a model name is supported by faster-whisper"""
    valid_models = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
        "ct2-fast-whisper-tiny", "ct2-fast-whisper-small", "ct2-fast-whisper-medium",
        "deepdml/faster-whisper-large-v3-turbo-ct2"
    ]
    
    # Check if it's a direct model name or path
    return (
        model_name in valid_models or
        model_name.startswith("ct2-") or 
        "/" in model_name  # Assume it's a HuggingFace model path
    )


def estimate_memory_usage(model_name: str, compute_type: str = "float16") -> dict:
    """Estimate memory usage for a Whisper model"""
    # Rough estimates in MB
    model_sizes = {
        "tiny": {"float32": 150, "float16": 75, "int8": 40},
        "base": {"float32": 290, "float16": 145, "int8": 75},
        "small": {"float32": 970, "float16": 485, "int8": 245},
        "medium": {"float32": 3000, "float16": 1500, "int8": 750},
        "large": {"float32": 6000, "float16": 3000, "int8": 1500},
    }
    
    # Extract base model name
    base_name = model_name.split("-")[-1] if "-" in model_name else model_name
    base_name = base_name.replace(".en", "")
    
    if base_name in model_sizes:
        return {
            "model_size_mb": model_sizes[base_name].get(compute_type, 1000),
            "recommended_ram_mb": model_sizes[base_name].get(compute_type, 1000) * 2,
            "compute_type": compute_type
        }
    
    return {"model_size_mb": 1000, "recommended_ram_mb": 2000, "compute_type": compute_type}


# Language mapping for Whisper compatibility
LANGUAGE_MAPPING = {
    # Full name -> ISO code (for Whisper)
    'arabic': 'ar',
    'english': 'en',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'russian': 'ru',
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko',
    # Reverse mapping (ISO -> full name)
    'ar': 'arabic',
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'ru': 'russian',
    'zh': 'chinese',
    'ja': 'japanese',
    'ko': 'korean'
}


def map_language_to_iso(language: str) -> str:
    """Convert full language name to ISO code for Whisper"""
    if language in LANGUAGE_MAPPING:
        mapped = LANGUAGE_MAPPING[language]
        # If it's already an ISO code, return as is, otherwise return the mapped ISO code
        return mapped if len(mapped) == 2 else language
    return language  # Return as-is if not found


def map_language_to_full(language: str) -> str:
    """Convert ISO code to full language name"""
    return LANGUAGE_MAPPING.get(language, language) 