import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)

try:
    import soundfile as sf
    import scipy.signal
except ImportError:
    raise ImportError("soundfile and scipy are required. Install with: pip install soundfile scipy")

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("faster-whisper is required. Install with: pip install faster-whisper")

from .asr_engine import ASREngine, ASRResult, ASRSegment, calculate_audio_duration
from .language_mapping import map_language_to_iso, map_language_to_full

# Handle config import - try relative import first, then absolute
try:
    from ..config import get_config
except ImportError:
    try:
        from config import get_config
    except ImportError:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_audio_soundfile(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file using soundfile with pydub fallback for unsupported formats"""
    try:
        audio_data, original_sr = sf.read(audio_path)
    except Exception as e:
        print(f"Soundfile failed for {audio_path}, using pydub fallback: {e}")
        
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(audio_path)
            
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            original_sr = audio.frame_rate
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / (2**15)
            
        except ImportError:
            raise ImportError("pydub is required for .m4a files. Install with: pip install pydub")
        except Exception as pydub_error:
            raise RuntimeError(f"Failed to load audio file {audio_path} with both soundfile and pydub: {pydub_error}")
    
    # Convert to mono if stereo (for soundfile path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if original_sr != target_sr:
        resample_ratio = target_sr / original_sr
        audio_data = scipy.signal.resample(
            audio_data, 
            int(len(audio_data) * resample_ratio)
        ).astype(np.float32)
    
    audio_data = audio_data.astype(np.float32)
    return audio_data, target_sr


class WhisperCT2Model(ASREngine):
    """Faster-whisper implementation for ASR with streaming support"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.device = self._get_device()
        self.compute_type = self._get_compute_type()
        
    def _get_device(self) -> str:
        """Get GPU device for model execution"""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in PyTorch. GPU is required for this evaluation.")
        return "cuda"
    
    def _get_compute_type(self) -> str:
        """Get compute type for GPU"""
        if self.config.compute_type != "auto":
            return self.config.compute_type
        return "float16"
    
    def load_model(self) -> None:
        """Load the Whisper model"""
        if self.is_loaded:
            return
            
        try:
            print(f"Loading Whisper model: {self.config.model}")
            print(f"Device: {self.device}")
            print(f"Compute type: {self.compute_type}")
            
            self.model = WhisperModel(
                self.config.model,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=self.config.local_files_only
            )
            self.is_loaded = True
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def transcribe(self, audio: Union[np.ndarray, str], **kwargs) -> ASRResult:
        """Transcribe audio to text"""
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        if isinstance(audio, str):
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio}")
            
            audio_data, sr = load_audio_soundfile(audio, target_sr=16000)
            audio_duration = len(audio_data) / 16000
        elif isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32)
            audio_duration = calculate_audio_duration(audio_data, 16000)
        else:
            raise ValueError("Audio input must be numpy array or file path")
        
        # Map language to ISO code for Whisper compatibility
        whisper_language = None
        if self.language and self.language != 'auto':
            whisper_language = map_language_to_iso(self.language)
        
        # Transcription parameters
        transcribe_params = {
            'beam_size': self.config.beam_size,
            'best_of': self.config.best_of,
            'temperature': self.config.temperature,
            'condition_on_previous_text': self.config.condition_on_previous_text,
            'language': whisper_language,
            'word_timestamps': True,
            'vad_filter': getattr(self.config, 'vad_filter', True),
            'repetition_penalty': getattr(self.config, 'repetition_penalty', 1.1),
            **kwargs
        }
        
        try:
            # Perform transcription
            segments, info = self.model.transcribe(audio_data, **transcribe_params)
            segment_list = list(segments)
            
            asr_segments = []
            for segment in segment_list:
                detected_lang = info.language if hasattr(info, 'language') else whisper_language
                full_lang = map_language_to_full(detected_lang) if detected_lang else self.language
                
                asr_segment = ASRSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=getattr(segment, 'avg_logprob', None),
                    language=full_lang
                )
                
                if hasattr(segment, 'words') and segment.words:
                    asr_segment.tokens = [word.word for word in segment.words]
                    asr_segment.token_timestamps = [word.start for word in segment.words]
                
                asr_segments.append(asr_segment)
            
            confidences = [seg.confidence for seg in asr_segments if seg.confidence is not None]
            overall_confidence = sum(confidences) / len(confidences) if confidences else None
            
            processing_time = time.time() - start_time
            
            result = ASRResult(
                segments=asr_segments,
                language=map_language_to_full(info.language) if hasattr(info, 'language') else self.language,
                confidence=overall_confidence,
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
        """Transcribe audio in streaming mode"""
        if not self.is_loaded:
            self.load_model()
        
        # Streaming buffer for overlapping chunks
        buffer = np.array([], dtype=np.float32)
        chunk_index = 0
        
        # Streaming parameters
        chunk_length = self.config.chunk_length
        stride_length = self.config.stride_length
        sample_rate = 16000
        
        chunk_samples = int(chunk_length * sample_rate)
        stride_samples = int(stride_length * sample_rate)
        
        for audio_chunk in audio_chunks:
            chunk_index += 1
            buffer = np.concatenate([buffer, audio_chunk])
            
            # Process if we have enough samples
            if len(buffer) >= chunk_samples:
                process_chunk = buffer[:chunk_samples]
                
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
                        audio_duration=calculate_audio_duration(process_chunk, 16000),
                        processing_time=0.0
                    )
                
                if len(buffer) > stride_samples:
                    buffer = buffer[stride_samples:]
                else:
                    buffer = np.array([], dtype=np.float32)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'language': self.language,
            'is_loaded': self.is_loaded,
            'device': self.device,
            'compute_type': self.compute_type,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }


def create_whisper_asr(model_size: str = "large-v3", config=None) -> WhisperCT2Model:
    """Factory function to create a Whisper ASR engine"""
    if config is None:
        try:
            config = get_config()
            config = config.asr
        except:
            class MinimalConfig:
                def __init__(self, model_size):
                    self.model = f"./local_models/faster-whisper-{model_size}-ct2"
                    self.language = "auto"
                    self.compute_type = "float16"
                    self.local_files_only = True
                    self.beam_size = 5
                    self.best_of = 3
                    self.temperature = 0.0
                    self.condition_on_previous_text = True
                    self.chunk_length = 2
                    self.stride_length = 0.5
                    self.repetition_penalty = 1.1
                    self.vad_filter = True
            
            config = MinimalConfig(model_size)
    
    return WhisperCT2Model(config)
