import time
import numpy as np
from typing import List, Dict, Optional, Union, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Handle config import with fallback
try:
    from ..config import get_config
except ImportError:
    # Fallback for when running as standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import get_config


@dataclass
class ASRSegment:
    """Represents a transcribed segment with timing and confidence information"""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Transcribed text
    confidence: Optional[float] = None  # Confidence score (0-1)
    tokens: Optional[List[str]] = None  # Individual tokens
    token_timestamps: Optional[List[float]] = None  # Token-level timestamps
    language: Optional[str] = None  # Detected language


@dataclass
class ASRResult:
    """Complete ASR result with metadata and performance metrics"""
    segments: List[ASRSegment] = field(default_factory=list)
    language: Optional[str] = None
    confidence: Optional[float] = None
    
    # Performance metrics
    audio_duration: Optional[float] = None  # Input audio duration
    processing_time: Optional[float] = None  # Time taken to process
    real_time_factor: Optional[float] = None  # RTF = processing_time / audio_duration
    
    # Model metadata
    model_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def get_full_text(self) -> str:
        """Get complete transcribed text from all segments"""
        return " ".join(segment.text.strip() for segment in self.segments)
    
    def get_duration(self) -> float:
        """Get total duration covered by segments"""
        if not self.segments:
            return 0.0
        return max(segment.end for segment in self.segments) - min(segment.start for segment in self.segments)
    
    def get_average_confidence(self) -> Optional[float]:
        """Get average confidence across all segments"""
        confidences = [seg.confidence for seg in self.segments if seg.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'confidence': seg.confidence,
                    'language': seg.language
                }
                for seg in self.segments
            ],
            'language': self.language,
            'confidence': self.confidence,
            'audio_duration': self.audio_duration,
            'processing_time': self.processing_time,
            'real_time_factor': self.real_time_factor,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'full_text': self.get_full_text()
        }


class ASREngine(ABC):
    """Abstract base class for ASR engines"""
    
    def __init__(self, config=None):
        try:
            self.config = config or get_config().asr
        except:
            # Fallback config for testing
            self.config = type('Config', (), {
                'model': 'tiny',
                'language': 'ar',
                'compute_type': 'auto',
                'beam_size': 1,
                'best_of': 1,
                'temperature': 0.0,
                'condition_on_previous_text': True,
                'chunk_length': 30,
                'stride_length': 5
            })()
            
        self.model_name = self.config.model
        self.language = self.config.language
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the ASR model"""
        pass
    
    @abstractmethod
    def transcribe(self, audio: Union[np.ndarray, str], **kwargs) -> ASRResult:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio data (numpy array) or file path (string)
            **kwargs: Additional transcription parameters
            
        Returns:
            ASRResult with transcription and metadata
        """
        pass
    
    @abstractmethod
    def transcribe_streaming(self, audio_chunks: Iterator[np.ndarray], **kwargs) -> Iterator[ASRResult]:
        """
        Transcribe audio in streaming mode
        
        Args:
            audio_chunks: Iterator of audio chunks (numpy arrays)
            **kwargs: Additional transcription parameters
            
        Yields:
            ASRResult for each processed chunk
        """
        pass
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        self.is_loaded = False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'language': self.language,
            'is_loaded': self.is_loaded,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }


class ASRManager:
    """Manager class for handling ASR operations with performance monitoring"""
    
    def __init__(self, engine: ASREngine):
        self.engine = engine
        try:
            self.config = get_config()
        except:
            self.config = None
        self.performance_stats = {
            'total_requests': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'average_rtf': 0.0,
            'last_rtf': 0.0
        }
    
    def transcribe(self, audio: Union[np.ndarray, str], **kwargs) -> ASRResult:
        """
        Transcribe audio with performance monitoring
        
        Args:
            audio: Audio data or file path
            **kwargs: Additional parameters for transcription
            
        Returns:
            ASRResult with transcription and performance metrics
        """
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        # Perform transcription
        result = self.engine.transcribe(audio, **kwargs)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        if result.audio_duration and result.audio_duration > 0:
            result.real_time_factor = processing_time / result.audio_duration
        
        # Update performance statistics
        self._update_stats(result)
        
        return result
    
    def transcribe_streaming(self, audio_chunks: Iterator[np.ndarray], **kwargs) -> Iterator[ASRResult]:
        """
        Transcribe audio in streaming mode with performance monitoring
        
        Args:
            audio_chunks: Iterator of audio chunks
            **kwargs: Additional parameters
            
        Yields:
            ASRResult for each chunk with performance metrics
        """
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        for result in self.engine.transcribe_streaming(audio_chunks, **kwargs):
            self._update_stats(result)
            yield result
    
    def _update_stats(self, result: ASRResult) -> None:
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        
        if result.audio_duration:
            self.performance_stats['total_audio_duration'] += result.audio_duration
        
        if result.processing_time:
            self.performance_stats['total_processing_time'] += result.processing_time
        
        if result.real_time_factor:
            self.performance_stats['last_rtf'] = result.real_time_factor
            
            # Update average RTF
            total_requests = self.performance_stats['total_requests']
            if total_requests > 1:
                old_avg = self.performance_stats['average_rtf']
                self.performance_stats['average_rtf'] = (
                    old_avg * (total_requests - 1) + result.real_time_factor
                ) / total_requests
            else:
                self.performance_stats['average_rtf'] = result.real_time_factor
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        if stats['total_audio_duration'] > 0:
            stats['overall_rtf'] = stats['total_processing_time'] / stats['total_audio_duration']
        else:
            stats['overall_rtf'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.performance_stats = {
            'total_requests': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'average_rtf': 0.0,
            'last_rtf': 0.0
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the ASR engine and model"""
        return self.engine.get_model_info()


# Utility functions for ASR processing
def calculate_audio_duration(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """Calculate duration of audio array in seconds"""
    return len(audio) / sample_rate


def merge_segments(segments: List[ASRSegment], max_gap: float = 0.5) -> List[ASRSegment]:
    """
    Merge consecutive segments with small gaps
    
    Args:
        segments: List of ASR segments
        max_gap: Maximum gap in seconds to merge
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    merged = [segments[0]]
    
    for segment in segments[1:]:
        last_segment = merged[-1]
        gap = segment.start - last_segment.end
        
        if gap <= max_gap:
            # Merge segments
            merged_text = f"{last_segment.text.strip()} {segment.text.strip()}"
            merged_confidence = None
            if last_segment.confidence and segment.confidence:
                merged_confidence = (last_segment.confidence + segment.confidence) / 2
            
            merged[-1] = ASRSegment(
                start=last_segment.start,
                end=segment.end,
                text=merged_text,
                confidence=merged_confidence,
                language=last_segment.language or segment.language
            )
        else:
            merged.append(segment)
    
    return merged


def filter_by_confidence(segments: List[ASRSegment], min_confidence: float = 0.5) -> List[ASRSegment]:
    """Filter segments by minimum confidence threshold"""
    return [seg for seg in segments if seg.confidence is None or seg.confidence >= min_confidence] 