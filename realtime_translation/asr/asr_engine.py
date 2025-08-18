import time
import numpy as np
from typing import List, Dict, Optional, Union, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

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
            raise ValueError("No segments available")
        return max(segment.end for segment in self.segments) - min(segment.start for segment in self.segments)
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all segments"""
        confidences = [seg.confidence for seg in self.segments if seg.confidence is not None]
        if not confidences:
            raise ValueError("No confidence scores available")
        return sum(confidences) / len(confidences)
    
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
    
    def __init__(self, config):
        if config is None:
            raise ValueError("Configuration is required")
        self.config = config
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


# Utility functions for ASR processing
def calculate_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate duration of audio array in seconds"""
    return len(audio) / sample_rate


def merge_segments(segments: List[ASRSegment], max_gap: float) -> List[ASRSegment]:
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
