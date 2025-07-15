import time
import numpy as np
from typing import List, Dict, Optional, Union, Iterator, Tuple
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
    try:
        from config import get_config
    except ImportError:
        # Create a mock config for testing
        def get_config():
            return type('Config', (), {
                'vad': type('VADConfig', (), {
                    'model': 'pyannote',
                    'threshold': 0.5,
                    'min_speech_duration': 0.25,
                    'min_silence_duration': 0.1,
                    'window_size': 512,
                    'sample_rate': 16000,
                    'chunk_size': 1024
                })()
            })()


@dataclass
class VADSegment:
    """Represents a detected speech segment with timing information"""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    confidence: float  # VAD confidence score (0-1)
    is_speech: bool  # True if speech, False if silence
    audio_chunk: Optional[np.ndarray] = None  # Optional audio data
    
    def duration(self) -> float:
        """Get segment duration in seconds"""
        return self.end - self.start
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'is_speech': self.is_speech,
            'duration': self.duration()
        }


@dataclass
class VADResult:
    """Complete VAD result with segments and metadata"""
    segments: List[VADSegment] = field(default_factory=list)
    
    # Performance metrics
    audio_duration: Optional[float] = None  # Input audio duration
    processing_time: Optional[float] = None  # Time taken to process
    
    # VAD metadata
    model_name: Optional[str] = None
    threshold: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    def get_speech_segments(self) -> List[VADSegment]:
        """Get only the speech segments"""
        return [seg for seg in self.segments if seg.is_speech]
    
    def get_silence_segments(self) -> List[VADSegment]:
        """Get only the silence segments"""
        return [seg for seg in self.segments if not seg.is_speech]
    
    def get_total_speech_duration(self) -> float:
        """Get total duration of speech in seconds"""
        return sum(seg.duration() for seg in self.get_speech_segments())
    
    def get_total_silence_duration(self) -> float:
        """Get total duration of silence in seconds"""
        return sum(seg.duration() for seg in self.get_silence_segments())
    
    def get_speech_ratio(self) -> float:
        """Get ratio of speech to total audio (0-1)"""
        if not self.audio_duration or self.audio_duration == 0:
            return 0.0
        return self.get_total_speech_duration() / self.audio_duration
    
    def merge_consecutive_segments(self, same_type_only: bool = True) -> 'VADResult':
        """
        Merge consecutive segments of the same type
        
        Args:
            same_type_only: If True, only merge segments of same type (speech/silence)
        """
        if not self.segments:
            return self
        
        merged_segments = [self.segments[0]]
        
        for segment in self.segments[1:]:
            last_segment = merged_segments[-1]
            
            # Check if we should merge
            should_merge = True
            if same_type_only:
                should_merge = (last_segment.is_speech == segment.is_speech)
            
            if should_merge:
                # Merge segments
                merged_confidence = (last_segment.confidence + segment.confidence) / 2
                merged_segments[-1] = VADSegment(
                    start=last_segment.start,
                    end=segment.end,
                    confidence=merged_confidence,
                    is_speech=last_segment.is_speech
                )
            else:
                merged_segments.append(segment)
        
        # Create new result with merged segments
        return VADResult(
            segments=merged_segments,
            audio_duration=self.audio_duration,
            processing_time=self.processing_time,
            model_name=self.model_name,
            threshold=self.threshold,
            timestamp=self.timestamp
        )
    
    def filter_short_segments(self, min_duration: float = 0.1) -> 'VADResult':
        """Filter out segments shorter than min_duration"""
        filtered_segments = [
            seg for seg in self.segments 
            if seg.duration() >= min_duration
        ]
        
        return VADResult(
            segments=filtered_segments,
            audio_duration=self.audio_duration,
            processing_time=self.processing_time,
            model_name=self.model_name,
            threshold=self.threshold,
            timestamp=self.timestamp
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'segments': [seg.to_dict() for seg in self.segments],
            'audio_duration': self.audio_duration,
            'processing_time': self.processing_time,
            'model_name': self.model_name,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'speech_segments': len(self.get_speech_segments()),
            'silence_segments': len(self.get_silence_segments()),
            'total_speech_duration': self.get_total_speech_duration(),
            'total_silence_duration': self.get_total_silence_duration(),
            'speech_ratio': self.get_speech_ratio()
        }


class VADEngine(ABC):
    """Abstract base class for VAD engines"""
    
    def __init__(self, config=None):
        try:
            self.config = config or get_config().vad
        except:
            # Fallback config for testing
            self.config = type('Config', (), {
                'model': 'pyannote',
                'threshold': 0.5,
                'min_speech_duration': 0.25,
                'min_silence_duration': 0.1,
                'window_size': 512,
                'sample_rate': 16000,
                'chunk_size': 1024
            })()
            
        self.model_name = self.config.model
        self.threshold = self.config.threshold
        self.sample_rate = self.config.sample_rate
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the VAD model"""
        pass
    
    @abstractmethod
    def predict(self, audio: np.ndarray) -> Union[float, np.ndarray]:
        """
        Predict voice activity for audio chunk
        
        Args:
            audio: Audio data (numpy array, 16kHz)
            
        Returns:
            VAD score(s) - float for single prediction, array for frame-level
        """
        pass
    
    def detect_segments(self, audio: Union[np.ndarray, str], **kwargs) -> VADResult:
        """
        Detect speech segments in audio
        
        Args:
            audio: Audio data (numpy array) or file path (string)
            **kwargs: Additional parameters
            
        Returns:
            VADResult with detected segments
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Handle different input types
        if isinstance(audio, str):
            # Load audio file
            import librosa
            audio_data, sr = librosa.load(audio, sr=self.sample_rate)
        elif isinstance(audio, np.ndarray):
            audio_data = audio
        else:
            raise ValueError("Audio input must be numpy array or file path")
        
        audio_duration = len(audio_data) / self.sample_rate
        
        # Get frame-level predictions
        frame_predictions = self._get_frame_predictions(audio_data, **kwargs)
        
        # Convert predictions to segments
        segments = self._predictions_to_segments(
            frame_predictions, 
            audio_data, 
            **kwargs
        )
        
        processing_time = time.time() - start_time
        
        return VADResult(
            segments=segments,
            audio_duration=audio_duration,
            processing_time=processing_time,
            model_name=self.model_name,
            threshold=self.threshold
        )
    
    def detect_streaming(self, audio_chunks: Iterator[np.ndarray], **kwargs) -> Iterator[VADResult]:
        """
        Detect speech in streaming audio
        
        Args:
            audio_chunks: Iterator of audio chunks (16kHz numpy arrays)
            **kwargs: Additional parameters
            
        Yields:
            VADResult for each processed chunk
        """
        if not self.is_loaded:
            self.load_model()
        
        chunk_index = 0
        base_time = 0.0
        
        for audio_chunk in audio_chunks:
            chunk_index += 1
            
            try:
                # Process chunk
                result = self.detect_segments(audio_chunk, **kwargs)
                
                # Adjust timestamps for streaming context
                chunk_duration = len(audio_chunk) / self.sample_rate
                for segment in result.segments:
                    segment.start += base_time
                    segment.end += base_time
                
                base_time += chunk_duration
                yield result
                
            except Exception as e:
                print(f"⚠️ Error processing VAD chunk {chunk_index}: {e}")
                # Yield empty result to maintain stream
                yield VADResult(
                    segments=[],
                    model_name=self.model_name,
                    threshold=self.threshold,
                    audio_duration=len(audio_chunk) / self.sample_rate,
                    processing_time=0.0
                )
    
    def _get_frame_predictions(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Get frame-level VAD predictions"""
        window_size = kwargs.get('window_size', self.config.window_size)
        stride = kwargs.get('stride', window_size // 2)
        
        predictions = []
        
        # Process audio in windows
        for i in range(0, len(audio) - window_size + 1, stride):
            window = audio[i:i + window_size]
            
            # Get prediction for this window
            prediction = self.predict(window)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predictions_to_segments(self, predictions: np.ndarray, audio: np.ndarray, **kwargs) -> List[VADSegment]:
        """Convert frame-level predictions to speech segments"""
        window_size = kwargs.get('window_size', self.config.window_size)
        stride = kwargs.get('stride', window_size // 2)
        
        min_speech_duration = kwargs.get('min_speech_duration', self.config.min_speech_duration)
        min_silence_duration = kwargs.get('min_silence_duration', self.config.min_silence_duration)
        
        # Convert predictions to binary decisions
        is_speech = predictions > self.threshold
        
        segments = []
        current_segment_start = None
        current_is_speech = None
        
        # Convert frame indices to time
        frame_times = np.arange(len(predictions)) * stride / self.sample_rate
        
        for i, (frame_speech, confidence, frame_time) in enumerate(zip(is_speech, predictions, frame_times)):
            if current_is_speech is None:
                # Start first segment
                current_segment_start = frame_time
                current_is_speech = frame_speech
                
            elif current_is_speech != frame_speech:
                # Segment change detected
                segment_end = frame_time
                segment_duration = segment_end - current_segment_start
                
                # Check minimum duration requirements
                min_duration = min_speech_duration if current_is_speech else min_silence_duration
                
                if segment_duration >= min_duration:
                    # Create segment
                    avg_confidence = np.mean(predictions[max(0, i-10):i])  # Average over recent frames
                    
                    segments.append(VADSegment(
                        start=current_segment_start,
                        end=segment_end,
                        confidence=float(avg_confidence),
                        is_speech=current_is_speech
                    ))
                    
                    # Start new segment
                    current_segment_start = frame_time
                    current_is_speech = frame_speech
                else:
                    # Segment too short, continue current segment
                    pass
        
        # Add final segment
        if current_segment_start is not None:
            final_end = len(audio) / self.sample_rate
            segment_duration = final_end - current_segment_start
            
            min_duration = min_speech_duration if current_is_speech else min_silence_duration
            if segment_duration >= min_duration:
                avg_confidence = np.mean(predictions[-10:])  # Average over final frames
                
                segments.append(VADSegment(
                    start=current_segment_start,
                    end=final_end,
                    confidence=float(avg_confidence),
                    is_speech=current_is_speech
                ))
        
        return segments
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        self.is_loaded = False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'threshold': self.threshold,
            'sample_rate': self.sample_rate,
            'is_loaded': self.is_loaded,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }


class VADManager:
    """Manager class for handling VAD operations with performance monitoring"""
    
    def __init__(self, engine: VADEngine):
        self.engine = engine
        try:
            self.config = get_config()
        except:
            self.config = None
        self.performance_stats = {
            'total_requests': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'total_speech_detected': 0.0,
            'average_speech_ratio': 0.0
        }
    
    def detect_segments(self, audio: Union[np.ndarray, str], **kwargs) -> VADResult:
        """
        Detect speech segments with performance monitoring
        
        Args:
            audio: Audio data or file path
            **kwargs: Additional parameters for detection
            
        Returns:
            VADResult with segments and performance metrics
        """
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        # Perform detection
        result = self.engine.detect_segments(audio, **kwargs)
        
        # Update performance statistics
        self._update_stats(result)
        
        return result
    
    def detect_streaming(self, audio_chunks: Iterator[np.ndarray], **kwargs) -> Iterator[VADResult]:
        """
        Detect speech in streaming mode with performance monitoring
        
        Args:
            audio_chunks: Iterator of audio chunks
            **kwargs: Additional parameters
            
        Yields:
            VADResult for each chunk with performance metrics
        """
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        for result in self.engine.detect_streaming(audio_chunks, **kwargs):
            self._update_stats(result)
            yield result
    
    def _update_stats(self, result: VADResult) -> None:
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        
        if result.audio_duration:
            self.performance_stats['total_audio_duration'] += result.audio_duration
            
        if result.processing_time:
            self.performance_stats['total_processing_time'] += result.processing_time
        
        speech_duration = result.get_total_speech_duration()
        self.performance_stats['total_speech_detected'] += speech_duration
        
        # Update average speech ratio
        total_requests = self.performance_stats['total_requests']
        if total_requests > 1:
            old_avg = self.performance_stats['average_speech_ratio']
            self.performance_stats['average_speech_ratio'] = (
                old_avg * (total_requests - 1) + result.get_speech_ratio()
            ) / total_requests
        else:
            self.performance_stats['average_speech_ratio'] = result.get_speech_ratio()
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        if stats['total_audio_duration'] > 0:
            stats['overall_speech_ratio'] = stats['total_speech_detected'] / stats['total_audio_duration']
        else:
            stats['overall_speech_ratio'] = 0.0
            
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.performance_stats = {
            'total_requests': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'total_speech_detected': 0.0,
            'average_speech_ratio': 0.0
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the VAD engine and model"""
        return self.engine.get_model_info()


# Utility functions for VAD processing
def calculate_speech_statistics(segments: List[VADSegment]) -> Dict:
    """Calculate statistics about speech segments"""
    speech_segments = [seg for seg in segments if seg.is_speech]
    silence_segments = [seg for seg in segments if not seg.is_speech]
    
    return {
        'total_segments': len(segments),
        'speech_segments': len(speech_segments),
        'silence_segments': len(silence_segments),
        'total_speech_duration': sum(seg.duration() for seg in speech_segments),
        'total_silence_duration': sum(seg.duration() for seg in silence_segments),
        'average_speech_confidence': np.mean([seg.confidence for seg in speech_segments]) if speech_segments else 0.0,
        'average_silence_confidence': np.mean([seg.confidence for seg in silence_segments]) if silence_segments else 0.0,
        'longest_speech_segment': max((seg.duration() for seg in speech_segments), default=0.0),
        'longest_silence_segment': max((seg.duration() for seg in silence_segments), default=0.0),
    }


def merge_close_segments(segments: List[VADSegment], max_gap: float = 0.3) -> List[VADSegment]:
    """
    Merge speech segments that are close together
    
    Args:
        segments: List of VAD segments
        max_gap: Maximum gap in seconds to merge across
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x.start)
    
    merged = []
    current_group = [sorted_segments[0]]
    
    for segment in sorted_segments[1:]:
        last_segment = current_group[-1]
        gap = segment.start - last_segment.end
        
        if (gap <= max_gap and 
            last_segment.is_speech and 
            segment.is_speech):
            # Add to current group
            current_group.append(segment)
        else:
            # Finalize current group and start new one
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                # Merge group into single segment
                merged_segment = VADSegment(
                    start=current_group[0].start,
                    end=current_group[-1].end,
                    confidence=np.mean([seg.confidence for seg in current_group]),
                    is_speech=True  # We only merge speech segments
                )
                merged.append(merged_segment)
            
            current_group = [segment]
    
    # Handle final group
    if len(current_group) == 1:
        merged.append(current_group[0])
    else:
        merged_segment = VADSegment(
            start=current_group[0].start,
            end=current_group[-1].end,
            confidence=np.mean([seg.confidence for seg in current_group]),
            is_speech=True
        )
        merged.append(merged_segment)
    
    return merged 