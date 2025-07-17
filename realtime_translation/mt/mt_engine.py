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
                'mt': type('MTConfig', (), {
                    'model': 'Helsinki-NLP/opus-mt-ar-en',
                    'source_language': 'ar',
                    'target_language': 'en',
                    'max_length': 128,
                    'beam_size': 4,
                    'temperature': 0.0,
                    'wait_k': type('WaitKConfig', (), {
                        'k': 3,
                        'adaptive_k': True,
                        'active': 'dev'
                    })()
                })()
            })()


@dataclass
class MTToken:
    """Represents a single translation token with metadata"""
    text: str  # Token text
    position: int  # Position in sequence
    confidence: Optional[float] = None  # Token confidence score
    attention_weights: Optional[List[float]] = None  # Attention weights
    timestamp: Optional[float] = None  # When token was generated
    is_final: bool = False  # Whether this token is finalized
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'position': self.position,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'is_final': self.is_final
        }


@dataclass
class MTResult:
    """Complete MT result with metadata and performance metrics"""
    # Translation output
    translated_text: str = ""
    tokens: List[MTToken] = field(default_factory=list)
    
    # Input information
    source_text: str = ""
    source_language: str = "arabic"
    target_language: str = "english"
    
    # Quality metrics
    confidence: Optional[float] = None  # Overall translation confidence
    attention_entropy: Optional[float] = None  # Attention distribution entropy
    length_ratio: Optional[float] = None  # Target/source length ratio
    
    # Performance metrics
    processing_time: Optional[float] = None  # Time taken to translate
    tokens_per_second: Optional[float] = None  # Translation speed
    
    # Wait-k specific
    k_value: Optional[int] = None  # Wait-k value used
    partial_translation: bool = False  # Whether this is a partial result
    is_final: bool = False  # Whether translation is complete
    
    # Model metadata
    model_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def get_token_count(self) -> int:
        """Get number of tokens in translation"""
        return len(self.tokens)
    
    def get_final_tokens(self) -> List[MTToken]:
        """Get only finalized tokens"""
        return [token for token in self.tokens if token.is_final]
    
    def get_average_confidence(self) -> Optional[float]:
        """Get average confidence across all tokens"""
        confidences = [token.confidence for token in self.tokens if token.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None
    
    def get_final_text(self) -> str:
        """Get text from only finalized tokens"""
        final_tokens = self.get_final_tokens()
        return " ".join(token.text for token in final_tokens)
    
    def update_with_new_tokens(self, new_tokens: List[MTToken]) -> None:
        """Update result with new tokens from streaming translation"""
        # Add new tokens
        for token in new_tokens:
            # Update existing token or add new one
            existing_idx = next((i for i, t in enumerate(self.tokens) if t.position == token.position), None)
            if existing_idx is not None:
                self.tokens[existing_idx] = token
            else:
                self.tokens.append(token)
        
        # Sort tokens by position
        self.tokens.sort(key=lambda t: t.position)
        
        # Update translated text
        self.translated_text = " ".join(token.text for token in self.tokens)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'translated_text': self.translated_text,
            'tokens': [token.to_dict() for token in self.tokens],
            'source_text': self.source_text,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'confidence': self.confidence,
            'attention_entropy': self.attention_entropy,
            'length_ratio': self.length_ratio,
            'processing_time': self.processing_time,
            'tokens_per_second': self.tokens_per_second,
            'k_value': self.k_value,
            'partial_translation': self.partial_translation,
            'is_final': self.is_final,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'token_count': self.get_token_count(),
            'final_token_count': len(self.get_final_tokens()),
            'average_confidence': self.get_average_confidence()
        }


class MTEngine(ABC):
    """Abstract base class for MT engines"""
    
    def __init__(self, config=None):
        try:
            self.config = config or get_config().mt
        except:
            # Fallback config for testing
            self.config = type('Config', (), {
                'model': 'Helsinki-NLP/opus-mt-ar-en',
                'source_language': 'ar',
                'target_language': 'en',
                'max_length': 128,
                'beam_size': 4,
                'temperature': 0.0,
                'wait_k': type('WaitKConfig', (), {
                    'k': 3,
                    'adaptive_k': True,
                    'active': 'dev'
                })()
            })()
            
        self.model_name = self.config.model
        self.source_language = self.config.source_language
        self.target_language = self.config.target_language
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the MT model"""
        pass
    
    @abstractmethod
    def translate(self, text: str, **kwargs) -> MTResult:
        """
        Translate text from source to target language
        
        Args:
            text: Source text to translate
            **kwargs: Additional translation parameters
            
        Returns:
            MTResult with translation and metadata
        """
        pass
    
    @abstractmethod
    def translate_streaming(self, text_stream: Iterator[str], wait_k: int = 3, **kwargs) -> Iterator[MTResult]:
        """
        Translate text in streaming mode with Wait-k policy
        
        Args:
            text_stream: Iterator of text chunks
            wait_k: Number of tokens to wait before starting translation
            **kwargs: Additional translation parameters
            
        Yields:
            MTResult for each translation update
        """
        pass
    
    def translate_with_wait_k(self, tokens: List[str], k: int = 3, **kwargs) -> MTResult:
        """
        Translate with Wait-k policy - start translating after k tokens
        
        Args:
            tokens: List of source tokens
            k: Number of tokens to wait
            **kwargs: Additional parameters
            
        Returns:
            MTResult with partial translation
        """
        if len(tokens) < k:
            # Not enough tokens yet, return empty result
            return MTResult(
                source_text=" ".join(tokens),
                k_value=k,
                partial_translation=True,
                is_final=False,
                model_name=self.model_name
            )
        
        # Use first k tokens for translation
        partial_text = " ".join(tokens[:k])
        result = self.translate(partial_text, **kwargs)
        result.k_value = k
        result.partial_translation = True
        result.is_final = (len(tokens) == k)  # Final if we have exactly k tokens
        
        return result
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        self.is_loaded = False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'is_loaded': self.is_loaded,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }


class MTManager:
    """Manager class for handling MT operations with performance monitoring"""
    
    def __init__(self, engine: MTEngine):
        self.engine = engine
        try:
            self.config = get_config()
        except:
            self.config = None
        self.performance_stats = {
            'total_requests': 0,
            'total_source_tokens': 0,
            'total_target_tokens': 0,
            'total_processing_time': 0.0,
            'average_tokens_per_second': 0.0,
            'average_confidence': 0.0
        }
    
    def translate(self, text: str, **kwargs) -> MTResult:
        """
        Translate text with performance monitoring
        
        Args:
            text: Source text to translate
            **kwargs: Additional parameters for translation
            
        Returns:
            MTResult with translation and performance metrics
        """
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        # Perform translation
        result = self.engine.translate(text, **kwargs)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        if result.tokens:
            result.tokens_per_second = len(result.tokens) / processing_time
        
        # Update performance statistics
        self._update_stats(result)
        
        return result
    
    def translate_streaming(self, text_stream: Iterator[str], wait_k: int = 3, **kwargs) -> Iterator[MTResult]:
        """
        Translate text in streaming mode with performance monitoring
        
        Args:
            text_stream: Iterator of text chunks
            wait_k: Wait-k value
            **kwargs: Additional parameters
            
        Yields:
            MTResult for each translation update with performance metrics
        """
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        for result in self.engine.translate_streaming(text_stream, wait_k=wait_k, **kwargs):
            self._update_stats(result)
            yield result
    
    def translate_with_wait_k(self, tokens: List[str], k: int = 3, **kwargs) -> MTResult:
        """
        Translate with Wait-k policy with performance monitoring
        
        Args:
            tokens: List of source tokens
            k: Wait-k value
            **kwargs: Additional parameters
            
        Returns:
            MTResult with translation and performance metrics
        """
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.engine.is_loaded:
            self.engine.load_model()
        
        # Perform translation
        result = self.engine.translate_with_wait_k(tokens, k=k, **kwargs)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        if result.tokens:
            result.tokens_per_second = len(result.tokens) / processing_time
        
        # Update performance statistics
        self._update_stats(result)
        
        return result
    
    def _update_stats(self, result: MTResult) -> None:
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        
        # Count tokens
        if result.source_text:
            source_tokens = len(result.source_text.split())
            self.performance_stats['total_source_tokens'] += source_tokens
        
        if result.tokens:
            self.performance_stats['total_target_tokens'] += len(result.tokens)
        
        if result.processing_time:
            self.performance_stats['total_processing_time'] += result.processing_time
        
        # Update average tokens per second
        if result.tokens_per_second:
            total_requests = self.performance_stats['total_requests']
            old_avg = self.performance_stats['average_tokens_per_second']
            self.performance_stats['average_tokens_per_second'] = (
                old_avg * (total_requests - 1) + result.tokens_per_second
            ) / total_requests
        
        # Update average confidence
        if result.confidence:
            total_requests = self.performance_stats['total_requests']
            old_avg = self.performance_stats['average_confidence']
            self.performance_stats['average_confidence'] = (
                old_avg * (total_requests - 1) + result.confidence
            ) / total_requests
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        if stats['total_processing_time'] > 0:
            stats['overall_tokens_per_second'] = stats['total_target_tokens'] / stats['total_processing_time']
        else:
            stats['overall_tokens_per_second'] = 0.0
        
        if stats['total_source_tokens'] > 0:
            stats['average_length_ratio'] = stats['total_target_tokens'] / stats['total_source_tokens']
        else:
            stats['average_length_ratio'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.performance_stats = {
            'total_requests': 0,
            'total_source_tokens': 0,
            'total_target_tokens': 0,
            'total_processing_time': 0.0,
            'average_tokens_per_second': 0.0,
            'average_confidence': 0.0
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the MT engine and model"""
        return self.engine.get_model_info()


# Utility functions for MT processing
def tokenize_text(text: str, language: str = "arabic") -> List[str]:
    """Simple tokenization - can be enhanced with language-specific tokenizers"""
    # Basic whitespace tokenization
    tokens = text.strip().split()
    return tokens


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score between reference and hypothesis"""
    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score / 100.0  # Convert to 0-1 range
    except ImportError:
        # Fallback: simple word overlap
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if not ref_words:
            return 0.0
        overlap = len(ref_words.intersection(hyp_words))
        return overlap / len(ref_words)


def estimate_translation_quality(result: MTResult) -> Dict[str, float]:
    """Estimate translation quality using various metrics"""
    quality_metrics = {}
    
    # Confidence-based quality
    if result.confidence:
        quality_metrics['confidence_quality'] = result.confidence
    
    # Length ratio quality (reasonable ratio indicates good translation)
    if result.length_ratio:
        # Ideal ratio for Arabic->English is around 0.8-1.2
        ideal_ratio = 1.0
        ratio_quality = 1.0 - min(abs(result.length_ratio - ideal_ratio), 1.0)
        quality_metrics['length_ratio_quality'] = ratio_quality
    
    # Token confidence distribution
    token_confidences = [t.confidence for t in result.tokens if t.confidence is not None]
    if token_confidences:
        quality_metrics['token_confidence_mean'] = np.mean(token_confidences)
        quality_metrics['token_confidence_std'] = np.std(token_confidences)
        quality_metrics['token_confidence_min'] = np.min(token_confidences)
    
    # Attention entropy (if available)
    if result.attention_entropy:
        # Lower entropy might indicate more focused attention
        quality_metrics['attention_quality'] = 1.0 / (1.0 + result.attention_entropy)
    
    # Overall quality score (weighted average)
    if quality_metrics:
        weights = {
            'confidence_quality': 0.4,
            'length_ratio_quality': 0.2,
            'token_confidence_mean': 0.3,
            'attention_quality': 0.1
        }
        
        overall_quality = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics:
                overall_quality += quality_metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            quality_metrics['overall_quality'] = overall_quality / total_weight
    
    return quality_metrics


def merge_translation_results(results: List[MTResult]) -> MTResult:
    """Merge multiple MT results into a single result"""
    if not results:
        return MTResult()
    
    if len(results) == 1:
        return results[0]
    
    # Merge results
    merged = MTResult(
        source_text=" ".join(r.source_text for r in results if r.source_text),
        source_language=results[0].source_language,
        target_language=results[0].target_language,
        model_name=results[0].model_name
    )
    
    # Combine tokens
    all_tokens = []
    position_offset = 0
    
    for result in results:
        for token in result.tokens:
            new_token = MTToken(
                text=token.text,
                position=token.position + position_offset,
                confidence=token.confidence,
                timestamp=token.timestamp,
                is_final=token.is_final
            )
            all_tokens.append(new_token)
        position_offset += len(result.tokens)
    
    merged.tokens = all_tokens
    merged.translated_text = " ".join(token.text for token in all_tokens)
    
    # Average metrics
    confidences = [r.confidence for r in results if r.confidence is not None]
    if confidences:
        merged.confidence = sum(confidences) / len(confidences)
    
    processing_times = [r.processing_time for r in results if r.processing_time is not None]
    if processing_times:
        merged.processing_time = sum(processing_times)
    
    return merged 