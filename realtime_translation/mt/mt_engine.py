import time
import numpy as np
from typing import List, Dict, Optional, Union, Iterator, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from config import get_config


@dataclass
class MTResult:
    """Complete MT result with metadata and performance metrics"""
    # Translation output
    translated_text: str = ""
    
    # Input information
    source_text: str = ""
    source_language: str = ""
    target_language: str = ""
    
    # Performance metrics
    processing_time: Optional[float] = None  # Time taken to translate
    
    # Model metadata
    model_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'translated_text': self.translated_text,
            'source_text': self.source_text,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'processing_time': self.processing_time,
            'model_name': self.model_name,
            'timestamp': self.timestamp
        }


class MTEngine(ABC):
    """Abstract base class for MT engines"""
    
    def __init__(self, config):
        if config is None:
            raise ValueError("Configuration is required")
        self.config = config
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
        """Translate text from source to target language"""
        pass
    
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