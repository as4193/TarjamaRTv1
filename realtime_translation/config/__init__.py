import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    format: str = "wav"


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    model: str = "pyannote"
    threshold: float = 0.5
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.1
    speech_pad_ms: int = 30
    window_size: int = 512
    sample_rate: int = 16000
    chunk_size: int = 1024
    aggressiveness: int = 2
    frame_duration_ms: int = 30


@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration"""
    model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    language: str = "auto"  # Will be set from MT source_language
    compute_type: str = "float16" 
    local_files_only: bool = False 
    beam_size: int = 5  
    best_of: int = 3    
    temperature: float = 0.0  
    condition_on_previous_text: bool = True
    chunk_length: int = 2
    stride_length: float = 0.5
    repetition_penalty: float = 1.1  
    vad_filter: bool = True  


@dataclass
class MTConfig:
    """Machine Translation configuration (vLLM)"""
    model: str = "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit"
    host: str = "localhost"
    port: int = 8000
    source_language: str = ""  # Will be set dynamically
    target_language: str = ""   # Will be set dynamically
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    timeout: int = 30
    max_retries: int = 3


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = "" #Please add your own API key here
    model: str = "gpt-4o-mini"
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    frequency_penalty: float = 0.5
    presence_penalty: float = 0.3


@dataclass
class StreamingConfig:
    """Streaming configuration"""
    chunk_size: float = 2.0
    overlap_ratio: float = 0.5  # 0.5 seconds overlap
    sample_rate: int = 16000


class Config:
    """Main configuration class containing all sub-configurations"""
    audio: AudioConfig = None
    vad: VADConfig = None
    asr: ASRConfig = None
    streaming: StreamingConfig = None
    mt: MTConfig = None
    openai: OpenAIConfig = None


def get_config() -> Config:
    """Get the global configuration instance"""
    config = Config()
    config.audio = AudioConfig()
    config.vad = VADConfig()
    config.asr = ASRConfig()
    config.streaming = StreamingConfig()
    config.mt = MTConfig()
    config.openai = OpenAIConfig()
    return config


# Export main classes and functions
__all__ = [
    'Config', 'get_config', 'AudioConfig', 'VADConfig', 'ASRConfig', 'MTConfig', 'OpenAIConfig', 'StreamingConfig'
] 