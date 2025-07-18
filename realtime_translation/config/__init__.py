import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """System-level configuration"""
    device: str = "auto"
    log_level: str = "INFO"
    max_workers: int = 4
    buffer_size: int = 1024
    output_dir: str = "outputs"

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    chunk_duration: float = 1.0
    overlap_duration: float = 0.2
    format: str = "wav"

@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration"""
    model: str = "openai/whisper-tiny"
    language: str = "arabic"
    compute_type: str = "auto"
    local_files_only: bool = True  # Use only local files
    beam_size: int = 5  # Increased for better accuracy
    best_of: int = 3    # Consider more candidates
    temperature: float = 0.1  # Slight randomness to reduce repetitions
    condition_on_previous_text: bool = True
    chunk_length: int = 30
    stride_length: int = 5
    repetition_penalty: float = 1.1  # Penalty for repetitive outputs
    vad_filter: bool = True  # Enable VAD filtering in Whisper

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
    speech_pad_start: float = 0.1
    speech_pad_end: float = 0.3

@dataclass
class WaitKConfig:
    """Wait-k policy configuration"""
    mode: str = "fixed"
    k: int = 3              
    k_value: int = 3        
    max_k: int = 10
    min_k: int = 1
    adaptive_k: bool = True  

@dataclass
class MTConfig:
    """Machine Translation configuration"""
    dev_model: str = "Helsinki-NLP/opus-mt-ar-en"
    prod_model: str = "facebook/seamless-mt-large"
    model: str = "Helsinki-NLP/opus-mt-ar-en"
    source_language: str = "arabic"  
    target_language: str = "english"  
    source_lang: str = "arabic"      
    target_lang: str = "english"      
    wait_k: WaitKConfig = field(default_factory=WaitKConfig)
    max_length: int = 128
    local_files_only: bool = True  # Use only local files
    beam_size: int = 4           
    num_beams: int = 1           
    temperature: float = 0.0
    do_sample: bool = False
    min_confidence: float = 0.3
    retry_threshold: float = 0.5

@dataclass
class DevelopmentConfig:
    """Development and testing configuration"""
    save_intermediate_outputs: bool = True
    debug_mode: bool = False
    test_audio_path: str = "tests/test_audio.wav"
    output_dir: str = "logs/debug"

@dataclass
class Config:
    """Main configuration class containing all sub-configurations"""
    system: SystemConfig = field(default_factory=SystemConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


class ConfigLoader:
    """Configuration loader with YAML support and environment variable overrides"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config = None
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            "config/config_simple.yaml",  #simple config for testing
            "config/config.yaml",
            "config.yaml",
            os.path.join(os.path.dirname(__file__), "config_simple.yaml"),
            # os.path.join(os.path.dirname(__file__), "config.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration file not found in standard locations")
    
    def load(self) -> Config:
        """Load configuration from YAML file with environment variable overrides"""
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Apply environment variable overrides
            yaml_data = self._apply_env_overrides(yaml_data)
            
            # Create configuration object
            self._config = self._create_config_from_dict(yaml_data)
        
        return self._config
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        
        prefix = "RTTS_"  # Real-Time Translation System
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert RTTS_ASR_MODEL to asr.model
                config_key = key[len(prefix):].lower().replace('_', '.')
                self._set_nested_value(config_dict, config_key, value)
        
        return config_dict
    
    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: str):
        """Set nested configuration value from dot-separated key path"""
        keys = key_path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        final_key = keys[-1]
        if value.lower() in ('true', 'false'):
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            current[final_key] = float(value)
        else:
            current[final_key] = value
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """Create Config object from dictionary"""
        # Create sub-configs
        system_config = SystemConfig(**config_dict.get('system', {}))
        audio_config = AudioConfig(**config_dict.get('audio', {}))
        vad_config = VADConfig(**config_dict.get('vad', {}))
        asr_config = ASRConfig(**config_dict.get('asr', {}))
        mt_config = MTConfig(**config_dict.get('mt', {}))
        development_config = DevelopmentConfig(**config_dict.get('development', {}))
        
        return Config(
            system=system_config,
            audio=audio_config,
            vad=vad_config,
            asr=asr_config,
            mt=mt_config,
            development=development_config,
        )
    
    def reload(self) -> Config:
        """Reload configuration from file"""
        self._config = None
        return self.load()


# Global configuration instance
_config_loader = None
_config = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config_loader, _config
    if _config is None:
        _config_loader = ConfigLoader()
        _config = _config_loader.load()
    return _config


def reload_config() -> Config:
    """Reload the global configuration"""
    global _config_loader, _config
    if _config_loader is None:
        _config_loader = ConfigLoader()
    _config = _config_loader.reload()
    return _config


# Export main classes and functions
__all__ = [
    'Config', 'ConfigLoader', 'get_config', 'reload_config',
    'SystemConfig', 'AudioConfig', 'ASRConfig','DevelopmentConfig',
    'VADConfig', 'WaitKConfig', 'MTConfig'
] 