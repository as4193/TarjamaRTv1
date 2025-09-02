import os
import logging
import requests
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..config import MTConfig

logger = logging.getLogger(__name__)

@dataclass
class MTResult:
    """Machine translation result"""
    translated_text: str
    source_language: str
    target_language: str
    processing_time: float
    error: Optional[str] = None


class MTEngine:
    """Base class for machine translation engines"""
    
    def translate(self, text: str, **kwargs) -> MTResult:
        """Translate text from source to target language"""
        raise NotImplementedError("Subclasses must implement translate method")
    
    def load_model(self) -> bool:
        """Load the translation model"""
        raise NotImplementedError("Subclasses must implement load_model method")


class VLLMTranslator(MTEngine):
    """vLLM-based translation engine using Qwen3-4B-Instruct-2507-AWQ-4bit model"""
    
    def __init__(self, config: MTConfig):
        """Initialize vLLM translator"""
        self.config = config
        self.vllm_host = config.host
        self.vllm_port = config.port
        
        if hasattr(config, 'model'):
            if '/' in config.model:
                self.vllm_model = 'qwen3-4b-instruct-2507-awq-4bit'
            else:
                self.vllm_model = config.model
        else:
            self.vllm_model = os.getenv('VLLM_MODEL_NAME', 'qwen3-4b-instruct-2507-fp8')
            
        self.api_endpoint = f"http://{self.vllm_host}:{self.vllm_port}/v1/chat/completions"
        
        logger.info(f"VLLM Translator initialized:")
        logger.info(f"   Host: {self.vllm_host}:{self.vllm_port}")
        logger.info(f"   Model: {self.vllm_model}")
        logger.info(f"   Source: {config.source_language}")
        logger.info(f"   Target: {config.target_language}")
        logger.info(f"   Config object type: {type(config)}")
        logger.info(f"   Config attributes: {dir(config)}")
        logger.info(f"   Config.model value: {getattr(config, 'model', 'NOT_FOUND')}")
    
    def load_model(self) -> bool:
        """Check if vLLM service is available"""
        try:
            response = requests.get(f"http://{self.vllm_host}:{self.vllm_port}/health", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM service is available")
                return True
            else:
                logger.error(f"vLLM service unhealthy: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to vLLM service: {e}")
            return False
    
    def translate(self, text: str, **kwargs) -> MTResult:
        """Translate text using vLLM with Qwen3-4B-Instruct-2507-AWQ-4bit model"""
        import time
        start_time = time.time()
        
        if not text or not text.strip():
            return MTResult(
                translated_text="",
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                processing_time=0.0,
                error="Empty input text"
            )
        
        try:
            # Prepare the prompt for translation
            prompt = self._build_translation_prompt(text)
            
            # Make API request to vLLM
            payload = {
                "model": self.vllm_model,
                "messages": [
                    {"role": "system", "content": "You are a professional translator. Translate the given text accurately while preserving the meaning and style without any additional explanation or comments."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": False
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    translated_text = response_data['choices'][0]['message']['content'].strip()
                    
                    processing_time = time.time() - start_time
                    
                    return MTResult(
                        translated_text=translated_text,
                        source_language=self.config.source_language,
                        target_language=self.config.target_language,
                        processing_time=processing_time
                    )
                else:
                    raise ValueError("Invalid response format from vLLM API")
            else:
                raise requests.HTTPError(f"vLLM API request failed with status {response.status_code}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Translation failed: {e}")
            
            return MTResult(
                translated_text="",
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _build_translation_prompt(self, text: str) -> str:
        """Build translation prompt for the given text"""
        source_lang = self.config.source_language
        target_lang = self.config.target_language
        
        prompt = f"""Translate the following text from {source_lang} to {target_lang}. Provide only the {target_lang} translation without any explanations or additional text.

{source_lang} text: {text}

{target_lang} translation:"""
        
        return prompt


def create_vllm_mt(config: MTConfig) -> VLLMTranslator:
    """Factory function to create vLLM MT engine"""
    return VLLMTranslator(config)