import time
import numpy as np
from typing import Union, Optional, List, Iterator, Dict
from pathlib import Path

try:
    from transformers import MarianMTModel, MarianTokenizer, pipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    MarianMTModel = None
    MarianTokenizer = None
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    TRANSFORMERS_AVAILABLE = False

import torch

from .mt_engine import MTEngine, MTResult, MTToken
from .wait_k_policy import WaitKPolicy, create_wait_k_policy

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


class HelsinkiMTModel(MTEngine):
    """Helsinki-NLP MT implementation for Arabic to English translation"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
        # Helsinki model specific parameters
        self.max_length = self.config.max_length
        self.beam_size = getattr(self.config, 'beam_size', 4)
        self.temperature = getattr(self.config, 'temperature', 0.0)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for Helsinki MT. "
                "Install with: pip install transformers"
            )
    
    def _get_device(self) -> str:
        """Determine the best device for model execution"""
        try:
            system_config = get_config().system
            if system_config.device == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return system_config.device
        except:
            # Fallback
            return "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> None:
        """Load the Helsinki MT model"""
        if self.is_loaded:
            return
        
        print(f"Loading Helsinki MT model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Source: {self.source_language} → Target: {self.target_language}")
        
        start_time = time.time()
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            
            # Load model with safetensors to avoid PyTorch security issue
            print("Loading model...")
            self.model = MarianMTModel.from_pretrained(
                self.model_name,
                use_safetensors=True  # Use safetensors to avoid torch.load security issue
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            print(f"Helsinki MT model loaded successfully in {load_time:.2f} seconds")
            self.is_loaded = True
            
        except Exception as e:
            print(f"Failed to load Helsinki MT model: {e}")
            print("   Make sure you have transformers and internet connection")
            raise
    
    def translate(self, text: str, **kwargs) -> MTResult:
        """
        Translate text from Arabic to English
        
        Args:
            text: Source text to translate
            **kwargs: Additional translation parameters
            
        Returns:
            MTResult with translation and metadata
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Preprocessing
        text = text.strip()
        if not text:
            return MTResult(
                source_text=text,
                translated_text="",
                source_language=self.source_language,
                target_language=self.target_language,
                model_name=self.model_name,
                processing_time=0.0
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            generation_kwargs = {
                'max_length': kwargs.get('max_length', self.max_length),
                'num_beams': kwargs.get('beam_size', self.beam_size),
                'temperature': kwargs.get('temperature', self.temperature),
                'do_sample': kwargs.get('do_sample', self.temperature > 0),
                'early_stopping': True,
                'return_dict_in_generate': True,
                'output_scores': True,
                'output_attentions': kwargs.get('output_attentions', False)
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode translation
            generated_tokens = outputs.sequences[0]
            translated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate confidence scores
            tokens_with_scores = self._extract_token_scores(
                generated_tokens, 
                outputs.scores if hasattr(outputs, 'scores') else None
            )
            
            # Calculate overall confidence
            token_confidences = [t.confidence for t in tokens_with_scores if t.confidence is not None]
            overall_confidence = np.mean(token_confidences) if token_confidences else None
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            source_tokens = len(text.split())
            target_tokens = len(translated_text.split())
            length_ratio = target_tokens / source_tokens if source_tokens > 0 else 0.0
            
            return MTResult(
                translated_text=translated_text,
                tokens=tokens_with_scores,
                source_text=text,
                source_language=self.source_language,
                target_language=self.target_language,
                confidence=overall_confidence,
                length_ratio=length_ratio,
                processing_time=processing_time,
                tokens_per_second=target_tokens / processing_time if processing_time > 0 else 0.0,
                model_name=self.model_name,
                is_final=True
            )
            
        except Exception as e:
            print(f"Translation failed: {e}")
            raise
    
    def translate_streaming(self, text_stream: Iterator[str], wait_k: int = 3, **kwargs) -> Iterator[MTResult]:
        """
        Translate text in streaming mode with Wait-k policy
        
        Args:
            text_stream: Iterator of text chunks
            wait_k: Wait-k value for policy
            **kwargs: Additional translation parameters
            
        Yields:
            MTResult for each translation update
        """
        if not self.is_loaded:
            self.load_model()
        
        # Create Wait-k policy
        policy_type = kwargs.get('policy_type', 'adaptive')
        wait_k_policy = create_wait_k_policy(policy_type, initial_k=wait_k)
        
        accumulated_text = ""
        chunk_index = 0
        
        for text_chunk in text_stream:
            chunk_index += 1
            
            # Add new text
            if accumulated_text:
                accumulated_text += " " + text_chunk
            else:
                accumulated_text = text_chunk
            
            # Tokenize accumulated text
            tokens = accumulated_text.split()
            
            # Check if we can translate with current policy
            if wait_k_policy.can_start_translation():
                # Get available context based on policy
                context_tokens = wait_k_policy.get_source_context()
                context_text = " ".join(context_tokens)
                
                try:
                    # Translate current context
                    result = self.translate(context_text, **kwargs)
                    result.k_value = wait_k_policy.state.current_k
                    result.partial_translation = True
                    result.is_final = False  # Not final until stream ends
                    
                    # Update policy with feedback
                    if result.confidence:
                        wait_k_policy.update_k(
                            confidence=result.confidence,
                            latency=result.processing_time
                        )
                    
                    yield result
                    
                except Exception as e:
                    print(f"Error in streaming translation chunk {chunk_index}: {e}")
                    # Yield empty result to maintain stream
                    yield MTResult(
                        source_text=context_text,
                        k_value=wait_k_policy.state.current_k,
                        partial_translation=True,
                        is_final=False,
                        model_name=self.model_name
                    )
            
            # Add tokens to policy
            for token in text_chunk.split():
                wait_k_policy.add_source_token(token)
        
        # Final translation with all available text
        if accumulated_text:
            try:
                final_result = self.translate(accumulated_text, **kwargs)
                final_result.k_value = wait_k_policy.state.current_k
                final_result.partial_translation = False
                final_result.is_final = True
                yield final_result
            except Exception as e:
                print(f"Error in final translation: {e}")
    
    def translate_with_wait_k(self, tokens: List[str], k: int = 3, **kwargs) -> MTResult:
        """
        Translate with Wait-k policy using token list
        
        Args:
            tokens: List of source tokens
            k: Wait-k value
            **kwargs: Additional parameters
            
        Returns:
            MTResult with partial or complete translation
        """
        # Use parent class implementation
        return super().translate_with_wait_k(tokens, k=k, **kwargs)
    
    def _extract_token_scores(self, token_ids: torch.Tensor, scores: Optional[List[torch.Tensor]]) -> List[MTToken]:
        """Extract tokens with confidence scores"""
        tokens = []
        
        # Decode tokens
        token_texts = []
        for i, token_id in enumerate(token_ids):
            if token_id.item() == self.tokenizer.eos_token_id:
                break
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text:  # Skip empty tokens
                token_texts.append(token_text)
        
        # Calculate confidence scores
        for i, token_text in enumerate(token_texts):
            confidence = None
            
            if scores and i < len(scores):
                # Convert logits to probabilities
                probs = torch.softmax(scores[i][0], dim=-1)
                # Get probability of selected token
                if i + 1 < len(token_ids):  # +1 because scores are shifted
                    token_id = token_ids[i + 1]
                    confidence = probs[token_id].item()
            
            tokens.append(MTToken(
                text=token_text,
                position=i,
                confidence=confidence,
                timestamp=time.time(),
                is_final=True
            ))
        
        return tokens
    
    def batch_translate(self, texts: List[str], **kwargs) -> List[MTResult]:
        """
        Translate multiple texts in batch for efficiency
        
        Args:
            texts: List of source texts
            **kwargs: Additional translation parameters
            
        Returns:
            List of MTResult objects
        """
        if not self.is_loaded:
            self.load_model()
        
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # Tokenize all inputs
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translations
            generation_kwargs = {
                'max_length': kwargs.get('max_length', self.max_length),
                'num_beams': kwargs.get('beam_size', self.beam_size),
                'temperature': kwargs.get('temperature', self.temperature),
                'do_sample': kwargs.get('do_sample', self.temperature > 0),
                'early_stopping': True
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Process results
            results = []
            processing_time = time.time() - start_time
            
            for i, (source_text, output_tokens) in enumerate(zip(texts, outputs)):
                translated_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                
                # Create simple tokens (no individual scores in batch mode)
                tokens = [
                    MTToken(text=token, position=j, is_final=True)
                    for j, token in enumerate(translated_text.split())
                ]
                
                result = MTResult(
                    translated_text=translated_text,
                    tokens=tokens,
                    source_text=source_text,
                    source_language=self.source_language,
                    target_language=self.target_language,
                    processing_time=processing_time / len(texts),  # Average per text
                    model_name=self.model_name,
                    is_final=True
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Batch translation failed: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU memory if using CUDA
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        self.is_loaded = False
        print("Helsinki MT model unloaded successfully")
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        base_info = super().get_model_info()
        base_info.update({
            'device': self.device,
            'max_length': self.max_length,
            'beam_size': self.beam_size,
            'temperature': self.temperature,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'pytorch_version': torch.__version__ if torch else 'N/A',
            'cuda_available': torch.cuda.is_available() if torch else False
        })
        return base_info


def create_helsinki_mt(model_name: Optional[str] = None) -> HelsinkiMTModel:
    """
    Factory function to create a HelsinkiMTModel instance
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Configured HelsinkiMTModel instance
    """
    try:
        config = get_config().mt
        
        if model_name:
            # Create a copy of config with different model name
            import copy
            config = copy.deepcopy(config)
            config.model = model_name
    except:
        # Fallback config for testing
        config = type('Config', (), {
            'model': model_name or 'Helsinki-NLP/opus-mt-ar-en',
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
    
    return HelsinkiMTModel(config)


# Utility functions for Helsinki MT
def validate_helsinki_installation() -> bool:
    """Check if Helsinki MT dependencies are available"""
    return TRANSFORMERS_AVAILABLE


def list_available_helsinki_models() -> List[str]:
    """List available Helsinki-NLP models for different language pairs"""
    # Common Helsinki-NLP models
    models = [
        "Helsinki-NLP/opus-mt-ar-en",  
        "Helsinki-NLP/opus-mt-en-ar",  
        "Helsinki-NLP/opus-mt-ar-fr",  
        "Helsinki-NLP/opus-mt-ar-de",  
        "Helsinki-NLP/opus-mt-ar-es",  
        "Helsinki-NLP/opus-mt-ar-it",  
        "Helsinki-NLP/opus-mt-ar-ru",  
    ]
    return models


def estimate_helsinki_performance(text_length: int, device: str = "cpu") -> dict:
    """Estimate processing performance for Helsinki MT"""
    # Rough estimates based on text length and device
    words = text_length // 5  # Estimate words from character count
    
    if device == "cuda":
        tokens_per_second = 50  # GPU speed
    else:
        tokens_per_second = 15  # CPU speed
    
    estimated_time = words / tokens_per_second
    
    return {
        "estimated_processing_time": estimated_time,
        "estimated_tokens_per_second": tokens_per_second,
        "estimated_words": words,
        "device": device,
        "notes": "Helsinki models are optimized for quality over speed"
    }


def test_helsinki_model_loading(model_name: str = "Helsinki-NLP/opus-mt-ar-en") -> bool:
    """Test if a Helsinki model can be loaded"""
    try:
        model = create_helsinki_mt(model_name)
        model.load_model()
        model.unload_model()
        return True
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return False 