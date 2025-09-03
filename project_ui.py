import streamlit as st
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from typing import Dict, List, Optional
import threading
from collections import deque
from openai import OpenAI

from realtime_translation.config import get_config, MTConfig
from realtime_translation.vad import create_pyannote_vad
from realtime_translation.asr import create_whisper_asr
from realtime_translation.mt import create_vllm_mt
from realtime_translation.asr.context_aware_prompt import build_smart_context_prompt
from realtime_translation.asr.language_mapping import map_language_to_iso, map_language_to_full, LANGUAGES


# Get configuration
config = get_config()

# Configuration constants
AUDIO_CHUNK_DURATION = config.streaming.chunk_size if hasattr(config, 'streaming') else 2.0
AUDIO_OVERLAP_DURATION = config.streaming.overlap_ratio if hasattr(config, 'streaming') else 0.5
AUDIO_STEP_SIZE = 0.1
AUDIO_SILENCE_THRESHOLD = 0.01
AUDIO_SAMPLE_RATE = config.audio.sample_rate if hasattr(config, 'audio') else 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = np.float32
AUDIO_PROCESSING_SLEEP = 0.1  
AUDIO_QUEUE_SLEEP = 0.5     


# UI Configuration
UI_MAX_SEGMENTS = 100
UI_COLUMN_LAYOUTS = {
    'language_select': [1, 1],
    'recording_controls': [1, 2, 1],
    'asr_toggle': [1, 2, 1],
    'file_upload': [1, 2, 1],
    'results_display': [1, 1]
}

# Language options
SUPPORTED_LANGUAGES = list(LANGUAGES.values())

# ASR Correction settings
ASR_CORRECTION_SIMILARITY_THRESHOLD = 0.5


class LiveTranslationSystem:
    """
    Real-time audio translation system with configurable ASR correction
    and dynamic audio processing strategies.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Core engines
        self.vad_engine = None
        self.asr_engine = None
        self.mt_engine = None
        
        # Audio configuration
        self.sample_rate = self.config.audio.sample_rate
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Translation state
        self.current_segments = deque(maxlen=UI_MAX_SEGMENTS)
        self.current_source_lang = "arabic"
        self.current_target_lang = "english"
        
        # System state
        self.models_loaded = False
        self.asr_correction_enabled = True
        
        # File processing state
        self.file_processing = False
        self.file_audio_data = None
        self.file_processing_completed = False
        self.file_processing_time = 0
        self.file_processing_chunks = 0
        self.file_processing_start_time = 0
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize VAD, ASR, and MT engines with proper error handling."""
        try:
            with st.spinner("Loading VAD model..."):
                self.vad_engine = create_pyannote_vad(config=self.config.vad)
                self.vad_engine.load_model()
            
            with st.spinner("Loading ASR model..."):
                self.asr_engine = create_whisper_asr(config=self.config.asr)
                self.asr_engine.load_model()
            
            with st.spinner("Initializing MT engine..."):
                mt_config = MTConfig(
                    model=self.config.mt.model,
                    host=self.config.mt.host,
                    port=self.config.mt.port,
                    source_language="arabic",
                    target_language="english",
                    max_tokens=self.config.mt.max_tokens,
                    temperature=self.config.mt.temperature,
                    top_p=self.config.mt.top_p,
                    top_k=self.config.mt.top_k,
                    min_p=self.config.mt.min_p,
                    timeout=self.config.mt.timeout,
                    max_retries=self.config.mt.max_retries
                )
                self.mt_engine = create_vllm_mt(mt_config)
                
            st.success("✅ All models loaded successfully!")
            self.models_loaded = True
            
        except Exception as e:
            st.error(f"Engine initialization failed: {str(e)}")
            print(f"Engine initialization error: {e}")
            self.models_loaded = False
    
    def start_recording(self, source_lang: str, target_lang: str) -> bool:
        """Start the recording session with specified language pair."""
        if not self.models_loaded:
            st.error("Models not yet loaded. Please wait for initialization to complete.")
            return False
        
        if not all([self.vad_engine, self.asr_engine, self.mt_engine]):
            st.error("Engines not properly initialized")
            return False
        
        try:
            mt_config = MTConfig(
                model=self.config.mt.model,
                host=self.config.mt.host,
                port=self.config.mt.port,
                source_language=source_lang,
                target_language=target_lang,
                max_tokens=self.config.mt.max_tokens,
                temperature=self.config.mt.temperature,
                top_p=self.config.mt.top_p,
                top_k=self.config.mt.top_k,
                min_p=self.config.mt.min_p,
                timeout=self.config.mt.timeout,
                max_retries=self.config.mt.max_retries
            )
            self.mt_engine = create_vllm_mt(mt_config)
        except Exception as e:
            st.error(f"Failed to update MT engine: {e}")
            return False
        
        self.current_source_lang = source_lang
        self.current_target_lang = target_lang
        self.current_segments.clear()
        self.file_processing_completed = False
        self.is_recording = True
        return True
    
    def stop_recording(self):
        """Stop recording and clear all pending audio data."""
        self.is_recording = False
        self.audio_queue.queue.clear()
        self.current_segments.clear()
        print("Recording stopped - all audio data cleared")
    
    def _get_audio_processing_config(self):
        """Get audio processing configuration based on ASR correction status."""
        if self.asr_correction_enabled:
            return {
                'chunk_samples': int(self.sample_rate * AUDIO_CHUNK_DURATION),
                'step_samples': int(self.sample_rate * AUDIO_STEP_SIZE),
                'overlap_samples': int(self.sample_rate * AUDIO_OVERLAP_DURATION)
            }
        else:
            return {
                'chunk_samples': int(self.sample_rate * AUDIO_CHUNK_DURATION),
                'step_samples': int(self.sample_rate * AUDIO_STEP_SIZE),
                'overlap_samples': int(self.sample_rate * AUDIO_OVERLAP_DURATION)
            }
    
    def audio_recording_loop(self):
        """Main audio recording loop with dynamic chunk processing."""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=AUDIO_CHANNELS, 
                dtype=AUDIO_DTYPE
            ) as stream:
                chunk_count = 0
                audio_buffer = np.array([], dtype=AUDIO_DTYPE)
                
                while self.is_recording:
                    config = self._get_audio_processing_config()
                    
                    # Read audio step
                    audio_step, _ = stream.read(config['step_samples'])
                    if len(audio_step) > 0:
                        audio_buffer = np.concatenate([audio_buffer, audio_step.flatten()])
                        
                        if len(audio_buffer) >= config['chunk_samples']:
                            audio_chunk = audio_buffer[:config['chunk_samples']]
                            
                            if config['overlap_samples'] > 0:
                                audio_buffer = audio_buffer[config['chunk_samples'] - config['overlap_samples']:]
                            else:
                                audio_buffer = np.array([], dtype=AUDIO_DTYPE)
                            
                            chunk_count += 1
                            max_amplitude = np.max(np.abs(audio_chunk))
                            
                            overlap_info = "no overlap" if self.asr_correction_enabled else f"{config['overlap_samples']/self.sample_rate:.1f}s overlap"
                            print(f"Chunk {chunk_count}: {len(audio_chunk)} samples, amplitude: {max_amplitude:.4f}, {overlap_info}")
                            
                            self.audio_queue.put(audio_chunk)
                    
                    time.sleep(AUDIO_PROCESSING_SLEEP)
                    
        except Exception as e:
            st.error(f"Audio recording error: {str(e)}")
    
    def audio_processing_loop(self):
        """Process audio chunks from the queue."""
        processed_count = 0
        
        while self.is_recording:
            try:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()
                    processed_count += 1
                    
                    self._process_audio_chunk(audio_chunk)
                    
                time.sleep(AUDIO_QUEUE_SLEEP)
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process a single audio chunk through the VAD-ASR-MT pipeline."""
        try:
            vad_result = self.vad_engine.detect_segments(audio_chunk)
            speech_segments = vad_result.get_speech_segments()
            
            if not speech_segments:
                return
            
            for i, segment in enumerate(speech_segments):
                start_sample = int(segment.start * self.sample_rate)
                end_sample = int(segment.end * self.sample_rate)
                segment_audio = audio_chunk[start_sample:end_sample]
                
                max_amplitude = np.max(np.abs(segment_audio))
                if max_amplitude < AUDIO_SILENCE_THRESHOLD:
                    continue
                
                self._process_speech_segment(segment_audio)
                    
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
    
    def _process_speech_segment(self, segment_audio: np.ndarray):
        """Process a single speech segment through ASR and MT."""
        temp_file_path = None
        try:
            # Save segment to temporary file
            temp_file_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_file_path, segment_audio, self.sample_rate)
            
            # ASR transcription
            iso_language = map_language_to_iso(self.current_source_lang)
            asr_result = self.asr_engine.transcribe(temp_file_path, language=iso_language)
            
            if asr_result and asr_result.get_full_text().strip():
                original_text = asr_result.get_full_text().strip()
                
                # Apply ASR correction if enabled
                if self.asr_correction_enabled:
                    corrected_text = self._correct_asr_text(original_text)
                    similarity_score = self._calculate_text_similarity(original_text, corrected_text)
                    final_text = corrected_text if similarity_score >= ASR_CORRECTION_SIMILARITY_THRESHOLD else original_text
                    
                    # Debug: Show what's happening with ASR correction
                    print(f"ASR Correction Debug:")
                    print(f"  Original: '{original_text}'")
                    print(f"  Corrected: '{corrected_text}'")
                    print(f"  Similarity: {similarity_score:.3f} (threshold: {ASR_CORRECTION_SIMILARITY_THRESHOLD})")
                    print(f"  Using: {'CORRECTED' if similarity_score >= ASR_CORRECTION_SIMILARITY_THRESHOLD else 'ORIGINAL'}")
                else:
                    final_text = original_text
                
                # Translation
                translation = self._translate_text(final_text)
                
                # Store result
                result = {
                    'original': final_text, 
                    'translation': translation,
                    'timestamp': time.time()
                }
                
                self.current_segments.append(result)
                print(f"Processed: {original_text[:50]}... → {translation[:50]}...")
                
        except Exception as e:
            st.error(f"Error processing speech segment: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_file_path}: {e}")
    
    def _correct_asr_text(self, text: str) -> str:
        """Apply ASR text correction using OpenAI API."""
        try:
            full_language = map_language_to_full(self.current_source_lang)
            prompt = build_smart_context_prompt(
                target_text=text,
                language=full_language,
                chunk_before=self._get_context_before(),
                chunk_after=None
            )
            
            # Get API key from environment variable or config
            env_key = os.getenv('OPENAI_API_KEY')
            config_key = self.config.openai.api_key
            api_key = (env_key or config_key or "").strip()
            
            
            if not api_key:
                raise ValueError("OpenAI API key not found. Set it in config.yaml or OPENAI_API_KEY environment variable.")
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are an ASR text correction expert. Fix text by removing duplicate words and correcting spelling mistakes. NEVER add new words or combine text chunks. ONLY output the corrected target text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.openai.max_tokens,
                temperature=self.config.openai.temperature,
                top_p=self.config.openai.top_p,
                frequency_penalty=self.config.openai.frequency_penalty,
                presence_penalty=self.config.openai.presence_penalty
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            return text
                
        except Exception as e:
            st.error(f"ASR correction error: {str(e)}")
            return text
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        try:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def _translate_text(self, text: str) -> str:
        """Translate text using the MT engine."""
        try:
            mt_result = self.mt_engine.translate(
                text,
                source_language=self.current_source_lang,
                target_language=self.current_target_lang
            )
            
            if mt_result and mt_result.translated_text:
                return mt_result.translated_text
            return "Translation failed"
                
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return f"Translation error: {str(e)}"
    
    def _get_context_before(self) -> Optional[str]:
        """Get the previous segment's text for context-aware correction."""
        if len(self.current_segments) > 0:
            return self.current_segments[-1]['original']
        return None
    
    
    def get_results(self) -> List[Dict]:
        return list(self.current_segments)
    
    def toggle_asr_correction(self):
        self.asr_correction_enabled = not self.asr_correction_enabled
    
    def process_uploaded_file_from_bytes(self, file_data: bytes, filename: str, source_lang: str, target_lang: str) -> bool:
        """Process uploaded file data through the VAD-ASR-MT pipeline using background processing like live mic."""
        
        if not self.models_loaded:
            st.error("Models not yet loaded. Please wait for initialization to complete.")
            return False
        
        if not all([self.vad_engine, self.asr_engine, self.mt_engine]):
            st.error("Engines not properly initialized")
            return False
        
        try:
            mt_config = MTConfig(
                model=self.config.mt.model,
                host=self.config.mt.host,
                port=self.config.mt.port,
                source_language=source_lang,
                target_language=target_lang,
                max_tokens=self.config.mt.max_tokens,
                temperature=self.config.mt.temperature,
                top_p=self.config.mt.top_p,
                top_k=self.config.mt.top_k,
                min_p=self.config.mt.min_p,
                timeout=self.config.mt.timeout,
                max_retries=self.config.mt.max_retries
            )
            self.mt_engine = create_vllm_mt(mt_config)
            
            # Update current languages
            self.current_source_lang = source_lang
            self.current_target_lang = target_lang
            
            # Clear previous results and reset completion status
            self.current_segments.clear()
            self.file_processing_completed = False
            
            # Load and preprocess audio
            file_ext = os.path.splitext(filename)[1].lower()
            
            temp_original_path = tempfile.mktemp(suffix=file_ext)
            with open(temp_original_path, 'wb') as f:
                f.write(file_data)
            
            # Convert to WAV if needed
            temp_wav_path = tempfile.mktemp(suffix='.wav')
            
            try:
                if file_ext in ['.m4a', '.mp4', '.mp3', '.flac', '.ogg']:
                    try:
                        from pydub import AudioSegment
                        st.info(f"🔄 Converting {file_ext.upper()} to WAV format...")
                        print(f"Converting {file_ext} to WAV...")
                        audio = AudioSegment.from_file(temp_original_path)
                        audio.export(temp_wav_path, format="wav")
                        audio_data, sample_rate = sf.read(temp_wav_path)
                        st.success("✅ Conversion completed!")
                    except ImportError:
                        st.error("❌ pydub not installed. Please install: pip install pydub")
                        return False
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {str(e)}")
                        return False
                else:
                    audio_data, sample_rate = sf.read(temp_original_path)
                
                # Ensure mono audio 
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if necessary
                if sample_rate != self.sample_rate:
                    try:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
                    except ImportError:
                        import scipy.signal
                        resample_ratio = self.sample_rate / sample_rate
                        new_length = int(len(audio_data) * resample_ratio)
                        audio_data = scipy.signal.resample(audio_data, new_length).astype(np.float32)
                
                audio_data = audio_data.astype(np.float32)
                
                self.file_audio_data = audio_data
                self.file_processing = True
                self.file_processing_start_time = time.time()
                
                self._start_file_processing_thread()
                
                conversion_msg = f" (converted from {file_ext.upper()})" if file_ext in ['.m4a', '.mp4', '.mp3', '.flac', '.ogg'] else ""
                st.success(f"🔄 Started processing {filename}{conversion_msg} - {len(audio_data) / self.sample_rate:.1f}s audio")
                return True
                
            except Exception as e:
                st.error(f"Error loading audio file: {str(e)}")
                return False
                
            finally:
                # Clean up temporary files
                for temp_path in [temp_original_path, temp_wav_path]:
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            print(f"Warning: Could not delete temp file {temp_path}: {e}")
        
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
            return False
    
    def _start_file_processing_thread(self):
        """Start background thread for file processing - works like recording thread."""
        if hasattr(self, 'file_processing_thread') and self.file_processing_thread.is_alive():
            return
            
        self.file_processing_thread = threading.Thread(
            target=self._file_processing_loop,
            daemon=True
        )
        self.file_processing_thread.start()
    
    def _file_processing_loop(self):
        """Background file processing loop - mimics audio_processing_loop."""
        
        try:
            # Get audio processing configuration
            config = self._get_audio_processing_config()
            chunk_samples = config['chunk_samples']
            overlap_samples = config['overlap_samples']
            
            # Process the file in chunks
            chunk_count = 0
            audio_buffer = self.file_audio_data.copy()
            
            while len(audio_buffer) >= chunk_samples and self.file_processing:
                chunk_count += 1
                
                # Extract chunk
                audio_chunk = audio_buffer[:chunk_samples]
                

                
                self._process_audio_chunk(audio_chunk)
                
                # Handle overlap for next chunk
                if overlap_samples > 0:
                    step_size = chunk_samples - overlap_samples
                    audio_buffer = audio_buffer[step_size:]
                else:
                    audio_buffer = audio_buffer[chunk_samples:]
                
                time.sleep(0.5)
            
            # Process remaining audio
            if len(audio_buffer) > 0 and len(audio_buffer) >= self.sample_rate * 0.5 and self.file_processing:
                chunk_count += 1
                self._process_audio_chunk(audio_buffer)
            
            # Calculate total processing time
            end_time = time.time()
            total_time = end_time - self.file_processing_start_time
    
            
            # Store completion info for UI display
            self.file_processing_completed = True
            self.file_processing_time = total_time
            self.file_processing_chunks = chunk_count
            self.file_processing = False
            
        except Exception as e:
            st.error(f"File processing error: {str(e)}")
            self.file_processing = False
    

def main():
    st.set_page_config(
        page_title="TarjamaRTv1",
        page_icon="🎤",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("TarjamaRTv1")
    
    # Initialize system
    if 'translation_system' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.translation_system = LiveTranslationSystem()
    
    system = st.session_state.translation_system
    
    # Language selection
    _render_language_selection(system)
    
    # Recording controls
    _render_recording_controls(system)
    
    # ASR Correction toggle
    _render_asr_toggle(system)
    
    st.divider()
    
    # File upload section
    _render_file_upload(system)
    
    st.divider()
    
    # System status
    _render_system_status(system)
    
    st.divider()
    
    _render_translation_results(system)
    
    if ('is_recording' in st.session_state and st.session_state.is_recording) or \
       (hasattr(system, 'file_processing') and system.file_processing):
        time.sleep(1.0)  
        st.rerun()


def _render_language_selection(system):
    """Render language selection controls."""
    col1, col2 = st.columns(UI_COLUMN_LAYOUTS['language_select'])
    
    with col1:
        source_lang = st.selectbox(
            "Source language",
            SUPPORTED_LANGUAGES,
            index=SUPPORTED_LANGUAGES.index(system.current_source_lang)
        )
    
    with col2:
        target_lang = st.selectbox(
            "Target language",
            SUPPORTED_LANGUAGES,
            index=SUPPORTED_LANGUAGES.index(system.current_target_lang)
        )
    
    # Store in session state for access by other functions
    st.session_state.current_source_lang = source_lang
    st.session_state.current_target_lang = target_lang


def _render_recording_controls(system):
    """Render recording start/stop controls."""
    col1, col2, col3 = st.columns(UI_COLUMN_LAYOUTS['recording_controls'])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not system.models_loaded:
            st.button("🎤", disabled=True, use_container_width=True)
        elif 'is_recording' in st.session_state and st.session_state.is_recording:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                _stop_recording(system)
        else:
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                _start_recording(system)


def _render_asr_toggle(system):
    """Render ASR correction toggle control."""
    col1, col2, col3 = st.columns(UI_COLUMN_LAYOUTS['asr_toggle'])
    
    with col2:
        asr_status = "🟢 ASR Correction: Enabled" if system.asr_correction_enabled else "🔴 ASR Correction: Disabled"
        if st.button(asr_status, use_container_width=True):
            system.toggle_asr_correction()
            st.rerun()


def _render_system_status(system):
    """Render system status information."""
    if system.models_loaded:
        if 'is_recording' in st.session_state and st.session_state.is_recording:
            st.success("🟢 Recording... Speak now!")
        elif hasattr(system, 'file_processing') and system.file_processing:
            st.info("🔄 Processing uploaded file...")
        elif hasattr(system, 'file_processing_completed') and system.file_processing_completed:
            st.success(f"✅ File processing completed in {system.file_processing_time:.2f} seconds! "
                      f"({system.file_processing_chunks} chunks processed, "
                      f"{len(system.current_segments)} segments found)")
        else:
            st.info("⚪ Ready to record or upload file")
    else:
        st.warning("⏳ Loading models...")


def _render_translation_results(system):
    """Render translation results in a two-column layout."""
    current_segments = system.get_results()
    
    if current_segments:
        col1, col2 = st.columns(UI_COLUMN_LAYOUTS['results_display'])
        
        with col1:
            st.markdown("### 📝 Original Text")
            _render_text_segments(current_segments, 'original', 'blue')
        
        with col2:
            st.markdown("### 🌐 Translation")
            _render_text_segments(current_segments, 'translation', 'green')
    else:
        _render_empty_state()


def _render_text_segments(segments, field, color_scheme):
    """Render text segments with consistent styling and uniform box sizes."""
    color_configs = {
        'blue': {
            'background': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
            'border': '#4a90e2'
        },
        'green': {
            'background': 'linear-gradient(135deg, #2d5a27 0%, #4a7c59 100%)',
            'border': '#7cb342'
        }
    }
    
    config = color_configs[color_scheme]
    
    for i, result in enumerate(segments):
        text = result[field]
        
        system = st.session_state.translation_system
        
        is_arabic_text = False
        if field == 'original' and system.current_source_lang == 'arabic':
            is_arabic_text = True
        elif field == 'translation' and system.current_target_lang == 'arabic':
            is_arabic_text = True
        
        if is_arabic_text:
            if len(text) > 100:
                font_size = "22px"
            elif len(text) > 50:
                font_size = "26px"
            else:
                font_size = "30px"
        else:
            if len(text) > 100:
                font_size = "16px"
            elif len(text) > 50:
                font_size = "20px"
            else:
                font_size = "24px"
        
        with st.container():
            st.markdown(f"""
            <div style="
                background: {config['background']};
                border: 2px solid {config['border']};
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                color: white;
                font-weight: 500;
                height: 120px;
                display: flex;
                align-items: center;
                overflow: hidden;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    width: 100%;
                    height: 100%;
                ">
                    <strong style="color: #ffd700; font-size: 18px; margin-right: 10px;">{i+1}.</strong>
                    <div style="
                        font-size: {font_size};
                        line-height: 1.4;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        display: -webkit-box;
                        -webkit-line-clamp: 3;
                        -webkit-box-orient: vertical;
                    ">
                        {text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _render_empty_state():
    """Render empty state when no translations are available."""
    col1, col2 = st.columns(UI_COLUMN_LAYOUTS['results_display'])
    
    with col1:
        st.markdown("### 📝 Original Text")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            border: 2px solid #95a5a6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            color: #bdc3c7;
            font-style: italic;
        ">
            🎤 No audio processed yet
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌐 Translation")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            border: 2px solid #95a5a6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            color: #bdc3c7;
            font-style: italic;
        ">
            🌍 No translation yet
        </div>
        """, unsafe_allow_html=True)


def _start_recording(system):
    """Start recording with the current language selection."""
    source_lang = st.session_state.get('current_source_lang', system.current_source_lang)
    target_lang = st.session_state.get('current_target_lang', system.current_target_lang)
    
    if system.start_recording(source_lang, target_lang):
        st.session_state.is_recording = True
        
        # Start audio threads
        if 'audio_thread' not in st.session_state or not st.session_state.audio_thread.is_alive():
            st.session_state.audio_thread = threading.Thread(
                target=system.audio_recording_loop, 
                daemon=True
            )
            st.session_state.audio_thread.start()
        
        if 'processing_thread' not in st.session_state or not st.session_state.processing_thread.is_alive():
            st.session_state.processing_thread = threading.Thread(
                target=system.audio_processing_loop, 
                daemon=True
            )
            st.session_state.processing_thread.start()
        
        st.rerun()

def _stop_recording(system):
    """Stop recording and clean up threads."""
    system.stop_recording()
    st.session_state.is_recording = False
    
    if 'audio_thread' in st.session_state:
        del st.session_state.audio_thread
    if 'processing_thread' in st.session_state:
        del st.session_state.processing_thread
    
    st.rerun()


def _render_file_upload(system):
    """Render file upload section."""
    st.markdown("### 📁 Upload Audio File")
    
    col1, col2, col3 = st.columns(UI_COLUMN_LAYOUTS['file_upload'])
    
    with col2:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'mp4', 'flac', 'ogg', 'm4a'],
            help="Upload an audio file to process through the translation pipeline",
            key="audio_file_uploader"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.info(f"📎 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            button_disabled = not system.models_loaded
            button_text = "🚀 Process Audio File" if system.models_loaded else "⏳ Loading Models..."
            
            if st.button(
                button_text, 
                type="primary", 
                use_container_width=True, 
                disabled=button_disabled,
                key="process_audio_button"
            ):
                
                if not system.models_loaded:
                    st.error("Models not yet loaded. Please wait for initialization to complete.")
                else:
                    # Get current language selection
                    source_lang = st.session_state.get('current_source_lang', system.current_source_lang)
                    target_lang = st.session_state.get('current_target_lang', system.current_target_lang)
                    
                    
                    # Store file in session state to avoid read issues
                    if 'uploaded_file_data' not in st.session_state:
                        st.session_state.uploaded_file_data = uploaded_file.read()
                        st.session_state.uploaded_file_name = uploaded_file.name
                    
                    # Process the file
                    try:
                        success = system.process_uploaded_file_from_bytes(
                            st.session_state.uploaded_file_data,
                            st.session_state.uploaded_file_name,
                            source_lang, 
                            target_lang
                        )
                        
                        if success:
                            # Clear stored file data
                            if 'uploaded_file_data' in st.session_state:
                                del st.session_state.uploaded_file_data
                            if 'uploaded_file_name' in st.session_state:
                                del st.session_state.uploaded_file_name
                        else:
                            st.write("❌ Processing failed!")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        else:
            st.button(
                "📁 Upload an audio file first", 
                disabled=True, 
                use_container_width=True,
                key="upload_placeholder_button"
            )

if __name__ == "__main__":
    main()
