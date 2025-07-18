import streamlit as st
import numpy as np
import time
import tempfile
import os
import warnings
import threading
import queue
warnings.filterwarnings("ignore")

# Fix LLVM/SVML issues
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check audio packages
try:
    import pyaudio
    import soundfile as sf
    from pydub import AudioSegment
    import yt_dlp
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.info("Run: pip install pyaudio soundfile pydub yt-dlp")
    st.stop()

# Check ASR/VAD/MT modules
try:
    from asr import create_whisper_asr
    from vad import create_pyannote_vad
    from mt import create_helsinki_mt
except ImportError as e:
    st.error(f"ASR/VAD/MT modules not found: {e}")
    st.info("Make sure you're in the realtime_translation directory")
    st.stop()

st.set_page_config(
    page_title="Speech Transcription & Translation",
    page_icon="🗣️",
    layout="wide"
)

# Language options
LANGUAGES = {
    "auto": "Auto-detect",
    "ar": "Arabic", 
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch"
}

# Translation target languages
TARGET_LANGUAGES = {
    "en": "English",
    "ar": "Arabic"
}

class TranscriptionEngine:
    def __init__(self):
        self.vad_engine = None
        self.asr_engine = None
        self.mt_engine_ar_en = None  # Arabic to English
        self.mt_engine_en_ar = None  # English to Arabic
        self.loaded = False
        
    def load_models(self):
        try:
            st.info("Loading VAD model...")
            self.vad_engine = create_pyannote_vad()
            if self.vad_engine:
                self.vad_engine.load_model()
            
            st.info("Loading ASR model...")
            self.asr_engine = create_whisper_asr()
            if self.asr_engine:
                self.asr_engine.load_model()
            
            st.info("Loading MT models...")
            # Load Arabic to English model
            self.mt_engine_ar_en = create_helsinki_mt("Helsinki-NLP/opus-mt-ar-en")
            if self.mt_engine_ar_en:
                self.mt_engine_ar_en.load_model()
            
            # Load English to Arabic model
            self.mt_engine_en_ar = create_helsinki_mt("Helsinki-NLP/opus-mt-en-ar")
            if self.mt_engine_en_ar:
                self.mt_engine_en_ar.load_model()
                
            self.loaded = True
            st.success("Models loaded!")
            return True
            
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return False
    
    def transcribe(self, audio_path, language="auto"):
        if not self.loaded:
            raise RuntimeError("Models not loaded")
        
        start_time = time.time()
        
        try:
            # VAD processing
            vad_result = None
            if self.vad_engine:
                vad_result = self.vad_engine.detect_segments(audio_path)
            
            # ASR processing
            asr_language = None if language == "auto" else language
            asr_result = self.asr_engine.transcribe(audio_path, language=asr_language)
            
            return {
                "text": asr_result.get_full_text(),
                "language": asr_result.language,
                "processing_time": time.time() - start_time,
                "speech_duration": vad_result.get_total_speech_duration() if vad_result else 0,
                "total_duration": vad_result.audio_duration if vad_result else 0
            }
            
        except Exception as e:
            if "LLVM" in str(e) or "svml" in str(e):
                st.error("LLVM/CPU compatibility issue detected")
                st.info("💡 Try restarting Python or use CPU-only mode")
                raise RuntimeError(f"LLVM Error: {e}")
            else:
                raise e
    
    def transcribe_streaming(self, audio_path, language="auto", chunk_duration=10.0):
        """Stream transcription results as chunks are processed"""
        if not self.loaded:
            raise RuntimeError("Models not loaded")
        
        try:
            # Use file chunking method if available
            if hasattr(self.asr_engine, 'transcribe_file_chunked'):
                asr_language = None if language == "auto" else language
                self.asr_engine.language = asr_language
                
                full_text = ""
                for chunk_result in self.asr_engine.transcribe_file_chunked(audio_path, chunk_duration):
                    chunk_text = chunk_result.get_full_text()
                    if chunk_text.strip():
                        full_text += " " + chunk_text.strip()
                        
                        yield {
                            "text": chunk_text.strip(),
                            "full_text": full_text.strip(),
                            "language": chunk_result.language,
                            "chunk_start": getattr(chunk_result, 'start_time', 0),
                            "chunk_end": getattr(chunk_result, 'end_time', 0),
                            "is_final": False
                        }
                
                # Final result
                yield {"is_final": True, "full_text": full_text.strip()}
            else:
                # Fallback to regular transcription
                result = self.transcribe(audio_path, language)
                yield result
                yield {"is_final": True}
                
        except Exception as e:
            if "LLVM" in str(e) or "svml" in str(e):
                st.error("LLVM/CPU compatibility issue detected")
                st.info("💡 Try restarting Python or use CPU-only mode")
                raise RuntimeError(f"LLVM Error: {e}")
            else:
                raise e
    
    def translate(self, text, target_language="en"):
        """Translate text to target language"""
        if not self.loaded:
            raise RuntimeError("Models not loaded")
        
        if not text.strip():
            return {"translated_text": "", "processing_time": 0.0, "confidence": 0.0}
        
        start_time = time.time()
        
        try:
            # Select appropriate MT engine based on target language
            if target_language == "en":
                # Translate to English (Arabic → English)
                mt_engine = self.mt_engine_ar_en
            elif target_language == "ar":
                # Translate to Arabic (English → Arabic)
                mt_engine = self.mt_engine_en_ar
            else:
                # Default to Arabic → English
                mt_engine = self.mt_engine_ar_en
            
            if mt_engine:
                result = mt_engine.translate(text)
                return {
                    "translated_text": result.translated_text,
                    "processing_time": time.time() - start_time,
                    "confidence": result.confidence or 0.0,
                    "source_language": result.source_language,
                    "target_language": result.target_language
                }
            else:
                return {"translated_text": text, "processing_time": 0.0, "confidence": 0.0}
                
        except Exception as e:
            st.error(f"Translation failed: {e}")
            return {"translated_text": text, "processing_time": 0.0, "confidence": 0.0}

class RealTimeRecorder:
    def __init__(self, engine, language="auto", enable_translation=False, target_language="en"):
        self.engine = engine
        self.language = language
        self.enable_translation = enable_translation
        self.target_language = target_language
        self.recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.translation_queue = queue.Queue()  # New queue for translation tasks
        self.audio_stream = None
        self.audio_obj = None
        
    def start_recording(self):
        """Start real-time recording and transcription"""
        if self.recording:
            return
            
        self.recording = True
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        
        # Start audio recording thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        # Start translation thread if enabled
        if self.enable_translation:
            self.translation_thread = threading.Thread(target=self._translation_worker)
            self.translation_thread.daemon = True
            self.translation_thread.start()
    
    def stop_recording(self):
        """Stop recording and transcription"""
        self.recording = False
        
        # Clean up audio stream
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        if self.audio_obj:
            try:
                self.audio_obj.terminate()
            except:
                pass
    
    def _recording_worker(self):
        """Worker thread for audio recording"""
        try:
            self.audio_obj = pyaudio.PyAudio()
            self.audio_stream = self.audio_obj.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            chunk_size = 16000 * 2  
            current_chunk = []
            
            while self.recording:
                try:
                    data = self.audio_stream.read(1024, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    current_chunk.extend(audio_data)
                    
                    # Process chunk when it reaches target size
                    if len(current_chunk) >= chunk_size:
                        # Convert to float32 properly to avoid ONNX errors
                        audio_float = np.array(current_chunk, dtype=np.float32) / 32768.0
                        self.audio_queue.put(audio_float)
                        current_chunk = current_chunk[-int(chunk_size * 0.2):]  # Keep 20% overlap
                        
                except Exception as e:
                    if self.recording:  # Only log if we're still supposed to be recording
                        print(f"Recording error: {e}")
                    break
            
            # Process final chunk if exists
            if current_chunk and self.recording:
                audio_float = np.array(current_chunk, dtype=np.float32) / 32768.0
                self.audio_queue.put(audio_float)
                
        except Exception as e:
            self.result_queue.put({"error": f"Recording failed: {e}"})
        finally:
            self.stop_recording()
    
    def _transcription_worker(self):
        """Worker thread for transcription"""
        chunk_count = 0
        full_transcription = ""
        full_translation = ""
        
        while self.recording or not self.audio_queue.empty():
            try:
                # Get audio chunk with timeout
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Skip very short chunks (reduced threshold for microphone)
                if len(audio_chunk) < 4800:  # Less than 0.3 seconds
                    continue
                
                chunk_count += 1
                
                # Create temporary file with proper float32 format
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    try:
                        # Analyze audio levels for debugging
                        audio_rms = np.sqrt(np.mean(audio_chunk**2))
                        audio_max = np.max(np.abs(audio_chunk))
                        
                        # Boost quiet microphone audio (common issue)
                        if audio_max > 0:
                            # Normalize to use more of the dynamic range
                            boost_factor = min(0.8 / audio_max, 10.0)  # Cap boost at 10x
                            audio_boosted = audio_chunk * boost_factor
                        else:
                            audio_boosted = audio_chunk
                        
                        # Ensure audio is float32 and in correct range
                        audio_normalized = np.clip(audio_boosted, -1.0, 1.0).astype(np.float32)
                        sf.write(tmp.name, audio_normalized, 16000)
                        
                        # Debug info (only for first few chunks)
                        if chunk_count <= 3:
                            print(f"Chunk {chunk_count}: RMS={audio_rms:.4f}, Max={audio_max:.4f}, Boost={boost_factor:.2f}x")
                        
                        # Transcribe chunk
                        asr_language = None if self.language == "auto" else self.language
                        result = self.engine.asr_engine.transcribe(tmp.name, language=asr_language)
                        
                        if result and result.get_full_text().strip():
                            chunk_text = result.get_full_text().strip()
                            full_transcription += " " + chunk_text
                            
                            # Prepare result with transcription
                            result_data = {
                                "chunk": chunk_count,
                                "text": chunk_text,
                                "full_text": full_transcription.strip(),
                                "language": result.language,
                                "is_final": False,
                                "type": "transcription"
                            }
                            
                            # Send to translation queue if enabled (non-blocking)
                            if self.enable_translation and chunk_text:
                                try:
                                    self.translation_queue.put({
                                        "chunk": chunk_count,
                                        "text": chunk_text,
                                        "full_text": full_transcription.strip()
                                    })
                                except:
                                    pass  # Don't block ASR if translation queue is full
                            
                            self.result_queue.put(result_data)
                        else:
                            # Debug info for first few chunks
                            debug_msg = "(no speech)"
                            if chunk_count <= 3:
                                debug_msg += f" - chunk_len={len(audio_chunk)}, max_val={audio_max:.4f}"
                            
                            self.result_queue.put({
                                "chunk": chunk_count,
                                "text": debug_msg,
                                "full_text": full_transcription.strip(),
                                "is_final": False
                            })
                            
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
                        
            except Exception as e:
                self.result_queue.put({"error": f"Transcription error: {e}"})
        
        # Final result
        final_result = {
            "is_final": True, 
            "full_text": full_transcription.strip(),
            "total_chunks": chunk_count,
            "type": "transcription"
        }
        
        # Send final text to translation queue if enabled
        if self.enable_translation and full_transcription.strip():
            try:
                self.translation_queue.put({
                    "chunk": -1,  # Final chunk indicator
                    "text": full_transcription.strip(),
                    "full_text": full_transcription.strip(),
                    "is_final": True
                })
            except:
                pass
        
        self.result_queue.put(final_result)
    
    def _translation_worker(self):
        """Worker thread for translation (independent of ASR)"""
        full_translation = ""
        
        while self.recording or not self.translation_queue.empty():
            try:
                # Get translation task with timeout
                try:
                    task = self.translation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                chunk_num = task["chunk"]
                text = task["text"]
                is_final = task.get("is_final", False)
                
                if not text.strip():
                    continue
                
                try:
                    # Translate the text
                    translation_result = self.engine.translate(text, self.target_language)
                    
                    if translation_result["translated_text"]:
                        if chunk_num != -1:  # Regular chunk
                            full_translation += " " + translation_result["translated_text"]
                        else:  # Final chunk
                            full_translation = translation_result["translated_text"]
                        
                        # Prepare translation result
                        result_data = {
                            "chunk": chunk_num,
                            "translated_text": translation_result["translated_text"],
                            "full_translation": full_translation.strip(),
                            "translation_confidence": translation_result["confidence"],
                            "is_final": is_final,
                            "type": "translation"
                        }
                        
                        self.result_queue.put(result_data)
                    
                except Exception as e:
                    # Translation failed for this chunk
                    result_data = {
                        "chunk": chunk_num,
                        "translated_text": "",
                        "full_translation": full_translation.strip(),
                        "translation_confidence": 0.0,
                        "is_final": is_final,
                        "type": "translation",
                        "error": str(e)
                    }
                    self.result_queue.put(result_data)
                
            except Exception as e:
                self.result_queue.put({"error": f"Translation error: {e}", "type": "translation"})
        
        # Final translation result
        if full_translation.strip():
            self.result_queue.put({
                "final_translation": full_translation.strip(),
                "is_final": True,
                "type": "translation"
            })
    
    def get_results(self):
        """Get transcription results (non-blocking)"""
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results

def convert_audio(audio_file, target_path):
    """Convert audio file to proper format for ASR"""
    try:
        audio = AudioSegment.from_file(audio_file)
        # Convert to mono, 16kHz
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Export as wav with proper format
        audio.export(target_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        
        # Re-read and save as float32 to ensure compatibility
        audio_data, sr = sf.read(target_path)
        
        # Ensure float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure proper range [-1, 1]
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Re-save with correct format
        sf.write(target_path, audio_data, 16000, subtype='FLOAT')
        
        return True
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return False

def download_youtube(url):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # First try with ffmpeg
            opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
            }
            
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
                
                # Look for wav file
                for file in os.listdir(temp_dir):
                    if file.endswith('.wav'):
                        output_path = f"temp_youtube_{int(time.time())}.wav"
                        temp_file = os.path.join(temp_dir, file)
                        
                        # Convert to proper format
                        if convert_audio(temp_file, output_path):
                            return output_path
                        else:
                            return None
                        
            except Exception as ffmpeg_error:
                if "ffmpeg" in str(ffmpeg_error).lower():
                    st.warning("FFmpeg not found - trying fallback method...")
                    
                    # Fallback: download without conversion and convert with pydub
                    opts_fallback = {
                        'format': 'bestaudio[ext=m4a]/best[ext=mp4]/best',
                        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    }
                    
                    with yt_dlp.YoutubeDL(opts_fallback) as ydl:
                        ydl.download([url])
                    
                    # Convert any downloaded file to wav using pydub
                    for file in os.listdir(temp_dir):
                        if file.endswith(('.m4a', '.mp4', '.webm', '.ogg')):
                            input_path = os.path.join(temp_dir, file)
                            output_path = f"temp_youtube_{int(time.time())}.wav"
                            
                            try:
                                if convert_audio(input_path, output_path):
                                    return output_path
                            except PermissionError:
                                # File might be locked, wait a bit and try again
                                time.sleep(1)
                                try:
                                    if convert_audio(input_path, output_path):
                                        return output_path
                                except:
                                    continue
                else:
                    raise ffmpeg_error
            
            return None
            
    except Exception as e:
        st.error(f"YouTube download failed: {e}")
        if "ffmpeg" in str(e).lower():
            st.info("💡 To fix ffmpeg issues: See the troubleshooting section above")
        elif "process cannot access the file" in str(e).lower():
            st.info("💡 File access issue - try again in a few seconds")
        else:
            st.info("💡 Try a different YouTube URL or check your internet connection")
        return None

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = TranscriptionEngine()
    st.session_state.models_loaded = False

if 'recorder' not in st.session_state:
    st.session_state.recorder = None
    st.session_state.recording = False

# UI
st.title("🎤 Speech Transcription & Translation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    language = st.selectbox(
        "Input Language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]
    )
    
    st.subheader("Translation")
    enable_translation = st.checkbox("Enable Translation", value=False)
    
    target_language = st.selectbox(
        "Target Language",
        options=list(TARGET_LANGUAGES.keys()),
        format_func=lambda x: TARGET_LANGUAGES[x],
        disabled=not enable_translation
    )
    
    st.subheader("Models")
    if not st.session_state.models_loaded:
        if st.button("Load Models"):
            st.session_state.models_loaded = st.session_state.engine.load_models()
    else:
        st.success("Models ready")

# Main
if not st.session_state.models_loaded:
    st.warning("Load models first")
    st.stop()

# Input method
method = st.radio("Input:", ["Audio File", "YouTube Link", "Microphone (Real-time)"], horizontal=True)

if method == "Audio File":
    st.subheader("Upload Audio")
    
    file = st.file_uploader("Choose file", type=['wav', 'mp3', 'mp4', 'm4a', 'flac'])
    
    if file and st.button("Transcribe"):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            try:
                if convert_audio(file, tmp.name):
                    st.subheader("Transcription Results")
                    
                    # Placeholders for streaming results
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Side-by-Side columns for transcription and translation
                    if enable_translation:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Transcription")
                            result_container = st.empty()
                            full_text_container = st.empty()
                        with col2:
                            st.subheader(f"Translation ({TARGET_LANGUAGES[target_language]})")
                            translation_container = st.empty()
                            full_translation_container = st.empty()
                    else:
                        result_container = st.empty()
                        full_text_container = st.empty()
                        translation_container = None
                        full_translation_container = None
                    
                    start_time = time.time()
                    chunk_count = 0
                    full_translation = ""
                    
                    try:
                        final_transcription = ""
                        for result in st.session_state.engine.transcribe_streaming(tmp.name, language):
                            if result.get("is_final"):
                                progress_bar.progress(1.0)
                                status_text.success("Processing Complete!")
                                final_transcription = result.get("full_text", "")
                                
                                # Show final stats
                                processing_time = time.time() - start_time
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Time", f"{processing_time:.2f}s")
                                with col2:
                                    st.metric("Chunks Processed", chunk_count)
                                
                                if result.get("language"):
                                    st.info(f"Language: {LANGUAGES.get(result['language'], result['language'])}")
                                break
                            else:
                                chunk_count += 1
                                progress = min(chunk_count * 0.1, 0.9)  # Estimate progress
                                progress_bar.progress(progress)
                                
                                status_text.text(f"Processing chunk {chunk_count}... ({result.get('chunk_start', 0):.1f}s)")
                                
                                if result.get("text"):
                                    # Show current chunk
                                    result_container.info(f"**Chunk {chunk_count}:** {result['text']}")
                                    
                                    # Show accumulated text
                                    if result.get("full_text"):
                                        full_text_container.success(f"**Full Text:** {result['full_text']}")
                                    
                                    # Translate each chunk in parallel if enabled
                                    if enable_translation and translation_container and full_translation_container:
                                        try:
                                            translation_result = st.session_state.engine.translate(result['text'], target_language)
                                            if translation_result["translated_text"]:
                                                translation_container.info(f"**Chunk {chunk_count}:** {translation_result['translated_text']}")
                                                full_translation += " " + translation_result['translated_text']
                                                full_translation_container.success(f"**Full Translation:** {full_translation.strip()}")
                                        except Exception as e:
                                            translation_container.warning(f"Translation failed for chunk {chunk_count}")
                                
                                time.sleep(0.05) 
                        
                        if chunk_count == 0:
                            st.warning("No speech detected in the audio file")
                            
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        
            finally:
                # Safe file cleanup
                try:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
                except Exception as cleanup_error:
                    print(f"Cleanup warning: {cleanup_error}")

elif method == "YouTube Link":
    st.subheader("YouTube Transcription")
    
    with st.expander("🔧 Having download issues?"):
        st.markdown("""
        **If you get ffmpeg errors, try these solutions:**
        
        **Option 1 - Quick Fix (Restart PowerShell as Admin):**
        1. Close this terminal
        2. Open PowerShell as Administrator (right-click → Run as administrator)
        3. Run: `choco install ffmpeg`
        4. Restart the UI
        
        **Option 2 - Manual Install:**
        1. Download from: https://www.gyan.dev/ffmpeg/builds/
        2. Extract to C:\\ffmpeg\\
        3. Add C:\\ffmpeg\\bin\\ to Windows PATH
        4. Restart terminal
        
        **Option 3 - The fallback method should work automatically (slower)**
        """)
    
    url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if url and st.button("Download & Transcribe"):
        with st.spinner("Downloading YouTube audio..."):
            audio_file = download_youtube(url)
            
            if audio_file:
                st.success("Downloaded successfully!")
                st.subheader("Transcription Results")
                
                # Placeholders for streaming results
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Side-by-Side columns for transcription and translation
                if enable_translation:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Transcription")
                        result_container = st.empty()
                        full_text_container = st.empty()
                    with col2:
                        st.subheader(f"Translation ({TARGET_LANGUAGES[target_language]})")
                        translation_container = st.empty()
                        full_translation_container = st.empty()
                else:
                    result_container = st.empty()
                    full_text_container = st.empty()
                    translation_container = None
                    full_translation_container = None
                
                start_time = time.time()
                chunk_count = 0
                full_translation = ""
                
                try:
                    final_transcription = ""
                    for result in st.session_state.engine.transcribe_streaming(audio_file, language):
                        if result.get("is_final"):
                            progress_bar.progress(1.0)
                            status_text.success("Processing Complete!")
                            final_transcription = result.get("full_text", "")
                            
                            # Show final stats
                            processing_time = time.time() - start_time
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Time", f"{processing_time:.2f}s")
                            with col2:
                                st.metric("Chunks Processed", chunk_count)
                            
                            if result.get("language"):
                                st.info(f"Language: {LANGUAGES.get(result['language'], result['language'])}")
                            break
                        else:
                            chunk_count += 1
                            progress = min(chunk_count * 0.1, 0.9)  # Estimate progress
                            progress_bar.progress(progress)
                            
                            status_text.text(f"Processing chunk {chunk_count}... ({result.get('chunk_start', 0):.1f}s)")
                            
                            if result.get("text"):
                                # Show current chunk
                                result_container.info(f"**Chunk {chunk_count}:** {result['text']}")
                                
                                # Show accumulated text
                                if result.get("full_text"):
                                    full_text_container.success(f"**Full Text:** {result['full_text']}")
                                
                                # Translate each chunk in parallel if enabled
                                if enable_translation and translation_container and full_translation_container:
                                    try:
                                        translation_result = st.session_state.engine.translate(result['text'], target_language)
                                        if translation_result["translated_text"]:
                                            translation_container.info(f"**Chunk {chunk_count}:** {translation_result['translated_text']}")
                                            full_translation += " " + translation_result['translated_text']
                                            full_translation_container.success(f"**Full Translation:** {full_translation.strip()}")
                                    except Exception as e:
                                        translation_container.warning(f"Translation failed for chunk {chunk_count}")
                            
                            time.sleep(0.05) 
                    
                    if chunk_count == 0:
                        st.warning("No speech detected in the YouTube video")
                        
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                
                # Safe file cleanup
                try:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                except Exception as cleanup_error:
                    print(f"Cleanup warning: {cleanup_error}")

elif method == "Microphone (Real-time)":
    st.subheader("Real-time Speech Transcription")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                st.session_state.recorder = RealTimeRecorder(
                    st.session_state.engine, 
                    language, 
                    enable_translation, 
                    target_language
                )
                st.session_state.recorder.start_recording()
                st.session_state.recording = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                if st.session_state.recorder:
                    st.session_state.recorder.stop_recording()
                st.session_state.recording = False
                st.rerun()
    
    with col2:
        if st.session_state.recording:
            st.success("🎤 Recording... Speak now!")
        else:
            st.info("Ready to record")
    
    # Real-Time results
    if st.session_state.recording and st.session_state.recorder:
        # Side-by-Side columns for transcription and translation
        if enable_translation:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Live Transcription")
                status_container = st.empty()
                current_chunk_container = st.empty()
                full_text_container = st.empty()
            with col2:
                st.subheader(f"Live Translation ({TARGET_LANGUAGES[target_language]})")
                current_translation_container = st.empty()
                full_translation_container = st.empty()
        else:
            st.subheader("Live Transcription")
            status_container = st.empty()
            current_chunk_container = st.empty()
            full_text_container = st.empty()
            current_translation_container = None
            full_translation_container = None
        
        while st.session_state.recording:
            results = st.session_state.recorder.get_results()
            
            for result in results:
                if result.get("error"):
                    st.error(result["error"])
                elif result.get("is_final"):
                    result_type = result.get("type", "transcription")
                    
                    if result_type == "transcription":
                        status_container.success("Recording Complete!")
                        if result.get("full_text"):
                            full_text_container.success(f"**Final Transcription:** {result['full_text']}")
                        if result.get("total_chunks"):
                            st.info(f"Processed {result['total_chunks']} audio chunks")
                    elif result_type == "translation" and enable_translation and full_translation_container:
                        if result.get("final_translation"):
                            full_translation_container.success(f"**Final Translation ({TARGET_LANGUAGES[target_language]}):** {result['final_translation']}")
                else:
                    # Live results
                    result_type = result.get("type", "transcription")
                    chunk_num = result.get("chunk", 0)
                    
                    if result_type == "transcription":
                        status_container.info(f"Processing chunk {chunk_num}...")
                        
                        if result.get("text") and result["text"] != "(no speech)":
                            current_chunk_container.info(f"**Chunk {chunk_num}:** {result['text']}")
                            
                            if result.get("full_text"):
                                full_text_container.success(f"**Live Transcription:** {result['full_text']}")
                        else:
                            current_chunk_container.warning(f"Chunk {chunk_num}: (no speech detected)")
                    
                    elif result_type == "translation" and enable_translation:
                        if result.get("translated_text") and current_translation_container:
                            current_translation_container.info(f"**Translation {chunk_num}:** {result['translated_text']}")
                            
                            # Show accumulated translation
                            if result.get("full_translation") and full_translation_container:
                                full_translation_container.success(f"**Full Translation:** {result['full_translation']}")
            
            time.sleep(0.5)
            
            # Check if recording stopped
            if not st.session_state.recording:
                break
    
    elif not st.session_state.recording and 'recorder' in st.session_state and st.session_state.recorder:
        # Show any remaining results after stopping
        st.subheader("Final Results")
        results = st.session_state.recorder.get_results()
        
        for result in results:
            if result.get("is_final") and result.get("full_text"):
                if enable_translation and result.get("final_translation"):
                    # Show side-by-side final results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Complete Transcription:** {result['full_text']}")
                    with col2:
                        st.success(f"**Complete Translation ({TARGET_LANGUAGES[target_language]}):** {result['final_translation']}")
                else:
                    st.success(f"**Complete Transcription:** {result['full_text']}")
                    
                if result.get("total_chunks"):
                    st.info(f"Total chunks processed: {result['total_chunks']}")

st.markdown("---")
st.markdown("**Speech Transcription & Translation** | VAD + ASR + MT") 