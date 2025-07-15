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

# Check ASR/VAD modules
try:
    from asr import create_whisper_asr
    from vad import create_pyannote_vad
except ImportError as e:
    st.error(f"ASR/VAD modules not found: {e}")
    st.info("Make sure you're in the realtime_translation directory")
    st.stop()

st.set_page_config(
    page_title="Simple Speech Transcription",
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

class TranscriptionEngine:
    def __init__(self):
        self.vad_engine = None
        self.asr_engine = None
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

class RealTimeRecorder:
    def __init__(self, engine, language="auto"):
        self.engine = engine
        self.language = language
        self.recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.audio_stream = None
        self.audio_obj = None
        
    def start_recording(self):
        """Start real-time recording and transcription"""
        if self.recording:
            return
            
        self.recording = True
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Start audio recording thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
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
                            
                            self.result_queue.put({
                                "chunk": chunk_count,
                                "text": chunk_text,
                                "full_text": full_transcription.strip(),
                                "language": result.language,
                                "is_final": False
                            })
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
        self.result_queue.put({
            "is_final": True, 
            "full_text": full_transcription.strip(),
            "total_chunks": chunk_count
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
st.title("🎤 Simple Speech Transcription")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]
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
                    
                    # Create placeholders for streaming results
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    result_container = st.empty()
                    full_text_container = st.empty()
                    
                    start_time = time.time()
                    chunk_count = 0
                    
                    try:
                        for result in st.session_state.engine.transcribe_streaming(tmp.name, language):
                            if result.get("is_final"):
                                progress_bar.progress(1.0)
                                status_text.success("Transcription Complete!")
                                
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
                                
                                time.sleep(0.05)  # Small delay for better UX
                        
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
                
                # Create placeholders for streaming results
                progress_bar = st.progress(0)
                status_text = st.empty()
                result_container = st.empty()
                full_text_container = st.empty()
                
                start_time = time.time()
                chunk_count = 0
                
                try:
                    for result in st.session_state.engine.transcribe_streaming(audio_file, language):
                        if result.get("is_final"):
                            progress_bar.progress(1.0)
                            status_text.success("Transcription Complete!")
                            
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
                            
                            time.sleep(0.05)  # Small delay for better UX
                    
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
                st.session_state.recorder = RealTimeRecorder(st.session_state.engine, language)
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
    
    # Show real-time results
    if st.session_state.recording and st.session_state.recorder:
        st.subheader("Live Transcription")
        
        # Placeholders for results
        status_container = st.empty()
        current_chunk_container = st.empty()
        full_text_container = st.empty()
        
        # Auto-refresh to get new results
        placeholder = st.empty()
        
        while st.session_state.recording:
            results = st.session_state.recorder.get_results()
            
            for result in results:
                if result.get("error"):
                    st.error(result["error"])
                elif result.get("is_final"):
                    status_container.success("Recording Complete!")
                    if result.get("full_text"):
                        full_text_container.success(f"**Final Transcription:** {result['full_text']}")
                    if result.get("total_chunks"):
                        st.info(f"Processed {result['total_chunks']} audio chunks")
                    break
                else:
                    # Live results
                    chunk_num = result.get("chunk", 0)
                    status_container.info(f"Processing chunk {chunk_num}...")
                    
                    if result.get("text") and result["text"] != "(no speech)":
                        current_chunk_container.info(f"**Chunk {chunk_num}:** {result['text']}")
                        
                        if result.get("full_text"):
                            full_text_container.success(f"**Live Transcription:** {result['full_text']}")
                    else:
                        current_chunk_container.warning(f"Chunk {chunk_num}: (no speech detected)")
            
            # Sleep briefly before checking again
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
                st.success(f"**Complete Transcription:** {result['full_text']}")
                if result.get("total_chunks"):
                    st.info(f"Total chunks processed: {result['total_chunks']}")

st.markdown("---")
st.markdown("**Simple Speech Transcription** | VAD + ASR") 