import streamlit as st
import numpy as np
import time
import tempfile
import os
import warnings
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

def record_microphone(duration):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        data = []
        for i in range(int(16000 / 1024 * duration)):
            chunk = stream.read(1024)
            data.append(np.frombuffer(chunk, dtype=np.int16))
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        return np.concatenate(data)
        
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return np.array([])

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
                        os.rename(os.path.join(temp_dir, file), output_path)
                        return output_path
                        
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
                                # Convert with pydub
                                audio = AudioSegment.from_file(input_path)
                                audio = audio.set_frame_rate(16000).set_channels(1)
                                audio.export(output_path, format="wav")
                                return output_path
                            except PermissionError:
                                # File might be locked, wait a bit and try again
                                time.sleep(1)
                                try:
                                    audio = AudioSegment.from_file(input_path)
                                    audio = audio.set_frame_rate(16000).set_channels(1)
                                    audio.export(output_path, format="wav")
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

def convert_audio(audio_file, target_path):
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(target_path, format="wav")
        return True
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return False

# Initialize
if 'engine' not in st.session_state:
    st.session_state.engine = TranscriptionEngine()
    st.session_state.models_loaded = False

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
method = st.radio("Input:", ["Audio File", "YouTube Link", "Microphone"], horizontal=True)

if method == "Audio File":
    st.subheader("Upload Audio")
    
    file = st.file_uploader("Choose file", type=['wav', 'mp3', 'mp4', 'm4a', 'flac'])
    
    if file and st.button("Transcribe"):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
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
                    
                os.unlink(tmp.name)

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
                
                os.unlink(audio_file)

elif method == "Microphone":
    st.subheader("Record Audio")
    
    duration = st.slider("Duration (seconds):", 5, 60, 10)
    
    if st.button("Start Recording"):
        st.subheader("Live Recording & Transcription")
        
        # Create placeholders for streaming results
        progress = st.progress(0)
        status = st.empty()
        current_text = st.empty()
        accumulated_text = st.empty()
        
        audio_data = []
        chunk_size = int(5 * 16000)  # 5-second chunks for transcription
        frames_per_chunk = int(chunk_size / 1024)
        total_chunks = int(16000 / 1024 * duration)
        transcribed_chunks = 0
        full_transcription = ""
        
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            for i in range(total_chunks):
                chunk = stream.read(1024)
                audio_data.append(np.frombuffer(chunk, dtype=np.int16))
                
                progress.progress((i + 1) / total_chunks)
                remaining = duration * (1 - (i + 1) / total_chunks)
                status.text(f"Recording... {remaining:.1f}s left")
                
                # Process chunks every 5 seconds
                if len(audio_data) >= frames_per_chunk or i == total_chunks - 1:
                    if len(audio_data) > 0:
                        try:
                            # Create temporary audio file for this chunk
                            chunk_audio = np.concatenate(audio_data)
                            
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                sf.write(tmp.name, chunk_audio.astype(np.float32) / 32768.0, 16000)
                                
                                transcribed_chunks += 1
                                current_text.info(f"Processing chunk {transcribed_chunks}...")
                                
                                # Quick transcription of current chunk
                                result = st.session_state.engine.transcribe(tmp.name, language)
                                
                                if result["text"]:
                                    chunk_text = result["text"].strip()
                                    full_transcription += " " + chunk_text
                                    
                                    current_text.success(f"**Chunk {transcribed_chunks}:** {chunk_text}")
                                    accumulated_text.info(f"**Full Text:** {full_transcription.strip()}")
                                else:
                                    current_text.warning(f"Chunk {transcribed_chunks}: (no speech)")
                                
                                os.unlink(tmp.name)
                                
                        except Exception as chunk_error:
                            current_text.error(f"Chunk processing error: {chunk_error}")
                        
                        # Reset for next chunk (keep some overlap)
                        overlap = int(len(audio_data) * 0.2)  # 20% overlap
                        audio_data = audio_data[-overlap:] if overlap > 0 else []
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            status.success("Recording Complete!")
            
            if not full_transcription.strip():
                st.warning("No speech detected during recording")
            else:
                st.success(f"**Final Transcription:** {full_transcription.strip()}")
                    
        except Exception as e:
            st.error(f"Recording failed: {e}")

st.markdown("---")
st.markdown("**Simple Speech Transcription** | VAD + ASR") 