# TarjamaRTv1

A multilingual real-time speech translation system supporting bidirectional translation between 99 languages with context-aware ASR correction.

## Key Features

- **99 Languages**: Bidirectional translation supporting 9,801 translation directions
- **Context-Aware Correction**: LLM-powered ASR error correction using temporal context to fix boundary artifacts
- **Voice Activity Detection**: Reduces hallucinations by removing silence segments
- **Optimized Streaming**: 2-second chunks with 0.5-second overlap for continuous processing
- **Modular Architecture**: Cascaded pipeline allowing component-wise optimization

## System Architecture

The system implements a 5-component pipeline:
1. **Voice Activity Detection** (pyannote/voice-activity-detection)
2. **Automatic Speech Recognition** (deepdml/faster-whisper-large-v3-turbo-ct2)
3. **Context-Aware ASR Correction** (GPT-4o-mini)
4. **Jaccard Validation** 
5. **Machine Translation** (vLLM-served cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit)

## Requirements

Before setting up the project, ensure you have:

- **Python 3.11** (required)
- **GPU with 6-10 GB VRAM** (to run VAD + ASR + MT models)
- **OpenAI API Key** (required for ASR Correction)
- **~80 GB free storage** (for models, dependencies, and Docker images)

**Note:** This project was developed and tested on Windows. You may encounter issues when running on other operating systems.

## Setup

1. **Clone repository**
```bash
git clone https://github.com/as4193/TarjamaRTv1.git
cd TarjamaRTv1
```

2. **Install PyTorch with CUDA support**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. **Install remaining requirements**
```bash
pip install -r requirements.txt
```

4. **Set OpenAI API key**

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_openai_key_here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=your_openai_key_here
```

5. **Login to Hugging Face (for gated models)**
```bash
huggingface-cli login
```
Enter your Hugging Face token when prompted. This is required for accessing gated models like Pyannote.

6. **Run vLLM with OpenAI**
```bash
docker pull vllm/vllm-openai:latest
docker-compose up -d
#You should be in vllm_service folder 
```
**Note:** This step may take 10-20 minutes depending on your internet speed, as the model will be downloaded from Hugging Face and then loaded into GPU.

7. **Start Streamlit app**
```bash
streamlit run project_ui.py
```

8. **Open browser**
```bash
# Navigate to http://localhost:8501
