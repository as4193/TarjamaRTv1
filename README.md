# TarjamaRTv1

Real-time translation app using Streamlit and OpenAI.

## Requirements

Before setting up the project, ensure you have:

- **Python 3.11** (required)
- **GPU with 6-10 GB VRAM** (to run VAD + ASR + MT models)
- **OpenAI API Key** (required for ASR Correction)

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

7. **Start Streamlit app**
```bash
streamlit run project_ui.py
```

8. **Open browser**
```bash
# Navigate to http://localhost:8501
