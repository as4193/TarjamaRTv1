# TarjamaRTv1

Real-time translation app using Streamlit and OpenAI.

## Setup

1. **Clone repository**
```bash
git clone https://github.com/as4193/TarjamaRTv1.git
cd TarjamaRTv1
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Set OpenAI API key**
```bash
export OPENAI_API_KEY=your_openai_key_here
```

4. **Run vLLM with OpenAI**
```bash
docker pull vllm/vllm-openai:latest
docker-compose up -d
```

5. **Start Streamlit app**
```bash
streamlit run project_ui.py
```

6. **Open browser**
```bash
# Navigate to http://localhost:8501
```