FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    HF_HOME=/models MODELS_DIR=/models
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
# torch с CUDA 12.8 (нужен для RTX 50xx); остальное — из requirements.txt
RUN pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py certs.py sanek.py speech_ru.py index.html ./
COPY voices ./voices
EXPOSE 8443
CMD ["python", "server.py"]
