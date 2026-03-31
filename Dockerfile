FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 🔥 Instalar PyTorch con CUDA REAL
RUN pip install --no-cache-dir --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Descargar modelos
RUN mkdir -p models && \
    gdown "1RUwVH5SmDtIkv7Oz1NEWy4PxDYU44av8" -O models/ncf_model.pth && \
    gdown "1XBvJxJJMCCVh14ozwlbkHPS79JtusfSn" -O models/encoders.pkl

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]