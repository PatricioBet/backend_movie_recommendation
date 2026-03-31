FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos desde Google Drive
RUN pip install gdown && \
    mkdir -p models && \
    gdown "1RUwVH5SmDtIkv7Oz1NEWy4PxDYU44av8" -O models/ncf_model.pth && \
    gdown "1XBvJxJJMCCVh14ozwlbkHPS79JtusfSn" -O models/encoders.pkl

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]