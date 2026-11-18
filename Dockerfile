# EchoGesture (Peratral) - air-gapped, reproducible container (CPU-only PyTorch keeps the image small).
# Build WITH network (downloads wheels once), then run with the network disabled:
#   docker build -t echogesture:airgapped .
#   docker run --rm --network none echogesture:airgapped pytest tests -q
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 espeak && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch first (the default PyPI wheel drags in several GB of CUDA libraries), then everything else.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && grep -viE '^torch==' requirements.txt > /tmp/req_rest.txt \
 && pip install --no-cache-dir -r /tmp/req_rest.txt
# MediaPipe's runtime needs the EGL/GLES shared libraries even on CPU (added after pip so the pip layer stays cached).
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
COPY config.yaml pytest.ini ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY assets ./assets
COPY weights ./weights
COPY outputs ./outputs
COPY data/vocab.json ./data/vocab.json
COPY data/test_manifest.csv data/train_manifest.csv data/val_manifest.csv ./data/
ENV OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
CMD ["python", "-c", "from src.predict import predict; print('usage: predict(<clip_dir_or_video>)')"]
