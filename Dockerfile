# EchoGesture (Peratral) - air-gapped, reproducible container.
# NOT BUILT OR TESTED YET on this machine (no Docker installed). Build with network, then run with --network none.
#   docker build -t echogesture .
#   docker run --rm --network none -v "$PWD/data:/app/data" echogesture python scripts/benchmark.py --quick
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 espeak && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY weights ./weights
COPY data/vocab.json ./data/vocab.json
ENV OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
CMD ["python", "-c", "from src.predict import predict; print('usage: predict(<clip_dir_or_video>)')"]
