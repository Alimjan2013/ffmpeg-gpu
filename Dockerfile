
# Dockerfile for GPU-accelerated ffmpeg + Python Flask server
# Requires running with NVIDIA runtime:
#   docker run --gpus all ...

FROM jrottenberg/ffmpeg:8.0-nvidia

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    MY_VARIABLE="This is a default value"

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        nvidia-utils-525 || apt-get install -y nvidia-utils-470 || true && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.12 -m venv $VIRTUAL_ENV

RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --upgrade pip && \
    pip install flask requests boto3 ffmpeg-python==0.2.0

# Optional: check GPU visibility at build time (for debugging only)
RUN nvidia-smi || echo "nvidia-smi not available at build time"

COPY server.py /app/server.py

EXPOSE 5000

ENTRYPOINT ["python3.12", "server.py"]
