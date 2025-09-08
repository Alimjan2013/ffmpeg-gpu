

# Multi-stage Dockerfile: Python builder, ffmpeg from jrottenberg, CUDA runtime as final image

# Stage 1: Python builder
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Get ffmpeg with NVENC from jrottenberg image
FROM jrottenberg/ffmpeg:8.0-nvidia as ffmpeg

# Stage 3: Final runtime image
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
WORKDIR /app

# Copy Python environment
COPY --from=builder /venv /venv
COPY --from=builder /app /app

# Copy ffmpeg and ffprobe binaries
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /usr/local/bin/ffprobe /usr/local/bin/ffprobe

# (Optional) Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY server.py /app/server.py

EXPOSE 5000

ENTRYPOINT ["/venv/bin/python", "server.py"]
